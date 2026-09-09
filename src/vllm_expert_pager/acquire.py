"""Triton kernel that runs ``ExpertTable.acquire`` on both tables in one launch.

Used on the decode path (K <= S). The torch version (table.py) is the reference
implementation of the same rules and stays for prefill's ``lookup`` and for the
unit tests. Table contents and return values match the torch version
(tests/test_acquire.py).

The torch version launched about 40 small kernels per layer for each of the two
tables, which accounted for a third of one decode step.
"""

import torch
import triton
import triton.language as tl

from vllm_expert_pager.table import _PROTECT, ExpertTable

# Number of bits used to pack the slot number into the key. BLOCK_S <= 2**_SLOT_BITS.
_SLOT_BITS = 9
# Constants referenced from inside @triton.jit must be wrapped in tl.constexpr.
_PROTECT_C = tl.constexpr(_PROTECT)
_SLOT_BITS_C = tl.constexpr(_SLOT_BITS)


@triton.jit
def _acquire(
    topk_ptr,  # int[K]  expert ids of this step (with duplicates)
    K,
    slot_of_ptr,  # int64[E+1]  expert -> slot
    expert_in_ptr,  # int64[S+1]  slot -> expert
    last_used_ptr,  # int64[S+1]
    step_ptr,  # int64[]
    hits_ptr,  # int64[]
    misses_ptr,  # int64[]
    S,
    map_ptr,  # int32[E]  expert_map (when WRITE_MAP)
    slot_out_ptr,  # int64[E]  expert -> slot (not read for unneeded experts)
    todo_ptr,  # int64[K]  missed experts packed from the front, -1 after
    scratch_ptr,  # int64[BLOCK_S]
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_S: tl.constexpr,
    WRITE_MAP: tl.constexpr,
):
    e = tl.arange(0, BLOCK_E).to(tl.int64)
    e_ok = e < E
    s = tl.arange(0, BLOCK_S).to(tl.int64)
    s_ok = s < S

    # need[e]: e is needed in this step. used[s]: the expert in slot s is needed
    # (never evicted).
    occupant = tl.load(expert_in_ptr + s, mask=s_ok, other=-1)
    need = e < 0  # all False
    used = s < 0
    for k in range(K):
        t = tl.load(topk_ptr + k).to(tl.int64)
        need = need | (e == t)
        used = used | (occupant == t)
    need = need & e_ok
    used = used & (occupant >= 0)

    current = tl.load(slot_of_ptr + e, mask=e_ok, other=-1)
    hit = need & (current >= 0)
    miss = need & (current < 0)

    # Victim ordering: ascending last_used, slots in use at the end, ties by
    # slot number. Pack the slot number into the low bits before sorting so the
    # order matches the torch version's stable argsort.
    last = tl.load(last_used_ptr + s, mask=s_ok, other=0)
    key = last + used.to(tl.int64) * _PROTECT_C
    key = tl.where(s_ok, key, 2 * _PROTECT_C)  # padding slots go last
    order = tl.sort((key << _SLOT_BITS_C) | s, 0) & ((1 << _SLOT_BITS_C) - 1)
    tl.store(scratch_ptr + s, order)
    tl.debug_barrier()

    # Look up the victim by rank, the ordinal of the miss.
    rank = tl.cumsum(miss.to(tl.int64), 0) - 1
    victim = tl.load(scratch_ptr + tl.minimum(tl.maximum(rank, 0), S - 1))
    evicted = tl.load(expert_in_ptr + victim)

    tl.store(
        slot_of_ptr + evicted, tl.zeros_like(evicted) - 1, mask=miss & (evicted >= 0)
    )
    tl.debug_barrier()
    tl.store(slot_of_ptr + e, victim, mask=miss)
    tl.store(expert_in_ptr + victim, e, mask=miss)
    slot = tl.where(miss, victim, current)
    step = tl.load(step_ptr)
    tl.store(last_used_ptr + slot, tl.zeros_like(slot) + step, mask=need)
    tl.store(step_ptr, step + 1)
    tl.store(hits_ptr, tl.load(hits_ptr) + tl.sum(hit.to(tl.int64), 0))
    tl.store(misses_ptr, tl.load(misses_ptr) + tl.sum(miss.to(tl.int64), 0))

    tl.store(slot_out_ptr + e, slot, mask=e_ok)
    if WRITE_MAP:
        tl.store(map_ptr + e, tl.where(need, slot, -1).to(tl.int32), mask=e_ok)
    tl.store(todo_ptr + e, tl.zeros_like(e) - 1, mask=e < K)
    tl.debug_barrier()
    tl.store(todo_ptr + rank, e, mask=miss)


@triton.jit
def _acquire_pair_kernel(
    topk_ptr,
    K,
    a_slot_of,
    a_expert_in,
    a_last_used,
    a_step,
    a_hits,
    a_misses,
    a_S,
    a_map,
    a_slot_out,
    a_todo,
    b_slot_of,
    b_expert_in,
    b_last_used,
    b_step,
    b_hits,
    b_misses,
    b_S,
    b_slot_out,
    b_todo,
    scratch_ptr,  # int64[2, BLOCK_S]
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    if tl.program_id(0) == 0:
        _acquire(
            topk_ptr, K, a_slot_of, a_expert_in, a_last_used, a_step, a_hits, a_misses,
            a_S, a_map, a_slot_out, a_todo, scratch_ptr,
            E, BLOCK_E, BLOCK_S, True,
        )  # fmt: skip
    else:
        _acquire(
            topk_ptr, K, b_slot_of, b_expert_in, b_last_used, b_step, b_hits, b_misses,
            b_S, a_map, b_slot_out, b_todo, scratch_ptr + BLOCK_S,
            E, BLOCK_E, BLOCK_S, False,
        )  # fmt: skip


class AcquirePair:
    """Update the VRAM and RAM tables in one launch.

    Outputs go to buffers at fixed addresses.
    """

    def __init__(self, vram: ExpertTable, ram: ExpertTable) -> None:
        if vram.num_experts != ram.num_experts:
            raise ValueError("both tables must have the same number of experts")
        E, device = vram.num_experts, vram.device
        self.vram, self.ram = vram, ram
        self.block_e = 1 << (E - 1).bit_length()
        self.block_s = 1 << (max(vram.num_slots, ram.num_slots) - 1).bit_length()
        if self.block_s > 1 << _SLOT_BITS:
            raise ValueError(f"at most {1 << _SLOT_BITS} slots per table")
        i64 = {"dtype": torch.int64, "device": device}
        # Values of unneeded experts are never read.
        self.expert_map = torch.full((E,), -1, dtype=torch.int32, device=device)
        self.vram_slot = torch.zeros((E,), **i64)
        self.ram_slot = torch.zeros((E,), **i64)
        self._vram_todo = torch.full((vram.num_slots,), -1, **i64)
        self._ram_todo = torch.full((ram.num_slots,), -1, **i64)
        self._scratch = torch.zeros((2, self.block_s), **i64)

    def __call__(self, topk_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Update both tables and return ``(vram_todo, ram_todo)`` of length K.

        ``expert_map``, ``vram_slot`` and ``ram_slot`` are written to the
        attribute buffers. Requires ``topk_ids.numel() <= vram.num_slots``.
        """
        K = topk_ids.numel()
        a, b = self.vram, self.ram
        _acquire_pair_kernel[(2,)](
            topk_ids.reshape(-1), K,
            a.slot_of, a.expert_in, a.last_used, a.step, a.hits, a.misses, a.num_slots,
            self.expert_map, self.vram_slot, self._vram_todo,
            b.slot_of, b.expert_in, b.last_used, b.step, b.hits, b.misses, b.num_slots,
            self.ram_slot, self._ram_todo,
            self._scratch,
            E=a.num_experts, BLOCK_E=self.block_e, BLOCK_S=self.block_s,
        )  # fmt: skip
        return self._vram_todo[:K], self._ram_todo[:K]
