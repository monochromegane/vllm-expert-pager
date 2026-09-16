"""Unit tests for ExpertTable. Runs on CPU tensors."""

import random

import pytest
import torch

from vllm_expert_pager.table import ExpertTable

E = 16


class ReferenceLRU:
    """LRU written in plain Python, for comparison.

    The earlier ExpertCache ordered recency by position within one call; the
    table stamps every expert of a step with the same time. Among equal stamps
    the lower slot number is evicted first (the same rule as the table's stable
    argsort). Hits and misses mean the same thing.
    """

    def __init__(self, num_slots: int) -> None:
        self.num_slots = num_slots
        self.slot_of: dict[int, int] = {}
        self.stamp = [0] * num_slots  # 0 is free
        self.step = 1
        self.hits = 0
        self.misses = 0

    def acquire(self, experts: set[int]) -> None:
        hits = {e for e in experts if e in self.slot_of}
        misses = sorted(experts - hits)
        self.hits += len(hits)
        self.misses += len(misses)
        protected = {self.slot_of[e] for e in hits}
        for e in misses:
            victim = min(
                (s for s in range(self.num_slots) if s not in protected),
                key=lambda s: (self.stamp[s], s),
            )
            for old, s in list(self.slot_of.items()):
                if s == victim:
                    del self.slot_of[old]
            self.slot_of[e] = victim
            protected.add(victim)
        for e in experts:
            self.stamp[self.slot_of[e]] = self.step
        self.step += 1

    def cached(self) -> set[int]:
        return set(self.slot_of)


def unique_experts(topk_ids: torch.Tensor) -> set[int]:
    return set(topk_ids.reshape(-1).tolist())


def check_invariants(table: ExpertTable) -> None:
    """slot_of and expert_in are inverses of each other and no slot is shared."""
    E, S = table.num_experts, table.num_slots
    slot_of = table.slot_of[:E].tolist()
    expert_in = table.expert_in[:S].tolist()
    used = [s for s in slot_of if s >= 0]
    assert len(used) == len(set(used)), "two experts share a slot"
    for e, s in enumerate(slot_of):
        if s >= 0:
            assert 0 <= s < S
            assert expert_in[s] == e
    for s, e in enumerate(expert_in):
        if e >= 0:
            assert slot_of[e] == s


def check_acquire_result(table, topk_ids, expert_map, slot, todo, before):
    """The return values of acquire give every needed expert a distinct slot."""
    need = unique_experts(topk_ids)
    emap = expert_map.tolist()
    slots = [emap[e] for e in need]
    assert len(slots) == len(set(slots))
    for e in range(table.num_experts):
        if e in need:
            assert 0 <= emap[e] < table.num_slots
            assert slot[e].item() == emap[e]
            assert table.slot_of[e].item() == emap[e]
        else:
            assert emap[e] == -1
    # todo holds only the missed experts, packed from the front, with -1 after.
    misses = sorted(e for e in need if e not in before)
    todo_list = todo.tolist()
    assert todo.shape[0] == topk_ids.numel()
    assert sorted(x for x in todo_list if x >= 0) == misses
    n = len(misses)
    assert all(x >= 0 for x in todo_list[:n])
    assert all(x == -1 for x in todo_list[n:])
    # None of the experts needed this step has been evicted.
    assert need <= {e for e in range(table.num_experts) if table.slot_of[e] >= 0}


@pytest.mark.parametrize("num_slots", [1, 4, 8])
def test_acquire_matches_reference_lru(num_slots: int) -> None:
    rng = random.Random(0)
    table = ExpertTable(E, num_slots, torch.device("cpu"))
    ref = ReferenceLRU(num_slots)
    for _ in range(300):
        k = rng.randint(1, num_slots)
        # Build topk_ids with duplicates (one token, top_k = k).
        ids = torch.tensor([[rng.randrange(E) for _ in range(k)]], dtype=torch.int32)
        before = {e for e in range(E) if table.slot_of[e] >= 0}
        assert before == ref.cached()

        expert_map, slot, todo = table.acquire(ids)
        check_invariants(table)
        check_acquire_result(table, ids, expert_map, slot, todo, before)

        ref.acquire(unique_experts(ids))
        assert table.hits.item() == ref.hits
        assert table.misses.item() == ref.misses
        assert {e for e in range(E) if table.slot_of[e] >= 0} == ref.cached()


def test_acquire_evicts_least_recently_used() -> None:
    table = ExpertTable(E, 2, torch.device("cpu"))
    table.acquire(torch.tensor([[0]]))
    table.acquire(torch.tensor([[1]]))
    table.acquire(torch.tensor([[0]]))  # touch 0 again; 1 becomes the oldest
    table.acquire(torch.tensor([[2]]))  # 1 is evicted
    cached = {e for e in range(E) if table.slot_of[e] >= 0}
    assert cached == {0, 2}
    assert table.hits.item() == 1
    assert table.misses.item() == 3


def test_acquire_keeps_all_needed_when_full() -> None:
    # With every slot occupied, requesting a set that overlaps the cached
    # experts still fits everyone.
    table = ExpertTable(E, 4, torch.device("cpu"))
    table.acquire(torch.tensor([[0, 1, 2, 3]]))
    expert_map, _, todo = table.acquire(torch.tensor([[2, 3, 4, 5]]))
    check_invariants(table)
    assert {e for e in range(E) if table.slot_of[e] >= 0} == {2, 3, 4, 5}
    assert sorted(x for x in todo.tolist() if x >= 0) == [4, 5]
    emap = expert_map.tolist()
    assert emap[0] == -1 and emap[1] == -1
    assert sorted(emap[e] for e in (2, 3, 4, 5)) == [0, 1, 2, 3]


def test_lookup_compacts_without_touching_table() -> None:
    table = ExpertTable(E, 2, torch.device("cpu"))
    table.acquire(torch.tensor([[5, 9]]))
    snapshot = (table.slot_of.clone(), table.expert_in.clone(), table.last_used.clone())

    ids = torch.tensor([[9, 3, 3], [12, 5, 9]])  # 6 elements > 2 slots
    expert_map, slot, cached_slot, todo = table.lookup(ids)

    need = sorted(unique_experts(ids))  # [3, 5, 9, 12]
    emap = expert_map.tolist()
    assert [emap[e] for e in need] == [0, 1, 2, 3]
    assert all(emap[e] == -1 for e in range(E) if e not in need)
    assert [slot[e].item() for e in need] == [0, 1, 2, 3]
    # Position j in todo equals the packed slot.
    assert todo.shape[0] == min(E, ids.numel())
    assert todo.tolist() == need + [-1] * (todo.shape[0] - len(need))
    # Only the hits, 5 and 9, have a slot on the cache slab.
    assert cached_slot[5].item() >= 0 and cached_slot[9].item() >= 0
    assert cached_slot[3].item() == -1 and cached_slot[12].item() == -1
    # The table is unchanged.
    for a, b in zip(snapshot, (table.slot_of, table.expert_in, table.last_used)):
        assert torch.equal(a, b)
    assert table.hits.item() == 0 and table.misses.item() == 2


def test_lookup_todo_is_capped_at_num_experts() -> None:
    table = ExpertTable(E, 2, torch.device("cpu"))
    ids = torch.arange(E).repeat(3).reshape(3, E)  # 48 elements, every expert
    expert_map, _, _, todo = table.lookup(ids)
    assert todo.shape[0] == E
    assert todo.tolist() == list(range(E))
    assert expert_map.tolist() == list(range(E))


def test_rejects_zero_slots() -> None:
    with pytest.raises(ValueError):
        ExpertTable(E, 0, torch.device("cpu"))


def test_pack_compacts_mask() -> None:
    table = ExpertTable(E, 2, torch.device("cpu"))
    mask = torch.zeros(E, dtype=torch.bool)
    mask[[3, 7, 12]] = True
    assert table.pack(mask).tolist() == [3, 7, 12] + [-1] * (E - 3)


def test_seed_puts_frequent_and_vram_experts_in_ram() -> None:
    R, S = 6, 2
    ram = ExpertTable(E, R, torch.device("cpu"))
    vram = ExpertTable(E, S, torch.device("cpu"))
    ram.acquire(torch.tensor([[0, 1, 2, 3, 4, 5]]))
    # 4 is in RAM. 9 is not, but the prefill references it.
    vram.acquire(torch.tensor([[4, 9]]))
    ids = torch.tensor([[8, 7, 6, 5], [8, 7, 6, 5], [8, 7, 0, 9]])  # 12 elements > R
    need = ram.need_mask(ids)
    before = ram.slot_of[:E].clone()

    ram.seed(vram, ids, need)
    check_invariants(ram)
    cached = {e for e in range(E) if ram.slot_of[e] >= 0}
    # The R - S = 4 most referenced (8, 7, 6, 5) and the VRAM tier's 4, 9.
    assert cached == {8, 7, 6, 5, 4, 9}
    # The VRAM-tier experts are newer (not evicted first in decode).
    assert ram.last_used[ram.slot_of[4]] > ram.last_used[ram.slot_of[8]]
    assert ram.last_used[ram.slot_of[9]] > ram.last_used[ram.slot_of[8]]
    # The experts that stayed kept their slots.
    assert ram.slot_of[4] == before[4] and ram.slot_of[5] == before[5]
    # Seeding does not count as references.
    assert ram.hits.item() == 0 and ram.misses.item() == 6


def test_seed_skips_vram_experts_that_have_no_row_to_read() -> None:
    R, S = 4, 2
    ram = ExpertTable(E, R, torch.device("cpu"))
    vram = ExpertTable(E, S, torch.device("cpu"))
    ram.acquire(torch.tensor([[0, 1, 2, 3]]))
    vram.acquire(
        torch.tensor([[0, 9]])
    )  # 9 is not in RAM and the prefill does not reference it
    ids = torch.tensor([[5, 6], [5, 6], [7, 0]])
    ram.seed(vram, ids, ram.need_mask(ids))
    check_invariants(ram)
    cached = {e for e in range(E) if ram.slot_of[e] >= 0}
    assert 9 not in cached
    assert {5, 6, 0} <= cached


def simulate_working_step(ram, vram, ids, rows, ram_base, working_base):
    """Mimic forward's order (copy, fetch, copy the rest) on the CPU.

    Returns the working rows and the table state. ``rows`` holds the contents of
    the RAM-tier rows (expert numbers, -1 for never written). The VRAM slab is
    assumed to hold the rows of the experts in its table.
    """
    need = ram.need_mask(ids)
    _, dst_row, cached_row, _ = vram.lookup(ids)
    before = ram.slot_of[:E].clone()
    if ids.numel() <= ram.num_slots:
        ram.acquire(ids)
    else:
        ram.seed(vram, ids, need)
    src_row, todo, ssd_todo, late = ram.plan_working(
        before, need, cached_row, dst_row, ram_base, working_base
    )
    working = torch.full((E,), -1)

    def copy(part_list):
        for e in part_list.tolist():
            if e >= 0:
                working[dst_row[e]] = e if cached_row[e] >= 0 else rows[src_row[e]]

    copy(todo)
    for e in ssd_todo.tolist():
        if e >= 0:
            rows[src_row[e]] = e  # the host reads it from SSD
    copy(late)
    return need, dst_row, working, ssd_todo


def check_working_step(ram, need, dst_row, working, rows, ram_base):
    for e in need.nonzero().flatten().tolist():
        assert working[dst_row[e]] == e, f"working row of expert {e} is wrong"
    for e in range(E):
        s = ram.slot_of[e].item()
        if s >= 0:
            assert rows[ram_base + s] == e, f"RAM row of expert {e} is stale"


def test_plan_working_keeps_rows_truthful_through_a_seed() -> None:
    R, S, L = 4, 2, 3
    layer = 1
    ram_base, working_base = layer * R, L * R
    ram = ExpertTable(E, R, torch.device("cpu"))
    vram = ExpertTable(E, S, torch.device("cpu"))
    rows = torch.full((L * R + E,), -1)
    ram.acquire(torch.tensor([[0, 1, 2, 3]]))
    for e in range(4):
        rows[ram_base + ram.slot_of[e]] = e
    vram.acquire(torch.tensor([[0, 9]]))
    # 5 and 6 rank highest and evict 0 and 1. 0 is in the VRAM tier and is
    # referenced, so it moves to another slot. 1 is evicted but referenced, so
    # it must be copied before fetch overwrites its slot.
    ids = torch.tensor([[5, 6], [5, 6], [5, 0], [7, 1]])
    need, dst_row, working, ssd_todo = simulate_working_step(
        ram, vram, ids, rows, ram_base, working_base
    )
    check_invariants(ram)
    check_working_step(ram, need, dst_row, working, rows, ram_base)
    cached = {e for e in range(E) if ram.slot_of[e] >= 0}
    assert cached == {5, 6, 0, 3}
    # Read from SSD: the newly seeded 5 and 6, the moved 0, and 7, which is not
    # in the table. 1, 2, 3 are not read.
    assert sorted(e for e in ssd_todo.tolist() if e >= 0) == [0, 5, 6, 7]
    # 7 was read into a shared working row.
    assert rows[working_base + dst_row[7]] == 7


def test_plan_working_reads_only_misses_when_ram_acquires() -> None:
    R, S, L = 4, 2, 2
    ram_base, working_base = 0, L * R
    ram = ExpertTable(E, R, torch.device("cpu"))
    vram = ExpertTable(E, S, torch.device("cpu"))
    rows = torch.full((L * R + E,), -1)
    ram.acquire(torch.tensor([[0, 1, 2, 3]]))
    for e in range(4):
        rows[ram_base + ram.slot_of[e]] = e
    vram.acquire(torch.tensor([[2, 3]]))
    ids = torch.tensor([[2, 5], [3, 3]])  # 4 elements <= R: the RAM tier acquires
    need, dst_row, working, ssd_todo = simulate_working_step(
        ram, vram, ids, rows, ram_base, working_base
    )
    check_invariants(ram)
    check_working_step(ram, need, dst_row, working, rows, ram_base)
    assert sorted(e for e in ssd_todo.tolist() if e >= 0) == [5]


def test_negative_ids_are_ignored() -> None:
    # vLLM uses -1 for invalid entries. They go to the sink and are not
    # confused with expert E-1.
    R, S = 4, 2
    ram = ExpertTable(E, R, torch.device("cpu"))
    vram = ExpertTable(E, S, torch.device("cpu"))
    ids = torch.tensor([[3, -1], [3, -1], [5, -1]], dtype=torch.int32)
    need = ram.need_mask(ids)
    assert need.nonzero().flatten().tolist() == [3, 5]
    ram.seed(vram, ids, need)
    check_invariants(ram)
    cached = {e for e in range(E) if ram.slot_of[e] >= 0}
    assert cached == {3, 5}
