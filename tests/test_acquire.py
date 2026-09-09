"""Check that AcquirePair (Triton) matches the torch ExpertTable.acquire. Needs a GPU."""

import random

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from vllm_expert_pager.acquire import AcquirePair
from vllm_expert_pager.table import ExpertTable


def make_tables(E, S, R, warm):
    dev = torch.device("cuda")
    vram, ram = ExpertTable(E, S, dev), ExpertTable(E, R, dev)
    if warm:
        # Same as the warm start in experts.py: expert e < R goes in slot e.
        ram.slot_of[:R] = ram.experts[:R]
        ram.expert_in[:R] = ram.experts[:R]
    return vram, ram


def state(t: ExpertTable):
    # The trailing sink (index E / S) gets garbage from the torch version and is
    # never read, so leave it out.
    E, S = t.num_experts, t.num_slots
    return [t.slot_of[:E], t.expert_in[:S], t.last_used[:S], t.step, t.hits, t.misses]


@pytest.mark.parametrize(
    "E,S,R,warm",
    [(16, 4, 8, False), (16, 4, 4, True), (256, 102, 256, True), (256, 32, 128, True)],
)
def test_matches_torch_acquire(E, S, R, warm):
    rng = random.Random(0)
    ref_vram, ref_ram = make_tables(E, S, R, warm)
    vram, ram = make_tables(E, S, R, warm)
    pair = AcquirePair(vram, ram)
    for _ in range(200):
        k = rng.randint(1, S)
        ids = torch.tensor(
            [[rng.randrange(E) for _ in range(k)]], dtype=torch.int32, device="cuda"
        )
        exp_map, exp_slot, exp_todo = ref_vram.acquire(ids)
        _, exp_src, exp_ssd = ref_ram.acquire(ids)
        todo, ssd_todo = pair(ids)
        torch.cuda.synchronize()

        for a, b in zip(state(ref_vram) + state(ref_ram), state(vram) + state(ram)):
            assert torch.equal(a, b)
        assert torch.equal(pair.expert_map, exp_map.to(torch.int32))
        need = exp_map >= 0
        assert torch.equal(pair.vram_slot[need], exp_slot[need])
        assert torch.equal(pair.ram_slot[need], exp_src[need])
        assert torch.equal(todo, exp_todo)
        assert torch.equal(ssd_todo, exp_ssd)
