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
