import pytest

from vllm_expert_pager.cache import ExpertCache


def test_first_acquire_is_all_misses():
    cache = ExpertCache(4)
    slots, misses = cache.acquire([7, 3])
    assert misses == [0, 1]
    assert sorted(slots) != [-1, -1] and len(set(slots)) == 2
    assert (cache.hits, cache.misses) == (0, 2)


def test_second_acquire_hits_same_slots():
    cache = ExpertCache(4)
    first, _ = cache.acquire([7, 3])
    second, misses = cache.acquire([7, 3])
    assert misses == []
    assert second == first
    assert (cache.hits, cache.misses) == (2, 2)


def test_evicts_least_recently_used():
    cache = ExpertCache(2)
    cache.acquire([1, 2])
    cache.acquire([1])  # refresh 1 -> 2 is evicted next
    slots, misses = cache.acquire([3])
    assert misses == [0]
    assert cache.peek(2) is None
    assert cache.peek(1) is not None
    assert slots[0] == cache.peek(3)


def test_experts_in_the_same_request_are_never_evicted():
    """Experts used in this step must not be evicted within the same step."""
    cache = ExpertCache(4)
    cache.acquire([0, 1, 2, 3])
    # Two hits and two misses; the misses must land in 0 and 1, which are not
    # used in this step.
    slots, misses = cache.acquire([2, 3, 8, 9])
    assert misses == [2, 3]
    assert len(set(slots)) == 4
    for expert in (2, 3, 8, 9):
        assert cache.peek(expert) is not None


def test_peek_does_not_change_lru_order():
    cache = ExpertCache(2)
    cache.acquire([1, 2])
    cache.peek(1)  # peek does not refresh 1
    cache.acquire([3])
    assert cache.peek(1) is None


def test_rejects_more_experts_than_slots():
    cache = ExpertCache(2)
    with pytest.raises(ValueError, match="only 2 slots"):
        cache.acquire([1, 2, 3])


def test_rejects_duplicate_experts():
    cache = ExpertCache(4)
    with pytest.raises(ValueError, match="unique"):
        cache.acquire([1, 1])


def test_rejects_zero_slots():
    with pytest.raises(ValueError, match="num_slots"):
        ExpertCache(0)


def test_slots_stay_within_capacity():
    cache = ExpertCache(3)
    for step in range(50):
        experts = [(step + i) % 20 for i in range(3)]
        slots, _ = cache.acquire(experts)
        assert len(set(slots)) == 3
        assert all(0 <= slot < 3 for slot in slots)
