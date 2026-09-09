"""LRU cache that manages the assignment of expert slots.

A slot is an integer naming the region on the slab that holds the weights of
one expert. This module only tracks the mapping between numbers and never
touches tensors, so it can be unit-tested without a GPU.
"""

from collections import OrderedDict
from collections.abc import Sequence


class ExpertCache:
    """Manages the expert slots of one layer with an LRU policy."""

    def __init__(self, num_slots: int) -> None:
        if num_slots < 1:
            raise ValueError(f"num_slots must be >= 1, got {num_slots}")
        self.num_slots = num_slots
        # expert -> slot, in LRU order: the first entry is the least recently used.
        self._slots: OrderedDict[int, int] = OrderedDict()
        self._free = list(range(num_slots))
        self.hits = 0
        self.misses = 0

    def peek(self, expert: int) -> int | None:
        """Look up a slot without changing the LRU order.

        Used when an expert is referenced from outside the cache slab.
        """
        return self._slots.get(expert)

    def acquire(self, experts: Sequence[int]) -> tuple[list[int], list[int]]:
        """Reserve a slot for every entry in ``experts``.

        Args:
            experts: Expert ids without duplicates. At most ``num_slots`` of them.

        Returns:
            ``(slots, misses)``. ``slots[i]`` is the slot of ``experts[i]``.
            ``misses`` lists the indices into ``experts`` whose weights must be
            loaded.

        Experts used in this step are never evicted. Hits are moved to the tail
        of the LRU first, so as long as ``len(experts) <= num_slots`` the
        eviction candidates are guaranteed to be experts not used in this step.
        """
        if len(experts) > self.num_slots:
            raise ValueError(
                f"{len(experts)} experts requested but only {self.num_slots} slots"
            )
        if len(set(experts)) != len(experts):
            raise ValueError(f"experts must be unique, got {list(experts)}")

        slots = [-1] * len(experts)
        misses: list[int] = []
        for i, expert in enumerate(experts):
            slot = self._slots.get(expert)
            if slot is None:
                misses.append(i)
            else:
                self._slots.move_to_end(expert)
                slots[i] = slot
        self.hits += len(experts) - len(misses)
        self.misses += len(misses)

        for i in misses:
            expert = experts[i]
            if self._free:
                slot = self._free.pop()
            else:
                _, slot = self._slots.popitem(last=False)
            self._slots[expert] = slot
            slots[i] = slot
        return slots, misses
