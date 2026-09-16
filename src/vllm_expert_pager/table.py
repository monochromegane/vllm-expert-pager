"""Expert-to-slot table kept on device tensors.

Replaces the earlier ``ExpertCache`` (an LRU built on Python's ``OrderedDict``)
with a form that can run under CUDA graphs. Everything is written with
fixed-shape torch ops and the host never reads a value. ``torch.unique`` and
indexing with a bool mask are avoided because their output shape depends on the
values (which forces a sync).

With CPU tensors the table can be unit-tested without a GPU.
"""

import torch

# Added to the victim-selection key to push "slots used in this step" to the
# end of the ordering.
_PROTECT = 1 << 40


class ExpertTable:
    """Expert -> slot table for one layer. Replacement is LRU.

    Every table has one extra element; the last index (``num_experts`` or
    ``num_slots``) is a sink. Writes whose length would depend on the values
    become fixed-shape scatters by sending the non-matching elements to the
    sink. The sink's value is never read.
    """

    def __init__(self, num_experts: int, num_slots: int, device: torch.device) -> None:
        if num_slots < 1:
            raise ValueError(f"num_slots must be >= 1, got {num_slots}")
        self.num_experts = num_experts
        self.num_slots = num_slots
        self.device = device
        i64 = {"dtype": torch.int64, "device": device}

        # expert -> slot. -1 means not cached.
        self.slot_of = torch.full((num_experts + 1,), -1, **i64)
        # slot -> expert. -1 means free.
        self.expert_in = torch.full((num_slots + 1,), -1, **i64)
        # Step in which the slot was last used. Free slots stay at 0.
        self.last_used = torch.zeros((num_slots + 1,), **i64)
        # Starts at 1, so last_used of a used slot is always greater than that
        # of a free slot.
        self.step = torch.ones((), **i64)

        # Cumulative counts, one per expert reference. Counted only on the cache
        # path (as before).
        self.hits = torch.zeros((), **i64)
        self.misses = torch.zeros((), **i64)

        # Constants. Creating torch.tensor(...) inside forward is a host ->
        # device copy, which is not allowed during capture, so prepare them here.
        self.experts = torch.arange(num_experts, **i64)
        self.no_cache = torch.full((num_experts,), -1, **i64)

    # ---- Shared ----

    def _ids(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """Flatten ``topk_ids`` to int64, sending negative ids (vLLM's -1 for invalid entries) to the sink E."""
        ids = topk_ids.reshape(-1).to(torch.int64)
        return torch.where(ids >= 0, ids, self.num_experts)

    def need_mask(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """Return the set of experts needed in this step as a bool mask."""
        need = torch.zeros(
            (self.num_experts + 1,), dtype=torch.bool, device=self.device
        )
        # Writing the same value to duplicate indices, so the order does not
        # matter.
        need.index_fill_(0, self._ids(topk_ids), True)
        return need[: self.num_experts]

    # ---- Cache path ----

    def acquire(
        self, topk_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign a slot to every needed expert and update the table.

        Requires ``topk_ids.numel() <= num_slots``; the caller guarantees this
        statically.

        Returns:
            ``(expert_map, slot, todo)``.
            ``expert_map[e]`` is the slot of expert ``e`` (-1 if not needed).
            ``slot[e]`` holds some value even for experts that are not needed,
            so read it only through ``expert_map`` or ``todo``.
            ``todo`` has length ``topk_ids.numel()`` and lists the experts that
            must be loaded (padded with -1 at the end).
        """
        E, S = self.num_experts, self.num_slots
        K = topk_ids.numel()

        need = self.need_mask(topk_ids)
        current = self.slot_of[:E]
        hit = need & (current >= 0)
        miss = need & (current < 0)

        # Slots holding an expert used in this step are never evicted.
        occupant = self.expert_in[:S]
        slot_used = (occupant >= 0) & hit[occupant.clamp(min=0)]

        # Victim ordering: free slots (last_used=0) -> unused slots, oldest
        # first -> slots in use. Slots used in the same step are ordered by slot
        # number (stable). Each miss takes the entry at its ordinal from the
        # front. hit + miss <= K <= S, so this never reaches the slots in use.
        key = self.last_used[:S] + slot_used.to(torch.int64) * _PROTECT
        order = torch.argsort(key, stable=True)
        rank = torch.cumsum(miss.to(torch.int64), 0) - 1
        victim = order[rank.clamp(0, S - 1)]

        # Update the tables. Non-miss elements are written to the sink.
        evicted = self.expert_in[victim]  # -1 if the slot was free
        self.slot_of.index_fill_(0, torch.where(miss & (evicted >= 0), evicted, E), -1)
        self.slot_of.index_put_((torch.where(miss, self.experts, E),), victim)
        self.expert_in.index_put_((torch.where(miss, victim, S),), self.experts)

        slot = torch.where(miss, victim, current)
        self.last_used.index_put_((torch.where(need, slot, S),), self.step)
        self.step += 1

        self.hits += hit.sum()
        self.misses += miss.sum()

        expert_map = torch.where(need, slot, -1)
        todo = torch.full((K + 1,), -1, dtype=torch.int64, device=self.device)
        todo.index_put_((torch.where(miss, rank, K),), self.experts)
        return expert_map, slot, todo[:K]

    # ---- Working-buffer path ----

    def lookup(
        self, topk_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign the needed experts to slots packed from 0 without updating the table.

        Used when ``topk_ids.numel() > num_slots``, i.e. when the experts may
        not fit in the cache slab. Experts that hit can be copied from the cache
        slab within VRAM, so their slot numbers there are returned as well.

        Returns:
            ``(expert_map, slot, cached_slot, todo)``.
            ``slot[e]`` is the packed slot, ``cached_slot[e]`` is the slot on
            the cache slab (-1 if absent), and ``todo`` lists the needed experts
            with length ``min(num_experts, topk_ids.numel())`` (padded with -1
            at the end).
        """
        E = self.num_experts
        K = min(E, topk_ids.numel())

        need = self.need_mask(topk_ids)
        slot = torch.cumsum(need.to(torch.int64), 0) - 1
        expert_map = torch.where(need, slot, -1)
        return expert_map, slot, self.slot_of[:E], self.pack(need)[:K]

    def pack(self, mask: torch.Tensor) -> torch.Tensor:
        """The expert numbers set in ``mask``, packed from the front and padded with -1 to length E."""
        E = self.num_experts
        rank = torch.cumsum(mask.to(torch.int64), 0) - 1
        out = torch.full((E + 1,), -1, dtype=torch.int64, device=self.device)
        out.index_put_((torch.where(mask, rank, E),), self.experts)
        return out[:E]

    def seed(
        self, vram: "ExpertTable", topk_ids: torch.Tensor, need: torch.Tensor
    ) -> None:
        """Update this table (the RAM tier) from the prefill's routing.

        Acquires the R - S most referenced experts of ``topk_ids``, then the
        experts that are in the VRAM tier ``vram``. The latter get the newer
        timestamp, so in the following decode steps the RAM tier never evicts a
        VRAM-tier expert first and RAM stays a superset of VRAM. VRAM-tier
        experts that are neither in the RAM tier nor referenced this step
        (``need``) are left out: a slot for them would never get its row read.
        Seeding is not a reference, so hits / misses are not counted.
        """
        E, R, S = self.num_experts, self.num_slots, vram.num_slots
        hits, misses = self.hits.clone(), self.misses.clone()
        # Reference counts. bincount sizes its output from the values (a sync),
        # so index_add_ into a fixed length instead.
        ids = self._ids(topk_ids)
        counts = torch.zeros((E + 1,), dtype=torch.int64, device=self.device)
        counts.index_add_(0, ids, torch.ones_like(ids))
        counts = counts[:E]
        top = counts.topk(max(R - S, 1)).indices
        # Fill unreferenced entries with the most frequent expert (need_mask
        # removes the duplicates).
        top = torch.where(counts[top] > 0, top, top[:1])
        if R > S:
            self.acquire(top)
        v = vram.expert_in[:S]
        ok = (v >= 0) & ((self.slot_of[v.clamp(min=0)] >= 0) | need[v.clamp(min=0)])
        self.acquire(torch.where(ok, v, top[:1]))
        self.hits.copy_(hits)
        self.misses.copy_(misses)

    def plan_working(
        self,
        before: torch.Tensor,
        need: torch.Tensor,
        cached_row: torch.Tensor,
        dst_row: torch.Tensor,
        ram_base: int,
        working_base: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign rows on the working-buffer path after the table was updated.

        ``before`` is ``slot_of[:E]`` from before the update. Returns
        ``(src_row, todo, ssd_todo, late)``.

        - Experts whose old slot is still usable (unchanged, or evicted so the
          slot is being freed) are copied from that slot. Evicted rows must be
          copied before fetch overwrites them, so they go into ``todo``
        - The other needed experts, and experts that moved to another slot, are
          read from SSD (``ssd_todo``). They land in their table slot, or in
          the shared working row ``dst_row + working_base`` when not in the
          table
        - ``todo`` lists the experts that need no SSD wait (including VRAM
          hits), ``late`` those copied after the read
        """
        after = self.slot_of[: self.num_experts]
        old_ok = (before >= 0) & ((after < 0) | (after == before))
        ssd = ~old_ok & (need | (before >= 0))
        src_row = torch.where(
            old_ok,
            before + ram_base,
            torch.where(after >= 0, after + ram_base, dst_row + working_base),
        )
        todo = self.pack(need & (old_ok | (cached_row >= 0)))
        return src_row, todo, self.pack(ssd), self.pack(ssd & need)
