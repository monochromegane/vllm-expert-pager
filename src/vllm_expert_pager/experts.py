"""Hook ``RoutedExperts`` so that expert weights are referenced through a slab.

Expert weights live in three tiers: the VRAM slab, pinned RAM, and a paging
file on SSD. The plugin owns their placement from load time on and does not
use vLLM's CPU offload. The RAM and SSD tiers hold the rows losslessly
compressed, and the GPU expands them when copying into VRAM.

- Construction: the (E, ...) weights allocated by create_weights are replaced
  with one-row placeholders
- Loading: weight_loader is intercepted and each expert's weights are
  compressed and written to the paging file and to the RAM tier (experts
  numbered below R)
- Inference: only the experts needed in the step are copied into staging and
  expanded into the slab, and expert_map renumbers experts to slots before the
  kernel runs. Decisions are made on device tensors and the host never reads a
  value. For experts not in RAM the GPU asks a C thread on the host to read
  them from SSD. A prefill that does not fit in the cache slab runs its experts
  through a working slab of W rows in chunks
"""

import os

import torch
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

from vllm_expert_pager.store import Store
from vllm_expert_pager.table import ExpertTable

# vLLM only configures handlers and levels for the "vllm" logger
# (DEFAULT_LOGGING_CONFIG in logger.py), so take a name under it. A bare
# __name__ falls outside that hierarchy, the effective level becomes the root's
# WARNING, and every info message is dropped.
logger = init_logger(f"vllm.{__name__}")

# Number of cache slots per layer. One slot holds one expert (about 3 MiB for
# 35B-A3B).
CACHE_SLOTS = int(os.environ.get("VLLM_EXPERT_PAGER_CACHE_SLOTS", "32"))
# Number of RAM-tier slots per layer. Unset means every expert (RAM only; the
# SSD is never read).
RAM_SLOTS = os.environ.get("VLLM_EXPERT_PAGER_RAM_SLOTS")
# Paging file. Required when RAM_SLOTS is smaller than the number of experts.
SSD_PATH = os.environ.get("VLLM_EXPERT_PAGER_SSD_PATH")
# Whether to compress the RAM and SSD tiers losslessly. auto compresses only
# when the SSD tier is used (RAM_SLOTS below the number of experts). When every
# expert fits in RAM there are no SSD reads, and the expansion (about
# 10 ms/token) costs more than the 12% saved on PCIe. Without an SSD wait there
# is nothing to hide the expansion under either.
COMPRESS = os.environ.get("VLLM_EXPERT_PAGER_COMPRESS", "auto")
# Fixed length of a compressed row as a ratio of the raw row. If an expert does
# not fit, loading stops and reports the ratio needed. How much a row compresses
# is a property of the model: Qwen3.8-Flash-Next-FP8 fits at 0.877 over all
# 24,576 of its experts, while Qwen3.6-35B-A3B-FP8 does not fit at 0.88 -- the
# first row to overflow asks for 0.8880, and loading stops there, so all that is
# known is that 0.89 holds every row. The default covers both; a model that
# compresses better can be given a smaller ratio.
PITCH_RATIO = float(os.environ.get("VLLM_EXPERT_PAGER_PITCH", "0.89"))
# Number of rows W of the working slab used by prefill (the working-buffer
# path). The needed experts are split into chunks of this many and the MoE runs
# once per chunk. The decode staging for compressed rows (S rows x pitch) also
# borrows the head of this slab, so W x raw row >= S x pitch is required. There
# is one slab for every layer, so what is saved here can go to
# VLLM_EXPERT_PAGER_CACHE_SLOTS.
WORKING_ROWS = int(os.environ.get("VLLM_EXPERT_PAGER_WORKING_ROWS", "64"))
# Interval for logging the hit rate, in Python calls of forward per layer. 0
# disables it. CUDA graph replay does not run Python, so the log only appears
# when forward is called eagerly (prefill etc.). The counters themselves live
# on the device and include replays.
LOG_INTERVAL = int(os.environ.get("VLLM_EXPERT_PAGER_LOG_INTERVAL", "1000"))

# Expert weights referenced through the slab. Scales stay resident in VRAM and
# a copy in slot order is shown alongside the weights.
PARAMS = ("w13_weight", "w2_weight")

_store: Store | None = None
# Working slab (W rows of weights and scales) shared across all layers, and the
# staging for prefill (W rows). Layers run sequentially on the same stream, so
# there is no cross-layer race.
_working_slab: dict[str, torch.Tensor] | None = None
_prefill_staging: list[torch.Tensor] | None = None


class ExpertPagerRoutedExperts(RoutedExperts):
    """``RoutedExperts`` that references expert weights through a slab.

    Uses the cache slab when ``topk_ids`` has no more elements than there are
    slots, and the working slab otherwise. The decision is made on the shape
    alone: checking ``|U|`` would require the host to read a value, which does
    not work under CUDA graphs.
    """

    _expert_pager_expert_map: torch.Tensor | None = None
    _expert_pager_calls = 0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.use_ep or self.local_num_experts != self.global_num_experts:
            raise RuntimeError(
                f"{self.layer_name}: vllm-expert-pager does not support expert "
                f"parallelism (local={self.local_num_experts}, "
                f"global={self.global_num_experts})"
            )
        if self.moe_config.moe_parallel_config.tp_size != 1:
            raise RuntimeError(
                f"{self.layer_name}: vllm-expert-pager does not support tensor "
                "parallelism"
            )
        E = self.global_num_experts
        params = {name: getattr(self, name) for name in PARAMS}

        global _store
        if _store is None:
            R = int(RAM_SLOTS) if RAM_SLOTS else E
            if R < CACHE_SLOTS:
                raise ValueError(
                    f"VLLM_EXPERT_PAGER_RAM_SLOTS={R} must be >= "
                    f"VLLM_EXPERT_PAGER_CACHE_SLOTS={CACHE_SLOTS}"
                )
            if WORKING_ROWS < 1:
                raise ValueError(
                    f"VLLM_EXPERT_PAGER_WORKING_ROWS must be >= 1, got {WORKING_ROWS}"
                )
            if COMPRESS not in ("auto", "on", "off"):
                raise ValueError(
                    "VLLM_EXPERT_PAGER_COMPRESS must be auto, on or off, "
                    f"got {COMPRESS!r}"
                )
            compress = R < E if COMPRESS == "auto" else COMPRESS == "on"
            _store = Store(
                num_layers=get_current_vllm_config().model_config.hf_text_config.num_hidden_layers,
                num_experts=E,
                ram_slots=R,
                # Two faces of shared working rows: chunk k+1 reads into the
                # face whose copies finished with chunk k-1.
                working_rows=2 * WORKING_ROWS,
                row_bytes={
                    n: p[0].numel() * p.element_size() for n, p in params.items()
                },
                compress=compress,
                pitch_ratio=PITCH_RATIO,
                path=SSD_PATH,
                device=params["w13_weight"].device,
            )
        self._expert_pager_store = _store
        self._expert_pager_layer = _store.add_layer()

        # Compression assumes raw fp8 rows. Rows repacked by Marlin scatter the
        # exponent bits and do not compress.
        if _store.compress and (
            getattr(self.quant_method, "fp8_backend", None) == Fp8MoeBackend.MARLIN
        ):
            raise RuntimeError(
                f"{self.layer_name}: vllm-expert-pager compresses raw fp8 rows and "
                "does not support the MARLIN backend; set "
                "VLLM_EXPERT_PAGER_COMPRESS=off"
            )
        # Free the (E, ...) allocated by create_weights and keep only the shape.
        # process_weights_after_loading looks at every expert, so present E rows
        # through a stride-0 expand. _expert_pager_setup shrinks it back to one
        # row afterwards.
        for p in params.values():
            p.data = torch.empty(
                (1, *p.shape[1:]), dtype=p.dtype, device=p.device
            ).expand(p.shape)
        # w13 arrives as separate gate and up shards, so stage per expert until
        # both are in.
        self._expert_pager_stage: dict[int, tuple[torch.Tensor, set[str]]] = {}

        # quant_method is a per-layer instance. Set up the slab right after the
        # post-load processing.
        original = self.quant_method.process_weights_after_loading

        def process_weights_after_loading(layer: RoutedExperts) -> None:
            original(layer)
            layer._expert_pager_setup()

        self.quant_method.process_weights_after_loading = process_weights_after_loading

    # ---- expert_map override ----

    @property
    def expert_map(self) -> torch.Tensor | None:
        if self._expert_pager_expert_map is not None:
            return self._expert_pager_expert_map
        return super().expert_map

    # ---- Loading ----

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        if param is self.w13_weight:
            N, H = self.intermediate_size_per_partition, self.hidden_size
            buf, seen = self._expert_pager_stage.setdefault(
                expert_id,
                (
                    torch.empty((2 * N, H), dtype=loaded_weight.dtype, device="cpu"),
                    set(),
                ),
            )
            # w1 (gate) goes in the first half, w3 (up) in the second. Same as
            # RoutedExperts._load_w13.
            half = 0 if shard_id == "w1" else N
            buf[half : half + N].copy_(loaded_weight)
            seen.add(shard_id)
            if seen != {"w1", "w3"}:
                return True if return_success else None
            del self._expert_pager_stage[expert_id]
            self._expert_pager_store.write(
                self._expert_pager_layer, expert_id, "w13_weight", buf
            )
        elif param is self.w2_weight:
            self._expert_pager_store.write(
                self._expert_pager_layer, expert_id, "w2_weight", loaded_weight
            )
        else:
            return super().weight_loader(
                param, loaded_weight, weight_name, shard_id, expert_id, return_success
            )
        return True if return_success else None

    # ---- After loading ----

    def _expert_pager_setup(self) -> None:
        # triton only exists in GPU environments, so import it here.
        from vllm_expert_pager.acquire import AcquirePair
        from vllm_expert_pager.gather import as_rows

        store = self._expert_pager_store
        params = {name: getattr(self, name) for name in PARAMS}
        for name, p in params.items():
            # Undo the expand and shrink back to one row. The leading slice is
            # contiguous, so contiguous() would not copy and the original
            # (E, ...) would stay alive.
            p.data = p.data[:1].clone()
            if p.data.numel() * p.element_size() != store.row_bytes[name]:
                raise RuntimeError(
                    f"{self.layer_name}: {name} row has "
                    f"{p.data.numel() * p.element_size()} B after "
                    f"process_weights_after_loading, expected {store.row_bytes[name]}"
                )
        device = params["w13_weight"].device
        E = self.global_num_experts

        self._expert_pager_table = ExpertTable(E, CACHE_SLOTS, device)
        self._expert_pager_ram = ExpertTable(E, store.ram_slots, device)
        # Warm start: at load time expert e < R was placed in RAM-tier slot e.
        R = store.ram_slots
        self._expert_pager_ram.slot_of[:R] = self._expert_pager_ram.experts[:R]
        self._expert_pager_ram.expert_in[:R] = self._expert_pager_ram.experts[:R]
        # For decode (K <= S) both tables are updated by one kernel.
        self._expert_pager_acquire = AcquirePair(
            self._expert_pager_table, self._expert_pager_ram
        )
        # The expert_map read by the kernel. int32 at a fixed address. acquire
        # writes it directly; the working-buffer path updates it per chunk with
        # copy_.
        self._expert_pager_map = self._expert_pager_acquire.expert_map

        # Scales stay in VRAM as (E, ...). The kernel indexes scales with the
        # same slot number as the weights (off_experts in fused_moe_kernel), so
        # keep a copy in slot order alongside the slab and let gather copy it
        # together with the weights.
        scale_name = getattr(self.quant_method, "weight_scale_name", None)
        if scale_name is None:
            raise RuntimeError(
                f"{self.layer_name}: {type(self.quant_method).__name__} has no "
                "weight_scale_name; vllm-expert-pager only knows how to stage FP8 "
                "scales"
            )
        # Same order as PARAMS (w13, w2). gather takes two-element lists in this
        # order.
        self._expert_pager_scale_names = [f"{w}_{scale_name}" for w in ("w13", "w2")]
        scales = {name: getattr(self, name) for name in self._expert_pager_scale_names}
        staged = {**params, **scales}

        self._expert_pager_cache_slab = {
            name: torch.empty((CACHE_SLOTS, *p.shape[1:]), dtype=p.dtype, device=device)
            for name, p in staged.items()
        }
        W = WORKING_ROWS
        global _working_slab, _prefill_staging
        if _working_slab is None:
            if store.compress:
                least = max(
                    -(-CACHE_SLOTS * store.pitch[n] // store.row_bytes[n])
                    for n in PARAMS
                )
                if W < least:
                    raise ValueError(
                        f"VLLM_EXPERT_PAGER_WORKING_ROWS={W} cannot hold the decode "
                        f"staging of VLLM_EXPERT_PAGER_CACHE_SLOTS={CACHE_SLOTS} "
                        f"compressed rows; set it to at least {least}"
                    )
            _working_slab = {
                name: torch.empty((W, *p.shape[1:]), dtype=p.dtype, device=device)
                for name, p in staged.items()
            }
            if store.compress:
                _prefill_staging = [
                    torch.empty((W, store.pitch[n]), dtype=torch.uint8, device=device)
                    for n in PARAMS
                ]
        for name, p in staged.items():
            if _working_slab[name].shape[1:] != p.shape[1:]:
                raise RuntimeError(
                    f"{self.layer_name}: {name} shape differs from other layers"
                )
        self._expert_pager_working_slab = _working_slab

        # (S, words) / (W, words) int32 views handed to the gather kernel and
        # (S, N) / (W, N) uint8 views handed to decode_rows, as lists in name
        # order.
        order = list(PARAMS) + self._expert_pager_scale_names
        self._expert_pager_cache_rows = [
            as_rows(self._expert_pager_cache_slab[n]) for n in order
        ]
        self._expert_pager_working_rows = [as_rows(_working_slab[n]) for n in order]
        self._expert_pager_scale_rows = [
            as_rows(scales[n].data) for n in self._expert_pager_scale_names
        ]
        self._expert_pager_cache_bytes = [
            self._expert_pager_cache_slab[n].view(torch.uint8).reshape(CACHE_SLOTS, -1)
            for n in PARAMS
        ]
        self._expert_pager_working_bytes = [
            _working_slab[n].view(torch.uint8).reshape(W, -1) for n in PARAMS
        ]
        # In decode (K <= S) the shared working slab is idle, so its head
        # serves as the staging for compressed rows (S rows x pitch; checked
        # above to fit). Prefill expands into the working slab itself, so it
        # has its own staging. Without compression no staging is needed
        # (RAM-tier rows are copied straight into the slab).
        self._expert_pager_decode_staging = (
            [
                _working_slab[n]
                .view(torch.uint8)
                .reshape(-1)[: CACHE_SLOTS * store.pitch[n]]
                .view(CACHE_SLOTS, store.pitch[n])
                for n in PARAMS
            ]
            if store.compress
            else None
        )
        self._expert_pager_prefill_staging = _prefill_staging
        self._expert_pager_params = staged
        # Prefill intercepts moe_sum to take the top-k sum once across the
        # chunks (_expert_pager_forward_working). Only TritonExperts'
        # moe_sum(cache3, output) is known.
        experts = self.quant_method.moe_kernel.fused_experts
        if getattr(type(experts), "moe_sum", None) is not TritonExperts.moe_sum:
            raise RuntimeError(
                f"{self.layer_name}: vllm-expert-pager chunks the prefill through "
                f"TritonExperts.moe_sum, but the kernel is {type(experts).__name__}"
            )
        self._expert_pager_experts = experts

        logger.info_once(
            "vllm-expert-pager: %d cache slots/layer, %d RAM slots/layer, "
            "%d working rows, rows %s",
            CACHE_SLOTS,
            R,
            W,
            f"compressed to {'+'.join(str(store.pitch[n]) for n in PARAMS)} B"
            if store.compress
            else "raw",
        )

    # ---- forward ----

    def forward_modular(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm_expert_pager.codec import decode_rows
        from vllm_expert_pager.gather import gather_rows

        store = self._expert_pager_store
        if store.failed is not None:
            raise RuntimeError(f"vllm-expert-pager: SSD read failed\n{store.failed}")
        vram = self._expert_pager_table
        # Choose the path from the shape alone, so capture and replay of the
        # graph take the same path.
        if topk_ids.numel() > vram.num_slots:
            return self._expert_pager_forward_working(
                x, topk_weights, topk_ids, shared_experts, shared_experts_input
            )

        # Decode. Both tables are updated by one kernel, and expert_map is
        # written directly into self._expert_pager_map. The RAM tier follows the
        # same rule as the VRAM tier and is touched every step, so RAM stays a
        # superset of VRAM.
        layer = self._expert_pager_layer
        staging = self._expert_pager_decode_staging
        todo, ssd_todo = self._expert_pager_acquire(topk_ids)
        dst_row, src_row = (
            self._expert_pager_acquire.vram_slot,
            self._expert_pager_acquire.ram_slot,
        )
        cached_row = vram.no_cache  # read everything from the RAM tier
        base = layer * store.ram_slots
        lut = store.layer_lut(layer)
        rows = (
            [store.view[n] for n in PARAMS],
            self._expert_pager_cache_rows[:2],
            self._expert_pager_cache_rows[:2],
            [t.view(torch.int32) for t in staging] if staging is not None else None,
            self._expert_pager_scale_rows,
            self._expert_pager_cache_rows[2:],
        )

        # Overlap the SSD read request and the gather of the experts that are
        # in RAM in a single launch. With compression the rows go to staging and
        # a separate launch expands them: it only reads and writes device
        # memory, so it is fast and overlaps other kernels. Fusing the copy into
        # the expansion issued the copy's requests all at once and halved the
        # bandwidth.
        gather_rows(
            todo, dst_row, cached_row, src_row, base, *rows,
            fetch=(ssd_todo, layer, store),
        )  # fmt: skip
        if store.compress:
            decode_rows(
                todo, dst_row, cached_row, staging, self._expert_pager_cache_bytes,
                lut, skip=ssd_todo,
            )  # fmt: skip
        # Experts that were in RAM are done by the launch above. Copy the ones
        # read from SSD.
        gather_rows(ssd_todo, dst_row, cached_row, src_row, base, *rows)
        if store.compress:
            decode_rows(
                ssd_todo, dst_row, cached_row, staging,
                self._expert_pager_cache_bytes, lut,
            )  # fmt: skip

        self._expert_pager_log()
        return self._expert_pager_run(
            self._expert_pager_cache_slab, x, topk_weights, topk_ids,
            shared_experts, shared_experts_input,
        )  # fmt: skip

    def _expert_pager_forward_working(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        """Prefill (K > S): run the experts through the W-row working slab in chunks.

        The experts may not fit in the slots, so the VRAM table is not updated.
        Experts that hit are copied from the cache slab within VRAM. The RAM
        tier acquires when K <= R and is otherwise seeded with this step's most
        referenced experts. Experts read from SSD land in their table slot, not
        in the shared working rows.

        Chunk k: copy the experts that need no wait -> publish chunk k+1's SSD
        read -> wait for chunk k's read -> copy the experts that were read ->
        run the MoE with only chunk k's experts in expert_map. The number of
        chunks does not depend on the values, so this runs under graphs too.
        The shared experts are passed to the first chunk only (this vLLM
        returns the routed output alone and keeps the shared output inside
        SharedExperts).

        To match a single run bit for bit, no chunk runs the MoE to the end.
        The kernel writes each expert's contribution (token, k) to the bf16
        intermediate_cache3 and only the top-k sum (moe_sum) is taken in fp32
        before one rounding to bf16. Running moe_sum per chunk would round a
        partial sum once per chunk, so moe_sum is intercepted: the
        contributions are collected into one buffer (the kernel writes zeros
        for the rows of experts outside the chunk, so adding them rounds
        nothing) and the real moe_sum runs once, in the last chunk. The last
        chunk's return value is the answer.
        """
        from vllm_expert_pager.codec import decode_rows
        from vllm_expert_pager.gather import (
            fetch_publish_rows,
            fetch_wait_rows,
            gather_rows,
        )

        store = self._expert_pager_store
        vram, ram = self._expert_pager_table, self._expert_pager_ram
        E, W = self.global_num_experts, WORKING_ROWS
        R, layer = store.ram_slots, self._expert_pager_layer

        need = ram.need_mask(topk_ids)
        cached_row = vram.slot_of[:E]
        before = ram.slot_of[:E].clone()
        if topk_ids.numel() <= R:
            ram.acquire(topk_ids)
        else:
            ram.seed(vram, topk_ids, need)
        dst_row, src_row, maps, todo, ssd, late = ram.plan_chunks(
            before, need, cached_row, W, layer * R, store.working_base
        )

        staging = self._expert_pager_prefill_staging
        lut = store.layer_lut(layer)
        rows = (
            [store.view[n] for n in PARAMS],
            self._expert_pager_cache_rows[:2],
            self._expert_pager_working_rows[:2],
            [t.view(torch.int32) for t in staging] if staging is not None else None,
            self._expert_pager_scale_rows,
            self._expert_pager_working_rows[2:],
        )
        tickets = store.ticket_chunk

        def copy(part: torch.Tensor) -> None:
            """Copy from the RAM tier (and the shared rows read from SSD); with
            compression, into staging and then expand. VRAM hits are copied raw
            from the cache slab."""
            gather_rows(part, dst_row, cached_row, src_row, 0, *rows)
            if store.compress:
                decode_rows(
                    part, dst_row, cached_row, staging,
                    self._expert_pager_working_bytes, lut,
                )  # fmt: skip

        def chunk(t: torch.Tensor, k: int) -> torch.Tensor:
            return t[k * W : (k + 1) * W]

        def ticket(k: int) -> torch.Tensor:
            return tickets[k % 2 : k % 2 + 1]

        N = maps.shape[0]
        experts = self._expert_pager_experts
        moe_sum = type(experts).moe_sum
        acc = None

        def collect(cache3: torch.Tensor, output: torch.Tensor) -> None:
            """Called by chunk k's MoE. Collects the contributions and runs the
            real moe_sum in the last chunk."""
            nonlocal acc
            acc = cache3.clone() if acc is None else acc.add_(cache3)
            if k == N - 1:
                moe_sum(experts, acc, output)

        fetch_publish_rows(chunk(ssd, 0), src_row, 0, layer, store, ticket(0))
        # The instance attribute shadows the class method, for the chunks only.
        experts.moe_sum = collect
        try:
            for k in range(N):
                copy(chunk(todo, k))
                if k + 1 < N:
                    fetch_publish_rows(
                        chunk(ssd, k + 1), src_row, 0, layer, store, ticket(k + 1)
                    )
                fetch_wait_rows(chunk(ssd, k), store, ticket(k))
                copy(chunk(late, k))
                self._expert_pager_map.copy_(maps[k])
                # The return values before the last chunk did not go through
                # moe_sum and are not read.
                out = self._expert_pager_run(
                    self._expert_pager_working_slab, x, topk_weights, topk_ids,
                    shared_experts if k == 0 else None,
                    shared_experts_input if k == 0 else None,
                )  # fmt: skip
        finally:
            del experts.moe_sum
        self._expert_pager_log()
        return out

    def _expert_pager_run(
        self,
        slabs: dict[str, torch.Tensor],
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        """Point the weights at the slab and run the original MoE."""
        saved = {name: param.data for name, param in self._expert_pager_params.items()}
        try:
            for name, param in self._expert_pager_params.items():
                param.data = slabs[name]
            self._expert_pager_expert_map = self._expert_pager_map
            return super().forward_modular(
                x=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts=shared_experts,
                shared_experts_input=shared_experts_input,
            )
        finally:
            self._expert_pager_expert_map = None
            for name, data in saved.items():
                self._expert_pager_params[name].data = data

    def forward_monolithic(self, *args, **kwargs) -> torch.Tensor:
        raise RuntimeError(
            f"{self.layer_name}: vllm-expert-pager only supports modular MoE kernels, "
            f"but {type(self.quant_method).__name__} is monolithic. Expert weights "
            "would not be staged."
        )

    def _expert_pager_log(self) -> None:
        self._expert_pager_calls += 1
        if LOG_INTERVAL <= 0 or self._expert_pager_calls % LOG_INTERVAL:
            return
        # Reading the counters is a device -> host sync, which is not allowed
        # during capture.
        if torch.cuda.is_current_stream_capturing():
            return
        vram, ram = self._expert_pager_table, self._expert_pager_ram
        vh, vm, rh, rm = torch.stack(
            [vram.hits, vram.misses, ram.hits, ram.misses]
        ).tolist()
        logger.info(
            "vllm-expert-pager %s: vram hit %.1f%% (%d/%d), ram hit %.1f%% (%d/%d)",
            self.layer_name,
            100.0 * vh / (vh + vm) if vh + vm else 0.0,
            vh,
            vh + vm,
            100.0 * rh / (rh + rm) if rh + rm else 0.0,
            rh,
            rh + rm,
        )
        # SSD-tier wait time. There is one store for every layer, so only the
        # first layer reports it.
        store = self._expert_pager_store
        if self._expert_pager_layer == 0 and store.fetches:
            logger.info(
                "vllm-expert-pager ssd: %d fetches, %d reads, %.3f s, %.2f ms/fetch, "
                "%d re-publishes, %d give-ups, %d missed",
                store.fetches,
                store.reads,
                store.io_seconds,
                1e3 * store.io_seconds / store.fetches,
                int(store.req[store.retry_at]),
                int(store.req[store.giveup_at]),
                store.missed,
            )


def register() -> None:
    RoutedExperts.register_oot(ExpertPagerRoutedExperts)
