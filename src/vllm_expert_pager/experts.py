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
  value. For experts not in RAM the GPU asks a host thread to read them from
  SSD
"""

import os

import torch
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
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
# not fit, loading stops and reports the ratio needed. The largest ratio seen so
# far (over 24,576 experts) is 0.877.
PITCH_RATIO = float(os.environ.get("VLLM_EXPERT_PAGER_PITCH", "0.88"))
# Number of staging rows for copying compressed rows in prefill (the working
# buffer path). Rows are expanded this many at a time.
_PREFILL_STAGING_ROWS = 16
# Interval for logging the hit rate, in Python calls of forward per layer. 0
# disables it. CUDA graph replay does not run Python, so the log only appears
# when forward is called eagerly (prefill etc.). The counters themselves live
# on the device and include replays.
LOG_INTERVAL = int(os.environ.get("VLLM_EXPERT_PAGER_LOG_INTERVAL", "1000"))

# Expert weights referenced through the slab. Scales stay resident in VRAM and
# a copy in slot order is shown alongside the weights.
PARAMS = ("w13_weight", "w2_weight")

_store: Store | None = None
# Working buffer (weights and scales) shared across all layers, and the staging
# for prefill. Layers run sequentially on the same stream, so there is no
# cross-layer race.
_working_slab: dict[str, torch.Tensor] | None = None
_prefill_staging: list[torch.Tensor] | None = None


class ExpertPagerRoutedExperts(RoutedExperts):
    """``RoutedExperts`` that references expert weights through a slab.

    Uses the cache slab when ``topk_ids`` has no more elements than there are
    slots, and the working buffer otherwise. The decision is made on the shape
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
        # writes it directly; the lookup path updates it with copy_.
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
        global _working_slab, _prefill_staging
        if _working_slab is None:
            _working_slab = {
                name: torch.empty((E, *p.shape[1:]), dtype=p.dtype, device=device)
                for name, p in staged.items()
            }
            if store.compress:
                _prefill_staging = [
                    torch.empty(
                        (_PREFILL_STAGING_ROWS, store.pitch[n]),
                        dtype=torch.uint8,
                        device=device,
                    )
                    for n in PARAMS
                ]
        for name, p in staged.items():
            if _working_slab[name].shape[1:] != p.shape[1:]:
                raise RuntimeError(
                    f"{self.layer_name}: {name} shape differs from other layers"
                )
        self._expert_pager_working_slab = _working_slab

        # (S, W) / (E, W) int32 views handed to the gather kernel and (S, N) /
        # (E, N) uint8 views handed to decode_rows, as lists in name order.
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
            _working_slab[n].view(torch.uint8).reshape(E, -1) for n in PARAMS
        ]
        # In decode (K <= S) the shared working buffer is idle, so its head
        # serves as the staging for compressed rows (S rows x pitch). Prefill
        # expands into the working buffer itself, so it has its own staging.
        # Without compression no staging is needed (RAM-tier rows are copied
        # straight into the slab).
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

        logger.info_once(
            "vllm-expert-pager: %d cache slots/layer, %d RAM slots/layer, rows %s",
            CACHE_SLOTS,
            R,
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
        from vllm_expert_pager.gather import fetch_decode_rows, fetch_rows, gather_rows

        store = self._expert_pager_store
        if store.failed is not None:
            raise RuntimeError(f"vllm-expert-pager: SSD read failed\n{store.failed}")
        vram, ram = self._expert_pager_table, self._expert_pager_ram
        K = topk_ids.numel()
        R, layer = store.ram_slots, self._expert_pager_layer

        # Choose the path from the shape alone, so capture and replay of the
        # graph take the same path. The RAM tier follows the same rule as the
        # VRAM tier and is touched every step, so RAM stays a superset of VRAM.
        if K <= vram.num_slots:
            # Decode. Both tables are updated by one kernel, and expert_map is
            # written directly into self._expert_pager_map.
            slabs = self._expert_pager_cache_slab
            dst_rows = self._expert_pager_cache_rows
            dst_bytes = self._expert_pager_cache_bytes
            staging = self._expert_pager_decode_staging
            todo, ssd_todo = self._expert_pager_acquire(topk_ids)
            dst_row, src_row = (
                self._expert_pager_acquire.vram_slot,
                self._expert_pager_acquire.ram_slot,
            )
            cached_row = vram.no_cache  # read everything from the RAM tier
            base = layer * R
            decode = True
        else:
            # May not fit in the slots, so use the working buffer without
            # updating the table. Experts that hit are copied from the cache
            # slab within VRAM.
            slabs = self._expert_pager_working_slab
            dst_rows = self._expert_pager_working_rows
            dst_bytes = self._expert_pager_working_bytes
            staging = self._expert_pager_prefill_staging
            expert_map, dst_row, cached_row, todo = vram.lookup(topk_ids)
            self._expert_pager_map.copy_(expert_map)
            if K <= R:
                _, src_row, ssd_todo = ram.acquire(topk_ids)
                base = layer * R
            else:
                # Does not fit in the RAM tier (prefill). Misses are read into
                # the shared working rows.
                _, packed, in_ram, need = ram.lookup(topk_ids)
                src_row = torch.where(
                    in_ram >= 0, in_ram + layer * R, packed + store.working_base
                )
                ssd_todo = torch.where(in_ram[need.clamp(min=0)] >= 0, -1, need)
                base = 0
            decode = False
        src = [store.view[n] for n in PARAMS]
        lut = store.layer_lut(layer)
        rows = (
            src,
            self._expert_pager_cache_rows[:2],
            dst_rows[:2],
            [t.view(torch.int32) for t in staging] if staging is not None else None,
            self._expert_pager_scale_rows,
            dst_rows[2:],
        )
        if decode and store.compress:
            # Overlap the SSD read request with the gather and expansion of the
            # experts that are in RAM in a single launch.
            fetch_decode_rows(
                todo, ssd_todo, dst_row, src_row, base, layer, store,
                src, dst_bytes, staging, self._expert_pager_scale_rows, dst_rows[2:], lut,
            )  # fmt: skip
        elif decode:
            # Overlap the SSD read request and the gather of the experts that
            # are in RAM in a single launch.
            gather_rows(
                todo, dst_row, cached_row, src_row, base, *rows,
                fetch=(ssd_todo, layer, store),
            )  # fmt: skip
        else:
            fetch_rows(ssd_todo, src_row, base, layer, store)
        if store.path is not None and not torch.cuda.is_current_stream_capturing():
            # In eager mode Python keeps queuing kernels for later layers, and
            # once the launch queue is full that launch blocks while holding the
            # GIL (Triton's launcher does not release it). The GPU is waiting in
            # fetch for the host's done, and the host thread is waiting for the
            # GIL: a deadlock (reproduced on WSL2). Wait for the fetch to finish
            # with the GIL released. Graph replay does not run Python per layer,
            # so it needs none of this.
            torch.cuda.current_stream().synchronize()

        if decode:
            # Experts that were in RAM are done by the launch above. Copy the
            # ones read from SSD.
            gather_rows(ssd_todo, dst_row, cached_row, src_row, base, *rows)
            if store.compress:
                decode_rows(ssd_todo, dst_row, cached_row, staging, dst_bytes, lut)
        elif store.compress:
            # Prefill. A staging's worth of rows at a time, copy from the RAM
            # tier (and the shared rows read from SSD) into staging and expand.
            # VRAM hits are copied raw from the cache slab.
            B = staging[0].shape[0]
            for b in range(0, todo.shape[0], B):
                part = todo[b : b + B]
                gather_rows(part, dst_row, cached_row, src_row, base, *rows)
                decode_rows(part, dst_row, cached_row, staging, dst_bytes, lut)
        else:
            gather_rows(todo, dst_row, cached_row, src_row, base, *rows)

        self._expert_pager_log()

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
