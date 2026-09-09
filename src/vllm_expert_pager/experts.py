"""Hook ``RoutedExperts`` so that expert weights are referenced through a slab.

Expert weights live in three tiers: the VRAM slab, pinned RAM, and a paging
file on SSD. The plugin owns their placement from load time on and does not
use vLLM's CPU offload.

- Construction: the (E, ...) weights allocated by create_weights are replaced
  with one-row placeholders
- Loading: weight_loader is intercepted and each expert's weights are written
  to the paging file and to the RAM tier (experts numbered below R)
- Inference: only the experts needed in the step are copied into the slab, and
  expert_map renumbers experts to slots before the kernel runs. Decisions are
  made on device tensors and the host never reads a value. For experts not in
  RAM the GPU asks a host thread to read them from SSD
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
# Interval for logging the hit rate, in Python calls of forward per layer. 0
# disables it. CUDA graph replay does not run Python, so the log only appears
# when forward is called eagerly (prefill etc.). The counters themselves live
# on the device and include replays.
LOG_INTERVAL = int(os.environ.get("VLLM_EXPERT_PAGER_LOG_INTERVAL", "1000"))

# Expert weights referenced through the slab. Scales stay resident in VRAM.
PARAMS = ("w13_weight", "w2_weight")

_store: Store | None = None
# Working buffer shared across all layers. Layers run sequentially on the same
# stream, so there is no cross-layer race.
_working_slab: dict[str, torch.Tensor] | None = None


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
            _store = Store(
                num_layers=get_current_vllm_config().model_config.hf_text_config.num_hidden_layers,
                num_experts=E,
                ram_slots=R,
                row_bytes={
                    n: p[0].numel() * p.element_size() for n, p in params.items()
                },
                path=SSD_PATH,
                device=params["w13_weight"].device,
            )
        self._expert_pager_store = _store
        self._expert_pager_layer = _store.add_layer()

        # Free the (E, ...) allocated by create_weights and keep only the shape.
        # Marlin's process_weights_after_loading repacks every expert, so
        # present E rows through a stride-0 expand. _expert_pager_setup shrinks
        # it back to one row afterwards.
        for p in params.values():
            p.data = torch.empty(
                (1, *p.shape[1:]), dtype=p.dtype, device=p.device
            ).expand(p.shape)
        self._expert_pager_marlin = (
            getattr(self.quant_method, "fp8_backend", None) == Fp8MoeBackend.MARLIN
        )
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
                self._expert_pager_layer,
                expert_id,
                "w13_weight",
                self._expert_pager_pack(buf),
            )
        elif param is self.w2_weight:
            self._expert_pager_store.write(
                self._expert_pager_layer,
                expert_id,
                "w2_weight",
                self._expert_pager_pack(loaded_weight),
            )
        else:
            return super().weight_loader(
                param, loaded_weight, weight_name, shard_id, expert_id, return_success
            )
        return True if return_success else None

    def _expert_pager_pack(self, w: torch.Tensor) -> torch.Tensor:
        """Turn one expert's (n, k) into the form the kernel reads.

        As-is for Triton, repacked for Marlin.
        """
        if not self._expert_pager_marlin:
            return w
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            pack_fp8_to_int32,
        )

        # repack_weight from prepare_fp8_moe_layer_for_marlin applied to a
        # single expert.
        q = pack_fp8_to_int32(w.cuda(), size_k_first=False).T.contiguous()
        perm = torch.empty(0, dtype=torch.int, device=q.device)
        return ops.gptq_marlin_repack(
            b_q_weight=q, perm=perm, size_k=w.shape[1], size_n=w.shape[0], num_bits=8
        ).cpu()

    # ---- After loading ----

    def _expert_pager_setup(self) -> None:
        # triton only exists in GPU environments, so import it here.
        from vllm_expert_pager.gather import as_rows

        store = self._expert_pager_store
        params = {name: getattr(self, name) for name in PARAMS}
        for name, p in params.items():
            # Shrink the (E, ...) repacked by Marlin back to one row; for Triton
            # this undoes the expand. The leading slice is contiguous, so
            # contiguous() would not copy and the original (E, ...) would stay
            # alive.
            p.data = p.data[:1].clone()
            if as_rows(p.data).shape[1] != store.pinned[name].shape[1]:
                raise RuntimeError(
                    f"{self.layer_name}: {name} row has {as_rows(p.data).shape[1]} words "
                    f"after process_weights_after_loading, RAM rows have "
                    f"{store.pinned[name].shape[1]}"
                )
        device = params["w13_weight"].device
        E = self.global_num_experts

        self._expert_pager_table = ExpertTable(E, CACHE_SLOTS, device)
        self._expert_pager_ram = ExpertTable(E, store.ram_slots, device)
        # Warm start: at load time expert e < R was placed in RAM-tier slot e.
        R = store.ram_slots
        self._expert_pager_ram.slot_of[:R] = self._expert_pager_ram.experts[:R]
        self._expert_pager_ram.expert_in[:R] = self._expert_pager_ram.experts[:R]

        self._expert_pager_cache_slab = {
            name: torch.empty((CACHE_SLOTS, *p.shape[1:]), dtype=p.dtype, device=device)
            for name, p in params.items()
        }
        global _working_slab
        if _working_slab is None:
            _working_slab = {
                name: torch.empty((E, *p.shape[1:]), dtype=p.dtype, device=device)
                for name, p in params.items()
            }
        for name, p in params.items():
            if _working_slab[name].shape[1:] != p.shape[1:]:
                raise RuntimeError(
                    f"{self.layer_name}: {name} shape differs from other layers"
                )
        self._expert_pager_working_slab = _working_slab

        # (S, W) / (E, W) int32 views handed to the gather kernel.
        self._expert_pager_cache_rows = {
            n: as_rows(t) for n, t in self._expert_pager_cache_slab.items()
        }
        self._expert_pager_working_rows = {
            n: as_rows(t) for n, t in _working_slab.items()
        }

        # The expert_map read by the kernel. int32 at a fixed address, updated
        # with copy_ every step.
        self._expert_pager_map = torch.empty((E,), dtype=torch.int32, device=device)
        self._expert_pager_params = params

        logger.info_once(
            "vllm-expert-pager: %d cache slots/layer, %d RAM slots/layer, %s",
            CACHE_SLOTS,
            R,
            "marlin repack" if self._expert_pager_marlin else "raw rows",
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
        from vllm_expert_pager.gather import fetch_rows, gather_rows

        store = self._expert_pager_store
        if store.failed is not None:
            raise RuntimeError(f"vllm-expert-pager: SSD read failed\n{store.failed}")
        vram, ram = self._expert_pager_table, self._expert_pager_ram
        K = topk_ids.numel()
        R, layer = store.ram_slots, self._expert_pager_layer

        # Choose the path from the shape alone, so capture and replay of the
        # graph take the same path.
        if K <= vram.num_slots:
            slabs = self._expert_pager_cache_slab
            dst_rows = self._expert_pager_cache_rows
            expert_map, dst_row, todo = vram.acquire(topk_ids)
            cached_row = vram.no_cache  # read everything from the RAM tier
        else:
            # May not fit in the slots, so use the working buffer without
            # updating the table. Experts that hit are copied from the cache
            # slab within VRAM.
            slabs = self._expert_pager_working_slab
            dst_rows = self._expert_pager_working_rows
            expert_map, dst_row, cached_row, todo = vram.lookup(topk_ids)

        # The RAM tier follows the same rule. It is touched every step, so RAM
        # stays a superset of VRAM.
        if K <= R:
            _, src_row, ssd_todo = ram.acquire(topk_ids)
            base = layer * R
        else:
            # Does not fit in the RAM tier (prefill). Misses are read into the
            # shared working rows.
            _, packed, in_ram, need = ram.lookup(topk_ids)
            src_row = torch.where(
                in_ram >= 0, in_ram + layer * R, packed + store.working_base
            )
            ssd_todo = torch.where(in_ram[need.clamp(min=0)] >= 0, -1, need)
            base = 0
        fetch_rows(ssd_todo, src_row, base, layer, store)

        for name in slabs:
            gather_rows(
                todo,
                dst_row,
                cached_row,
                src_row,
                base,
                store.view[name],
                self._expert_pager_cache_rows[name],
                dst_rows[name],
            )
        self._expert_pager_map.copy_(expert_map)

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


def register() -> None:
    RoutedExperts.register_oot(ExpertPagerRoutedExperts)
