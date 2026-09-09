"""Hook ``RoutedExperts`` so that expert weights are referenced through a slab.

Only the experts needed in the current step are copied into the slab, and
``expert_map`` renumbers experts to slots before the kernel runs. Evicting the
weights to the CPU side is left to vLLM's standard
``--cpu-offload-params w13_weight w2_weight``.
"""

import os

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

from vllm_expert_pager.cache import ExpertCache

# vLLM only configures handlers and levels for the "vllm" logger
# (DEFAULT_LOGGING_CONFIG in logger.py), so take a name under it. A bare
# __name__ falls outside that hierarchy, the effective level becomes the root's
# WARNING, and every info message is dropped.
logger = init_logger(f"vllm.{__name__}")

# Number of cache slots per layer. One slot holds one expert (about 3 MiB for
# 35B-A3B).
CACHE_SLOTS = int(os.environ.get("VLLM_EXPERT_PAGER_CACHE_SLOTS", "32"))
# Interval for logging the hit rate, in forward calls per layer. 0 disables it.
LOG_INTERVAL = int(os.environ.get("VLLM_EXPERT_PAGER_LOG_INTERVAL", "1000"))

# Parameters that have an expert axis but are not expert weights. The same set
# that RoutedExperts.get_expert_weights() excludes from EPLB.
_NON_EXPERT_PARAMS = frozenset(
    {
        "e_score_correction_bias",
        "w13_input_scale",
        "w2_input_scale",
        "hash_indices_table",
    }
)

# Working buffers shared across all layers. Layers with identical shapes reuse
# one buffer. Layers run sequentially on the same stream, so there is no
# cross-layer race.
_working_slabs: dict[tuple, dict[str, torch.Tensor]] = {}


class ExpertPagerRoutedExperts(RoutedExperts):
    """``RoutedExperts`` that references expert weights through a slab.

    Uses the cache slab when the number of experts needed fits in the slots,
    and the working buffer otherwise.
    """

    _expert_pager_params: dict[str, torch.nn.Parameter] | None = None
    _expert_pager_expert_map: torch.Tensor | None = None
    _expert_pager_calls = 0

    # ---- expert_map override ----

    @property
    def expert_map(self) -> torch.Tensor | None:
        if self._expert_pager_expert_map is not None:
            return self._expert_pager_expert_map
        return super().expert_map

    # ---- Setup (on the first forward) ----

    def _expert_pager_setup(self, device: torch.device) -> None:
        if self.use_ep or self.local_num_experts != self.global_num_experts:
            raise RuntimeError(
                f"{self.layer_name}: vllm-expert-pager does not support expert "
                f"parallelism (local={self.local_num_experts}, "
                f"global={self.global_num_experts})"
            )

        params = {
            name: param
            for name, param in self.named_parameters()
            if param.dim() >= 2
            and param.shape[0] == self.local_num_experts
            and name not in _NON_EXPERT_PARAMS
        }
        missing = {"w13_weight", "w2_weight"} - params.keys()
        if missing:
            raise RuntimeError(
                f"{self.layer_name}: per-expert parameters {sorted(missing)} not found; "
                f"got {sorted(params)}"
            )

        # p.data is swapped for the slab later, so keep the original tensors as
        # the copy source.
        self._expert_pager_src = {name: param.data for name, param in params.items()}
        self._expert_pager_cache = ExpertCache(CACHE_SLOTS)
        self._expert_pager_cache_slab = {
            name: torch.empty(
                (CACHE_SLOTS, *param.shape[1:]), dtype=param.dtype, device=device
            )
            for name, param in params.items()
        }

        signature = tuple(
            (name, tuple(param.shape[1:]), param.dtype, str(device))
            for name, param in sorted(params.items())
        ) + (self.local_num_experts,)
        slab = _working_slabs.get(signature)
        if slab is None:
            slab = {
                name: torch.empty(
                    (self.local_num_experts, *param.shape[1:]),
                    dtype=param.dtype,
                    device=device,
                )
                for name, param in params.items()
            }
            _working_slabs[signature] = slab
        self._expert_pager_working_slab = slab

        self._expert_pager_map_host = torch.full(
            (self.global_num_experts,), -1, dtype=torch.int32
        )
        self._expert_pager_map = torch.empty(
            (self.global_num_experts,), dtype=torch.int32, device=device
        )

        # Weights offloaded through UVA live in pinned host memory, but they are
        # referenced through a device-visible view, so `.device` reports cuda.
        # They cannot be told apart from VRAM-resident weights. vLLM sets
        # `_vllm_is_uva_offloaded` on the Parameter (offloader/uva.py), but
        # quantization backends that re-register parameters in
        # process_weights_after_loading (Marlin) drop this attribute. So a
        # missing marker does not mean "resident".
        on_cpu = sorted(
            name for name, src in self._expert_pager_src.items() if src.device != device
        )
        uva = sorted(
            name
            for name, param in params.items()
            if name not in on_cpu and getattr(param, "_vllm_is_uva_offloaded", False)
        )
        unknown = sorted(params.keys() - set(on_cpu) - set(uva))

        if on_cpu:
            logger.warning_once(
                "vllm-expert-pager: expert weights %s are on plain CPU tensors rather "
                "than a UVA view. vLLM also patches the decoder layer forward in this "
                "case, which moves the whole layer to GPU every step. Check that "
                "VLLM_WEIGHT_OFFLOADING_DISABLE_UVA is unset and UVA is available.",
                ", ".join(on_cpu),
            )
        # The *_once variants memoize with lru_cache, so the arguments must be
        # hashable.
        logger.info_once(
            "vllm-expert-pager: %d slots/layer, slabbed params %s "
            "(on cpu: %s / uva: %s / vram or uva-without-marker: %s)",
            CACHE_SLOTS,
            ", ".join(sorted(params)),
            ", ".join(on_cpu) or "none",
            ", ".join(uva) or "none",
            ", ".join(unknown) or "none",
        )
        self._expert_pager_params = params

    # ---- Transfers ----

    def _expert_pager_load(
        self, dst: dict[str, torch.Tensor], slot: int, expert: int
    ) -> None:
        """Copy one expert from the source (CPU or UVA view) into the slab."""
        for name, slab in dst.items():
            slab[slot].copy_(self._expert_pager_src[name][expert], non_blocking=True)

    def _expert_pager_reuse(
        self, dst: dict[str, torch.Tensor], slot: int, cached_slot: int
    ) -> None:
        """Copy from the cache slab into the working buffer (cheap: stays in VRAM)."""
        for name, slab in dst.items():
            slab[slot].copy_(
                self._expert_pager_cache_slab[name][cached_slot], non_blocking=True
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
        if self._expert_pager_params is None:
            self._expert_pager_setup(x.device)

        # This forces a device -> host sync, which is why this cannot run under
        # CUDA graphs.
        experts = [
            e
            for e in torch.unique(topk_ids).tolist()
            if 0 <= e < self.global_num_experts
        ]

        if len(experts) <= self._expert_pager_cache.num_slots:
            slabs = self._expert_pager_cache_slab
            slots, misses = self._expert_pager_cache.acquire(experts)
            for i in misses:
                self._expert_pager_load(slabs, slots[i], experts[i])
        else:
            # Does not fit in the slots, so use the working buffer without
            # updating the cache.
            slabs = self._expert_pager_working_slab
            slots = list(range(len(experts)))
            for slot, expert in enumerate(experts):
                cached = self._expert_pager_cache.peek(expert)
                if cached is None:
                    self._expert_pager_load(slabs, slot, expert)
                else:
                    self._expert_pager_reuse(slabs, slot, cached)

        self._expert_pager_map_host.fill_(-1)
        if experts:
            self._expert_pager_map_host[experts] = torch.tensor(
                slots, dtype=torch.int32
            )
        self._expert_pager_map.copy_(self._expert_pager_map_host, non_blocking=True)

        self._expert_pager_log(len(experts))

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

    def _expert_pager_log(self, num_experts: int) -> None:
        self._expert_pager_calls += 1
        if LOG_INTERVAL <= 0 or self._expert_pager_calls % LOG_INTERVAL:
            return
        cache = self._expert_pager_cache
        total = cache.hits + cache.misses
        logger.info(
            "vllm-expert-pager %s: hit %.1f%% (%d/%d), last |U|=%d",
            self.layer_name,
            100.0 * cache.hits / total if total else 0.0,
            cache.hits,
            total,
            num_experts,
        )


def register() -> None:
    RoutedExperts.register_oot(ExpertPagerRoutedExperts)
