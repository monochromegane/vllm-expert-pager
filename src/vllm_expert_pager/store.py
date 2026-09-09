"""RAM and SSD tiers for expert weights.

The RAM tier is an array of rows in pinned memory. The GPU reads it through a
UVA view and the host writes it through a numpy view. Each layer has R rows,
followed by E rows for prefill (shared by all layers); row numbers are unique
across layers so the gather kernel has a single source.

The SSD tier is a paging file of fixed-length records (the w13 row followed by
the w2 row) in [layer][expert] order. weight_loader writes it at load time; at
inference a host thread serves requests from the GPU with O_DIRECT preadv.
O_DIRECT keeps the page cache from holding a second copy of the RAM tier.

The GPU and host communicate through two words in pinned memory. The GPU writes
a request (layer, and the list of experts and rows) and advances req[0] (seq);
once the host has finished reading it writes the same value to done. The GPU
side is the fetch kernel in gather.py.
"""

import os
import threading
import time
import traceback

import torch
from vllm.logger import init_logger
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(f"vllm.{__name__}")

# Alignment required by O_DIRECT. Rows, records and the pinned base address
# must be multiples of this.
_ALIGN = 4096
# torch's pinned allocator (CachingHostAllocator) rounds allocation sizes up to
# the next power of two. The RAM tier takes tens of GiB per name in a single
# allocation, so a 20.5 GiB request would consume 32 GiB. Change the allocator
# setting so pinned allocations of this size or more are not rounded.
_PINNED_ROUND_LIMIT_MB = 1024


def _pinned_aligned(rows: int, words: int) -> torch.Tensor:
    """Pinned (rows, words) int32 tensor whose start is aligned to _ALIGN.

    torch's pinned allocator may carve a piece out of a larger block, so the
    start is not necessarily on a page boundary. Allocate extra and use the
    aligned part.
    """
    # The default device is cuda while the model is being built, so name the
    # CPU explicitly.
    slack = _ALIGN // 4
    flat = torch.empty(
        (rows * words + slack,), dtype=torch.int32, device="cpu", pin_memory=True
    )
    skip = (-flat.data_ptr() % _ALIGN) // 4
    return flat[skip : skip + rows * words].view(rows, words)


class Store:
    """One per model. The RAM tier's pinned buffer, the paging file and the host thread."""

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        ram_slots: int,
        row_bytes: dict[str, int],
        path: str | None,
        device: torch.device,
    ) -> None:
        L, E, R = num_layers, num_experts, ram_slots
        if not 1 <= R <= E:
            raise ValueError(
                f"VLLM_EXPERT_PAGER_RAM_SLOTS must be in [1, {E}], got {R}"
            )
        if R < E and path is None:
            raise ValueError(
                "VLLM_EXPERT_PAGER_SSD_PATH is required when "
                "VLLM_EXPERT_PAGER_RAM_SLOTS < num_experts"
            )
        for name, nbytes in row_bytes.items():
            if nbytes % _ALIGN:
                raise ValueError(
                    f"{name}: row of {nbytes} B is not a multiple of {_ALIGN}"
                )

        self.num_layers, self.num_experts, self.ram_slots = L, E, R
        self.layers = 0
        self.names = list(row_bytes)
        self.record_bytes = sum(row_bytes.values())
        self.offset = {}
        pos = 0
        for name in self.names:
            self.offset[name] = pos
            pos += row_bytes[name]

        # Rows [0, L*R) are the RAM tier (slot s of layer l is l*R + s); rows
        # [L*R, L*R+E) are the shared working rows.
        self.working_base = L * R
        torch._C._accelerator_setAllocatorSettings(
            f"pinned_max_round_threshold_mb:{_PINNED_ROUND_LIMIT_MB}"
        )
        logger.info(
            "vllm-expert-pager: RAM tier %d rows x %d B = %.2f GiB pinned",
            L * R + E,
            self.record_bytes,
            (L * R + E) * self.record_bytes / 2**30,
        )
        self.pinned = {
            name: _pinned_aligned(L * R + E, nbytes // 4)
            for name, nbytes in row_bytes.items()
        }
        self.view = {
            n: get_accelerator_view_from_cpu_tensor(t) for n, t in self.pinned.items()
        }
        self._np = {n: t.numpy() for n, t in self.pinned.items()}

        # req = [seq, layer, expert[B], row[B]]. B is the fetch kernel's BLOCK
        # (the power of two >= E).
        self.block = 1 << (E - 1).bit_length()
        self.req = torch.zeros(
            (2 + 2 * self.block,), dtype=torch.int64, device="cpu", pin_memory=True
        )
        self.done = torch.zeros((1,), dtype=torch.int64, device="cpu", pin_memory=True)
        self.req_view = get_accelerator_view_from_cpu_tensor(self.req)
        self.done_view = get_accelerator_view_from_cpu_tensor(self.done)
        self.seq = torch.zeros((1,), dtype=torch.int64, device=device)
        # Traceback of an I/O failure on the host thread. forward checks it and
        # raises.
        self.failed: str | None = None

        self.path = path if R < E else None
        self.fd_w = self.fd_r = None
        if self.path is not None:
            size = L * E * self.record_bytes
            # An existing file is only overwritten if it is a previous paging
            # file (same size).
            if os.path.exists(path) and os.path.getsize(path) != size:
                raise FileExistsError(
                    f"{path} exists and is not a paging file of {size} B; remove it first"
                )
            self.fd_w = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            self.fd_r = os.open(path, os.O_RDONLY | os.O_DIRECT)
            logger.info(
                "vllm-expert-pager: paging file %s (%.1f GiB)", path, size / 2**30
            )
        threading.Thread(
            target=self._serve, daemon=True, name="vllm-expert-pager-ssd"
        ).start()

    def add_layer(self) -> int:
        """Hand out a layer number. Layers register in construction order."""
        if self.layers >= self.num_layers:
            raise RuntimeError(
                f"more MoE layers than num_hidden_layers={self.num_layers}"
            )
        self.layers += 1
        return self.layers - 1

    # ---- Load time ----

    def write(self, layer: int, expert: int, name: str, row: torch.Tensor) -> None:
        """Put one expert's weights in the paging file and, if e < R, in RAM-tier slot e."""
        words = row.contiguous().view(torch.int32).reshape(-1)
        if words.numel() != self.pinned[name].shape[1]:
            raise ValueError(
                f"{name}: expert {expert} of layer {layer} has {words.numel()} words, "
                f"expected {self.pinned[name].shape[1]}"
            )
        if expert < self.ram_slots:
            self.pinned[name][layer * self.ram_slots + expert].copy_(words)
        if self.fd_w is not None:
            offset = (
                layer * self.num_experts + expert
            ) * self.record_bytes + self.offset[name]
            os.pwrite(self.fd_w, memoryview(words.numpy()), offset)

    # ---- Inference (host thread) ----

    def _serve(self) -> None:
        req, done, B = self.req.numpy(), self.done.numpy(), self.block
        last = 0
        while True:
            seq = int(req[0])
            if seq == last:
                time.sleep(50e-6)
                continue
            try:
                layer = int(req[1])
                for e, row in zip(
                    req[2 : 2 + B].tolist(), req[2 + B : 2 + 2 * B].tolist()
                ):
                    if e >= 0:
                        self._read(layer, e, row)
            except Exception:  # noqa: BLE001  keep the thread alive; always release the GPU
                self.failed = traceback.format_exc()
                logger.error("vllm-expert-pager: SSD read failed\n%s", self.failed)
            # Release the GPU even on failure. The next forward that runs Python
            # sees failed and stops.
            done[0] = seq
            last = seq

    def _read(self, layer: int, expert: int, row: int) -> None:
        if self.fd_r is None:
            raise RuntimeError(
                f"expert {expert} of layer {layer} is not in RAM and there is no paging file"
            )
        offset = (layer * self.num_experts + expert) * self.record_bytes
        n = os.preadv(self.fd_r, [self._np[name][row] for name in self.names], offset)
        if n != self.record_bytes:
            raise OSError(f"short read: {n} of {self.record_bytes} B at {offset}")
