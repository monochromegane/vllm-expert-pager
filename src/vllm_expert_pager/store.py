"""RAM and SSD tiers for expert weights. Both hold the rows compressed (codec.py).

The RAM tier is an array of rows in pinned memory. The GPU reads it through a
UVA view and the host writes it through a numpy view. Each layer has R rows,
followed by E rows for prefill (shared by all layers); row numbers are unique
across layers so the gather kernel has a single source.

The SSD tier is a paging file of fixed-length records (the w13 row followed by
the w2 row, each compressed and rounded up to its pitch) in [layer][expert]
order. weight_loader writes it at load time; at inference a host thread serves
requests from the GPU with O_DIRECT preadv, and the experts of one layer are
read concurrently by a thread pool. O_DIRECT keeps the page cache from holding
a second copy of the RAM tier. The host does not expand the rows; the GPU does,
after copying them into VRAM.

The Huffman LUT differs per row but is only 512 B, so it is not part of the
record. It lives in a VRAM-resident table like the scales (``lut``, indexed by
layer and expert). When gather also expands, it copies per group, and a LUT per
group would add 3.6% to the PCIe traffic (codec.py).

With ``compress=False`` the rows are stored raw. When every expert fits in RAM
there are no SSD reads, and the expansion costs more than the 12% saved on PCIe.

The GPU and host communicate through two words in pinned memory. The GPU writes
a request (layer, and the list of experts and rows) and advances req[0] (seq);
once the host has finished reading it writes the same value to done. The GPU
side is the fetch kernel in gather.py.
"""

import faulthandler
import os
import signal
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, wait

import torch
from vllm.logger import init_logger
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from vllm_expert_pager import codec

logger = init_logger(f"vllm.{__name__}")

# Alignment required by O_DIRECT. Rows, records and the pinned base address
# must be multiples of this.
_ALIGN = 4096
# torch's pinned allocator (CachingHostAllocator) rounds allocation sizes up to
# the next power of two. The RAM tier takes tens of GiB per name in a single
# allocation, so a 20.5 GiB request would consume 32 GiB. Change the allocator
# setting so pinned allocations of this size or more are not rounded.
_PINNED_ROUND_LIMIT_MB = 1024
# Number of threads that read the experts of one layer concurrently. The NVMe
# of the test machine (WSL2) delivers about 2.9 GB/s with a single outstanding
# request and saturates at about 4.7 GB/s from eight requests on. Splitting one
# record into pieces does not make it faster.
_READ_WORKERS = 8
# Warn when serving one request takes longer than this (to tell a stall from
# slow I/O).
_SLOW_SECONDS = 1.0


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
        compress: bool,
        pitch_ratio: float,
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
            unit = codec.GROUP if compress else _ALIGN
            if nbytes % unit:
                raise ValueError(
                    f"{name}: row of {nbytes} B is not a multiple of {unit}"
                )

        self.num_layers, self.num_experts, self.ram_slots = L, E, R
        self.layers = 0
        self.device = device
        self.compress = compress
        self.names = list(row_bytes)
        # Raw (expanded) rows, and the fixed length of the rows in the tiers.
        # The pitch is a multiple of _ALIGN, so it satisfies O_DIRECT.
        self.row_bytes = dict(row_bytes)
        self.pitch = {
            name: codec.pitch_of(nbytes, pitch_ratio, _ALIGN) if compress else nbytes
            for name, nbytes in row_bytes.items()
        }
        self.record_bytes = sum(self.pitch.values())
        self.offset = {}
        pos = 0
        for name in self.names:
            self.offset[name] = pos
            pos += self.pitch[name]

        # Rows [0, L*R) are the RAM tier (slot s of layer l is l*R + s); rows
        # [L*R, L*R+E) are the shared working rows.
        self.working_base = L * R
        torch._C._accelerator_setAllocatorSettings(
            f"pinned_max_round_threshold_mb:{_PINNED_ROUND_LIMIT_MB}"
        )
        logger.info(
            "vllm-expert-pager: RAM tier %d rows x %d B (%s) = %.2f GiB pinned",
            L * R + E,
            self.record_bytes,
            f"compressed from {sum(row_bytes.values())} B" if compress else "raw",
            (L * R + E) * self.record_bytes / 2**30,
        )
        self.pinned = {
            name: _pinned_aligned(L * R + E, nbytes // 4)
            for name, nbytes in self.pitch.items()
        }
        # Per-row Huffman LUTs: lut[l, e, i] belongs to expert e of layer l,
        # name i. Without compression they are never read, so allocate just
        # enough to have a pointer to pass to the kernel.
        self.lut = torch.zeros(
            (L if compress else 1, E, len(self.names), codec.LUT_BYTES),
            dtype=torch.uint8,
            device=device,
        )
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
        # Statistics. fetches counts the requests, reads the experts read, and
        # io_seconds the total time from seeing a request to writing done. Only
        # the host thread writes them.
        self.fetches = self.reads = 0
        self.io_seconds = 0.0

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
        self._pool = ThreadPoolExecutor(
            max_workers=_READ_WORKERS, thread_name_prefix="vllm-expert-pager-read"
        )
        # For diagnosing a stall: kill -USR2 <EngineCore pid> dumps the Python
        # stack of every thread to stderr (the server log). It works even when
        # a thread is spinning in C code while holding the GIL.
        if threading.current_thread() is threading.main_thread():
            faulthandler.register(signal.SIGUSR2, all_threads=True)
        threading.Thread(
            target=self._serve, daemon=True, name="vllm-expert-pager-ssd"
        ).start()

    def layer_lut(self, layer: int) -> torch.Tensor:
        """The LUTs ``(E, names, 512)`` of layer ``layer``. Unused without compression."""
        return self.lut[layer if self.compress else 0]

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
        """Put one expert's weights (compressed if compress) in the paging file and the RAM tier.

        The RAM tier gets them only when e < R (slot e).
        """
        raw = row.contiguous().view(torch.uint8).reshape(-1)
        if raw.numel() != self.row_bytes[name]:
            raise ValueError(
                f"{name}: expert {expert} of layer {layer} has {raw.numel()} B, "
                f"expected {self.row_bytes[name]}"
            )
        if self.compress:
            record, lut = codec.encode(raw.to(self.device), self.pitch[name])
            self.lut[layer, expert, self.names.index(name)] = lut
            words = record.view(torch.int32).cpu()
        else:
            words = raw.view(torch.int32)
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
            t0 = time.perf_counter()
            layer = int(req[1])
            pairs = [
                (e, row)
                for e, row in zip(
                    req[2 : 2 + B].tolist(), req[2 + B : 2 + 2 * B].tolist()
                )
                if e >= 0
            ]
            futures = [self._pool.submit(self._read, layer, e, row) for e, row in pairs]
            # Even when a read fails, write done only after all of them finished.
            wait(futures)
            if time.perf_counter() - t0 > _SLOW_SECONDS:
                logger.warning(
                    "vllm-expert-pager: seq %d layer %d: %d SSD reads took %.1f s",
                    seq,
                    layer,
                    len(pairs),
                    time.perf_counter() - t0,
                )
            try:
                for f in futures:
                    f.result()
            except Exception:  # noqa: BLE001  keep the thread alive; always release the GPU
                self.failed = traceback.format_exc()
                logger.error("vllm-expert-pager: SSD read failed\n%s", self.failed)
            self.fetches += 1
            self.reads += len(pairs)
            self.io_seconds += time.perf_counter() - t0
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
