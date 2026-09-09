"""Triton kernels that copy one expert into a slab and request reads from the host.

The source is either the RAM tier (pinned memory seen from the device as a UVA
view) or the cache slab; the destination is the cache slab or the working
buffer. Which experts to copy is passed as index tensors on the device, so the
host needs to know nothing and the kernels run under CUDA graphs.

The SMs copy with loads and stores instead of ``cudaMemcpyAsync``. A memcpy has
fixed addresses, so putting one in a graph would mean "copy the same expert
every time".
"""

import torch
import triton
import triton.language as tl

# Number of words one program copies (int32, so 16 KiB).
_BLOCK = 4096


@triton.jit
def _gather_rows_kernel(
    todo_ptr,  # int64[K]  experts to copy. -1 is an empty entry
    dst_row_ptr,  # int64[E]  expert -> destination row
    cache_row_ptr,  # int64[E]  expert -> row on the cache slab; -1 reads from RAM
    src_row_ptr,  # int64[E]  expert -> RAM-tier row (before adding base)
    base,  # added to RAM-tier rows (first row of the layer)
    src_ptr,  # int32[*, W]  RAM tier (UVA view)
    cache_ptr,  # int32[S, W]  cache slab
    dst_ptr,  # int32[*, W]  destination
    W,  # words per row
    BLOCK: tl.constexpr,
):
    j = tl.program_id(0)
    chunk = tl.program_id(1)

    e = tl.load(todo_ptr + j)
    active = e >= 0
    # Keep the index in range for empty entries; the value read is discarded.
    e = tl.maximum(e, 0)
    dst_row = tl.maximum(tl.load(dst_row_ptr + e), 0)
    cache_row = tl.load(cache_row_ptr + e)
    src_row = tl.load(src_row_ptr + e) + base

    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    in_row = active & (offs < W)
    from_cache = in_row & (cache_row >= 0)
    from_src = in_row & (cache_row < 0)

    # A masked-off load does not touch memory, so issuing both reads only one.
    x_cache = tl.load(
        cache_ptr + tl.maximum(cache_row, 0) * W + offs, mask=from_cache, other=0
    )
    x_src = tl.load(src_ptr + src_row * W + offs, mask=from_src, other=0)
    x = tl.where(cache_row >= 0, x_cache, x_src)
    tl.store(dst_ptr + dst_row * W + offs, x, mask=in_row)


@triton.jit
def _fetch_kernel(
    todo_ptr,  # int64[K]  experts to read from SSD. -1 is an empty entry
    row_ptr,  # int64[E]  expert -> RAM-tier row (before adding base)
    K,
    base,
    layer,
    seq_ptr,  # int64[1]  sequence number on the device
    req_ptr,  # int64[2 + 2*BLOCK]  pinned. [seq, layer, expert[BLOCK], row[BLOCK]]
    done_ptr,  # int64[1]  pinned. The host writes the seq it has finished
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    e = tl.load(todo_ptr + offs, mask=offs < K, other=-1)
    if tl.sum((e >= 0).to(tl.int32), axis=0) > 0:
        row = tl.load(row_ptr + tl.maximum(e, 0)) + base
        # Write all of BLOCK so leftovers from the previous request are never
        # read.
        tl.store(req_ptr + 2 + offs, e)
        tl.store(req_ptr + 2 + BLOCK + offs, tl.where(e >= 0, row, -1))
        seq = tl.load(seq_ptr) + 1
        tl.store(seq_ptr, seq)
        # Triton may specialize layer into a constant, on which dtype conversion
        # cannot be called, so add it to an int64 zero to match the type.
        tl.store(req_ptr + 1, seq * 0 + layer)
        # Publish seq only after every thread has finished writing. The release
        # makes the preceding writes visible to the host.
        tl.debug_barrier()
        tl.atomic_xchg(req_ptr, seq, sem="release", scope="sys")
        d = tl.load(done_ptr, volatile=True)
        while d < seq:
            d = tl.load(done_ptr, volatile=True)


def as_rows(t: torch.Tensor) -> torch.Tensor:
    """View a tensor whose leading axis is the expert axis as ``(E, W)`` int32.

    Lets the copy treat the data as raw bytes regardless of dtype, so the same
    kernel works for fp8 and for Marlin's int32.
    """
    if not t.is_contiguous():
        raise ValueError("expert weights must be contiguous")
    row_bytes = t[0].numel() * t.element_size()
    if row_bytes % 4:
        raise ValueError(f"row size {row_bytes} B is not a multiple of 4")
    return t.view(torch.int32).reshape(t.shape[0], -1)


def gather_rows(
    todo: torch.Tensor,
    dst_row: torch.Tensor,
    cache_row: torch.Tensor,
    src_row: torch.Tensor,
    base: int,
    src: torch.Tensor,
    cache: torch.Tensor,
    dst: torch.Tensor,
) -> None:
    """Copy the rows of the experts listed in ``todo`` into ``dst``.

    ``src``, ``cache`` and ``dst`` are ``(*, W)`` int32 views made by
    ``as_rows``. ``cache_row`` decides whether each row is read from ``cache``
    or ``src``.
    """
    W = src.shape[1]
    grid = (todo.shape[0], triton.cdiv(W, _BLOCK))
    _gather_rows_kernel[grid](
        todo, dst_row, cache_row, src_row, base, src, cache, dst, W, BLOCK=_BLOCK
    )


def fetch_rows(
    todo: torch.Tensor, row: torch.Tensor, base: int, layer: int, store
) -> None:
    """Ask the host to read the experts in ``todo`` from SSD and wait.

    Each expert ``e`` lands in RAM-tier row ``row[e] + base``. Returns
    immediately if ``todo`` is empty.
    """
    _fetch_kernel[(1,)](
        todo,
        row,
        todo.shape[0],
        base,
        layer,
        store.seq,
        store.req_view,
        store.done_view,
        BLOCK=store.block,
    )
