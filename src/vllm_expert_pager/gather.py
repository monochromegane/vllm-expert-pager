"""Triton kernel that copies one expert's weights into a slab.

The source is either a UVA view (pinned host memory seen from the device) or
the cache slab; the destination is the cache slab or the working buffer. Which
experts to copy is passed as index tensors on the device, so the host needs to
know nothing and the kernel runs under CUDA graphs.

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
    cache_row_ptr,  # int64[E]  expert -> row on the cache slab; -1 reads from UVA
    uva_ptr,  # int32[E, W]  source (UVA view or VRAM resident)
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

    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    in_row = active & (offs < W)
    from_cache = in_row & (cache_row >= 0)
    from_uva = in_row & (cache_row < 0)

    # A masked-off load does not touch memory, so issuing both reads only one.
    x_cache = tl.load(
        cache_ptr + tl.maximum(cache_row, 0) * W + offs, mask=from_cache, other=0
    )
    x_uva = tl.load(uva_ptr + e * W + offs, mask=from_uva, other=0)
    x = tl.where(cache_row >= 0, x_cache, x_uva)
    tl.store(dst_ptr + dst_row * W + offs, x, mask=in_row)


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
    uva: torch.Tensor,
    cache: torch.Tensor,
    dst: torch.Tensor,
) -> None:
    """Copy the rows of the experts listed in ``todo`` into ``dst``.

    The arguments are ``(*, W)`` int32 views made by ``as_rows``. ``cache_row``
    decides whether each row is read from ``uva`` or ``cache``.
    """
    W = uva.shape[1]
    grid = (todo.shape[0], triton.cdiv(W, _BLOCK))
    _gather_rows_kernel[grid](
        todo, dst_row, cache_row, uva, cache, dst, W, BLOCK=_BLOCK
    )
