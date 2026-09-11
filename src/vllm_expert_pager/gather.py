"""Triton kernels that copy one expert into a slab and request reads from the host.

The source is either the RAM tier (pinned memory seen from the device as a UVA
view) or the cache slab; the destination is the cache slab or the working
buffer. Scales (VRAM resident, in expert order) are copied to the same slot,
because the MoE kernel indexes weights and scales with the same slot number.
Which experts to copy is passed as index tensors on the device, so the host
needs to know nothing and the kernels run under CUDA graphs.

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
def _copy_row(
    chunk,
    W,  # words per row
    src_ptr,  # int32[*, W]  RAM tier (UVA view)
    cache_ptr,  # int32[S, W]  cache slab
    dst_ptr,  # int32[*, W]  destination
    src_row,
    cache_row,
    dst_row,
    active,
    BLOCK: tl.constexpr,
):
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
def _copy_scale(
    SW,  # words per scale row
    scale_ptr,  # int32[E, SW]  scales of every expert (VRAM resident)
    sdst_ptr,  # int32[*, SW]  scales in slot order
    e,
    dst_row,
    active,
    SBLOCK: tl.constexpr,
):
    offs = tl.arange(0, SBLOCK)
    m = active & (offs < SW)
    x = tl.load(scale_ptr + e * SW + offs, mask=m, other=0)
    tl.store(sdst_ptr + dst_row * SW + offs, x, mask=m)


@triton.jit
def _gather_one(
    j, c,  # index into todo and chunk number
    todo_ptr, dst_row_ptr, cache_row_ptr, src_row_ptr, base,
    src0_ptr, cache0_ptr, dst0_ptr, W0, C0,
    src1_ptr, cache1_ptr, dst1_ptr, W1,
    scale0_ptr, sdst0_ptr, SW0,
    scale1_ptr, sdst1_ptr, SW1,
    ssd_ptr, K,  # with SKIP_SSD: experts in this list are not copied (a separate launch copies them after the SSD read)
    BLOCK: tl.constexpr,
    SBLOCK: tl.constexpr,
    SKIP_SSD: tl.constexpr,
):  # fmt: skip
    e = tl.load(todo_ptr + j)
    active = e >= 0
    if SKIP_SSD:
        for k in range(K):
            active = active & (tl.load(ssd_ptr + k) != e)
    # Keep the index in range for empty entries; the value read is discarded.
    e = tl.maximum(e, 0)
    dst_row = tl.maximum(tl.load(dst_row_ptr + e), 0)
    cache_row = tl.load(cache_row_ptr + e)
    src_row = tl.load(src_row_ptr + e) + base

    # Weights are copied per name in chunks of BLOCK words. The program for the
    # first chunk of each name copies that name's whole scale row (it only needs
    # updating when the slot's contents change).
    if c < C0:
        _copy_row(c, W0, src0_ptr, cache0_ptr, dst0_ptr, src_row, cache_row, dst_row, active, BLOCK)  # fmt: skip
        if c == 0:
            _copy_scale(SW0, scale0_ptr, sdst0_ptr, e, dst_row, active, SBLOCK)
    else:
        _copy_row(c - C0, W1, src1_ptr, cache1_ptr, dst1_ptr, src_row, cache_row, dst_row, active, BLOCK)  # fmt: skip
        if c == C0:
            _copy_scale(SW1, scale1_ptr, sdst1_ptr, e, dst_row, active, SBLOCK)


@triton.jit
def _gather_rows_kernel(
    todo_ptr,  # int64[K]  experts to copy. -1 is an empty entry
    dst_row_ptr,  # int64[E]  expert -> destination row
    cache_row_ptr,  # int64[E]  expert -> row on the cache slab; -1 reads from RAM
    src_row_ptr,  # int64[E]  expert -> RAM-tier row (before adding base)
    base,  # added to RAM-tier rows (first row of the layer)
    src0_ptr, cache0_ptr, dst0_ptr, W0, C0,  # name 0 (w13). C0 is the chunk count
    src1_ptr, cache1_ptr, dst1_ptr, W1,  # name 1 (w2)
    scale0_ptr, sdst0_ptr, SW0,  # scales of name 0
    scale1_ptr, sdst1_ptr, SW1,  # scales of name 1
    ssd_ptr, K,  # with FETCH: experts to read from SSD (length K, padded with -1)
    layer, seq_ptr, req_ptr, done_ptr,  # with FETCH: arguments of _fetch
    BLOCK: tl.constexpr,
    SBLOCK: tl.constexpr,
    FETCH: tl.constexpr,
    FBLOCK: tl.constexpr,
):  # fmt: skip
    j = tl.program_id(0)
    c = tl.program_id(1)
    if FETCH:
        # Column 0 is reserved for the request. While program (0, 0) asks the
        # host to read from SSD and waits, the programs of the other columns
        # (other blocks) copy the experts that are in RAM, so the SSD wait and
        # the PCIe gather overlap. Blocks are dispatched in order, so column 0
        # runs first.
        if c == 0:
            if j == 0:
                _fetch(
                    ssd_ptr,
                    src_row_ptr,
                    K,
                    base,
                    layer,
                    seq_ptr,
                    req_ptr,
                    done_ptr,
                    FBLOCK,
                )
        else:
            _gather_one(
                j, c - 1, todo_ptr, dst_row_ptr, cache_row_ptr, src_row_ptr, base,
                src0_ptr, cache0_ptr, dst0_ptr, W0, C0, src1_ptr, cache1_ptr, dst1_ptr, W1,
                scale0_ptr, sdst0_ptr, SW0, scale1_ptr, sdst1_ptr, SW1,
                ssd_ptr, K, BLOCK, SBLOCK, True,
            )  # fmt: skip
    else:
        _gather_one(
            j, c, todo_ptr, dst_row_ptr, cache_row_ptr, src_row_ptr, base,
            src0_ptr, cache0_ptr, dst0_ptr, W0, C0, src1_ptr, cache1_ptr, dst1_ptr, W1,
            scale0_ptr, sdst0_ptr, SW0, scale1_ptr, sdst1_ptr, SW1,
            ssd_ptr, K, BLOCK, SBLOCK, False,
        )  # fmt: skip


@triton.jit
def _fetch(
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


@triton.jit
def _fetch_kernel(
    todo_ptr, row_ptr, K, base, layer, seq_ptr, req_ptr, done_ptr, BLOCK: tl.constexpr
):
    _fetch(todo_ptr, row_ptr, K, base, layer, seq_ptr, req_ptr, done_ptr, BLOCK)


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
    src: list[torch.Tensor],
    cache: list[torch.Tensor],
    dst: list[torch.Tensor],
    scale: list[torch.Tensor],
    scale_dst: list[torch.Tensor],
    fetch: tuple[torch.Tensor, int, object] | None = None,
) -> None:
    """Copy the rows and scales of the experts in ``todo`` into ``dst`` / ``scale_dst``.

    Each list has two elements in name order (w13, w2), each a ``(*, W)`` int32
    view made by ``as_rows``. ``cache_row`` decides whether a row is read from
    ``cache`` or ``src``. Scales are always read from ``scale`` (every expert).

    With ``fetch=(ssd_todo, layer, store)`` the same launch also asks the host
    to read the experts in ``ssd_todo`` from SSD into the RAM tier and waits for
    it (as ``fetch_rows`` does); meanwhile it copies the experts in ``todo``
    that are not in ``ssd_todo`` (those in RAM). The experts in ``ssd_todo`` are
    copied afterwards by a separate ``gather_rows(ssd_todo, ...)``.
    """
    W0, W1 = src[0].shape[1], src[1].shape[1]
    C0, C1 = triton.cdiv(W0, _BLOCK), triton.cdiv(W1, _BLOCK)
    SW0, SW1 = scale[0].shape[1], scale[1].shape[1]
    K = todo.shape[0]
    if fetch is None:
        ssd, layer, store = todo, 0, None
        seq = req = done = todo  # unused
        grid, fblock = (K, C0 + C1), 16
    else:
        ssd, layer, store = fetch
        if ssd.shape[0] != K:
            raise ValueError(f"ssd_todo has {ssd.shape[0]} entries, todo has {K}")
        seq, req, done = store.seq, store.req_view, store.done_view
        grid, fblock = (K, C0 + C1 + 1), store.block
    _gather_rows_kernel[grid](
        todo, dst_row, cache_row, src_row, base,
        src[0], cache[0], dst[0], W0, C0,
        src[1], cache[1], dst[1], W1,
        scale[0], scale_dst[0], SW0,
        scale[1], scale_dst[1], SW1,
        ssd, K, layer, seq, req, done,
        BLOCK=_BLOCK, SBLOCK=1 << (max(SW0, SW1) - 1).bit_length(),
        FETCH=fetch is not None, FBLOCK=fblock,
    )  # fmt: skip


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
