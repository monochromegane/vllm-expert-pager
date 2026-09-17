"""Triton kernels that copy one expert and request reads from the host.

The source is either the RAM tier (pinned memory seen from the device as a UVA
view, compressed) or the cache slab (raw). Scales (VRAM resident, in expert
order) are copied to the same slot, because the MoE kernel indexes weights and
scales with the same slot number. Which experts to copy is passed as index
tensors on the device, so the host needs to know nothing and the kernels run
under CUDA graphs.

``gather_rows`` only copies rows. With compression, RAM-tier rows go to staging
(``codec.decode_rows`` expands them in a separate launch); without it they go
straight into the slab. Raw rows from the cache slab always go to dst. With
``fetch`` it copies while waiting for the SSD read (the decode path).

The SMs copy with loads and stores instead of ``cudaMemcpyAsync``. A memcpy has
fixed addresses, so putting one in a graph would mean "copy the same expert
every time".

Only ``_PROGRAMS`` copying programs are launched, and they walk the (row, chunk)
pairs in turn. Launching one program per chunk and letting them all read at
once fills the memory system with requests waiting on PCIe: the copy tops out
at 12 GB/s, and the memory accesses of other kernels queue behind it, so
nothing overlaps on another stream. Keeping about 64 KiB in flight gives 23 to
24 GB/s (the same as the copy engine) and overlaps with other kernels.
"""

import torch
import triton
import triton.language as tl

# Words per copy (int32, so 16 KiB) and the number of copying programs. The
# amount in flight is _PROGRAMS x _BLOCK x 4 B = 64 KiB. One row or 200 rows
# both reach 23 to 24 GB/s; more or less in flight is slower.
_BLOCK = 4096
_PROGRAMS = 4
# After this many idle spins waiting for done, the request is published again.
# One spin is one volatile load across PCIe, measured at 514 spins/ms (GPU
# otherwise idle) to 220 spins/ms (overlapping compute), so 0.5 to 1.2 s. It is
# long so that a fetch that is merely slow (prefill reads hundreds of experts in
# one request) is not re-published. Re-publishing never reads twice, but the
# count marks that a handoff was lost. Kept as constexpr so the kernel sees a
# constant.
_SPIN_RETRY = tl.constexpr(1 << 18)
# After this many re-publishes without done, stop waiting and move on (4 to
# 10 s). When the handoff with the host is broken in both directions the
# re-publish does not arrive either, and this keeps the server from stalling
# forever. The rows of that layer stay stale, so one token's output is wrong.
_SPIN_GIVEUP = tl.constexpr(8)


@triton.jit
def _copy_row(
    chunk,
    W,  # words per raw row
    P,  # words per compressed row (pitch)
    src_ptr,  # int32[*, P]  RAM tier (UVA view, compressed)
    cache_ptr,  # int32[S, W]  cache slab (raw)
    dst_ptr,  # int32[*, W]  raw destination
    stg_ptr,  # int32[*, P]  compressed destination (row j, the index into todo)
    src_row,
    cache_row,
    dst_row,
    j,
    active,
    BLOCK: tl.constexpr,
    COMPRESS: tl.constexpr,
):
    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    from_cache = active & (cache_row >= 0) & (offs < W)
    from_src = active & (cache_row < 0) & (offs < P)

    # A masked-off load does not touch memory, so issuing both runs only one.
    x = tl.load(
        cache_ptr + tl.maximum(cache_row, 0) * W + offs, mask=from_cache, other=0
    )
    tl.store(dst_ptr + dst_row * W + offs, x, mask=from_cache)
    x = tl.load(src_ptr + src_row * P + offs, mask=from_src, other=0)
    if COMPRESS:
        tl.store(stg_ptr + j * P + offs, x, mask=from_src)
    else:
        # A raw row is already in the form the kernel reads, so copy it straight
        # into the slab (P == W).
        tl.store(dst_ptr + dst_row * P + offs, x, mask=from_src)


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
def _gather_loop(
    pid, K, C,  # program number, number of rows, chunks per row (C0 + C1)
    todo_ptr, dst_row_ptr, cache_row_ptr, src_row_ptr, base,
    src0_ptr, cache0_ptr, dst0_ptr, stg0_ptr, W0, P0, C0,
    src1_ptr, cache1_ptr, dst1_ptr, stg1_ptr, W1, P1,
    scale0_ptr, sdst0_ptr, SW0,
    scale1_ptr, sdst1_ptr, SW1,
    ssd_ptr,  # experts to read from SSD (length K, padded with -1)
    BLOCK: tl.constexpr,
    SBLOCK: tl.constexpr,
    COMPRESS: tl.constexpr,
    SKIP_SSD: tl.constexpr,  # experts in the list are not copied (a separate launch copies them after the SSD read)
    PROGRAMS: tl.constexpr,
):  # fmt: skip
    """Walk the rows of todo in order; of each row, copy the chunks pid, pid + PROGRAMS, ...

    Weights are copied per name in chunks of BLOCK words. The program for the
    first chunk of each name copies that name's whole scale row (it only needs
    updating when the slot's contents change).
    """
    for j in range(K):
        e = tl.load(todo_ptr + j)
        active = e >= 0
        if SKIP_SSD:
            for k in range(K):
                active = active & (tl.load(ssd_ptr + k) != e)
        if active:
            dst_row = tl.maximum(tl.load(dst_row_ptr + e), 0)
            cache_row = tl.load(cache_row_ptr + e)
            src_row = tl.load(src_row_ptr + e) + base
            jj = j.to(tl.int64)
            for c in range(pid, C, PROGRAMS):
                if c < C0:
                    _copy_row(c, W0, P0, src0_ptr, cache0_ptr, dst0_ptr, stg0_ptr, src_row, cache_row, dst_row, jj, True, BLOCK, COMPRESS)  # fmt: skip
                    if c == 0:
                        _copy_scale(
                            SW0, scale0_ptr, sdst0_ptr, e, dst_row, True, SBLOCK
                        )
                else:
                    _copy_row(c - C0, W1, P1, src1_ptr, cache1_ptr, dst1_ptr, stg1_ptr, src_row, cache_row, dst_row, jj, True, BLOCK, COMPRESS)  # fmt: skip
                    if c == C0:
                        _copy_scale(
                            SW1, scale1_ptr, sdst1_ptr, e, dst_row, True, SBLOCK
                        )


@triton.jit
def _gather_rows_kernel(
    todo_ptr,  # int64[K]  experts to copy. -1 is an empty entry
    dst_row_ptr,  # int64[E]  expert -> destination row
    cache_row_ptr,  # int64[E]  expert -> row on the cache slab; -1 reads from RAM
    src_row_ptr,  # int64[E]  expert -> RAM-tier row (before adding base)
    base,  # added to RAM-tier rows (first row of the layer)
    src0_ptr, cache0_ptr, dst0_ptr, stg0_ptr, W0, P0, C0,  # name 0 (w13). C0 is the chunk count
    src1_ptr, cache1_ptr, dst1_ptr, stg1_ptr, W1, P1,  # name 1 (w2)
    scale0_ptr, sdst0_ptr, SW0,  # scales of name 0
    scale1_ptr, sdst1_ptr, SW1,  # scales of name 1
    ssd_ptr, K, C,  # experts to read from SSD (length K, padded with -1). C is the chunks per row
    layer, seq_ptr, req_ptr, done_ptr,  # with FETCH: arguments of _fetch
    BLOCK: tl.constexpr,
    SBLOCK: tl.constexpr,
    COMPRESS: tl.constexpr,
    FETCH: tl.constexpr,
    FBLOCK: tl.constexpr,
    RING: tl.constexpr,
    PROGRAMS: tl.constexpr,
):  # fmt: skip
    pid = tl.program_id(0)
    if FETCH:
        # While the last program asks the host to read from SSD and waits, the
        # other programs copy the experts that are in RAM, so the SSD wait and
        # the PCIe gather overlap.
        if pid == PROGRAMS:
            _fetch(ssd_ptr, src_row_ptr, K, base, layer, seq_ptr, req_ptr, done_ptr, FBLOCK, RING)  # fmt: skip
        else:
            _gather_loop(
                pid, K, C,
                todo_ptr, dst_row_ptr, cache_row_ptr, src_row_ptr, base,
                src0_ptr, cache0_ptr, dst0_ptr, stg0_ptr, W0, P0, C0,
                src1_ptr, cache1_ptr, dst1_ptr, stg1_ptr, W1, P1,
                scale0_ptr, sdst0_ptr, SW0, scale1_ptr, sdst1_ptr, SW1,
                ssd_ptr, BLOCK, SBLOCK, COMPRESS, True, PROGRAMS,
            )  # fmt: skip
    else:
        _gather_loop(
            pid, K, C,
            todo_ptr, dst_row_ptr, cache_row_ptr, src_row_ptr, base,
            src0_ptr, cache0_ptr, dst0_ptr, stg0_ptr, W0, P0, C0,
            src1_ptr, cache1_ptr, dst1_ptr, stg1_ptr, W1, P1,
            scale0_ptr, sdst0_ptr, SW0, scale1_ptr, sdst1_ptr, SW1,
            ssd_ptr, BLOCK, SBLOCK, COMPRESS, False, PROGRAMS,
        )  # fmt: skip


@triton.jit
def _fetch_publish(
    todo_ptr,  # int64[K]  experts to read from SSD. -1 is an empty entry
    row_ptr,  # int64[E]  expert -> RAM-tier row (before adding base)
    K,
    base,
    layer,
    seq_ptr,  # int64[1]  sequence number on the device
    ticket_ptr,  # int64[1]  device. Receives the seq of the published request (_fetch_wait waits for it)
    req_ptr,  # int64[2 + RING*(2+2B)]  pinned. [re-publishes, give-ups] + RING slots of [seq, layer, expert[B], row[B]]
    BLOCK: tl.constexpr,
    RING: tl.constexpr,
):
    """Ask the host to read the experts in ``todo`` from SSD, without waiting.

    Does nothing if ``todo`` is empty. The request is written to slot
    ``seq % RING``. The host serves requests in seq order, so up to RING of
    them can be outstanding (store.py).
    """
    offs = tl.arange(0, BLOCK)
    e = tl.load(todo_ptr + offs, mask=offs < K, other=-1)
    if tl.sum((e >= 0).to(tl.int32), axis=0) > 0:
        row = tl.load(row_ptr + tl.maximum(e, 0)) + base
        seq = tl.load(seq_ptr) + 1
        tl.store(seq_ptr, seq)
        tl.store(ticket_ptr, seq)
        slot = req_ptr + 2 + (seq % RING) * (2 + 2 * BLOCK)
        # Write all of BLOCK so leftovers from an earlier request are never
        # read.
        tl.store(slot + 2 + offs, e)
        tl.store(slot + 2 + BLOCK + offs, tl.where(e >= 0, row, -1))
        # Triton may specialize layer into a constant, on which dtype conversion
        # cannot be called, so add it to an int64 zero to match the type.
        tl.store(slot + 1, seq * 0 + layer)
        # Publish seq only after every thread has finished writing. The release
        # makes the preceding writes visible to the host.
        tl.debug_barrier()
        tl.atomic_xchg(slot, seq, sem="release", scope="sys")


@triton.jit
def _fetch_wait(
    todo_ptr,  # int64[K]  the list passed to _fetch_publish
    K,
    ticket_ptr,  # int64[1]  device. The seq written by _fetch_publish
    req_ptr,  # int64[2 + RING*(2+2B)]  pinned
    done_ptr,  # int64[1]  pinned. The host writes the seq it has finished
    BLOCK: tl.constexpr,
    RING: tl.constexpr,
):
    """Wait for the done of the request published by ``_fetch_publish``.

    Does nothing if ``todo`` is empty. done advances in order, so every request
    before the awaited seq has finished as well.
    """
    offs = tl.arange(0, BLOCK)
    e = tl.load(todo_ptr + offs, mask=offs < K, other=-1)
    if tl.sum((e >= 0).to(tl.int32), axis=0) > 0:
        seq = tl.load(ticket_ptr)
        slot = req_ptr + 2 + (seq % RING) * (2 + 2 * BLOCK)
        d = tl.load(done_ptr, volatile=True)
        n = 0
        tries = 0
        while (d < seq) & (tries < _SPIN_GIVEUP):
            n += 1
            if n >= _SPIN_RETRY:
                n = 0
                tries += 1
                # Still waiting: publish the request again and advance the
                # re-publish count. If the request had not arrived, the host
                # serves this as a new one; if it was already served
                # (seq <= last), the host sees the count change and rewrites
                # done. Either direction of a lost handoff recovers by itself.
                tl.atomic_xchg(slot, seq, sem="release", scope="sys")
                tl.atomic_add(req_ptr, 1, sem="release", scope="sys")
            d = tl.load(done_ptr, volatile=True)
        if d < seq:
            # Even the re-publishes did not get through. Stop waiting here, or
            # the server stays stuck. The rows of this layer stay stale, so
            # count it for the host to notice.
            tl.atomic_add(req_ptr + 1, 1, sem="release", scope="sys")


@triton.jit
def _fetch(
    todo_ptr, row_ptr, K, base, layer, seq_ptr, req_ptr, done_ptr,
    BLOCK: tl.constexpr, RING: tl.constexpr,
):  # fmt: skip
    """Publish a request and wait for its done (``_fetch_publish`` + ``_fetch_wait``). The ticket is seq itself."""
    _fetch_publish(todo_ptr, row_ptr, K, base, layer, seq_ptr, seq_ptr, req_ptr, BLOCK, RING)  # fmt: skip
    _fetch_wait(todo_ptr, K, seq_ptr, req_ptr, done_ptr, BLOCK, RING)


@triton.jit
def _fetch_kernel(
    todo_ptr, row_ptr, K, base, layer, seq_ptr, req_ptr, done_ptr,
    BLOCK: tl.constexpr, RING: tl.constexpr,
):  # fmt: skip
    _fetch(todo_ptr, row_ptr, K, base, layer, seq_ptr, req_ptr, done_ptr, BLOCK, RING)


@triton.jit
def _fetch_publish_kernel(
    todo_ptr, row_ptr, K, base, layer, seq_ptr, ticket_ptr, req_ptr,
    BLOCK: tl.constexpr, RING: tl.constexpr,
):  # fmt: skip
    _fetch_publish(todo_ptr, row_ptr, K, base, layer, seq_ptr, ticket_ptr, req_ptr, BLOCK, RING)  # fmt: skip


@triton.jit
def _fetch_wait_kernel(
    todo_ptr, K, ticket_ptr, req_ptr, done_ptr,
    BLOCK: tl.constexpr, RING: tl.constexpr,
):  # fmt: skip
    _fetch_wait(todo_ptr, K, ticket_ptr, req_ptr, done_ptr, BLOCK, RING)


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
    stg: list[torch.Tensor] | None,
    scale: list[torch.Tensor],
    scale_dst: list[torch.Tensor],
    fetch: tuple[torch.Tensor, int, object] | None = None,
) -> None:
    """Copy the rows and scales of the experts in ``todo``.

    Each list has two elements in name order (w13, w2), each an int32 view made
    by ``as_rows``. Experts with ``cache_row[e] >= 0`` are copied from the cache
    slab ``cache`` (raw, ``(S, W)``) to row ``dst_row[e]`` of ``dst``
    (``(*, W)``). The others are copied from the RAM tier ``src``: with
    compression to row j (the index into todo) of the staging ``stg``
    (``(*, P)``), without it (``stg`` is None) to row ``dst_row[e]`` of ``dst``.
    Compressed rows are expanded by ``codec.decode_rows``. Scales are always
    copied from ``scale`` (every expert) to row ``dst_row[e]`` of ``scale_dst``.

    With ``fetch=(ssd_todo, layer, store)`` the same launch also asks the host
    to read the experts in ``ssd_todo`` from SSD into the RAM tier and waits for
    it (as ``fetch_rows`` does); meanwhile it copies the experts in ``todo``
    that are not in ``ssd_todo`` (those in RAM). The experts in ``ssd_todo`` are
    copied afterwards by a separate ``gather_rows(ssd_todo, ...)``.
    """
    W0, W1 = dst[0].shape[1], dst[1].shape[1]
    P0, P1 = src[0].shape[1], src[1].shape[1]
    compress = stg is not None
    if compress:
        if (P0, P1) != (stg[0].shape[1], stg[1].shape[1]):
            raise ValueError("staging rows must have the RAM tier's pitch")
    else:
        if (P0, P1) != (W0, W1):
            raise ValueError("uncompressed RAM rows must have the slab's row size")
        stg = dst  # unused
    C0, C1 = triton.cdiv(max(W0, P0), _BLOCK), triton.cdiv(max(W1, P1), _BLOCK)
    SW0, SW1 = scale[0].shape[1], scale[1].shape[1]
    K = todo.shape[0]
    if fetch is None:
        ssd, layer, store = todo, 0, None
        seq = req = done = todo  # unused
        grid, fblock, ring = (_PROGRAMS,), 16, 1
    else:
        ssd, layer, store = fetch
        if ssd.shape[0] != K:
            raise ValueError(f"ssd_todo has {ssd.shape[0]} entries, todo has {K}")
        seq, req, done = store.seq, store.req_view, store.done_view
        # One extra program at the end publishes the request and waits.
        grid, fblock, ring = (_PROGRAMS + 1,), store.block, store.ring
    _gather_rows_kernel[grid](
        todo, dst_row, cache_row, src_row, base,
        src[0], cache[0], dst[0], stg[0], W0, P0, C0,
        src[1], cache[1], dst[1], stg[1], W1, P1,
        scale[0], scale_dst[0], SW0,
        scale[1], scale_dst[1], SW1,
        ssd, K, C0 + C1, layer, seq, req, done,
        BLOCK=_BLOCK, SBLOCK=1 << (max(SW0, SW1) - 1).bit_length(),
        COMPRESS=compress, FETCH=fetch is not None, FBLOCK=fblock, RING=ring,
        PROGRAMS=_PROGRAMS,
    )  # fmt: skip


def fetch_rows(
    todo: torch.Tensor, row: torch.Tensor, base: int, layer: int, store
) -> None:
    """Ask the host to read the experts in ``todo`` from SSD and wait.

    Each expert ``e`` lands in RAM-tier row ``row[e] + base``. Returns
    immediately if ``todo`` is empty.
    """
    _fetch_kernel[(1,)](
        todo, row, todo.shape[0], base, layer, store.seq, store.req_view, store.done_view,
        BLOCK=store.block, RING=store.ring,
    )  # fmt: skip


def fetch_publish_rows(
    todo: torch.Tensor,
    row: torch.Tensor,
    base: int,
    layer: int,
    store,
    ticket: torch.Tensor,
) -> None:
    """Publish the request of ``fetch_rows`` without waiting; ``fetch_wait_rows`` waits.

    Work that does not depend on the read (copies of rows that are in RAM, the
    MoE of the previous chunk) goes between the publish and the wait, so the SSD
    read overlaps it. ``ticket`` (one int64 word on the device) receives the seq
    of the published request. If ``todo`` is empty no request is published and
    the current seq is written (waiting for it covers every earlier request).
    Another request may be published before waiting, but at most ``store.ring``
    requests can be outstanding.
    """
    _fetch_publish_kernel[(1,)](
        todo, row, todo.shape[0], base, layer, store.seq, ticket, store.req_view,
        BLOCK=store.block, RING=store.ring,
    )  # fmt: skip


def fetch_wait_rows(todo: torch.Tensor, store, ticket: torch.Tensor) -> None:
    """Wait for done to reach the seq in ``ticket``. Does nothing if ``todo`` is empty."""
    _fetch_wait_kernel[(1,)](
        todo, todo.shape[0], ticket, store.req_view, store.done_view,
        BLOCK=store.block, RING=store.ring,
    )  # fmt: skip
