"""Lossless compression of fp8 e4m3 expert rows.

The SSD and RAM tiers hold rows in this format, and the GPU expands them when
copying into the VRAM slab.

Each byte sign<<7 | exp<<3 | man is split into the 4 bits of sign + mantissa
(which do not compress) and the 4 bits of the exponent (about 2.6 bits of
entropy), and only the exponent is Huffman coded. The table is built per row:
the exponent distribution differs between experts, and with a single static
table some experts do not shrink. Each name (w13, w2) is independent.

Decoding cuts a row of N values into chunks of CHUNK values and processes one
chunk per lane, LANES lanes (one warp) per group, in parallel. A lane keeps the
code bit stream in a 64-bit buffer and refills it with one 32-bit word whenever
fewer than LIMIT bits remain. The encoder computes the refill order (symbol
index, lane) with the same rule as the decoder and lays the words out in that
order, so lanes that refill at the same time read adjacent words. The sign +
mantissa nibbles are likewise stored as 32-bit words of 8 values in lane order,
and the output is written 8 values at a time as 64-bit words. Every load and
store is warp-coalesced.

Record (N values, G = N / (CHUNK * LANES) groups):

  [0, 4(G+1))            int32[G+1]  first Huffman word (word index) of group g; the last entry is the total word count
  [H, H + N/2)           int32[N/8]  sign<<3 | man nibbles, 8 per word, in (group, word index, lane) order
  [H + N/2, ...)         int32[*]    Huffman words, per group in refill order      (H = 4(G+1))

**The LUT (512 B) is not part of the record; it lives in a VRAM-resident table
like the scales** (``Store.lut``, indexed by expert). Copying it per group would
add 3.6% of PCIe traffic to a 14 KB group and halve the gain of the fused path
(below).

Decoding group g needs only this LUT and two contiguous ranges of the record
(the fixed-width words and the Huffman words), so ``copy_and_decode_group`` can
copy those two ranges from the RAM tier into staging and expand them right
away. When the gather kernel calls it, the expansion hides under the SSD wait.

Code lengths are limited to LIMIT bits; canonical Huffman codes are
bit-reversed and packed from the LSB. Records have a fixed length ``pitch``; a
row that does not fit stops the load.
"""

import heapq

import torch
import triton
import triton.language as tl

# Maximum code length. The LUT has 2**LIMIT entries.
LIMIT = 9
LUT_BYTES = 1 << LIMIT
# Values handled by one lane. Smaller means more lanes and more speed, but more
# leftover words per chunk.
CHUNK = 512
# Lanes per group. One warp.
LANES = 32
# The number of values in a row must be a multiple of this.
GROUP = CHUNK * LANES
# Words copied per load/store (in the copy part).
COPY_BLOCK = 1024


def _huffman_lengths(counts: list[float]) -> list[int]:
    heap = [(c, i, (i,)) for i, c in enumerate(counts)]
    heapq.heapify(heap)
    lengths = [0] * len(counts)
    n = len(counts)
    while len(heap) > 1:
        c1, _, s1 = heapq.heappop(heap)
        c2, _, s2 = heapq.heappop(heap)
        for s in s1 + s2:
            lengths[s] += 1
        heapq.heappush(heap, (c1 + c2, n, s1 + s2))
        n += 1
    return lengths


def code_lengths(hist: list[int], limit: int = LIMIT) -> list[int]:
    """Length-limited Huffman code lengths. Symbols that never occur also get a code."""
    counts = [float(max(c, 1)) for c in hist]
    k = 0
    while True:
        lengths = _huffman_lengths(counts)
        if max(lengths) <= limit:
            return lengths
        # Raise the probability of rare symbols to make the tree shallower.
        k += 1
        floor = sum(counts) / 2 ** (limit - k)
        counts = [max(c, floor) for c in counts]


class Code:
    """Huffman code for the 16 exponent symbols, built from a histogram."""

    def __init__(self, hist: list[int]) -> None:
        self.lengths = code_lengths(hist)
        # Canonical: assign codes from 0 in (length, symbol) order.
        order = sorted(range(16), key=lambda s: (self.lengths[s], s))
        canon = [0] * 16
        code = 0
        for i, s in enumerate(order):
            if i:
                code = (code + 1) << (self.lengths[s] - self.lengths[order[i - 1]])
            canon[s] = code
        # Store the codes bit-reversed so the first bit is the LSB in the stream.
        self.codes = [
            int(f"{canon[s]:0{self.lengths[s]}b}"[::-1], 2) for s in range(16)
        ]
        self.lut = [-1] * LUT_BYTES
        for s in range(16):
            n = self.lengths[s]
            for hi in range(1 << (LIMIT - n)):
                x = self.codes[s] | (hi << n)
                assert self.lut[x] < 0, "codes are not prefix-free"
                self.lut[x] = (s << 4) | n
        assert min(self.lut) >= 0, "codes do not fill the LUT (Kraft sum < 1)"


def pitch_of(n_values: int, ratio: float, align: int) -> int:
    """Fixed length of a compressed record: ratio times the raw size, rounded up to align."""
    return -(-int(n_values * ratio) // align) * align


# ---- Encoding (at load time, with torch ops) ----


def encode(
    raw: torch.Tensor, pitch: int, chunk: int = CHUNK
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress a uint8 row (N,). Returns ``(record of pitch bytes, 512 B LUT)``.

    Raises ValueError if the row does not fit in the record.
    """
    N = raw.numel()
    if N % (chunk * LANES):
        raise ValueError(f"row of {N} values is not a multiple of {chunk * LANES}")
    C, G = N // chunk, N // (chunk * LANES)
    dev = raw.device
    raw = raw.reshape(-1)
    exp = ((raw >> 3) & 15).long()

    # Sign + mantissa: 32-bit words of 8 values in (group, word, lane) order.
    sm = ((raw >> 7) << 3) | (raw & 7)
    fixed = (sm[0::2] | (sm[1::2] << 4)).view(G, LANES, chunk // 8, 4)
    fixed = fixed.transpose(1, 2).contiguous().reshape(-1)

    code = Code(torch.bincount(exp, minlength=16).tolist())
    table = torch.tensor([code.lengths, code.codes], dtype=torch.int64, device=dev)
    n = table[0][exp].view(C, chunk)
    before = torch.cumsum(n, 1) - n  # first bit of each symbol within the chunk

    # The decoder refills a 32-bit word before symbol i when fewer than LIMIT
    # bits remain. The k-th refill happens at the first symbol with
    # 32k - before[i] < LIMIT, that is before[i] >= 32k - (LIMIT - 1). nw is the
    # number of words in the chunk.
    nw = (before[:, -1] + (LIMIT - 1)) // 32 + 1
    kmax = int(nw.max())
    thr = 32 * torch.arange(kmax, device=dev) - (LIMIT - 1)
    at = torch.searchsorted(before, thr.expand(C, kmax).contiguous())  # (C, kmax)
    valid = at < chunk
    total_words = int(nw.sum())
    head = 4 * (G + 1) + N // 2
    nbytes = head + 4 * total_words
    if nbytes > pitch:
        raise ValueError(
            f"compressed row is {nbytes} B, pitch is {pitch} B; "
            f"VLLM_EXPERT_PAGER_PITCH must be at least {nbytes / N:.4f}"
        )

    # First build a contiguous word sequence per chunk. The bit ranges of the
    # codes do not overlap, so adding equals OR.
    wbase = torch.cumsum(nw, 0) - nw
    start = (wbase * 32).view(C, 1) + before
    v = table[1][exp].view(C, chunk)
    w, sh = start >> 5, start & 31
    words = torch.zeros((total_words + 1,), dtype=torch.int64, device=dev)
    words.scatter_add_(0, w.reshape(-1), ((v << sh) & 0xFFFFFFFF).reshape(-1))
    words.scatter_add_(0, (w + 1).reshape(-1), (v >> (32 - sh)).reshape(-1))
    # Reorder the words (c, k) within each group by (refilling symbol index, lane).
    c = torch.arange(C, device=dev).view(C, 1).expand(C, kmax)
    key = ((c // LANES) * chunk + at) * LANES + (c % LANES)
    src = (wbase.view(C, 1) + torch.arange(kmax, device=dev))[valid]
    interleaved = words[src[torch.argsort(key[valid])]]
    per_group = nw.view(G, LANES).sum(1)
    # The last entry is the total word count. The words of group g are
    # [gstart[g], gstart[g+1]), the range the fused path copies.
    gstart = torch.cat(
        [torch.cumsum(per_group, 0) - per_group, per_group.sum().view(1)]
    )

    out = torch.zeros((pitch,), dtype=torch.uint8, device=dev)
    out[: 4 * (G + 1)] = _int32_bytes(gstart)
    out[4 * (G + 1) : head] = fixed
    out[head:nbytes] = _int32_bytes(interleaved)
    return out, torch.tensor(code.lut, dtype=torch.uint8, device=dev)


def _int32_bytes(x: torch.Tensor) -> torch.Tensor:
    """Bytes of int64 values in [0, 2**32) as two's-complement int32."""
    return torch.where(x >= 1 << 31, x - (1 << 32), x).to(torch.int32).view(torch.uint8)


# ---- Decoding (at inference, Triton) ----


@triton.jit
def _decode_group(
    lut8,  # uint8*  this row's LUT (512 B, VRAM resident)
    src32,  # int32*  this row's compressed record
    dst64,  # int64*  destination row
    G,  # number of groups
    g,  # this program's group
    rp,  # first Huffman word of the group (word index)
    CHUNK,  # values per chunk (multiple of 8)
    LANES: tl.constexpr,
    LUT_BYTES: tl.constexpr,
    LIMIT: tl.constexpr,
):
    lane = tl.arange(0, LANES)
    nib32 = src32 + (G + 1) + (g * (CHUNK // 8)) * LANES
    huff32 = src32 + (G + 1) + (G * CHUNK * LANES) // 8
    buf = tl.zeros([LANES], dtype=tl.int64)
    level = tl.zeros([LANES], dtype=tl.int32)
    dst64 = dst64 + (g * LANES + lane) * (CHUNK // 8)
    for m in range(CHUNK // 8):
        nib = tl.load(nib32 + m * LANES + lane)
        out = tl.zeros([LANES], dtype=tl.int64)
        for k in tl.static_range(8):
            # Refilling lanes read adjacent words from the group's read position on.
            refill = level < LIMIT
            cnt = refill.to(tl.int32)
            word = tl.load(huff32 + rp + tl.cumsum(cnt, 0) - 1, mask=refill, other=0)
            rp += tl.sum(cnt, 0)
            buf = tl.where(
                refill,
                buf | ((word.to(tl.int64) & 0xFFFFFFFF) << level.to(tl.int64)),
                buf,
            )
            level = tl.where(refill, level + 32, level)
            ent = tl.load(lut8 + (buf & (LUT_BYTES - 1)).to(tl.int32)).to(tl.int32)
            n = ent & 15
            buf = buf >> n.to(tl.int64)
            level -= n
            s = (nib >> (4 * k)) & 15
            o = ((s & 8) << 4) | ((ent >> 4) << 3) | (s & 7)
            out |= o.to(tl.int64) << (8 * k)
        tl.store(dst64 + m, out)


@triton.jit
def _copy_range(src32, dst32, start, count, BLOCK: tl.constexpr):
    """Copy ``src32[start : start+count]`` to the same position in ``dst32``."""
    for off in range(0, count, BLOCK):
        offs = start + off + tl.arange(0, BLOCK)
        m = offs < start + count
        tl.store(dst32 + offs, tl.load(src32 + offs, mask=m, other=0), mask=m)


@triton.jit
def copy_and_decode_group(
    lut8,  # uint8*  this row's LUT (VRAM resident)
    src32,  # int32*  compressed record in the RAM tier (UVA)
    stg32,  # int32*  the same row in staging
    dst64,  # int64*  destination row (raw)
    G, g, CHUNK,
    LANES: tl.constexpr, LUT_BYTES: tl.constexpr, LIMIT: tl.constexpr,
    COPY_BLOCK: tl.constexpr,
):  # fmt: skip
    """Copy the two ranges group g needs from the RAM tier into staging and expand into the slab.

    Called from the gather kernel, the expansion proceeds under the SSD wait.
    """
    nib = (G + 1) + (g * (CHUNK // 8)) * LANES
    huff = (G + 1) + (G * CHUNK * LANES) // 8
    rp = tl.load(src32 + g)
    rp_end = tl.load(src32 + g + 1)
    _copy_range(src32, stg32, nib, (CHUNK // 8) * LANES, COPY_BLOCK)
    _copy_range(src32, stg32, huff + rp, rp_end - rp, COPY_BLOCK)
    # Other lanes of the same warp read the words copied here, so line up the writes.
    tl.debug_barrier()
    _decode_group(lut8, stg32, dst64, G, g, rp, CHUNK, LANES, LUT_BYTES, LIMIT)


@triton.jit
def _decode_rows_kernel(
    todo_ptr,  # int64[K]  experts to expand. -1 is an empty entry
    dst_row_ptr,  # int64[E]  expert -> destination row
    cache_row_ptr,  # int64[E]  expert -> row on the cache slab; >= 0 means it was copied raw, so nothing to do
    lut_ptr,  # uint8[E, 2, 512]  this layer's LUTs
    stg0_32, P0, dst0_64, N0, G0,  # name 0: staging (int32 view), pitch (B), destination (int64), row bytes, group count
    stg1_32, P1, dst1_64, N1, G1,  # name 1
    ssd_ptr, K,  # with SKIP_SSD: experts in this list are skipped
    CHUNK,
    LANES: tl.constexpr,
    LUT_BYTES: tl.constexpr,
    LIMIT: tl.constexpr,
    SKIP_SSD: tl.constexpr,
):  # fmt: skip
    j = tl.program_id(0).to(tl.int64)
    g = tl.program_id(1)
    e = tl.load(todo_ptr + j)
    active = e >= 0
    if SKIP_SSD:
        for k in range(K):
            active = active & (tl.load(ssd_ptr + k) != e)
    e = tl.maximum(e, 0)
    active = active & (tl.load(cache_row_ptr + e) < 0)
    dst_row = tl.maximum(tl.load(dst_row_ptr + e), 0)
    lut = lut_ptr + e * (2 * LUT_BYTES)
    # todo has a fixed length K (S etc.) and may be mostly empty. Programs for
    # empty entries exit without doing anything (looping under a mask costs
    # 0.1 ms per launch).
    if active:
        if g < G0:
            row = stg0_32 + j * (P0 // 4)
            _decode_group(
                lut, row, dst0_64 + dst_row * (N0 // 8),
                G0, g, tl.load(row + g), CHUNK, LANES, LUT_BYTES, LIMIT,
            )  # fmt: skip
        else:
            row = stg1_32 + j * (P1 // 4)
            _decode_group(
                lut + LUT_BYTES, row, dst1_64 + dst_row * (N1 // 8),
                G1, g - G0, tl.load(row + (g - G0)), CHUNK, LANES, LUT_BYTES, LIMIT,
            )  # fmt: skip


def groups(n_bytes: int, chunk: int = CHUNK) -> int:
    """Number of groups in one row."""
    return n_bytes // (chunk * LANES)


def decode_rows(
    todo: torch.Tensor,
    dst_row: torch.Tensor,
    cache_row: torch.Tensor,
    staging: list[torch.Tensor],
    dst: list[torch.Tensor],
    lut: torch.Tensor,
    skip: torch.Tensor | None = None,
    chunk: int = CHUNK,
) -> None:
    """Expand ``staging[name][j]`` (the j-th compressed row of todo) into ``dst[name][dst_row[e]]``.

    Each list has two elements in name order (w13, w2). ``staging`` is
    ``(*, pitch)`` uint8, ``dst`` is ``(*, N)`` uint8, and ``lut`` is this
    layer's ``(E, 2, 512)`` uint8. Experts with ``cache_row[e] >= 0`` were
    copied raw by gather and are skipped. With ``skip``, the experts in that
    list are skipped too.
    """
    K = todo.shape[0]
    P0, P1 = staging[0].shape[1], staging[1].shape[1]
    N0, N1 = dst[0].shape[1], dst[1].shape[1]
    G0, G1 = groups(N0, chunk), groups(N1, chunk)
    _decode_rows_kernel[(K, G0 + G1)](
        todo, dst_row, cache_row, lut,
        staging[0].view(torch.int32), P0, dst[0].view(torch.int64), N0, G0,
        staging[1].view(torch.int32), P1, dst[1].view(torch.int64), N1, G1,
        todo if skip is None else skip, K,
        chunk,
        LANES=LANES, LUT_BYTES=LUT_BYTES, LIMIT=LIMIT, SKIP_SSD=skip is not None,
        num_warps=LANES // 32,
    )  # fmt: skip
