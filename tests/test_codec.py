"""Check the codec's encoder (torch) and decoder (Triton) against a pure-Python reference decoder. Needs a GPU.

The reference decoder walks the format described in codec.py's docstring one
lane at a time, sequentially, with the same refill rule as the Triton version
(a 32-bit word when fewer than LIMIT bits remain; lane order for the same
symbol index).
"""

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from vllm_expert_pager import codec

CHUNK = 64
GROUP = CHUNK * codec.LANES
N = 2 * GROUP
dev = torch.device("cuda")

# An exponent distribution close to real data (13 is the mode; nothing below 4).
_EXP_P = [0.0] * 4 + [
    0.001,
    0.002,
    0.004,
    0.008,
    0.016,
    0.032,
    0.064,
    0.125,
    0.228,
    0.315,
    0.188,
    0.015,
]


def rows(seed: int, n: int = N, p=_EXP_P) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    exp = torch.multinomial(
        torch.tensor(p, dtype=torch.float64), n, replacement=True, generator=g
    )
    sm = torch.randint(0, 256, (n,), dtype=torch.uint8, generator=g) & 0x87
    return sm | (exp.to(torch.uint8) << 3)


def decode_ref(rec: bytes, lut: bytes, n: int, chunk: int) -> list[int]:
    lanes = codec.LANES
    G = n // (chunk * lanes)

    def i32(off: int) -> int:
        return int.from_bytes(rec[off : off + 4], "little", signed=True)

    nib_off = 4 * (G + 1)
    huff_off = nib_off + n // 2
    out = [0] * n
    for g in range(G):
        rp = i32(4 * g)
        buf, level = [0] * lanes, [0] * lanes
        for i in range(chunk):
            m, k = divmod(i, 8)
            for lane in range(lanes):
                if level[lane] < codec.LIMIT:
                    buf[lane] |= (i32(huff_off + 4 * rp) & 0xFFFFFFFF) << level[lane]
                    level[lane] += 32
                    rp += 1
                ent = lut[buf[lane] & 511]
                buf[lane] >>= ent & 15
                level[lane] -= ent & 15
                nib = i32(nib_off + 4 * ((g * (chunk // 8) + m) * lanes + lane))
                s = (nib >> (4 * k)) & 15
                out[(g * lanes + lane) * chunk + i] = (
                    ((s & 8) << 4) | ((ent >> 4) << 3) | (s & 7)
                )
        # The group's words fit in [gstart[g], gstart[g+1]).
        assert rp <= i32(4 * (g + 1))
    return out


@pytest.mark.parametrize(
    "raw",
    [
        rows(0),
        # 16 symbols, uniform: every code length is 4
        rows(1, p=[1.0] * 16),
        # geometric: plain Huffman would produce 15-bit codes, so the length limit applies
        rows(2, p=[2.0**k for k in range(16)]),
        # a single symbol
        torch.full((N,), 13 << 3, dtype=torch.uint8),
    ],
    ids=["realistic", "uniform", "geometric", "constant"],
)
def test_reference_decodes_gpu_encoding(raw):
    pitch = codec.pitch_of(
        N, 1.25, 4096
    )  # the uniform distribution does not shrink, and small rows exceed the raw size by the header
    rec, lut = codec.encode(raw.to(dev), pitch, CHUNK)
    got = decode_ref(rec.cpu().numpy().tobytes(), lut.cpu().numpy().tobytes(), N, CHUNK)
    assert got == raw.tolist()


def test_code_lengths_are_limited():
    lengths = codec.code_lengths([2**k for k in range(16)])
    assert max(lengths) <= codec.LIMIT
    assert sum(2.0**-n for n in lengths) == 1.0


def test_encode_rejects_rows_over_pitch():
    # The fixed nibbles alone need N/2 bytes, so half that pitch can never fit.
    with pytest.raises(ValueError, match="VLLM_EXPERT_PAGER_PITCH must be at least"):
        codec.encode(rows(3).to(dev), N // 4, CHUNK)
    with pytest.raises(ValueError, match="not a multiple"):
        codec.encode(rows(3, n=N + 8).to(dev), 8 * N, CHUNK)


def test_decode_rows_follows_todo_and_skips():
    # Name 0 has N values, name 1 has N/2. Expert 3 goes to row 1; expert 5
    # counts as a VRAM hit and is skipped; expert 6 is in the skip list and is
    # skipped.
    E = 8
    ns = (N, N // 2)
    pitches = [codec.pitch_of(n, 1.25, 4096) for n in ns]
    raws = {e: [rows(10 * e + k, n) for k, n in enumerate(ns)] for e in (3, 5, 6)}
    staging = [torch.zeros((3, p), dtype=torch.uint8, device=dev) for p in pitches]
    lut = torch.zeros((E, 2, codec.LUT_BYTES), dtype=torch.uint8, device=dev)
    for j, e in enumerate((3, 5, 6)):
        for k in range(2):
            staging[k][j], lut[e, k] = codec.encode(
                raws[e][k].to(dev), pitches[k], CHUNK
            )
    dst = [torch.zeros((2, n), dtype=torch.uint8, device=dev) for n in ns]
    todo = torch.tensor([3, 5, 6], dtype=torch.int64, device=dev)
    dst_row = torch.zeros((E,), dtype=torch.int64, device=dev)
    dst_row[3], dst_row[5], dst_row[6] = 1, 0, 0
    cache_row = torch.full((E,), -1, dtype=torch.int64, device=dev)
    cache_row[5] = 2
    skip = torch.tensor([6, -1, -1], dtype=torch.int64, device=dev)

    codec.decode_rows(
        todo, dst_row, cache_row, staging, dst, lut, skip=skip, chunk=CHUNK
    )
    torch.cuda.synchronize()
    for k in range(2):
        assert torch.equal(dst[k][1].cpu(), raws[3][k])
        assert not dst[k][0].any()

    # Without skip, expert 6 is expanded into row 0. Expert 5 still has
    # cache_row >= 0 and is skipped.
    codec.decode_rows(todo, dst_row, cache_row, staging, dst, lut, chunk=CHUNK)
    torch.cuda.synchronize()
    for k in range(2):
        assert torch.equal(dst[k][0].cpu(), raws[6][k])
