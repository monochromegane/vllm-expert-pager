"""Check that gather_rows copies compressed RAM-tier rows into staging, raw cache-slab rows into
the slab, and the scales into the slot. Needs a GPU."""

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from vllm_expert_pager.gather import gather_rows

E, S = 8, 4
WORDS = (8192, 4096)  # raw rows (w13, w2). Two chunks and one chunk with _BLOCK=4096
PITCH = (6144, 3072)  # compressed rows
SCALE_WORDS = (32, 16)


def test_gather_copies_rows_and_scales():
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)

    def rand(*shape):
        return torch.randint(
            -(2**31), 2**31 - 1, shape, dtype=torch.int32, device=dev, generator=g
        )

    src = [rand(16, p) for p in PITCH]
    cache = [rand(S, w) for w in WORDS]
    dst = [torch.zeros((S, w), dtype=torch.int32, device=dev) for w in WORDS]
    stg = [torch.zeros((S, p), dtype=torch.int32, device=dev) for p in PITCH]
    scale = [rand(E, w) for w in SCALE_WORDS]
    scale_dst = [
        torch.zeros((S, w), dtype=torch.int32, device=dev) for w in SCALE_WORDS
    ]

    # Expert 3 goes from RAM-tier row 7 (+base 2 = 9) to staging row 0 (its
    # index in todo), expert 5 from cache row 0 to slot 2. The remaining todo
    # entries are empty.
    todo = torch.tensor([3, 5, -1, -1], dtype=torch.int64, device=dev)
    dst_row = torch.zeros((E,), dtype=torch.int64, device=dev)
    dst_row[3], dst_row[5] = 1, 2
    cache_row = torch.full((E,), -1, dtype=torch.int64, device=dev)
    cache_row[5] = 0
    src_row = torch.zeros((E,), dtype=torch.int64, device=dev)
    src_row[3] = 7

    gather_rows(
        todo, dst_row, cache_row, src_row, 2, src, cache, dst, stg, scale, scale_dst
    )
    torch.cuda.synchronize()

    for i in range(2):
        assert torch.equal(stg[i][0], src[i][9])
        assert torch.equal(dst[i][2], cache[i][0])
        assert torch.equal(scale_dst[i][1], scale[i][3])
        assert torch.equal(scale_dst[i][2], scale[i][5])
        # Untouched rows stay 0. Compressed rows never land in the slab.
        assert not dst[i][0].any() and not dst[i][1].any() and not dst[i][3].any()
        assert not stg[i][1:].any()
        assert not scale_dst[i][0].any() and not scale_dst[i][3].any()
