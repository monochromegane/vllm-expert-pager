"""Round-trip test of Store and the fetch kernel. Needs a GPU and Linux (O_DIRECT).

The GPU writes a request to pinned memory, the host thread reads it, preadv's
from the paging file into a RAM-tier row, and the GPU sees done and exits. Run
that sequence many times through CUDA graph replay and check the row contents.
"""

import os

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available() or not hasattr(os, "O_DIRECT"):
    pytest.skip("needs CUDA and O_DIRECT", allow_module_level=True)

from vllm_expert_pager.gather import fetch_rows
from vllm_expert_pager.store import Store

L, E, R = 2, 8, 4
ROW_BYTES = {"w13_weight": 8192, "w2_weight": 4096}


def pattern(layer: int, expert: int, name: str) -> torch.Tensor:
    words = ROW_BYTES[name] // 4
    return torch.full(
        (words,), layer * 1000 + expert * 10 + len(name), dtype=torch.int32
    )


@pytest.fixture
def store(tmp_path):
    s = Store(
        L, E, R, ROW_BYTES, str(tmp_path / "expert_pager.bin"), torch.device("cuda")
    )
    for layer in range(L):
        for e in range(E):
            for name in ROW_BYTES:
                s.write(layer, e, name, pattern(layer, e, name))
    return s


def test_warm_start_rows(store):
    for layer in range(L):
        for e in range(R):
            for name in ROW_BYTES:
                assert torch.equal(
                    store.pinned[name][layer * R + e], pattern(layer, e, name)
                )


def test_fetch_reads_missing_experts_in_graph(store):
    dev = torch.device("cuda")
    todo = torch.full((R,), -1, dtype=torch.int64, device=dev)
    row = torch.zeros((E,), dtype=torch.int64, device=dev)
    layer = 1

    # Only addresses are baked into the graph; swap the contents and replay.
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fetch_rows(todo, row, layer * R, layer, store)  # warmup (empty request)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            fetch_rows(todo, row, layer * R, layer, store)
    torch.cuda.synchronize()

    for i in range(20):
        # Experts 4..7 are not in RAM. Read them into slots i % R and (i+1) % R.
        a, b = 4 + i % 4, 4 + (i + 1) % 4
        sa, sb = i % R, (i + 1) % R
        todo.copy_(torch.tensor([a, b, -1, -1], device=dev))
        row[a], row[b] = sa, sb
        g.replay()
        torch.cuda.synchronize()
        assert store.failed is None, store.failed
        for name in ROW_BYTES:
            assert torch.equal(
                store.pinned[name][layer * R + sa], pattern(layer, a, name)
            )
            assert torch.equal(
                store.pinned[name][layer * R + sb], pattern(layer, b, name)
            )

    # An empty request never reaches the host.
    seq_before = int(store.req[0])
    todo.fill_(-1)
    g.replay()
    torch.cuda.synchronize()
    assert int(store.req[0]) == seq_before


def test_large_pinned_is_not_rounded_to_pow2(store):
    # Store turns off the allocator's power-of-two rounding for large
    # allocations (store.py). 1.5 GiB would become 2 GiB if rounded.
    n = 3 * 2**29
    key = "allocated_bytes.current"
    before = torch.cuda.host_memory_stats()[key]
    t = torch.empty((n,), dtype=torch.uint8, device="cpu", pin_memory=True)
    assert torch.cuda.host_memory_stats()[key] - before == n
    del t


def test_rejects_foreign_file(tmp_path):
    path = tmp_path / "expert_pager.bin"
    path.write_bytes(b"not a paging file")
    with pytest.raises(FileExistsError):
        Store(L, E, R, ROW_BYTES, str(path), torch.device("cuda"))
