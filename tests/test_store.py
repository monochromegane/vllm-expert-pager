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
from vllm_expert_pager.store import _READ_WORKERS, Store

L, E, R = 2, 8, 4
ROW_BYTES = {"w13_weight": 8192, "w2_weight": 4096}


def pattern(layer: int, expert: int, name: str) -> torch.Tensor:
    words = ROW_BYTES[name] // 4
    return torch.full(
        (words,), layer * 1000 + expert * 10 + len(name), dtype=torch.int32
    )


def make_store(tmp_path, num_layers: int, num_experts: int, ram_slots: int) -> Store:
    s = Store(
        num_layers,
        num_experts,
        ram_slots,
        ROW_BYTES,
        str(tmp_path / "expert_pager.bin"),
        torch.device("cuda"),
    )
    for layer in range(num_layers):
        for e in range(num_experts):
            for name in ROW_BYTES:
                s.write(layer, e, name, pattern(layer, e, name))
    return s


@pytest.fixture
def store(tmp_path):
    return make_store(tmp_path, L, E, R)


def capture_fetch(store: Store, todo, row, layer: int) -> torch.cuda.CUDAGraph:
    """Capture fetch into a graph.

    Only addresses are baked into the graph; the contents are swapped before
    replay.
    """
    base = layer * store.ram_slots
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fetch_rows(todo, row, base, layer, store)  # warmup (empty request)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            fetch_rows(todo, row, base, layer, store)
    torch.cuda.synchronize()
    return g


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
    g = capture_fetch(store, todo, row, layer)

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
    assert store.fetches == 20 and store.reads == 40

    # An empty request never reaches the host.
    seq_before = int(store.req[0])
    todo.fill_(-1)
    g.replay()
    torch.cuda.synchronize()
    assert int(store.req[0]) == seq_before


def test_fetch_reads_many_experts_at_once(tmp_path):
    # Read more experts than there are threads in one request and check that
    # done is written only once all of them are in. Expert R+s goes to slot s.
    E2, R2 = 4 * _READ_WORKERS, 2 * _READ_WORKERS
    store = make_store(tmp_path, 1, E2, R2)
    dev = torch.device("cuda")
    todo = torch.full((R2,), -1, dtype=torch.int64, device=dev)
    row = torch.zeros((E2,), dtype=torch.int64, device=dev)
    g = capture_fetch(store, todo, row, 0)

    missing = list(range(R2, E2))
    todo.copy_(torch.tensor(missing, device=dev))
    row[R2:] = torch.arange(R2, device=dev)
    g.replay()
    torch.cuda.synchronize()
    assert store.failed is None, store.failed
    assert store.fetches == 1 and store.reads == R2
    for s, e in enumerate(missing):
        for name in ROW_BYTES:
            assert torch.equal(store.pinned[name][s], pattern(0, e, name))


def test_failed_read_releases_gpu(store):
    # Make one read fail by pointing its row outside the RAM tier. The host
    # finishes the other reads, then writes done and releases the GPU (without
    # that, synchronize would never return).
    dev = torch.device("cuda")
    todo = torch.full((R,), -1, dtype=torch.int64, device=dev)
    row = torch.zeros((E,), dtype=torch.int64, device=dev)
    layer = 1
    g = capture_fetch(store, todo, row, layer)

    todo.copy_(torch.tensor([4, 5, -1, -1], device=dev))
    row[4], row[5] = 10**6, 0
    g.replay()
    torch.cuda.synchronize()
    assert store.failed is not None and "IndexError" in store.failed
    for name in ROW_BYTES:
        assert torch.equal(store.pinned[name][layer * R + 0], pattern(layer, 5, name))


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
