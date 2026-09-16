"""Round-trip test of Store and the fetch kernel. Needs a GPU and Linux (O_DIRECT).

The GPU writes a request to pinned memory, the host thread reads it, preadv's
from the paging file into a RAM-tier row, and the GPU sees done and exits. Run
that sequence many times through CUDA graph replay and check the row contents.
The RAM-tier rows are compressed, so the expected values are built with
codec.encode, or the rows are expanded with decode_rows and compared with the
raw rows.
"""

import os
import threading
import time

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available() or not hasattr(os, "O_DIRECT"):
    pytest.skip("needs CUDA and O_DIRECT", allow_module_level=True)

from vllm_expert_pager import codec
from vllm_expert_pager.gather import fetch_decode_rows, fetch_rows, gather_rows
from vllm_expert_pager.store import _READ_WORKERS, Store

L, E, R = 2, 8, 4
# Rows are multiples of codec.GROUP (16 KiB).
ROW_BYTES = {"w13_weight": 2 * codec.GROUP, "w2_weight": codec.GROUP}
# Small rows do not shrink (the header adds to them), so allow the raw size.
PITCH = 1.0


def pattern(layer: int, expert: int, name: str) -> torch.Tensor:
    words = ROW_BYTES[name] // 4
    return torch.full(
        (words,), layer * 1000 + expert * 10 + len(name), dtype=torch.int32
    )


def encoded(store: Store, layer: int, expert: int, name: str) -> torch.Tensor:
    """The contents (int32 words) a RAM-tier row is expected to hold."""
    row = pattern(layer, expert, name)
    if not store.compress:
        return row
    raw = row.view(torch.uint8).cuda()
    return codec.encode(raw, store.pitch[name])[0].view(torch.int32).cpu()


def make_store(
    tmp_path,
    num_layers: int,
    num_experts: int,
    ram_slots: int,
    compress: bool = True,
) -> Store:
    s = Store(
        num_layers,
        num_experts,
        ram_slots,
        ROW_BYTES,
        compress,
        PITCH,
        str(tmp_path / "expert_pager.bin"),
        torch.device("cuda"),
    )
    for layer in range(num_layers):
        for e in range(num_experts):
            for name in ROW_BYTES:
                s.write(layer, e, name, pattern(layer, e, name))
    return s


@pytest.fixture(params=[True, False], ids=["compressed", "raw"])
def store(tmp_path, request):
    return make_store(tmp_path, L, E, R, compress=request.param)


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
                    store.pinned[name][layer * R + e], encoded(store, layer, e, name)
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
                store.pinned[name][layer * R + sa], encoded(store, layer, a, name)
            )
            assert torch.equal(
                store.pinned[name][layer * R + sb], encoded(store, layer, b, name)
            )
    assert store.fetches == 20 and store.reads == 40

    # An empty request never reaches the host.
    seq_before = int(store.req[0])
    todo.fill_(-1)
    g.replay()
    torch.cuda.synchronize()
    assert int(store.req[0]) == seq_before


def test_fetch_decode_expands_ram_hits_while_reading_ssd(store):
    # The decode shape: the first launch asks for the SSD read while copying
    # (and, with compression, expanding) the experts that are in RAM, and the
    # second launch copies the experts once they are read. Driven through graph
    # replay; the slab contents must equal the raw rows.
    dev = torch.device("cuda")
    layer, S = 1, 3
    base = layer * R
    W = [n // 4 for n in ROW_BYTES.values()]
    src = [store.view[name] for name in ROW_BYTES]
    cache = [torch.zeros((S, w), dtype=torch.int32, device=dev) for w in W]
    dst = [
        torch.zeros((S, n), dtype=torch.uint8, device=dev) for n in ROW_BYTES.values()
    ]
    dst_rows = [d.view(torch.int32) for d in dst]
    staging = [
        torch.zeros((S, store.pitch[name]), dtype=torch.uint8, device=dev)
        for name in ROW_BYTES
    ]
    stg = [t.view(torch.int32) for t in staging]
    scale = [
        torch.arange(E * 8, dtype=torch.int32, device=dev).view(E, 8) * (i + 1)
        for i in range(2)
    ]
    scale_dst = [torch.zeros((S, 8), dtype=torch.int32, device=dev) for _ in range(2)]
    todo = torch.full((S,), -1, dtype=torch.int64, device=dev)  # VRAM misses
    ssd = torch.full((S,), -1, dtype=torch.int64, device=dev)  # those read from SSD
    dst_row = torch.zeros((E,), dtype=torch.int64, device=dev)
    cache_row = torch.full((E,), -1, dtype=torch.int64, device=dev)
    src_row = torch.zeros((E,), dtype=torch.int64, device=dev)
    lut = store.layer_lut(layer)

    def step() -> None:
        if store.compress:
            fetch_decode_rows(todo, ssd, dst_row, src_row, base, layer, store,
                              src, dst, staging, scale, scale_dst, lut)  # fmt: skip
        else:
            gather_rows(todo, dst_row, cache_row, src_row, base, src, cache, dst_rows,
                        None, scale, scale_dst, fetch=(ssd, layer, store))  # fmt: skip
        gather_rows(ssd, dst_row, cache_row, src_row, base, src, cache, dst_rows,
                    stg if store.compress else None, scale, scale_dst)  # fmt: skip
        if store.compress:
            codec.decode_rows(ssd, dst_row, cache_row, staging, dst, lut)

    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        step()  # warmup (empty request)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            step()
    torch.cuda.synchronize()

    for i in range(6):
        # Experts a (< R, RAM slot a) and b (< R) come from RAM; expert c (>= R)
        # is first read from SSD into RAM slot R-1. Slab rows: a -> 0, c -> 1,
        # b -> 2.
        a, b, c = i % 2, 2, R + i % (E - R)
        todo.copy_(torch.tensor([a, c, b], device=dev))
        ssd.copy_(torch.tensor([c, -1, -1], device=dev))
        src_row[a], src_row[b], src_row[c] = a, b, R - 1
        dst_row[a], dst_row[c], dst_row[b] = 0, 1, 2
        dst[0].zero_(), dst[1].zero_()
        g.replay()
        torch.cuda.synchronize()
        assert store.failed is None, store.failed
        for k, name in enumerate(ROW_BYTES):
            raw = [pattern(layer, x, name).view(torch.uint8).to(dev) for x in (a, c, b)]
            for row in range(3):
                assert torch.equal(dst[k][row], raw[row])
            assert torch.equal(scale_dst[k][0], scale[k][a])
            assert torch.equal(scale_dst[k][1], scale[k][c])
            assert torch.equal(scale_dst[k][2], scale[k][b])
    assert store.fetches == 6 and store.reads == 6


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
            assert torch.equal(store.pinned[name][s], encoded(store, 0, e, name))


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
        assert torch.equal(
            store.pinned[name][layer * R + 0], encoded(store, layer, 5, name)
        )


def test_slow_host_makes_the_fetch_republish_the_request(store):
    # When the host answers slowly, the waiting fetch publishes its request
    # again. The request had arrived, so the re-publish does not change seq:
    # nothing is read twice and nothing is lost.
    dev = torch.device("cuda")
    todo = torch.full((R,), -1, dtype=torch.int64, device=dev)
    row = torch.zeros((E,), dtype=torch.int64, device=dev)
    layer = 1
    g = capture_fetch(store, todo, row, layer)

    read = store._read

    def slow(*args):
        time.sleep(1.5)  # longer than the re-publish interval (0.5 to 1.2 s)
        read(*args)

    store._read = slow
    todo.copy_(torch.tensor([4, -1, -1, -1], device=dev))
    row[4] = 0
    g.replay()
    torch.cuda.synchronize()

    assert store.failed is None, store.failed
    assert int(store.req[store.retry_at]) >= 1
    assert store.fetches == 1 and store.reads == 1
    for name in ROW_BYTES:
        assert torch.equal(
            store.pinned[name][layer * R], encoded(store, layer, 4, name)
        )


def test_fetch_gives_up_when_the_host_never_answers(store):
    # Simulates a handoff broken in both directions (the host never answers).
    # The fetch stops waiting after a few seconds, so the server does not
    # stall. The give-up count stays in pinned memory, and the host notices
    # from the gap in seq at the next request.
    dev = torch.device("cuda")
    todo = torch.full((R,), -1, dtype=torch.int64, device=dev)
    row = torch.zeros((E,), dtype=torch.int64, device=dev)
    g = capture_fetch(store, todo, row, 1)

    release = threading.Event()
    store._read = lambda *args: release.wait(60)
    todo.copy_(torch.tensor([4, -1, -1, -1], device=dev))
    row[4] = 0
    t0 = time.perf_counter()
    g.replay()
    torch.cuda.synchronize()
    waited = time.perf_counter() - t0
    release.set()

    assert waited < 30, f"the fetch waited {waited:.1f} s"
    assert int(store.req[store.giveup_at]) == 1
    assert int(store.req[store.retry_at]) >= 1


def test_serve_resends_done_when_the_fetch_is_still_spinning(store):
    # When done did not reach the GPU. The GPU publishes its request again, so
    # the host rewrites done even for a seq it has already served and releases
    # the GPU. Reproduced by resetting done to 0.
    dev = torch.device("cuda")
    todo = torch.full((R,), -1, dtype=torch.int64, device=dev)
    row = torch.zeros((E,), dtype=torch.int64, device=dev)
    g = capture_fetch(store, todo, row, 1)
    todo.copy_(torch.tensor([4, -1, -1, -1], device=dev))
    row[4] = 0
    g.replay()
    torch.cuda.synchronize()

    served = int(store.done[0])
    assert served > 0
    store.done[0] = 0
    store.req[store.retry_at] += 1
    for _ in range(200):
        if int(store.done[0]) == served:
            break
        time.sleep(0.01)
    assert int(store.done[0]) == served


def test_large_pinned_is_not_rounded_to_pow2(tmp_path):
    # Store turns off the allocator's power-of-two rounding for large
    # allocations (store.py). 1.5 GiB would become 2 GiB if rounded. The
    # allocation happens only once (a second one hits the allocator's cache and
    # shows no increase), so this does not use the parametrized store fixture.
    make_store(tmp_path, 1, 2, 1)
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
        Store(L, E, R, ROW_BYTES, True, PITCH, str(path), torch.device("cuda"))


def test_rejects_rows_over_pitch(tmp_path):
    # A row that does not shrink (uniform random) does not fit the pitch, and
    # the error reports the ratio needed.
    s = Store(
        1,
        2,
        1,
        ROW_BYTES,
        True,
        0.5,
        str(tmp_path / "expert_pager.bin"),
        torch.device("cuda"),
    )
    row = torch.randint(
        -(2**31), 2**31 - 1, (ROW_BYTES["w2_weight"] // 4,), dtype=torch.int32
    )
    with pytest.raises(ValueError, match="VLLM_EXPERT_PAGER_PITCH must be at least"):
        s.write(0, 0, "w2_weight", row)
