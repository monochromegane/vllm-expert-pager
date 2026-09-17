"""RAM and SSD tiers for expert weights. Both hold the rows compressed (codec.py).

The RAM tier is an array of rows in pinned memory. The GPU reads it through a
UVA view and the host writes it through a numpy view. Each layer has R rows,
followed by the shared working rows for prefill (all layers; the SSD reads of
experts that are not in the table land there); row numbers are unique across
layers so the gather kernel has a single source.

The SSD tier is a paging file of fixed-length records (the w13 row followed by
the w2 row, each compressed and rounded up to its pitch) in [layer][expert]
order. weight_loader writes it at load time; at inference a C thread on the
host serves requests from the GPU with O_DIRECT preadv, and the experts of one
layer are read concurrently by worker threads. O_DIRECT keeps the page cache
from holding a second copy of the RAM tier. The host does not expand the rows;
the GPU does, after copying them into VRAM.

The Huffman LUT differs per row but is only 512 B, so it is not part of the
record. It lives in a VRAM-resident table like the scales (``lut``, indexed by
layer and expert). When gather also expands, it copies per group, and a LUT per
group would add 3.6% to the PCIe traffic (codec.py).

With ``compress=False`` the rows are stored raw. When every expert fits in RAM
there are no SSD reads, and the expansion costs more than the 12% saved on PCIe.

The GPU and host communicate through a request ring and a done word in pinned
memory. The GPU writes a request (layer, and the list of experts and rows) to
slot ``seq % RING`` and publishes its seq; the host serves requests in seq
order and writes the served seq to done. Up to RING requests can be
outstanding. done advances in order, so waiting for a seq also covers every
request before it. The GPU side is the fetch kernel in gather.py.

The host side is a C thread with no Python in between (built by ``native.py``).
When the GPU waits on a Python thread, a cycle through the GIL can stall both.
With R = E (RAM only) no requests are made, so the server is not started.
"""

import ctypes
import faulthandler
import os
import signal
import threading

import torch
from vllm.logger import init_logger
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from vllm_expert_pager import codec, native

logger = init_logger(f"vllm.{__name__}")

# Alignment required by O_DIRECT. Rows, records and the pinned base address
# must be multiples of this.
_ALIGN = 4096
# torch's pinned allocator (CachingHostAllocator) rounds allocation sizes up to
# the next power of two. The RAM tier takes tens of GiB per name in a single
# allocation, so a 20.5 GiB request would consume 32 GiB. Change the allocator
# setting so pinned allocations of this size or more are not rounded.
_PINNED_ROUND_LIMIT_MB = 1024
# Number of threads that read the experts of one layer concurrently (the server
# itself and the workers). The NVMe of the test machine (WSL2) delivers about
# 2.9 GB/s with a single outstanding request and saturates at about 4.7 GB/s
# from eight requests on. Splitting one record into pieces does not make it
# faster.
_READ_WORKERS = 8
# Number of slots in the request ring. The fetch kernel has one request
# outstanding at a time; 4 leaves room for more.
RING = 4
_C_SOURCE = r"""
#define _GNU_SOURCE
#include <errno.h>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

/* req = [re-publishes, give-ups] + RING slots of [seq, layer, expert[B], row[B]] */
#define HEAD 2
#define MAX_NAMES 4

typedef struct {
  int fd;
  volatile int64_t *req, *done;
  long ring, block, slot_words;
  long L, E, rows, record_bytes;
  int names;
  uint8_t *pinned[MAX_NAMES];
  long pitch[MAX_NAMES];

  /* Statistics. fetches counts the requests served, reads the experts read, io_seconds the
   * total time from seeing a request to writing done, and missed the requests the GPU gave
   * up on before they arrived. */
  long fetches, reads, missed;
  double io_seconds;
  char err[512];            /* description of the first failure */
  int failing, failed;

  /* The current request's reads. Taking a job and updating finished happen under mu, so a
   * worker that wakes up late cannot take a job of the next request by mistake. */
  long layer, njobs, next, finished;
  long *job_expert, *job_row;
  long gen;                 /* request generation; the workers wait for it to move */
  pthread_mutex_t mu;
  pthread_cond_t cv;
  int nworkers;
  pthread_t *tids, server;
  int stop;
  volatile double test_delay;  /* for tests: delay handling a request by this many seconds after seeing it. Reset to 0 to leave */
} ssd_t;

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static void pause_or_yield(long *spins) {
  if (++*spins > 200000) { sched_yield(); *spins = 0; } else _mm_pause();
}

static void fail(ssd_t *s, const char *fmt, ...) {
  if (__atomic_exchange_n(&s->failing, 1, __ATOMIC_ACQ_REL)) return;  /* keep only the first failure */
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(s->err, sizeof s->err, fmt, ap);
  va_end(ap);
  __atomic_store_n(&s->failed, 1, __ATOMIC_RELEASE);
}

/* Read one expert (the w13 row followed by the w2 row) into a RAM-tier row. */
static void read_one(ssd_t *s, long j) {
  long layer = s->layer, e = s->job_expert[j], row = s->job_row[j];
  if (layer < 0 || layer >= s->L || e < 0 || e >= s->E) {
    fail(s, "expert %ld of layer %ld is out of range", e, layer);
    return;
  }
  if (row < 0 || row >= s->rows) {
    fail(s, "row %ld for expert %ld of layer %ld is outside the RAM tier of %ld rows",
         row, e, layer, s->rows);
    return;
  }
  struct iovec iov[MAX_NAMES];
  for (int i = 0; i < s->names; i++) {
    iov[i].iov_base = s->pinned[i] + row * s->pitch[i];
    iov[i].iov_len = (size_t)s->pitch[i];
  }
  off_t off = (off_t)(layer * s->E + e) * s->record_bytes;
  ssize_t n = preadv(s->fd, iov, s->names, off);
  if (n < 0)
    fail(s, "preadv of expert %ld of layer %ld failed: %s", e, layer, strerror(errno));
  else if (n != s->record_bytes)
    fail(s, "short read: %zd of %ld B for expert %ld of layer %ld", n, s->record_bytes, e, layer);
}

/* Take jobs of the current request until none is left. Both the server and the workers run this. */
static void run_jobs(ssd_t *s) {
  for (;;) {
    pthread_mutex_lock(&s->mu);
    long j = s->next < s->njobs ? s->next++ : -1;
    pthread_mutex_unlock(&s->mu);
    if (j < 0) return;
    read_one(s, j);
    pthread_mutex_lock(&s->mu);
    s->finished++;
    pthread_mutex_unlock(&s->mu);
  }
}

static void *worker_main(void *arg) {
  ssd_t *s = (ssd_t *)arg;
  long seen = 0;
  for (;;) {
    pthread_mutex_lock(&s->mu);
    while (s->gen == seen && !s->stop) pthread_cond_wait(&s->cv, &s->mu);
    seen = s->gen;
    int stop = s->stop;
    pthread_mutex_unlock(&s->mu);
    if (stop) return NULL;
    run_jobs(s);
  }
}

/* Serve the GPU's requests in seq order. Wait for requests by spinning (a sleep adds 100 us
 * to the round trip). */
static void *server_main(void *arg) {
  ssd_t *s = (ssd_t *)arg;
  int64_t last = 0, retries = 0;
  for (;;) {
    volatile int64_t *slot = s->req + HEAD + ((last + 1) % s->ring) * s->slot_words;
    long spins = 0, idle = 0;
    int64_t seq;
    while ((seq = __atomic_load_n(slot, __ATOMIC_ACQUIRE)) <= last) {
      if (__atomic_load_n(&s->stop, __ATOMIC_ACQUIRE)) return NULL;
      /* done may not have reached a waiting GPU. When the re-publish count moves, or nothing
       * arrives for a long time, rewrite the served done. The value does not change. */
      int64_t r = __atomic_load_n(&s->req[0], __ATOMIC_ACQUIRE);
      if (r != retries || ++idle > (1L << 25)) {
        retries = r;
        idle = 0;
        __atomic_store_n(s->done, last, __ATOMIC_RELEASE);
      }
      pause_or_yield(&spins);
    }
    double t0 = now();
    /* If the slot holds a seq beyond last+1, the requests in between were given up by the GPU
     * and never visible to the host until the slot was overwritten. The rows of those layers
     * are stale, so count them. */
    if (seq > last + 1) s->missed += (long)(seq - last - 1);
    while (s->test_delay > 0 && now() - t0 < s->test_delay) {
      if (__atomic_load_n(&s->stop, __ATOMIC_ACQUIRE)) return NULL;
      usleep(1000);
    }

    long n = 0;
    pthread_mutex_lock(&s->mu);
    s->layer = (long)slot[1];
    for (long j = 0; j < s->block; j++) {
      int64_t e = slot[2 + j];
      if (e < 0) continue;
      s->job_expert[n] = (long)e;
      s->job_row[n] = (long)slot[2 + s->block + j];
      n++;
    }
    s->njobs = n;
    s->next = s->finished = 0;
    if (n > 1) {  /* a single job is read here without waking anyone */
      s->gen++;
      pthread_cond_broadcast(&s->cv);
    }
    pthread_mutex_unlock(&s->mu);
    run_jobs(s);
    spins = 0;
    while (__atomic_load_n(&s->finished, __ATOMIC_ACQUIRE) < n) pause_or_yield(&spins);

    s->fetches++;
    s->reads += n;
    s->io_seconds += now() - t0;
    /* Re-publishes while the read was taking long are not lost handoffs, so absorb them. */
    retries = __atomic_load_n(&s->req[0], __ATOMIC_ACQUIRE);
    /* Release the GPU even when a read failed. The next forward that runs Python sees the
     * failure and stops. */
    __atomic_store_n(s->done, seq, __ATOMIC_RELEASE);
    last = seq;
  }
}

ssd_t *ssd_start(int fd, void *req, void *done,
                 long ring, long block, long L, long E, long rows, long record_bytes,
                 int names, void **pinned, long *pitch, int nworkers) {
  if (names > MAX_NAMES) return NULL;
  ssd_t *s = (ssd_t *)calloc(1, sizeof *s);
  if (!s) return NULL;
  s->fd = fd;
  s->req = (volatile int64_t *)req;
  s->done = (volatile int64_t *)done;
  s->ring = ring;
  s->block = block;
  s->slot_words = 2 + 2 * block;
  s->L = L; s->E = E; s->rows = rows; s->record_bytes = record_bytes;
  s->names = names;
  for (int i = 0; i < names; i++) {
    s->pinned[i] = (uint8_t *)pinned[i];
    s->pitch[i] = pitch[i];
  }
  s->job_expert = (long *)calloc((size_t)block, sizeof(long));
  s->job_row = (long *)calloc((size_t)block, sizeof(long));
  s->nworkers = nworkers;
  s->tids = (pthread_t *)calloc((size_t)nworkers, sizeof(pthread_t));
  if (!s->job_expert || !s->job_row || !s->tids) return NULL;
  pthread_mutex_init(&s->mu, NULL);
  pthread_cond_init(&s->cv, NULL);
  for (int i = 0; i < nworkers; i++) pthread_create(&s->tids[i], NULL, worker_main, s);
  pthread_create(&s->server, NULL, server_main, s);
  return s;
}

void ssd_stop(ssd_t *s) {
  pthread_mutex_lock(&s->mu);
  s->stop = 1;
  pthread_cond_broadcast(&s->cv);
  pthread_mutex_unlock(&s->mu);
  pthread_join(s->server, NULL);
  for (int i = 0; i < s->nworkers; i++) pthread_join(s->tids[i], NULL);
  free(s->job_expert);
  free(s->job_row);
  free(s->tids);
  free(s);
}

void ssd_stats(ssd_t *s, long *fetches, long *reads, double *io_seconds, long *missed) {
  *fetches = s->fetches;
  *reads = s->reads;
  *io_seconds = s->io_seconds;
  *missed = s->missed;
}

/* The description of the failed read, or NULL if none failed. */
const char *ssd_error(ssd_t *s) {
  return __atomic_load_n(&s->failed, __ATOMIC_ACQUIRE) ? s->err : NULL;
}

void ssd_test_delay(ssd_t *s, double seconds) { s->test_delay = seconds; }
"""


def _build() -> ctypes.CDLL:
    so = native.build("ssd", _C_SOURCE)
    so.ssd_start.restype = ctypes.c_void_p
    so.ssd_start.argtypes = (
        [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        + [ctypes.c_long] * 6
        + [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    )
    so.ssd_stop.argtypes = [ctypes.c_void_p]
    so.ssd_stats.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_long),
    ]
    so.ssd_error.restype = ctypes.c_char_p
    so.ssd_error.argtypes = [ctypes.c_void_p]
    so.ssd_test_delay.argtypes = [ctypes.c_void_p, ctypes.c_double]
    return so


def _pinned_aligned(rows: int, words: int) -> torch.Tensor:
    """Pinned (rows, words) int32 tensor whose start is aligned to _ALIGN.

    torch's pinned allocator may carve a piece out of a larger block, so the
    start is not necessarily on a page boundary. Allocate extra and use the
    aligned part.
    """
    # The default device is cuda while the model is being built, so name the
    # CPU explicitly.
    slack = _ALIGN // 4
    flat = torch.empty(
        (rows * words + slack,), dtype=torch.int32, device="cpu", pin_memory=True
    )
    skip = (-flat.data_ptr() % _ALIGN) // 4
    return flat[skip : skip + rows * words].view(rows, words)


class Store:
    """One per model. The RAM tier's pinned buffer, the paging file and the C host thread."""

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        ram_slots: int,
        working_rows: int,
        row_bytes: dict[str, int],
        compress: bool,
        pitch_ratio: float,
        path: str | None,
        device: torch.device,
    ) -> None:
        L, E, R = num_layers, num_experts, ram_slots
        if not 1 <= R <= E:
            raise ValueError(
                f"VLLM_EXPERT_PAGER_RAM_SLOTS must be in [1, {E}], got {R}"
            )
        if working_rows < 1:
            raise ValueError(f"working_rows must be >= 1, got {working_rows}")
        if R < E and path is None:
            raise ValueError(
                "VLLM_EXPERT_PAGER_SSD_PATH is required when "
                "VLLM_EXPERT_PAGER_RAM_SLOTS < num_experts"
            )
        for name, nbytes in row_bytes.items():
            unit = codec.GROUP if compress else _ALIGN
            if nbytes % unit:
                raise ValueError(
                    f"{name}: row of {nbytes} B is not a multiple of {unit}"
                )

        self.num_layers, self.num_experts, self.ram_slots = L, E, R
        self.layers = 0
        self.device = device
        self.compress = compress
        self.names = list(row_bytes)
        # Raw (expanded) rows, and the fixed length of the rows in the tiers.
        # The pitch is a multiple of _ALIGN, so it satisfies O_DIRECT.
        self.row_bytes = dict(row_bytes)
        self.pitch = {
            name: codec.pitch_of(nbytes, pitch_ratio, _ALIGN) if compress else nbytes
            for name, nbytes in row_bytes.items()
        }
        self.record_bytes = sum(self.pitch.values())
        self.offset = {}
        pos = 0
        for name in self.names:
            self.offset[name] = pos
            pos += self.pitch[name]

        # Rows [0, L*R) are the RAM tier (slot s of layer l is l*R + s); the
        # working_rows rows after them are the shared working rows.
        self.working_base = L * R
        self.rows = L * R + working_rows
        torch._C._accelerator_setAllocatorSettings(
            f"pinned_max_round_threshold_mb:{_PINNED_ROUND_LIMIT_MB}"
        )
        logger.info(
            "vllm-expert-pager: RAM tier %d rows x %d B (%s) = %.2f GiB pinned",
            self.rows,
            self.record_bytes,
            f"compressed from {sum(row_bytes.values())} B" if compress else "raw",
            self.rows * self.record_bytes / 2**30,
        )
        self.pinned = {
            name: _pinned_aligned(self.rows, nbytes // 4)
            for name, nbytes in self.pitch.items()
        }
        # Per-row Huffman LUTs: lut[l, e, i] belongs to expert e of layer l,
        # name i. Without compression they are never read, so allocate just
        # enough to have a pointer to pass to the kernel.
        self.lut = torch.zeros(
            (L if compress else 1, E, len(self.names), codec.LUT_BYTES),
            dtype=torch.uint8,
            device=device,
        )
        self.view = {
            n: get_accelerator_view_from_cpu_tensor(t) for n, t in self.pinned.items()
        }

        # req = [re-publishes, give-ups] + RING slots of [seq, layer, expert[B],
        # row[B]]. B is the fetch kernel's BLOCK (the power of two >= E). The
        # first two words advance when a spinning fetch gives up waiting and
        # publishes its request again, and when even that does not get through
        # and it stops waiting.
        self.block = 1 << (E - 1).bit_length()
        self.ring = RING
        self.retry_at, self.giveup_at = 0, 1
        self.slot_words = 2 + 2 * self.block
        self.req = torch.zeros(
            (2 + RING * self.slot_words,),
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        self.done = torch.zeros((1,), dtype=torch.int64, device="cpu", pin_memory=True)
        self.req_view = get_accelerator_view_from_cpu_tensor(self.req)
        self.done_view = get_accelerator_view_from_cpu_tensor(self.done)
        # Sequence number of the requests (device). Publishing advances it and
        # also writes the published seq to a ticket, which the waiter waits for.
        # Prefill publishes the next chunk's read before waiting for the current
        # chunk, so it alternates between two tickets (ticket_chunk).
        self.seq = torch.zeros((1,), dtype=torch.int64, device=device)
        self.ticket_chunk = torch.zeros((2,), dtype=torch.int64, device=device)

        self.path = path if R < E else None
        self.fd_w = self.fd_r = None
        self._handle = None
        if self.path is not None:
            size = L * E * self.record_bytes
            # An existing file is only overwritten if it is a previous paging
            # file (same size).
            if os.path.exists(path) and os.path.getsize(path) != size:
                raise FileExistsError(
                    f"{path} exists and is not a paging file of {size} B; remove it first"
                )
            self.fd_w = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            self.fd_r = os.open(path, os.O_RDONLY | os.O_DIRECT)
            logger.info(
                "vllm-expert-pager: paging file %s (%.1f GiB)", path, size / 2**30
            )
            self._start_server()
        # For diagnosing a stall: kill -USR2 <EngineCore pid> dumps the Python
        # stack of every thread to stderr (the server log).
        if threading.current_thread() is threading.main_thread():
            faulthandler.register(signal.SIGUSR2, all_threads=True)

    def _start_server(self) -> None:
        n = len(self.names)
        pinned = (ctypes.c_void_p * n)(*(self.pinned[k].data_ptr() for k in self.names))
        pitch = (ctypes.c_long * n)(*(self.pitch[k] for k in self.names))
        self._so = _build()
        self._handle = self._so.ssd_start(
            self.fd_r,
            ctypes.c_void_p(self.req.data_ptr()),
            ctypes.c_void_p(self.done.data_ptr()),
            self.ring,
            self.block,
            self.num_layers,
            self.num_experts,
            self.rows,
            self.record_bytes,
            n,
            pinned,
            pitch,
            # The server reads too, so _READ_WORKERS reads run concurrently.
            _READ_WORKERS - 1,
        )
        if not self._handle:
            raise RuntimeError("vllm-expert-pager: cannot start the SSD server")

    def close(self) -> None:
        """Stop the host thread. It normally ends with the process; this is for tests and GC."""
        if getattr(self, "_handle", None):
            self._so.ssd_stop(ctypes.c_void_p(self._handle))
            self._handle = None

    __del__ = close

    def test_delay(self, seconds: float) -> None:
        """For tests. Delay the host between seeing a request and handling it by ``seconds``. 0 clears it."""
        self._so.ssd_test_delay(ctypes.c_void_p(self._handle), seconds)

    # ---- Statistics (written by the host thread) ----

    def _stats(self) -> tuple[int, int, float, int]:
        if not self._handle:
            return 0, 0, 0.0, 0
        fetches, reads, missed = (ctypes.c_long() for _ in range(3))
        io_seconds = ctypes.c_double()
        self._so.ssd_stats(
            ctypes.c_void_p(self._handle),
            ctypes.byref(fetches),
            ctypes.byref(reads),
            ctypes.byref(io_seconds),
            ctypes.byref(missed),
        )
        return fetches.value, reads.value, io_seconds.value, missed.value

    @property
    def fetches(self) -> int:
        """Number of requests served."""
        return self._stats()[0]

    @property
    def reads(self) -> int:
        """Number of experts read."""
        return self._stats()[1]

    @property
    def io_seconds(self) -> float:
        """Total time from seeing a request to writing done."""
        return self._stats()[2]

    @property
    def missed(self) -> int:
        """Requests the GPU gave up on before they arrived (non-zero means a handoff was lost)."""
        return self._stats()[3]

    @property
    def failed(self) -> str | None:
        """Description of a failed read, if any. forward checks it and raises."""
        if not self._handle:
            return None
        err = self._so.ssd_error(ctypes.c_void_p(self._handle))
        return None if err is None else err.decode(errors="replace")

    def layer_lut(self, layer: int) -> torch.Tensor:
        """The LUTs ``(E, names, 512)`` of layer ``layer``. Unused without compression."""
        return self.lut[layer if self.compress else 0]

    def add_layer(self) -> int:
        """Hand out a layer number. Layers register in construction order."""
        if self.layers >= self.num_layers:
            raise RuntimeError(
                f"more MoE layers than num_hidden_layers={self.num_layers}"
            )
        self.layers += 1
        return self.layers - 1

    # ---- Load time ----

    def write(self, layer: int, expert: int, name: str, row: torch.Tensor) -> None:
        """Put one expert's weights (compressed if compress) in the paging file and the RAM tier.

        The RAM tier gets them only when e < R (slot e).
        """
        raw = row.contiguous().view(torch.uint8).reshape(-1)
        if raw.numel() != self.row_bytes[name]:
            raise ValueError(
                f"{name}: expert {expert} of layer {layer} has {raw.numel()} B, "
                f"expected {self.row_bytes[name]}"
            )
        if self.compress:
            record, lut = codec.encode(raw.to(self.device), self.pitch[name])
            self.lut[layer, expert, self.names.index(name)] = lut
            words = record.view(torch.int32).cpu()
        else:
            words = raw.view(torch.int32)
        if expert < self.ram_slots:
            self.pinned[name][layer * self.ram_slots + expert].copy_(words)
        if self.fd_w is not None:
            offset = (
                layer * self.num_experts + expert
            ) * self.record_bytes + self.offset[name]
            os.pwrite(self.fd_w, memoryview(words.numpy()), offset)
