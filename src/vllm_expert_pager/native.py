"""Build the host-side C code with ``cc`` on first use and load it through ctypes.

Like Triton, this wants ``-march=native``, so the package ships no extension
module and builds in the running environment. The result is cached under
``~/.cache/vllm_expert_pager`` and reused for the same source. Used by
``store.py`` (the server that reads the SSD).
"""

import ctypes
import hashlib
import os
import subprocess
import tempfile

from vllm.logger import init_logger

logger = init_logger(f"vllm.{__name__}")

_FLAGS = ["-O3", "-march=native", "-funroll-loops", "-fPIC", "-shared", "-pthread"]


def build(name: str, source: str) -> ctypes.CDLL:
    """Compile ``source`` into a shared library and load it. ``name`` prefixes the cached file."""
    key = hashlib.sha256((source + " ".join(_FLAGS)).encode()).hexdigest()[:16]
    cache = os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "vllm_expert_pager",
    )
    os.makedirs(cache, exist_ok=True)
    lib = os.path.join(cache, f"{name}-{key}.so")
    if not os.path.exists(lib):
        # Build next to the cache (/tmp may be on a different device).
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, f"{name}.c")
            with open(src, "w") as f:
                f.write(source)
            out = f"{lib}.{os.getpid()}"
            r = subprocess.run(
                [os.environ.get("CC", "cc"), *_FLAGS, src, "-o", out, "-lm"],
                capture_output=True,
                text=True,
                check=False,
            )
            if r.returncode:
                raise RuntimeError(
                    f"vllm-expert-pager: cannot build {name}\n{r.stderr}"
                )
            os.replace(out, lib)
        logger.info("vllm-expert-pager: built %s at %s", name, lib)
    return ctypes.CDLL(lib)
