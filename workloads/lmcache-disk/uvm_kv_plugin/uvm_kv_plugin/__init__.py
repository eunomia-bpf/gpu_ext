"""vLLM general plugin: scoped UVM allocator pool for KV cache blocks.

Installed through the official ``vllm.general_plugins`` entry-point group.
When enabled, the GPU worker's ``_maybe_get_memory_pool_context`` is wrapped
so the ``kv_cache`` tag is backed by the same UVM pluggable allocator used
for weights (``uvm_allocator.so`` exporting ``uvm_malloc``/``uvm_free``)
inside a scoped ``torch.cuda.MemPool``. Every other memory-pool tag
(e.g. ``weights``) delegates to stock vLLM behavior.

Environment (read once, at install time, per process):
    UVM_KV_PLUGIN
        "1"/"true"/"yes"/"on" enables the plugin. Unset keeps it inert.
    UVM_KV_PLUGIN_SO
        Absolute path to ``uvm_allocator.so``. Required when enabled.
    UVM_KV_PLUGIN_COUNTERS
        "1"/"true"/"yes"/"on" logs allocator enter/exit counters.
"""

from __future__ import annotations

import atexit
import ctypes
import functools
import gc
import logging
import os
from contextlib import contextmanager

import torch

logger = logging.getLogger("uvm_kv_plugin")

KV_TAG = "kv_cache"
MALLOC_FN = "uvm_malloc"
FREE_FN = "uvm_free"

ENABLE_ENV = "UVM_KV_PLUGIN"
SO_ENV = "UVM_KV_PLUGIN_SO"
COUNTERS_ENV = "UVM_KV_PLUGIN_COUNTERS"

_TRUTHY = ("1", "true", "yes", "on")

_so_path: str | None = None
_counters_enabled = False
_keep_alive: list[tuple[object, object]] = []
_stats_lib: ctypes.CDLL | None = None
_installed = False


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _get_stats_lib() -> ctypes.CDLL:
    global _stats_lib
    if _stats_lib is None:
        lib = ctypes.CDLL(_so_path)
        for fn in (
            "uvm_get_allocated_bytes",
            "uvm_get_peak_allocated_bytes",
            "uvm_get_num_allocs",
            "uvm_get_num_frees",
        ):
            getattr(lib, fn).restype = ctypes.c_size_t
            getattr(lib, fn).argtypes = []
        _stats_lib = lib
    return _stats_lib


def _log_counters(phase: str) -> None:
    try:
        lib = _get_stats_lib()
        live = lib.uvm_get_allocated_bytes()
        peak = lib.uvm_get_peak_allocated_bytes()
        allocs = lib.uvm_get_num_allocs()
        frees = lib.uvm_get_num_frees()
    except Exception:
        logger.warning(
            "uvm_kv_plugin: cannot read allocator counters (%s)",
            phase,
            exc_info=True,
        )
        return
    logger.info(
        "uvm_kv_plugin: %s UVM kv_cache pool | allocs=%d frees=%d "
        "live=%d bytes (%.2f GiB) peak=%d bytes",
        phase,
        allocs,
        frees,
        live,
        live / 2**30,
        peak,
    )


@contextmanager
def _uvm_kv_pool():
    """Route allocations through the UVM allocator for the kv_cache context.

    The allocator and pool wrappers are kept alive for the process lifetime:
    ``MemPool`` holds a non-owning pointer to the C++ allocator owned by the
    ``CUDAPluggableAllocator`` wrapper, and the KV cache blocks stay
    allocated in the pool after the context exits.
    """
    allocator = torch.cuda.memory.CUDAPluggableAllocator(
        _so_path, MALLOC_FN, FREE_FN
    )
    pool = torch.cuda.memory.MemPool(allocator._allocator)
    _keep_alive.append((allocator, pool))
    if _counters_enabled:
        _log_counters("enter")
    try:
        with torch.cuda.memory.use_mem_pool(pool):
            yield
    finally:
        if _counters_enabled:
            _log_counters("exit")


def _release_keep_alive() -> None:
    """atexit hook: drop pool wrappers in the safe two-phase order."""
    if not _keep_alive:
        return
    pools = [pool for _, pool in _keep_alive]
    allocators = [allocator for allocator, _ in _keep_alive]
    _keep_alive.clear()
    pools.clear()
    gc.collect()
    allocators.clear()


def _worker_class():
    import vllm.v1.worker.gpu_worker as gpu_worker

    cls = getattr(gpu_worker, "Worker", None) or getattr(
        gpu_worker, "GPUWorker", None
    )
    if cls is None:
        raise RuntimeError("vllm.v1.worker.gpu_worker exposes no Worker class")
    return cls


def install() -> None:
    """``vllm.general_plugins`` entry point. Idempotent, fail-safe."""
    global _installed, _so_path, _counters_enabled
    if _installed:
        return
    _installed = True

    if not _env_enabled(ENABLE_ENV):
        logger.debug("uvm_kv_plugin: disabled (set %s to enable)", ENABLE_ENV)
        return

    so = os.environ.get(SO_ENV, "").strip()
    if not os.path.isabs(so):
        logger.error(
            "uvm_kv_plugin: %s must be an absolute path, got %r; staying off",
            SO_ENV,
            so,
        )
        return
    if not os.path.isfile(so):
        logger.error(
            "uvm_kv_plugin: %s does not exist: %s; staying off", SO_ENV, so
        )
        return
    if not torch.cuda.is_available():
        logger.warning("uvm_kv_plugin: CUDA not available; staying off")
        return

    _so_path = so
    _counters_enabled = _env_enabled(COUNTERS_ENV)

    try:
        cls = _worker_class()
        original = cls._maybe_get_memory_pool_context
    except Exception:
        logger.exception("uvm_kv_plugin: cannot wrap GPU worker; staying off")
        return
    if getattr(original, "_uvm_kv_plugin_wrapped", False):
        return

    @functools.wraps(original)
    def wrapped(self, tag: str):
        if tag == KV_TAG:
            return _uvm_kv_pool()
        return original(self, tag)

    wrapped._uvm_kv_plugin_wrapped = True
    cls._maybe_get_memory_pool_context = wrapped
    atexit.register(_release_keep_alive)
    logger.info(
        "uvm_kv_plugin: wrapping %s._maybe_get_memory_pool_context; "
        "'%s' tag uses UVM pool from %s",
        cls.__qualname__,
        KV_TAG,
        so,
    )


__all__ = ["install"]
