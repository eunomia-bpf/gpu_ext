"""Opt-in async GDS prefetch for LMCache 0.5.4 ``GdsBackend``.

Fills the two framework async seams used by
``StorageManager.async_lookup_and_prefetch`` (``batched_async_contains`` and
``batched_get_non_blocking``) so disk-populated KV is read asynchronously
through real GDS/cuFile methods while the existing gpubpf storage decision ABI
chooses submit or defer.  LMCache stays the owner of CuFile objects, file
descriptors, GPU addresses, streams and reference counts: reads run through
``GdsBackend.allocate``/``_load_bytes_from_disk_with_memory`` and completed
staging ``MemoryObj`` are consumed and released by the normal LMCache consumer
(``_async_process_tokens_internal`` -> ``batched_to_gpu`` ->
``ref_count_down``).

Reference ownership stays single-release: the bootstrap wraps
``LMCacheEngine.retrieve`` and ``LMCacheEngine._async_process_tokens_internal``
with a per-call consumption flag.  When token processing actually consumes a
lookup's completed ``EventManager`` LOADING event and the outer ``retrieve``
then returns without exception, the event is popped (without touching
refcounts), transferring it out of LMCache's abort-only
``cleanup_memory_objs``.  Early returns and failures leave the event
registered for that cleanup release path; the handoff adds no extra
reference and unconsumed lookups keep the original cleanup ownership.

Staging accounting uses the live allocator, not a second lifecycle framework:
``GPUMemoryAllocator.allocator.total_allocated_size`` counts every
allocated-but-unreleased buffer (in-flight reads plus completed-but-unconsumed
objects held by LMCache).  Reads queued before allocation do not yet show up
there, so admission and allocation of one lookup's prefix happen atomically
under the same bounded check; deferred reads keep banked bytes in
``pending_bytes`` until they allocate.  No path allocates outside the finite
pool.

Bootstrap, including from ``sitecustomize.py``::

    from lmcache_gds_async_prefetch_adapter import bootstrap_from_env
    bootstrap_from_env()

Set ``LMCACHE_GDS_ASYNC_PREFETCH=1`` to install; ``LMCACHE_GDS_POLICY_MODE``
selects the decider (``fifo`` = eager prefetch, ``native``/``bpf`` decide JIT
deferral).  Optional ``key.request_configs["lmcache.prefetch_deadline_ns"]``
is an explicitly application-provided HOST monotonic deadline; only an
existing deadline sets ``HINT_PREFETCH_JIT`` and yields a deferrable slack,
otherwise prefetch is eager.  Estimates come strictly from completed reads
(zero until one exists).
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import os
import threading
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from types import MethodType
from typing import Any, Callable, Mapping, Optional, Sequence

import lmcache_gds_policy_adapter as _policy
from lmcache_gds_policy_adapter import (
    ACTION_DEFER,
    ACTION_RECOMPUTE,
    ACTION_SUBMIT_NOW,
    FLAG_DEMAND,
    FLAG_SAFE_TO_DEFER,
    FLAG_SPECULATIVE,
    HINT_PREFETCH_JIT,
    Decider,
    PolicyRequest,
    build_decider,
)

__all__ = [
    "ASYNC_PREFETCH_ENV",
    "STAGING_BUDGET_ENV",
    "EXPECTED_LMCACHE_VERSION",
    "DEADLINE_CONFIG_KEY",
    "Estimates",
    "GdsAsyncPrefetchAdapter",
    "install_backend",
    "install_storage_manager",
    "bootstrap_from_env",
]

ASYNC_PREFETCH_ENV = "LMCACHE_GDS_ASYNC_PREFETCH"
STAGING_BUDGET_ENV = "LMCACHE_GDS_ASYNC_STAGING_BUDGET_BYTES"
EXPECTED_LMCACHE_VERSION = "0.5.4"
_ASYNC_ADAPTER_ATTR = "_gds_async_prefetch_adapter"
_ASYNC_CLASS_HOOK_ATTR = "_gds_async_prefetch_class_hook"
DEADLINE_CONFIG_KEY = "lmcache.prefetch_deadline_ns"
_EMBEDDED_HINT_KEY = "lmcache.embedded_hint"
_NO_DEBUG_KEY = object()
_ENGINE_EVENT_TRANSFER_ATTR = "_gds_async_prefetch_engine_event_transfer"
_CONSUMED_EVENT_ATTR = "_gds_async_prefetch_event_consumed"

logger = logging.getLogger(__name__)


def _env_bool(environ: Mapping[str, str], name: str) -> bool:
    raw = environ.get(name)
    if raw is None:
        return False
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} is not a boolean: {raw!r}")


class Estimates:
    """Completion-derived read-cost estimate (bytes/second EWMA).

    Starts at zero; only completed reads update it, so requests made before
    the first completion carry ``estimated_transfer_ns=0`` and native/BPF
    JIT policies treat them as not-slack-checkable (eager submit).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bps: float = 0.0
        self.completed_reads = 0
        self.completed_bytes = 0

    def observe(self, nbytes: int, elapsed_ns: int) -> None:
        if nbytes <= 0 or elapsed_ns <= 0:
            return
        sample = nbytes / (elapsed_ns / 1e9)
        with self._lock:
            self._bps = (
                sample
                if self._bps == 0.0
                else 0.5 * self._bps + 0.5 * sample
            )
            self.completed_reads += 1
            self.completed_bytes += nbytes

    def transfer_ns(self, nbytes: int) -> int:
        with self._lock:
            bps = self._bps
        if bps <= 0:
            return 0
        # nbytes/bps is seconds; the policy ABI carries nanoseconds.
        return max(1, int(nbytes / bps * 1e9))


@dataclass
class _PrefetchEntry:
    key: Any
    nbytes: int
    deadline_ns: Optional[int]
    future: "asyncio.Future[Any]"
    task: Optional["asyncio.Task[Any]"] = None
    demand: bool = False
    banked: bool = False  # deferred: bytes counted in the pending ledger
    state: str = "banked"  # banked -> reading -> done
    obj: Any = None  # completed staging MemoryObj (ref=1 until first claim)
    claimed: bool = False  # first consumer took the allocation reference
    wake: Optional["asyncio.Event"] = None

    def notify(self) -> None:
        if self.wake is not None:
            self.wake.set()


class GdsAsyncPrefetchAdapter:
    """Install the LMCache async lookup seams on one GDS backend."""

    def __init__(
        self,
        backend: Any,
        *,
        decider: Optional[Decider] = None,
        estimates: Optional[Estimates] = None,
        staging_budget_bytes: Optional[int] = None,
        decision_lock: Optional[threading.Lock] = None,
        clock: Optional[Any] = None,
    ) -> None:
        self.backend = backend
        mode = os.environ.get(
            _policy.MODE_ENV if hasattr(_policy, "MODE_ENV") else "LMCACHE_GDS_POLICY_MODE",
            "native",
        )
        self.mode = mode.strip().lower() or "native"
        self.decider = decider or build_decider(
            self.mode,
            uvm_device=os.environ.get("LMCACHE_GDS_POLICY_UVM_DEVICE", "/dev/nvidia-uvm"),
        )
        self.estimates = estimates or Estimates()
        capacity = int(getattr(backend, "config").gds_buffer_size) * 1024**2
        budget = os.environ.get(STAGING_BUDGET_ENV)
        if budget is not None:
            budget_bytes = int(budget, 0)
            if budget_bytes <= 0:
                raise ValueError(f"{STAGING_BUDGET_ENV} must be positive")
            capacity = min(capacity, budget_bytes)
        self.staging_capacity_bytes = capacity
        self._admission_lock = decision_lock or threading.Lock()
        self._clock = clock or time.monotonic_ns
        self._loop = backend.loop
        self._registry: dict[Any, _PrefetchEntry] = {}
        self._pending_bytes = 0
        self._request_id = 0
        self._installed = False
        self.stats = {
            "lookups": 0,
            "admitted_keys": 0,
            "prefetch_decisions": 0,
            "jit_submit": 0,
            "jit_defer": 0,
            "demand_adoptions": 0,
            "demand_decisions": 0,
            "alloc_retries": 0,
        }

    # Layout and live accounting helpers -----------------------------------

    def _inner_allocator(self) -> Any:
        allocator = self.backend.memory_allocator
        return getattr(allocator, "allocator", allocator)

    def _used_bytes(self) -> int:
        inner = self._inner_allocator()
        try:
            return int(inner.total_allocated_size)
        except AttributeError:
            return 0

    def _free_bytes(self) -> int:
        with self._admission_lock:
            return self.staging_capacity_bytes - self._used_bytes() - self._pending_bytes

    def _hot_entries(self, keys: Sequence[Any]) -> list[Optional[Any]]:
        """Resolve keys to DiskCacheMetadata (path/size/shape/dtype/fmt).

        Fast path reads hot_cache under one lock hold; cold misses fall back
        to ``_try_to_read_metadata`` (it takes the lock itself), so cold
        metadata discovery works without the placeholder crash.
        """
        with self.backend.hot_lock:
            cache = self.backend.hot_cache
            entries: list[Optional[Any]] = [cache.get(key) for key in keys]
        for index, entry in enumerate(entries):
            if entry is None:
                entries[index] = self.backend._try_to_read_metadata(keys[index])
        return entries

    def _install_methods(self) -> "GdsAsyncPrefetchAdapter":
        if self._installed:
            return self
        existing = getattr(self.backend, _ASYNC_ADAPTER_ATTR, None)
        if existing is not None and existing is not self:
            raise RuntimeError("GdsBackend already has an async-prefetch adapter")
        self.backend.batched_async_contains = MethodType(
            self._batched_async_contains, self.backend
        )
        self.backend.batched_get_non_blocking = MethodType(
            self._batched_get_non_blocking, self.backend
        )
        setattr(self.backend, _ASYNC_ADAPTER_ATTR, self)
        self._installed = True
        return self

    def install(self) -> "GdsAsyncPrefetchAdapter":
        return self._install_methods()

    def close(self) -> None:
        if not self._installed:
            self.decider.close()
            return
        self.backend.batched_async_contains = self.backend.__class__.batched_async_contains
        self.backend.batched_get_non_blocking = (
            self.backend.__class__.batched_get_non_blocking
        )
        try:
            delattr(self.backend, _ASYNC_ADAPTER_ATTR)
        except AttributeError:
            pass
        self._installed = False
        self.decider.close()

    # Decision construction ------------------------------------------------

    def _deadline_ns(self, key: Any) -> Optional[int]:
        request_configs = getattr(key, "request_configs", None)
        if not request_configs:
            return None
        raw = None
        if isinstance(request_configs, Mapping):
            raw = request_configs.get(DEADLINE_CONFIG_KEY)
        else:
            raw = getattr(request_configs, DEADLINE_CONFIG_KEY, None)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def _estimated_transfer_ns(self, nbytes: int) -> int:
        return self.estimates.transfer_ns(nbytes)

    def _build_request(
        self,
        *,
        entry_size: int,
        demand: bool,
        deadline_ns: Optional[int],
        in_flight: int,
    ) -> PolicyRequest:
        now_ns = self._clock()
        slack_ns = max(0, deadline_ns - now_ns) if deadline_ns is not None else 0
        if demand:
            flags = FLAG_DEMAND
            hint = 0
        else:
            flags = FLAG_SPECULATIVE | FLAG_SAFE_TO_DEFER
            hint = HINT_PREFETCH_JIT if deadline_ns is not None else 0
        try:
            pressure = (
                self._used_bytes() * 1000 // max(1, self.staging_capacity_bytes)
            )
        except Exception:
            pressure = 0
        return PolicyRequest(
            op=_policy.OP_READ,
            flags=flags,
            nbytes=entry_size,
            caller_hint=hint,
            slack_ns=slack_ns,
            estimated_transfer_ns=self._estimated_transfer_ns(entry_size),
            queue_depth=in_flight,
            hbm_pressure_permille=pressure,
        )

    def _decide(self, request: PolicyRequest, request_id: int) -> Any:
        self.stats["prefetch_decisions"] += 1
        return self.decider.decide(request, request_id)

    # Prefix admission -----------------------------------------------------

    async def _batched_async_contains(
        self,
        bound_backend: Any,
        lookup_id: str,
        keys: Sequence[Any],
        pin: bool = False,
    ) -> int:
        """Ordered contiguous-prefix lookup with bounded staging admission.

        Returns the number of leading keys present on disk whose combined
        physical bytes fit in the remaining staging budget.  Later keys are
        neither reported nor retrieved, so every admitted chunk stays a real
        contiguous prefix.
        """
        del bound_backend
        del pin  # GDS has no eviction; pin/unpin are no-ops upstream
        self.stats["lookups"] += 1
        self._purge_registry()
        entries = self._hot_entries(keys)
        admitted = 0
        for index, (key, entry) in enumerate(zip(keys, entries, strict=True)):
            if entry is None:
                break
            with self._admission_lock:
                free = (
                    self.staging_capacity_bytes
                    - self._used_bytes()
                    - self._pending_bytes
                )
                if entry.size > free:
                    break
                request = self._build_request(
                    entry_size=entry.size,
                    demand=False,
                    deadline_ns=self._deadline_ns(key),
                    in_flight=len(self._registry),
                )
                decision = self._decide(request, self._next_request_id())
                if decision.action == ACTION_DEFER:
                    self._bank_deferred_locked(key, entry, decision.defer_ns)
                    self.stats["jit_defer"] += 1
                elif decision.action == ACTION_SUBMIT_NOW:
                    if not self._start_read_locked(key, entry, None):
                        break
                    self.stats["jit_submit"] += 1
                else:
                    raise RuntimeError(
                        f"RECOMPUTE is invalid for a disk prefetch ({key})"
                    )
            admitted = index + 1
            self.stats["admitted_keys"] += 1
        return admitted

    # Deferred / reading execution ----------------------------------------

    def _bank_deferred_locked(
        self, key: Any, entry: Any, defer_ns: int
    ) -> bool:
        """Bank this prefix key for later (admission lock held).

        Banked bytes stay in ``_pending_bytes`` so concurrent admissions
        cannot assume the same free bytes before the deferred read allocates.
        """
        existing = self._registry.get(key)
        if existing is not None:
            return True
        wake = asyncio.Event()
        handle = _PrefetchEntry(
            key=key,
            nbytes=entry.size,
            deadline_ns=self._deadline_ns(key),
            future=self._loop.create_future(),
            state="banked",
            banked=True,
            wake=wake,
        )
        self._registry[key] = handle
        self._pending_bytes += handle.nbytes
        handle.task = self._loop.create_task(
            self._deferred_read(handle, entry, defer_ns)
        )
        return True

    async def _sleep_defer(self, handle: _PrefetchEntry, defer_ns: int) -> None:
        """Wait for the deferred slot or an early wake (continuation/demand).

        The wake is consumed after observation so a stale signal cannot turn
        the next defer cycle into a hot re-decision spin.
        """
        wake = handle.wake
        if wake is None:
            await asyncio.sleep(max(0.0, defer_ns / 1e9))
            return
        try:
            await asyncio.wait_for(wake.wait(), timeout=max(0.0, defer_ns / 1e9))
        except asyncio.TimeoutError:
            pass
        wake.clear()

    async def _deferred_read(
        self, handle: _PrefetchEntry, entry: Any, defer_ns: int
    ) -> None:
        try:
            while True:
                await self._sleep_defer(handle, defer_ns)
                fresh_entry = self._hot_entry(handle.key)
                if fresh_entry is None:
                    self._finish_failed(handle)
                    return
                entry = fresh_entry
                with self._admission_lock:
                    if handle.state != "banked":
                        return
                    request = self._build_request(
                        entry_size=entry.size,
                        demand=handle.demand,
                        deadline_ns=handle.deadline_ns,
                        in_flight=len(self._registry),
                    )
                    decision = self._decide(
                        request, self._next_request_id()
                    )
                    if decision.action == ACTION_DEFER:
                        defer_ns = decision.defer_ns
                        continue
                    if decision.action == ACTION_RECOMPUTE:
                        raise RuntimeError(
                            "RECOMPUTE is invalid for a disk prefetch"
                        )
                    # SUBMIT_NOW: try to allocate inline under the same lock.
                    if self._start_read_locked(handle.key, entry, handle):
                        return
                    no_space = True
                if no_space:
                    # Could not stage under the finite pool right now; retry
                    # on the next short defer slot instead of spinning hot,
                    # and stop when the request's lead time is exhausted.
                    if (
                        not handle.demand
                        and handle.deadline_ns is not None
                        and self._clock() >= handle.deadline_ns
                    ):
                        self._finish_failed(handle)
                        return
                    await self._sleep_defer(handle, 1_000_000)
        except asyncio.CancelledError:
            self._release_banked(handle)
            raise
        except BaseException as error:
            self._fail_future(handle, error)

    def _release_banked(self, handle: _PrefetchEntry) -> None:
        if handle.banked:
            self._pending_bytes -= handle.nbytes
            handle.banked = False
        self._registry.pop(handle.key, None)

    def _fail_future(self, handle: _PrefetchEntry, error: BaseException) -> None:
        self._release_banked(handle)
        if not handle.future.done():
            handle.future.set_exception(error)

    def _finish_failed(self, handle: _PrefetchEntry) -> None:
        self._release_banked(handle)
        if not handle.future.done():
            handle.future.set_result(None)

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    # Reading execution ----------------------------------------------------

    def _start_read_locked(
        self, key: Any, entry: Any, handle: Optional[_PrefetchEntry]
    ) -> Optional[_PrefetchEntry]:
        """Allocate staging storage and schedule the cuFile read.

        Admission lock held by the caller: the budget check and this
        allocation are atomic, so concurrent admissions cannot both assume
        the same free bytes.  Returns the handle on success, None when no
        space fits.
        """
        existing = self._registry.get(key)
        if (
            existing is not None
            and existing is not handle
            and (existing.state != "done" or existing.obj is not None)
        ):
            # Another active in-flight/deferred read, or still-staged data:
            # adopt it instead of duplicating I/O.
            return existing
        memory_obj = self.backend.allocate(
            entry.shape, entry.dtype, fmt=entry.fmt, busy_loop=False
        )
        if memory_obj is None:
            return None
        if handle is None:
            handle = _PrefetchEntry(
                key=key,
                nbytes=entry.size,
                deadline_ns=self._deadline_ns(key),
                future=self._loop.create_future(),
                state="reading",
                wake=asyncio.Event(),
            )
            self._registry[key] = handle
        else:
            handle.state = "reading"
            if handle.banked:
                self._pending_bytes -= handle.nbytes
                handle.banked = False
        handle.task = self._loop.create_task(
            self._execute_read(handle, entry, memory_obj)
        )
        return handle

    async def _execute_read(
        self, handle: _PrefetchEntry, entry: Any, memory_obj: Any
    ) -> None:
        started_ns = self._clock()
        inner = asyncio.ensure_future(
            self._run_io(
                lambda: self.backend._load_bytes_from_disk_with_memory(
                    handle.key, entry.path, memory_obj
                )
            )
        )
        try:
            loaded = await asyncio.shield(inner)
        except asyncio.CancelledError:
            # Never release staging while the executor thread may still be
            # writing into it: reap after the in-flight read lands, then
            # release our allocation reference.
            async def _reap() -> None:
                try:
                    await inner
                except BaseException:
                    pass
                try:
                    memory_obj.ref_count_down()
                except Exception:
                    pass

            asyncio.ensure_future(_reap())
            raise
        except BaseException as error:
            if not handle.future.done():
                handle.future.set_exception(error)
            raise
        elapsed_ns = self._clock() - started_ns
        if loaded is None:
            # backend released the buffer internally on read failure.
            handle.state = "done"
            if not handle.future.done():
                handle.future.set_result(None)
            return
        self.estimates.observe(memory_obj.get_size(), elapsed_ns)
        handle.obj = loaded
        handle.state = "done"
        if not handle.future.done():
            handle.future.set_result(loaded)

    async def _run_io(self, fn: Any) -> Any:
        pool = getattr(self.backend, "_thread_pool", None)
        if pool is not None:
            return await asyncio.get_running_loop().run_in_executor(pool, fn)
        return await asyncio.to_thread(fn)

    def _purge_registry(self) -> None:
        """Retire finished entries whose staging was freed by the consumer."""
        dead: list[Any] = []
        for key, handle in self._registry.items():
            if handle.state != "done":
                continue
            obj = handle.obj
            if obj is None:
                dead.append(key)
                continue
            try:
                if not obj.is_valid() or obj.get_ref_count() <= 0:
                    dead.append(key)
            except Exception:
                dead.append(key)
        for key in dead:
            self._registry.pop(key, None)

    def _hot_entry(self, key: Any) -> Optional[Any]:
        with self.backend.hot_lock:
            entry = self.backend.hot_cache.get(key)
        if entry is None:
            entry = self.backend._try_to_read_metadata(key)
        return entry

    # Continuation seam ----------------------------------------------------

    async def _batched_get_non_blocking(
        self,
        bound_backend: Any,
        lookup_id: str,
        keys: list[Any],
        transfer_spec: Any = None,
    ) -> list[Any]:
        """Prefetch continuation, not a demand signal.

        StorageManager.async_lookup_and_prefetch calls this right after
        batched_async_contains while the GPU consumer is still away, so each
        key here adopts its prefetch pipeline (deferred entries get an early
        wake and a fresh, still policy-governed re-decision).  True demand
        handling exists only through :meth:`mark_demand`.
        """
        del bound_backend, lookup_id, transfer_spec
        self._purge_registry()
        results: list[Any] = []
        for key in keys:
            results.append(await self._get_one(key))
        return results

    def mark_demand(self, key: Any) -> None:
        """Opt-in live demand feed for one key (unused by the framework).

        The framework continuation is not a demand signal; an external
        scheduler integration may call this to force the policy's demand
        branch on the next re-decision wake.  This adapter records the
        missing scheduler demand signal explicitly rather than simulating
        one.
        """
        handle = self._registry.get(key)
        if handle is not None:
            handle.demand = True
            handle.notify()

    async def _get_one(self, key: Any) -> Any:
        handle = self._registry.get(key)
        if handle is None:
            # No prefetch entry: lookup-miss race or a retired past
            # prefetch.  Execute a bounded demand-style read to satisfy
            # the continuation.
            return await self._demand_read(key)
        if handle.state == "banked":
            handle.notify()  # early wake; the policy still decides
        result = await handle.future
        if result is not None:
            if handle.claimed:
                result.ref_count_up()
            else:
                handle.claimed = True
        return result

    async def _demand_read(self, key: Any) -> Any:
        """Bounded demand-style read for an unstaged key.

        Same finite staging pool; overflow falls back once to the
        upstream-equivalent bounded busy-loop allocation run entirely on
        the executor thread.  Never allocates outside the pool.
        """
        entry = self._hot_entry(key)
        if entry is None:
            return None
        with self._admission_lock:
            request = self._build_request(
                entry_size=entry.size,
                demand=True,
                deadline_ns=self._deadline_ns(key),
                in_flight=len(self._registry),
            )
            self.stats["demand_decisions"] += 1
            decision = self.decider.decide(request, self._next_request_id())
            if decision.action != ACTION_SUBMIT_NOW:
                raise RuntimeError(
                    "demand read got a non-SUBMIT_NOW decision; refusing"
                )
            handle = self._start_read_locked(key, entry, None)
            if handle is None:
                handle = self._demand_read_slow(key, entry)
        if handle is None:
            return None
        return await handle.future

    def _demand_read_slow(
        self, key: Any, entry: Any
    ) -> _PrefetchEntry:
        handle = _PrefetchEntry(
            key=key,
            nbytes=entry.size,
            deadline_ns=self._deadline_ns(key),
            future=self._loop.create_future(),
            state="reading",
            demand=True,
            wake=asyncio.Event(),
        )
        self._registry[key] = handle
        handle.task = self._loop.create_task(
            self._execute_demand_read(handle, entry)
        )
        return handle

    async def _execute_demand_read(
        self, handle: _PrefetchEntry, entry: Any
    ) -> None:
        try:
            memory_obj = await self._run_io(
                lambda: self.backend.allocate(
                    entry.shape, entry.dtype, fmt=entry.fmt, busy_loop=True
                )
            )
            if memory_obj is None:
                handle.state = "done"
                if not handle.future.done():
                    handle.future.set_result(None)
                return
            loaded = await self._run_io(
                lambda: self.backend._load_bytes_from_disk_with_memory(
                    handle.key, entry.path, memory_obj
                )
            )
            handle.state = "done"
            if loaded is None:
                if not handle.future.done():
                    handle.future.set_result(None)
                return
            handle.obj = loaded
            if not handle.future.done():
                handle.future.set_result(loaded)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            if not handle.future.done():
                handle.future.set_exception(error)


def install_backend(
    backend: Any,
    *,
    decider: Optional[Decider] = None,
    estimates: Optional[Estimates] = None,
    staging_budget_bytes: Optional[int] = None,
    decision_lock: Optional[threading.Lock] = None,
    clock: Optional[Any] = None,
) -> "GdsAsyncPrefetchAdapter":
    """Install on an existing backend, returning the idempotent adapter."""
    existing = getattr(backend, _ASYNC_ADAPTER_ATTR, None)
    if existing is not None:
        return existing
    return GdsAsyncPrefetchAdapter(
        backend,
        decider=decider,
        estimates=estimates,
        staging_budget_bytes=staging_budget_bytes,
        decision_lock=decision_lock,
        clock=clock,
    ).install()


def install_storage_manager(
    manager: Any, **kwargs: Any
) -> list["GdsAsyncPrefetchAdapter"]:
    """Install async-prefetch seams on current GDS backends."""
    installed: list["GdsAsyncPrefetchAdapter"] = []
    for backend in manager.storage_backends.values():
        if backend.__class__.__name__ == "GdsBackend":
            installed.append(install_backend(backend, **kwargs))
    return installed


def _assert_signature(cls: Any, name: str, parameters: tuple[str, ...]) -> None:
    actual = tuple(inspect.signature(getattr(cls, name)).parameters)
    if actual != parameters:
        raise RuntimeError(
            f"LMCache {EXPECTED_LMCACHE_VERSION} {cls.__name__}.{name} "
            f"signature drift: {actual!r} != {parameters!r}"
        )


def _install_engine_event_transfer() -> None:
    """Transfer the completed LOADING event out of abort cleanup on success.

    LMCache 0.5.4 leaves a lookup's completed prefetch future registered in
    the ``EventManager`` after the engine consumed it: ``retrieve`` releases
    every staging ``MemoryObj`` exactly once (used objects after
    ``batched_to_gpu``, others through the "free the memory objects that are
    not hit" loop), while ``_async_process_tokens_internal`` only peeks the
    event via ``EventManager.get_event_future``.  The later ``lookup_unpin``
    (vLLM ``wait_for_save``) still finds status DONE, treats the lookup as
    aborted, and ``cleanup_memory_objs`` pops that future and releases the
    same objects a second time, driving ``ref_count`` negative once per
    staged object ("Double free occurred somewhere").

    The seam is minimal and consumption-gated: a wrapper on
    ``_async_process_tokens_internal`` records a per-call flag when the
    engine actually consumed the lookup's completed future (the wrapper
    re-peeks the same future and its result exactly as the engine does),
    and a wrapper on ``retrieve`` pops the DONE event -- without touching
    refcounts -- only when that flag is set AND the outer ``retrieve``
    returned without exception.  Early returns (e.g. the ``is_healthy``
    guard) do not set the flag. If the outer call raises, no pop occurs,
    even when inner processing already set the flag. This preserves the
    existing cleanup path; partial-failure ownership is not newly repaired.
    The handoff adds no extra reference to the events it transfers.
    """
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache.v1.event_manager import EventStatus, EventType

    if getattr(LMCacheEngine, _ENGINE_EVENT_TRANSFER_ATTR, None) is not None:
        return
    _assert_signature(
        LMCacheEngine,
        "retrieve",
        ("self", "tokens", "mask", "kwargs"),
    )
    _assert_signature(
        LMCacheEngine,
        "_async_process_tokens_internal",
        ("self", "tokens", "mask", "ret_mask", "kwargs"),
    )
    original_retrieve = LMCacheEngine.retrieve
    original_process_tokens = LMCacheEngine._async_process_tokens_internal

    @functools.wraps(original_process_tokens)
    def hooked_process_tokens(
        instance: Any,
        tokens: Any,
        mask: Any = None,
        ret_mask: Any = None,
        **kwargs: Any,
    ) -> Any:
        chunks, tot_kv_size = original_process_tokens(
            instance, tokens, mask, ret_mask, **kwargs
        )
        try:
            req_id = kwargs.get("req_id")
            if req_id is not None:
                # Idempotent re-peek of the same future the engine checks:
                # it completes only when token processing consumed a
                # successfully finished prefetch result.
                instance.event_manager.get_event_future(
                    EventType.LOADING, req_id
                ).result()
                setattr(instance, _CONSUMED_EVENT_ATTR, True)
        except Exception as error:
            logger.debug(
                "async-prefetch consumption check failed: %s", error
            )
        return chunks, tot_kv_size

    @functools.wraps(original_retrieve)
    def hooked_retrieve(
        instance: Any,
        tokens: Any,
        mask: Any = None,
        **kwargs: Any,
    ) -> Any:
        setattr(instance, _CONSUMED_EVENT_ATTR, False)
        result = original_retrieve(instance, tokens, mask, **kwargs)
        if not getattr(instance, _CONSUMED_EVENT_ATTR, False):
            return result
        req_id = kwargs.get("req_id")
        try:
            if instance.event_manager.get_event_status(
                EventType.LOADING, req_id
            ) == EventStatus.DONE:
                instance.event_manager.pop_event(EventType.LOADING, req_id)
        except Exception as error:
            logger.debug(
                "async-prefetch event transfer after retrieve failed: %s", error
            )
        return result

    LMCacheEngine._async_process_tokens_internal = hooked_process_tokens
    LMCacheEngine.retrieve = hooked_retrieve
    setattr(LMCacheEngine, _ENGINE_EVENT_TRANSFER_ATTR, hooked_retrieve)


_bootstrap_lock = threading.Lock()


def bootstrap_from_env(
    environ: Mapping[str, str] = os.environ,
) -> Optional[Callable[..., None]]:
    """Install the async-prefetch class hook on future GdsBackend instances.

    Returns the installed hook, or None when the opt-in env is absent.  The
    hook chains with any prior GdsBackend.__init__ wrapper (e.g. the
    admission adapter's) and installs disjoint seams.
    """
    if not _env_bool(environ, ASYNC_PREFETCH_ENV):
        return None

    try:
        installed_version = version("lmcache")
    except PackageNotFoundError as error:
        raise RuntimeError("LMCache is not installed") from error
    if installed_version != EXPECTED_LMCACHE_VERSION:
        raise RuntimeError(
            f"adapter requires LMCache {EXPECTED_LMCACHE_VERSION}, "
            f"found {installed_version}"
        )

    from lmcache.v1.storage_backend.gds_backend import GdsBackend

    with _bootstrap_lock:
        prior = getattr(GdsBackend, _ASYNC_CLASS_HOOK_ATTR, None)
        if prior is not None:
            return prior

        _assert_signature(
            GdsBackend,
            "__init__",
            ("self", "config", "metadata", "loop", "dst_device"),
        )
        _assert_signature(
            GdsBackend,
            "batched_async_contains",
            ("self", "lookup_id", "keys", "pin"),
        )
        _assert_signature(
            GdsBackend,
            "batched_get_non_blocking",
            ("self", "lookup_id", "keys", "transfer_spec"),
        )

        original_init = GdsBackend.__init__

        @functools.wraps(original_init)
        def hooked_init(instance: Any, *args: Any, **kwargs: Any) -> None:
            original_init(instance, *args, **kwargs)
            install_backend(instance)

        GdsBackend.__init__ = hooked_init
        setattr(GdsBackend, _ASYNC_CLASS_HOOK_ATTR, hooked_init)
        _install_engine_event_transfer()
        return hooked_init


# With gds-control on PYTHONPATH, importing this module early (sitecustomize)
# installs the hook when the opt-in env is present.  Absence keeps ordinary
# LMCache processes unchanged.
if _env_bool(os.environ, ASYNC_PREFETCH_ENV):
    bootstrap_from_env()
