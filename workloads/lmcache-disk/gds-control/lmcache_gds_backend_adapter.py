"""Upper-level gpu-storage admission for LMCache 0.5.4 ``GdsBackend``.

This integration deliberately wraps the storage backend, not
``GDSContext``.  A decision is made for each logical cache key before
``GdsBackend`` reaches ``_save_gds`` or ``_load_gds``.  LMCache therefore
continues to own CuFile objects, file descriptors, GPU addresses, CUDA streams,
and the lifetime of an I/O once it is submitted.

The cmd82 ABI and fifo/native/BPF deciders are imported from
``lmcache_gds_policy_adapter`` so this integration and the standalone executor
use the same 136-byte request and the same native policy implementation.

``LiveDemandRequestProvider`` extends the fixed-telemetry provider with live
demand feedback: a flagged write carries the live pending demand-read count
as ``queue_depth``, ``HINT_LIVE_DEMAND`` in ``caller_hint``, and the
remaining per-write delay budget as ``slack_ns``.  Those fields are demand
feedback, not measured HBM pressure.  After every wait, a deferred flagged
write is re-decided from the live state until the policy says SUBMIT.

Bootstrap, including from ``sitecustomize.py``::

    from lmcache_gds_backend_adapter import bootstrap_from_env
    bootstrap_from_env()

Set ``LMCACHE_GDS_POLICY_MODE`` to ``fifo``, ``native``, or ``bpf``.  Merely
importing this module also bootstraps when that variable is present.  The class
hook is intentionally version/signature checked and fails closed on LMCache
drift.  ``install_backend`` remains available for an already-created backend
or for dependency-free tests.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import os
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from types import MethodType
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence

from lmcache_gds_policy_adapter import (
    ACTION_DEFER,
    ACTION_RECOMPUTE,
    ACTION_SUBMIT_NOW,
    FLAG_DEMAND,
    FLAG_RECOMPUTABLE,
    FLAG_SAFE_TO_DEFER,
    FLAG_SPECULATIVE,
    HINT_LIVE_DEMAND,
    OP_READ,
    OP_WRITE,
    Decider,
    Decision,
    PolicyRequest,
    build_decider,
)

__all__ = [
    "MODE_ENV",
    "EXPECTED_LMCACHE_VERSION",
    "AdmissionError",
    "Telemetry",
    "EnvironmentRequestProvider",
    "LiveDemandRequestProvider",
    "WriteFeedback",
    "GdsBackendAdmissionAdapter",
    "install_backend",
    "install_storage_manager",
    "bootstrap_from_env",
]

MODE_ENV = "LMCACHE_GDS_POLICY_MODE"
EXPECTED_LMCACHE_VERSION = "0.5.4"
_ADAPTER_ATTR = "_gds_admission_adapter"
_CLASS_HOOK_ATTR = "_gds_admission_class_hook"

READ_DEMAND = "read_demand"
READ_SPECULATIVE = "read_speculative"
WRITE = "write"


class AdmissionError(RuntimeError):
    """The policy result cannot be represented safely by the LMCache API."""


@dataclass(frozen=True)
class Telemetry:
    """Policy inputs shared by the default per-object request provider."""

    priority: int = 0
    tenant_id: int = 0
    caller_hint: int = 0
    deadline_ns: int = 0
    slack_ns: int = 0
    estimated_transfer_ns: int = 0
    recompute_ns: int = 0
    queue_depth: int = 0
    hbm_pressure_permille: int = 0
    speculative_recomputable: bool = False


def _env_int(environ: Mapping[str, str], suffix: str, default: int = 0) -> int:
    raw = environ.get("LMCACHE_GDS_POLICY_" + suffix)
    if raw is None:
        return default
    value = int(raw, 0)
    if value < 0:
        raise ValueError(f"LMCACHE_GDS_POLICY_{suffix} must be non-negative")
    return value


def _env_bool(environ: Mapping[str, str], suffix: str, default: bool = False) -> bool:
    raw = environ.get("LMCACHE_GDS_POLICY_" + suffix)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"LMCACHE_GDS_POLICY_{suffix} is not a boolean: {raw!r}")


class EnvironmentRequestProvider:
    """Build conservative requests from fixed process-level telemetry.

    Blocking reads are demand requests and can never be deferred/recomputed by
    the policy.  The existing ``StorageManager.get_non_blocking`` path is the
    explicit speculative-read seam.  Writes are safe to defer because
    ``submit_put_task`` already retains their ``MemoryObj`` asynchronously.
    """

    def __init__(self, telemetry: Telemetry = Telemetry()) -> None:
        self.telemetry = telemetry

    @classmethod
    def from_environ(
        cls, environ: Mapping[str, str] = os.environ
    ) -> "EnvironmentRequestProvider":
        return cls(
            Telemetry(
                priority=_env_int(environ, "PRIORITY"),
                tenant_id=_env_int(environ, "TENANT_ID"),
                caller_hint=_env_int(environ, "CALLER_HINT"),
                deadline_ns=_env_int(environ, "DEADLINE_NS"),
                slack_ns=_env_int(environ, "SLACK_NS"),
                estimated_transfer_ns=_env_int(environ, "ESTIMATED_TRANSFER_NS"),
                recompute_ns=_env_int(environ, "RECOMPUTE_NS"),
                queue_depth=_env_int(environ, "QUEUE_DEPTH"),
                hbm_pressure_permille=_env_int(
                    environ, "HBM_PRESSURE_PERMILLE"
                ),
                speculative_recomputable=_env_bool(
                    environ, "SPECULATIVE_RECOMPUTABLE"
                ),
            )
        )

    def __call__(
        self, kind: str, key: Any, memory_obj: Any, backend: Any
    ) -> PolicyRequest:
        if kind == WRITE:
            op = OP_WRITE
            flags = FLAG_SAFE_TO_DEFER
        elif kind == READ_DEMAND:
            op = OP_READ
            flags = FLAG_DEMAND
        elif kind == READ_SPECULATIVE:
            op = OP_READ
            flags = FLAG_SPECULATIVE | FLAG_SAFE_TO_DEFER
            if self.telemetry.speculative_recomputable:
                flags |= FLAG_RECOMPUTABLE
        else:
            raise ValueError(f"unknown admission kind {kind!r}")

        nbytes = 0
        if memory_obj is not None:
            nbytes = int(memory_obj.get_size())
        elif kind.startswith("read"):
            # Metadata contains only size/path/type information.  No fd, GPU
            # pointer, tensor, or stream crosses the admission boundary.
            with backend.hot_lock:
                entry = backend.hot_cache.get(key) if key is not None else None
            if entry is not None:
                nbytes = int(entry.size)

        t = self.telemetry
        return PolicyRequest(
            op=op,
            flags=flags,
            priority=t.priority,
            nbytes=nbytes,
            tenant_id=t.tenant_id,
            caller_hint=t.caller_hint,
            deadline_ns=t.deadline_ns,
            slack_ns=t.slack_ns,
            estimated_transfer_ns=t.estimated_transfer_ns,
            recompute_ns=t.recompute_ns,
            queue_depth=t.queue_depth,
            hbm_pressure_permille=t.hbm_pressure_permille,
        )


@dataclass(frozen=True)
class WriteFeedback:
    """Per-write admission record for later runner integration.

    ``pending_demand_reads`` and ``remaining_budget_ns`` are demand feedback,
    not measured HBM pressure.  No key digest or raw key string is kept.
    """

    request_id: int
    pending_demand_reads: int
    remaining_budget_ns: int
    decision: int
    requested_wait_ns: int


class _WriteDelayBudget:
    """Monotonic delay budget for one deferred write, started at admission."""

    __slots__ = ("_start_ns", "_total_ns")

    def __init__(self, start_ns: int, total_ns: int) -> None:
        self._start_ns = start_ns
        self._total_ns = total_ns

    def remaining_ns(self, now_ns: int) -> int:
        return max(0, self._total_ns - (now_ns - self._start_ns))


class LiveDemandRequestProvider(EnvironmentRequestProvider):
    """EnvironmentRequestProvider with live demand feedback for writes.

    Writes carry the live pending demand-read count as ``queue_depth``,
    ``HINT_LIVE_DEMAND`` in ``caller_hint``, and the remaining per-write
    delay budget as ``slack_ns``.  Those fields are demand feedback, not
    measured HBM pressure.  Reads keep the parent's fixed-telemetry requests.
    """

    def __init__(
        self,
        telemetry: Telemetry = Telemetry(),
        total_write_delay_budget_ns: int = 10_000_000,
    ) -> None:
        super().__init__(telemetry)
        if total_write_delay_budget_ns < 0:
            raise ValueError("total_write_delay_budget_ns must be non-negative")
        self.total_write_delay_budget_ns = int(total_write_delay_budget_ns)
        self._pending_lock = threading.Lock()
        self._pending_demand_reads = 0

    @classmethod
    def from_environ(
        cls, environ: Mapping[str, str] = os.environ
    ) -> "LiveDemandRequestProvider":
        provider = super().from_environ(environ)
        provider.total_write_delay_budget_ns = _env_int(
            environ, "WRITE_DELAY_BUDGET_NS", 10_000_000
        )
        return provider

    def demand_read_started(self) -> None:
        with self._pending_lock:
            self._pending_demand_reads += 1

    def demand_read_finished(self) -> None:
        with self._pending_lock:
            self._pending_demand_reads = max(0, self._pending_demand_reads - 1)

    @property
    def pending_demand_reads(self) -> int:
        with self._pending_lock:
            return self._pending_demand_reads

    def begin_write_delay(self) -> _WriteDelayBudget:
        return _WriteDelayBudget(
            int(time.monotonic() * 1_000_000_000),
            self.total_write_delay_budget_ns,
        )

    def __call__(
        self,
        kind: str,
        key: Any,
        memory_obj: Any,
        backend: Any,
        write_budget: Optional[_WriteDelayBudget] = None,
    ) -> PolicyRequest:
        if kind != WRITE:
            return super().__call__(kind, key, memory_obj, backend)

        t = self.telemetry
        nbytes = 0
        if memory_obj is not None:
            nbytes = int(memory_obj.get_size())
        if write_budget is not None:
            slack_ns = write_budget.remaining_ns(
                int(time.monotonic() * 1_000_000_000)
            )
        else:
            slack_ns = self.total_write_delay_budget_ns
        return PolicyRequest(
            op=OP_WRITE,
            flags=FLAG_SAFE_TO_DEFER,
            priority=t.priority,
            nbytes=nbytes,
            tenant_id=t.tenant_id,
            caller_hint=HINT_LIVE_DEMAND | t.caller_hint,
            deadline_ns=t.deadline_ns,
            slack_ns=slack_ns,
            estimated_transfer_ns=t.estimated_transfer_ns,
            recompute_ns=t.recompute_ns,
            queue_depth=self.pending_demand_reads,
            hbm_pressure_permille=t.hbm_pressure_permille,
        )


RequestProvider = Callable[[str, Any, Any, Any], PolicyRequest]
AsyncWait = Callable[[float], Awaitable[Any]]


class GdsBackendAdmissionAdapter:
    """Install safe policy admission on one LMCache ``GdsBackend`` instance."""

    def __init__(
        self,
        backend: Any,
        *,
        mode: str = "native",
        decider: Optional[Decider] = None,
        request_provider: Optional[RequestProvider] = None,
        uvm_device: str = "/dev/nvidia-uvm",
        async_wait: AsyncWait = asyncio.sleep,
    ) -> None:
        self.backend = backend
        self.decider = decider or build_decider(mode, uvm_device=uvm_device)
        self.request_provider = request_provider or EnvironmentRequestProvider()
        self._async_wait = async_wait
        self._decision_lock = threading.Lock()
        self._request_id = 0
        self._installed = False
        self._original_submit = backend.submit_put_task
        self._original_get = backend.get_blocking
        self._original_get_non_blocking = backend.get_non_blocking
        self._original_batched_get = backend.batched_get_blocking
        self._original_async_save = backend._async_save_bytes_to_disk
        self.stats = {
            "decisions": 0,
            "submit_now": 0,
            "defer": 0,
            "recompute": 0,
            "blocking_defer_miss": 0,
        }

    def install(self) -> "GdsBackendAdmissionAdapter":
        if self._installed:
            return self
        existing = getattr(self.backend, _ADAPTER_ATTR, None)
        if existing is not None and existing is not self:
            raise RuntimeError("GdsBackend already has a policy adapter")
        self.backend.submit_put_task = MethodType(self._submit_put_task, self.backend)
        self.backend.get_blocking = MethodType(self._get_blocking, self.backend)
        self.backend.get_non_blocking = MethodType(
            self._get_non_blocking, self.backend
        )
        self.backend.batched_get_blocking = MethodType(
            self._batched_get_blocking, self.backend
        )
        setattr(self.backend, _ADAPTER_ATTR, self)
        self._installed = True
        return self

    def close(self) -> None:
        if self._installed:
            self.backend.submit_put_task = self._original_submit
            self.backend.get_blocking = self._original_get
            self.backend.get_non_blocking = self._original_get_non_blocking
            self.backend.batched_get_blocking = self._original_batched_get
            try:
                delattr(self.backend, _ADAPTER_ATTR)
            except AttributeError:
                pass
            self._installed = False
        self.decider.close()

    def _admit(self, kind: str, key: Any, memory_obj: Any = None) -> tuple[PolicyRequest, Decision]:
        request = self.request_provider(kind, key, memory_obj, self.backend)
        expected_op = OP_WRITE if kind == WRITE else OP_READ
        if request.op != expected_op:
            raise AdmissionError(
                f"request provider returned op={request.op} for {kind}"
            )
        if kind == READ_DEMAND and not request.flags & FLAG_DEMAND:
            raise AdmissionError("blocking reads must be marked demand")
        with self._decision_lock:
            request_id = self._request_id
            self._request_id += 1
            decision = self.decider.decide(request, request_id)
            self.stats["decisions"] += 1
            if decision.action == ACTION_SUBMIT_NOW:
                self.stats["submit_now"] += 1
            elif decision.action == ACTION_DEFER:
                self.stats["defer"] += 1
            elif decision.action == ACTION_RECOMPUTE:
                self.stats["recompute"] += 1
            else:
                raise AdmissionError(f"unknown policy action {decision.action}")
        if request.flags & FLAG_DEMAND and decision.action != ACTION_SUBMIT_NOW:
            raise AdmissionError("policy violated the demand-read SUBMIT_NOW contract")
        return request, decision

    # MethodType supplies ``bound_backend``; all operations use saved LMCache
    # methods so an admitted object cannot recursively receive a second decision.
    def _submit_put_task(
        self,
        bound_backend: Any,
        key: Any,
        memory_obj: Any,
        on_complete_callback: Optional[Callable[[Any], None]] = None,
    ) -> Future:
        del bound_backend
        _request, decision = self._admit(WRITE, key, memory_obj)
        if decision.action == ACTION_SUBMIT_NOW:
            return self._original_submit(key, memory_obj, on_complete_callback)
        if decision.action == ACTION_RECOMPUTE:
            raise AdmissionError("RECOMPUTE is invalid for a GDS write")

        # This is the exact ownership boundary used by LMCache 0.5.4's
        # submit_put_task.  StorageManager may now release its reference while
        # this one remains live through the existing _async_save finally block.
        if memory_obj.tensor is None:
            raise AdmissionError("GDS write requires a live tensor")
        memory_obj.ref_count_up()
        with self.backend.put_lock:
            self.backend.put_tasks.add(key)

        async def delayed_save() -> None:
            handed_to_lmcache = False
            try:
                if decision.defer_ns:
                    await self._async_wait(decision.defer_ns / 1_000_000_000)
                handed_to_lmcache = True
                await self._original_async_save(
                    key, memory_obj, on_complete_callback
                )
            finally:
                # If cancellation happens before LMCache's coroutine starts,
                # reproduce its cleanup.  Once handed off, its own finally owns it.
                if not handed_to_lmcache:
                    memory_obj.ref_count_down()
                    with self.backend.put_lock:
                        self.backend.put_tasks.discard(key)

        coroutine = delayed_save()
        try:
            return asyncio.run_coroutine_threadsafe(coroutine, self.backend.loop)
        except BaseException:
            coroutine.close()
            memory_obj.ref_count_down()
            with self.backend.put_lock:
                self.backend.put_tasks.discard(key)
            raise

    def _get_blocking(self, bound_backend: Any, key: Any) -> Any:
        del bound_backend
        _request, decision = self._admit(READ_DEMAND, key)
        if decision.action == ACTION_SUBMIT_NOW:
            return self._original_get(key)
        # RECOMPUTE is an explicit miss before allocation/GPU copy.  DEFER is
        # unreachable for a conforming demand policy and also fails closed.
        return None

    def _get_non_blocking(
        self, bound_backend: Any, key: Any, location: Optional[str] = None
    ) -> Future:
        del bound_backend, location
        _request, decision = self._admit(READ_SPECULATIVE, key)
        if decision.action == ACTION_RECOMPUTE:
            future: Future = Future()
            future.set_result(None)
            return future

        async def admitted_read() -> Any:
            if decision.action == ACTION_DEFER and decision.defer_ns:
                await self._async_wait(decision.defer_ns / 1_000_000_000)
            return await asyncio.to_thread(self._original_get, key)

        coroutine = admitted_read()
        try:
            return asyncio.run_coroutine_threadsafe(coroutine, self.backend.loop)
        except BaseException:
            coroutine.close()
            raise

    def _batched_get_blocking(
        self, bound_backend: Any, keys: Sequence[Any]
    ) -> list[Any]:
        del bound_backend
        submitted_keys: list[Any] = []
        submitted_indices: list[int] = []
        output: list[Any] = [None] * len(keys)
        for index, key in enumerate(keys):
            _request, decision = self._admit(READ_DEMAND, key)
            if decision.action == ACTION_SUBMIT_NOW:
                submitted_keys.append(key)
                submitted_indices.append(index)
            elif decision.action == ACTION_DEFER:
                self.stats["blocking_defer_miss"] += 1
        if submitted_keys:
            loaded = self._original_batched_get(submitted_keys)
            if len(loaded) != len(submitted_keys):
                raise AdmissionError("GdsBackend returned a misaligned batch")
            for index, memory_obj in zip(submitted_indices, loaded, strict=True):
                output[index] = memory_obj
        return output


def install_backend(
    backend: Any,
    *,
    mode: str = "native",
    decider: Optional[Decider] = None,
    request_provider: Optional[RequestProvider] = None,
    uvm_device: str = "/dev/nvidia-uvm",
    async_wait: AsyncWait = asyncio.sleep,
) -> GdsBackendAdmissionAdapter:
    """Install on an existing backend, returning the idempotent adapter."""
    existing = getattr(backend, _ADAPTER_ATTR, None)
    if existing is not None:
        return existing
    return GdsBackendAdmissionAdapter(
        backend,
        mode=mode,
        decider=decider,
        request_provider=request_provider,
        uvm_device=uvm_device,
        async_wait=async_wait,
    ).install()


def install_storage_manager(manager: Any, **kwargs: Any) -> list[GdsBackendAdmissionAdapter]:
    """Install on current GDS backends while retaining StorageManager's Future seam."""
    installed: list[GdsBackendAdmissionAdapter] = []
    for backend in manager.storage_backends.values():
        if backend.__class__.__name__ == "GdsBackend":
            installed.append(install_backend(backend, **kwargs))
    return installed


_bootstrap_lock = threading.Lock()


def _assert_signature(cls: Any, name: str, parameters: tuple[str, ...]) -> None:
    actual = tuple(inspect.signature(getattr(cls, name)).parameters)
    if actual != parameters:
        raise RuntimeError(
            f"LMCache {EXPECTED_LMCACHE_VERSION} {cls.__name__}.{name} "
            f"signature drift: {actual!r} != {parameters!r}"
        )


def bootstrap_from_env(
    environ: Mapping[str, str] = os.environ,
) -> Optional[Callable[..., None]]:
    """Patch future LMCache 0.5.4 GdsBackend instances from environment.

    Returns the installed class constructor hook, or ``None`` when the mode is
    absent/disabled.  Import this module early from ``sitecustomize`` or call
    this function explicitly before ``StorageManager`` creates its backends.
    """
    mode = environ.get(MODE_ENV, "").strip().lower()
    if mode in {"", "off", "disabled", "none"}:
        return None
    if mode not in {"fifo", "native", "bpf"}:
        raise ValueError(f"{MODE_ENV} must be fifo, native, bpf, or disabled")

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
        prior = getattr(GdsBackend, _CLASS_HOOK_ATTR, None)
        if prior is not None:
            prior_mode, hook = prior
            if prior_mode != mode:
                raise RuntimeError(
                    f"GdsBackend already bootstrapped in {prior_mode!r} mode"
                )
            return hook

        _assert_signature(
            GdsBackend,
            "__init__",
            ("self", "config", "metadata", "loop", "dst_device"),
        )
        _assert_signature(
            GdsBackend,
            "submit_put_task",
            ("self", "key", "memory_obj", "on_complete_callback"),
        )
        _assert_signature(
            GdsBackend,
            "_async_save_bytes_to_disk",
            ("self", "key", "memory_obj", "on_complete_callback"),
        )
        _assert_signature(GdsBackend, "get_blocking", ("self", "key"))
        _assert_signature(
            GdsBackend, "get_non_blocking", ("self", "key", "location")
        )
        _assert_signature(GdsBackend, "batched_get_blocking", ("self", "keys"))

        original_init = GdsBackend.__init__
        provider = EnvironmentRequestProvider.from_environ(environ)
        uvm_device = environ.get(
            "LMCACHE_GDS_POLICY_UVM_DEVICE", "/dev/nvidia-uvm"
        )

        @functools.wraps(original_init)
        def hooked_init(instance: Any, *args: Any, **kwargs: Any) -> None:
            original_init(instance, *args, **kwargs)
            install_backend(
                instance,
                mode=mode,
                request_provider=provider,
                uvm_device=uvm_device,
            )

        GdsBackend.__init__ = hooked_init
        setattr(GdsBackend, _CLASS_HOOK_ATTR, (mode, hooked_init))
        return hooked_init


# With gds-control on PYTHONPATH, a sitecustomize file may simply import this
# module.  Absence of MODE_ENV keeps ordinary LMCache processes unchanged.
if MODE_ENV in os.environ:
    bootstrap_from_env()
