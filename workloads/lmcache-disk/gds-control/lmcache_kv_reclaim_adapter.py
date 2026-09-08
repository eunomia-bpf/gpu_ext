"""Opt-in disk-aware KV reclaim serving adapter (LMCache 0.5.4 + vLLM 0.27.1).

Serving glue connecting three real components, all inside the single-GPU
UniProc EngineCore process (scheduler, worker, and LMCache engine share
that process):

1. ``gds-control/lmcache_kv_backing_state.py`` - in-process registry of
   real per-request disk-backing state: contiguous ``prefix_tokens`` /
   ``prefix_bytes`` (whole-object sums, never prorated), ``coverage_known``
   (false when any linked chunk is unknown/not_submitted or lacks token
   extents), and actual GDS read costs (``_backing.read_stats()`` /
   ``read_stats_summary()``: ``mean_ns_per_kib`` is None until a real
   ``_load_gds`` read completed; write submit-to-completion intervals are
   never mixed into read stats).  Activated on the live engine/GDS backend
   only after the original ``LMCacheConnectorV1Impl.register_kv_caches``
   chain completes (that chain runs ``_manager.post_init()``, which is
   where ``LMCacheEngine.storage_manager`` comes into existence at
   cache_engine.py:301-332).
2. The installed scheduler seam ``Scheduler.preempt_victim_callback``
   (gds-control/vllm-preemption-seam.patch, attribute default None).  The
   adapter sets the attribute after the original ``Scheduler.__init__``
   returns; victim removal, same-step rollback, and block freeing stay
   entirely inside stock scheduler code.
3. ``gds-control/kv_reclaim_binding.py`` - ``KvReclaimNative`` /
   ``KvReclaimKernel`` scalar victim selection over at most 8
   same-priority-class running-request candidates, returning
   (index, cookie, route, estimated_ns).

Recovery routes (consumed on resume, not merely logged):

- ``ROUTE_FULL_RECOMPUTE`` (1): while the route is pending for a request,
  the scheduler-side client ``lookup_cache`` returns a real 0-hit answer
  for that lookup id, so the connector makes its own consistent zero-load
  ``LoadSpec`` (lmcache_cached_tokens=0, can_load=False) and the request
  recomputes every token.  No lookup/prefetch is sent to any worker, so
  no outstanding lookup is created.  The pending route is consumed in the
  connector's ``update_state_after_alloc`` (which also clears the client
  lookup status exactly as upstream does normally).  Repeated deferred
  scheduling attempts inside one resume cycle stay idempotent: the bypass
  returns the same 0-hit answer every time until the request is actually
  scheduled.
- ``ROUTE_DISK_PREFIX`` (2): nothing is forced; the real LMCache lookup
  and disk restore run as upstream, and the realized external token count
  is recorded (a miss is recorded as such, never relabeled).
- ``ROUTE_STOCK`` (0): the stock victim is kept (callback returns None).

Priority convention: vLLM keeps LOWER value = MORE important; the ABI
carries the worst (maximum) class value.  The seam only allows overriding
inside the stock victim's priority class, so the adapter filters
candidates to ``default_victim``'s class BEFORE calling the decider and
before recording any route or marking preemption; the explicit clamp
into 0..MAX_PRIORITY applies to one shared class value and can never
reorder candidates.  ``disk_backed_tokens`` handed to the decider is the
registry prefix capped to the request's live ``num_computed_tokens``
extent; ``disk_backed_bytes`` stays the actual whole-object transfer size
(GDS restores whole backing objects - no fractional proration).

Bootstrap, including from ``gds-control/bootstrap/sitecustomize.py``::

    from lmcache_kv_reclaim_adapter import bootstrap_from_env
    bootstrap_from_env()

Required environment (all checked, fail-fast - no defaults are invented):

- ``LMCACHE_KV_RECLAIM=1``                      opt-in master switch
- ``LMCACHE_KV_RECLAIM_MODE=native|bpf``        decider arm (no default)
- ``LMCACHE_KV_RECLAIM_RECOMPUTE_NS_PER_TOKEN`` positive int; measured
  end-to-end recompute proxy supplied by the runner, identical for every
  arm of a campaign.
- ``LMCACHE_KV_RECLAIM_UVM_DEVICE``             bpf mode only.

Optional:

- ``LMCACHE_KV_RECLAIM_DIAG_OUT``               path for the shutdown
  diagnostics JSON file (see ``write_diagnostics``).

At process exit the adapter writes one structured JSON diagnostics file
(for the EngineCore process, which holds the one scheduler policy and
the backing activation; processes with no installed policy and no
backing activation - e.g. the API/frontend - never write, so they
cannot clobber the populated EngineCore file at a later exit) containing
decisions, recovery records, counters, warnings, backing ``read_stats()``
and backing ``describe()`` snapshots, so the serving runner can collect
them without any in-process query service.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import threading
import time
import atexit
from collections import deque
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Mapping, Optional

import kv_reclaim_binding as _binding
import lmcache_kv_backing_state as _backing

__all__ = [
    "RECLAIM_ENV",
    "MODE_ENV",
    "RECOMPUTE_NS_PER_TOKEN_ENV",
    "UVM_DEVICE_ENV",
    "DIAG_OUT_ENV",
    "EXPECTED_LMCACHE_VERSION",
    "KvReclaimVictimPolicy",
    "bootstrap_from_env",
    "adapter_state",
    "write_diagnostics",
]

RECLAIM_ENV = "LMCACHE_KV_RECLAIM"
MODE_ENV = "LMCACHE_KV_RECLAIM_MODE"
RECOMPUTE_NS_PER_TOKEN_ENV = "LMCACHE_KV_RECLAIM_RECOMPUTE_NS_PER_TOKEN"
UVM_DEVICE_ENV = "LMCACHE_KV_RECLAIM_UVM_DEVICE"
DIAG_OUT_ENV = "LMCACHE_KV_RECLAIM_DIAG_OUT"
EXPECTED_LMCACHE_VERSION = "0.5.4"
BOOTSTRAP_STATE_ATTR = "_kv_reclaim_policy"
_SCHEDULER_HOOK_ATTR = "_kv_reclaim_scheduler_class_hook"
_REGISTER_HOOK_ATTR = "_kv_reclaim_register_class_hook"
_LOOKUP_CACHE_HOOK_ATTR = "_kv_reclaim_lookup_cache_hook"
_LOOKUP_HOOK_ATTR = "_kv_reclaim_lookup_hook"
_UPDATE_ALLOC_HOOK_ATTR = "_kv_reclaim_update_alloc_hook"
_REQUEST_FINISHED_HOOK_ATTR = "_kv_reclaim_request_finished_hook"

MAX_DECISION_RECORDS = 512
MAX_RECOVERY_RECORDS = 512
MAX_COOKIE_MAP = 100_000

logger = logging.getLogger(__name__)


def _env_bool(name: str, environ: Mapping[str, str]) -> bool:
    raw = environ.get(name)
    if raw is None:
        return False
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} is not a boolean: {raw!r}")


def _assert_signature(cls: Any, name: str, parameters: tuple[str, ...]) -> None:
    import inspect

    actual = tuple(inspect.signature(getattr(cls, name)).parameters)
    if actual != parameters:
        raise RuntimeError(
            f"{cls.__name__}.{name} signature drift: {actual!r} != {parameters!r}"
        )


def _require_lmcache_version() -> None:
    try:
        installed = version("lmcache")
    except PackageNotFoundError as error:
        raise RuntimeError("LMCache is not installed") from error
    if installed != EXPECTED_LMCACHE_VERSION:
        raise RuntimeError(
            f"adapter requires LMCache {EXPECTED_LMCACHE_VERSION}, "
            f"found {installed}"
        )


class _BootstrapState:
    """Parsed-once opt-in configuration shared by every class hook."""

    def __init__(self, environ: Mapping[str, str]) -> None:
        self.enabled = _env_bool(RECLAIM_ENV, environ)
        self.mode = ""
        self.recompute_ns_per_token: Optional[int] = None
        self.uvm_path = ""
        self.diag_out = environ.get(DIAG_OUT_ENV) or None
        if not self.enabled:
            return
        mode_raw = environ.get(MODE_ENV)
        if mode_raw is None:
            raise ValueError(f"{MODE_ENV} is required when {RECLAIM_ENV}=1")
        self.mode = mode_raw.strip().lower()
        if self.mode not in {"native", "bpf"}:
            raise ValueError(f"{MODE_ENV} must be native or bpf: {mode_raw!r}")
        raw_price = environ.get(RECOMPUTE_NS_PER_TOKEN_ENV)
        if raw_price is None:
            raise ValueError(
                f"{RECOMPUTE_NS_PER_TOKEN_ENV} is required when "
                f"{RECLAIM_ENV}=1 (measured end-to-end recompute proxy, "
                "no invented constants)"
            )
        price = int(raw_price)
        if price <= 0:
            raise ValueError(
                f"{RECOMPUTE_NS_PER_TOKEN_ENV} must be positive: {price}"
            )
        self.recompute_ns_per_token = price
        if self.mode == "bpf":
            path_raw = environ.get(UVM_DEVICE_ENV)
            self.uvm_path = path_raw or "/dev/nvidia-uvm"


class KvReclaimVictimPolicy:
    """Scheduler-side victim selection on the installed preemption seam.

    Created once after the original ``Scheduler.__init__`` returned, when
    the opt-in environment is set.  Validates the actual KV cache layout
    at construction (bytes per physical block derived from the real
    config, never hardcoded; unsupported layouts fail fast) and installs
    ``scheduler.preempt_victim_callback`` plus the resume-route hooks on
    the scheduler-side LMCache connector/lookup client.
    """

    def __init__(self, scheduler: Any, state: _BootstrapState) -> None:
        assert state.enabled and state.recompute_ns_per_token is not None
        self.mode = state.mode
        self.recompute_ns_per_token = state.recompute_ns_per_token
        if state.mode == "native":
            self.decider = _binding.KvReclaimNative()
        else:
            fd = os.open(state.uvm_path, os.O_RDWR)
            self.decider = _binding.KvReclaimKernel(fd)
        self._kv_cache_manager = scheduler.kv_cache_manager
        self._bytes_per_block = self._derive_bytes_per_block()
        self._install_recovery_hooks(scheduler)
        self._state_lock = threading.Lock()
        self._cookies: dict[str, int] = {}
        self._next_cookie = 0
        self._pending_route: dict[str, dict[str, Any]] = {}
        self._decision_seq = 0
        self._warnings: set[str] = set()
        self.counters: dict[str, int] = {
            "seam_invocations": 0,
            "override_picks": 0,
            "stock_kept": 0,
            "unsupported_running_size": 0,
            "coverage_unknown_candidates": 0,
            "read_pricing_missing_decisions": 0,
            "forced_zero_lookups": 0,
            "route_consumed_full_recompute": 0,
            "route_consumed_disk_prefix": 0,
            "disk_route_recompute_fallback": 0,
            "requests_finished_dropped": 0,
        }
        self.decisions: deque[Mapping[str, Any]] = deque(
            maxlen=MAX_DECISION_RECORDS
        )
        self.recovery: deque[Mapping[str, Any]] = deque(
            maxlen=MAX_RECOVERY_RECORDS
        )
        scheduler.preempt_victim_callback = self.pick_victim

    # -- layout -------------------------------------------------------------

    def _derive_bytes_per_block(self) -> int:
        # The installed scheduler keeps a per-layer spec in kv_cache_groups
        # (kv_cache_utils.generate_scheduler_kv_cache_config replaces an
        # UniformTypeKVCacheSpecs group spec with its first per-layer spec),
        # so the spec type cannot decide byte accounting here.  For the
        # target layout - one group, every tensor owning exactly one layer
        # (no cross-layer/group sharing), no packed layout - each physical
        # block covers all layers, so bytes per block is
        # sum(kv_cache_tensors.size) / num_blocks (the allocation source
        # get_kv_cache_config_from_groups builds exactly that shape for the
        # one-group uniform branch).  Anything else is refused: no guessed
        # bytes.
        config = self._kv_cache_manager.kv_cache_config
        groups = list(config.kv_cache_groups)
        if len(groups) != 1:
            raise RuntimeError(
                "kv reclaim byte accounting supports one kv cache group "
                f"layouts only; found {len(groups)}"
            )
        tensors = list(config.kv_cache_tensors)
        num_blocks = int(config.num_blocks)
        if num_blocks <= 0 or not tensors:
            raise RuntimeError(
                f"invalid kv cache allocation (num_blocks={num_blocks}, "
                f"tensors={len(tensors)})"
            )
        total_bytes = 0
        seen_layers: set[str] = set()
        for tensor in tensors:
            shared_by = list(getattr(tensor, "shared_by", ()) or ())
            if len(shared_by) != 1:
                raise RuntimeError(
                    "kv cache tensors share allocations across layers or "
                    f"groups (shared_by={shared_by}); bytes per physical "
                    "block is not derivable without guessed accounting"
                )
            block_stride = int(getattr(tensor, "block_stride", 0) or 0)
            if block_stride != 0:
                raise RuntimeError(
                    "packed kv cache layout (block_stride!=0) is not "
                    "supported by kv reclaim byte accounting"
                )
            if shared_by[0] in seen_layers:
                raise RuntimeError(
                    f"layer {shared_by[0]!r} appears in multiple kv cache "
                    "tensors; layout not supported"
                )
            seen_layers.add(shared_by[0])
            total_bytes += int(tensor.size)
        if total_bytes % num_blocks != 0:
            raise RuntimeError(
                f"kv cache tensors ({total_bytes} bytes) are not block "
                f"aligned for num_blocks={num_blocks}"
            )
        return total_bytes // num_blocks

    # -- resume-route hooks --------------------------------------------------

    def _install_recovery_hooks(self, scheduler: Any) -> None:
        connector = getattr(scheduler, "connector", None)
        if connector is None:
            raise RuntimeError(
                "scheduler has no KV connector; disk-aware KV reclaim "
                "requires the LMCache connector (kv_transfer_config)"
            )
        impl = getattr(connector, "_lmcache_engine", None)
        if impl is None:
            raise RuntimeError(
                "scheduler connector has no _lmcache_engine impl attribute"
            )
        impl_type = type(impl)
        impl_modules = (
            "lmcache.integration.vllm.vllm_v1_adapter",
        )
        if (
            impl_type.__name__ != "LMCacheConnectorV1Impl"
            or impl_type.__module__ not in impl_modules
        ):
            raise RuntimeError(
                "expected an LMCacheConnectorV1Impl scheduler impl from "
                f"{impl_modules}, found "
                f"{impl_type.__module__}.{impl_type.__qualname__}"
            )
        lookup_client = impl.lookup_client
        if lookup_client is None:
            raise RuntimeError(
                "scheduler-role LMCache connector has no lookup client; "
                "disk-aware reclaim cannot run"
            )

        policy = self

        if not getattr(lookup_client, _LOOKUP_CACHE_HOOK_ATTR, None):
            _assert_signature(
                type(lookup_client), "lookup_cache", ("self", "lookup_id")
            )
            original_lookup_cache = lookup_client.lookup_cache

            def hooked_lookup_cache(
                lookup_id: str,
                _original=original_lookup_cache,
                _policy=policy,
            ) -> Optional[int]:
                with _policy._state_lock:
                    entry = _policy._pending_route.get(lookup_id)
                    if entry is not None and entry["route"] == (
                        _binding.ROUTE_FULL_RECOMPUTE
                    ):
                        _policy.counters["forced_zero_lookups"] += 1
                        return 0
                return _original(lookup_id)

            lookup_client.lookup_cache = hooked_lookup_cache
            setattr(lookup_client, _LOOKUP_CACHE_HOOK_ATTR, True)

        if not getattr(lookup_client, _LOOKUP_HOOK_ATTR, None):
            _assert_signature(
                type(lookup_client),
                "lookup",
                ("self", "token_ids", "lookup_id", "request_configs"),
            )
            original_lookup = lookup_client.lookup

            def hooked_lookup(
                token_ids: Any,
                lookup_id: str,
                request_configs: Optional[dict] = None,
                _original=original_lookup,
                _policy=policy,
            ) -> Optional[int]:
                # FULL_RECOMPUTE only: a pending disk-prefix route must
                # still run the real LMCache lookup so the disk restore
                # can happen.
                with _policy._state_lock:
                    entry = _policy._pending_route.get(lookup_id)
                    if (
                        entry is not None
                        and entry["route"] == _binding.ROUTE_FULL_RECOMPUTE
                    ):
                        _policy.counters["forced_zero_lookups"] += 1
                        return 0
                return _original(token_ids, lookup_id, request_configs)

            lookup_client.lookup = hooked_lookup
            setattr(lookup_client, _LOOKUP_HOOK_ATTR, True)

        if not getattr(impl, _UPDATE_ALLOC_HOOK_ATTR, None):
            _assert_signature(
                type(impl),
                "update_state_after_alloc",
                ("self", "request", "num_external_tokens"),
            )
            original_update = impl.update_state_after_alloc

            def hooked_update(
                request: Any,
                num_external_tokens: int,
                _original=original_update,
                _policy=policy,
            ) -> None:
                _original(request, num_external_tokens)
                _policy._consume_route(request, num_external_tokens)

            impl.update_state_after_alloc = hooked_update
            setattr(impl, _UPDATE_ALLOC_HOOK_ATTR, True)

        if not getattr(impl, _REQUEST_FINISHED_HOOK_ATTR, None):
            _assert_signature(
                type(impl),
                "request_finished",
                ("self", "request", "block_ids"),
            )
            original_finished = impl.request_finished

            def hooked_finished(
                request: Any,
                block_ids: Any,
                _original=original_finished,
                _policy=policy,
            ) -> Any:
                result = _original(request, block_ids)
                _policy._on_request_finished(request)
                return result

            impl.request_finished = hooked_finished
            setattr(impl, _REQUEST_FINISHED_HOOK_ATTR, True)

    def _consume_route(self, request: Any, num_external_tokens: int) -> None:
        request_id = request.request_id
        with self._state_lock:
            route_entry = self._pending_route.pop(request_id, None)
        if route_entry is None:
            return
        record = {
            "t": time.monotonic_ns(),
            "request_id": request_id,
            "route": route_entry["route"],
            "route_name": route_entry["route_name"],
            "estimated_ns": route_entry["estimated_ns"],
            "decision_seq": route_entry["decision_seq"],
            "num_external_tokens": num_external_tokens,
        }
        if route_entry["route"] == _binding.ROUTE_FULL_RECOMPUTE:
            self.counters["route_consumed_full_recompute"] += 1
            if num_external_tokens != 0:
                self._warn_once(
                    f"forced zero-load resume for {request_id} observed "
                    f"{num_external_tokens} external tokens"
                )
                record["outcome"] = "full_recompute_zero_admitted_mismatch"
            else:
                # Admission stage: zero external tokens scheduled into the
                # resume step.  Completed I/O is only evidenced by actual
                # read stats (backing read_stats / read_cost_samples_s).
                record["outcome"] = "full_recompute_zero_admitted"
        elif route_entry["route"] == _binding.ROUTE_DISK_PREFIX:
            self.counters["route_consumed_disk_prefix"] += 1
            if num_external_tokens > 0:
                # Admission, not transfer completion: the scheduler has
                # granted a load of this many tokens; completed disk I/O
                # shows up in the backing registry's actual read stats.
                record["outcome"] = "disk_prefix_admitted"
            else:
                record["outcome"] = "disk_prefix_zero_admitted"
                self.counters["disk_route_recompute_fallback"] += 1
        else:
            record["outcome"] = "unknown_route"
        self.recovery.append(record)

    def _on_request_finished(self, request: Any) -> None:
        request_id = request.request_id
        with self._state_lock:
            had_entry = self._cookies.pop(request_id, None) is not None
            self._pending_route.pop(request_id, None)
        if had_entry:
            self.counters["requests_finished_dropped"] += 1
        try:
            _backing.finish_request(request_id)
        except Exception as error:
            self._warn_once(f"backing finish_request failed: {error!r}")

    # -- cookies ------------------------------------------------------------

    def _cookie_for(self, request_id: str) -> int:
        with self._state_lock:
            cookie = self._cookies.get(request_id)
            if cookie is None:
                if len(self._cookies) >= MAX_COOKIE_MAP:
                    raise RuntimeError("kv reclaim cookie map overflow")
                self._next_cookie += 1
                cookie = self._next_cookie
                self._cookies[request_id] = cookie
            return cookie

    def drop_request(self, request_id: str) -> None:
        with self._state_lock:
            self._cookies.pop(request_id, None)
            self._pending_route.pop(request_id, None)

    # -- candidate construction --------------------------------------------

    def _exclusive_blocks_bytes(self, request_id: str) -> int:
        blocks = self._kv_cache_manager.get_blocks(request_id).blocks
        seen: set[int] = set()
        count = 0
        for group in blocks:
            for block in group:
                if block.is_null:
                    continue
                identity = id(block)
                if identity in seen:
                    continue
                seen.add(identity)
                if block.ref_cnt == 1:
                    count += 1
        return count * self._bytes_per_block

    def _warn_once(self, message: str) -> None:
        with self._state_lock:
            first = message not in self._warnings
            self._warnings.add(message)
        if first:
            logger.warning("kv reclaim: %s", message)

    def _build_candidate(self, request: Any) -> dict[str, Any]:
        request_id = request.request_id
        summary = _backing.request_summary(request_id)
        computed_tokens = int(request.num_computed_tokens)
        backed_tokens_arg = 0
        backed_bytes_arg = 0
        flags = 0
        if summary.get("known") and summary.get("coverage_known"):
            prefix_tokens = int(summary.get("prefix_tokens") or 0)
            prefix_bytes = summary.get("prefix_bytes")
            if prefix_tokens > 0 and prefix_bytes is not None:
                # Coverage trusted by the registry (no unknown/not_submitted
                # chunks, no missing extents).  Cap the dispatchable prefix
                # at the request's live computed extent; bytes stay the
                # actual whole-object transfer size.
                backed_tokens_arg = min(prefix_tokens, computed_tokens)
                backed_bytes_arg = int(prefix_bytes)
                flags = _binding.FLAG_COVERAGE_KNOWN
            else:
                self.counters["coverage_unknown_candidates"] += 1
        else:
            self.counters["coverage_unknown_candidates"] += 1
        return {
            "cookie": self._cookie_for(request_id),
            "freeable_bytes": self._exclusive_blocks_bytes(request_id),
            "computed_tokens": computed_tokens,
            "disk_backed_tokens": backed_tokens_arg,
            "disk_backed_bytes": backed_bytes_arg,
            "priority": self._normalize_priority(request.priority),
            "flags": flags,
        }

    def _normalize_priority(self, vllm_priority: int) -> int:
        # vLLM: lower = more important.  ABI worst class = MAX_PRIORITY.
        # One decision always covers a single priority class (the seam
        # enforces it and this adapter pre-filters), so the clamp below
        # can never reorder candidates; it maps the shared class value
        # into the fixed-width field.
        priority = int(vllm_priority)
        if priority < 0:
            priority = 0
        return min(priority, _binding.MAX_PRIORITY)

    # -- the seam callback --------------------------------------------------

    def pick_victim(self, running: list[Any], default_victim: Any) -> Any:
        self.counters["seam_invocations"] += 1
        if not running:
            return None

        # Same-priority-class candidates only: vLLM lower value = more
        # important; the stock victim defines the worst class the seam
        # permits overriding within.
        class_members = [
            request
            for request in running
            if request.priority == default_victim.priority
        ]
        stock_index = -1
        for index, request in enumerate(class_members):
            if request is default_victim:
                stock_index = index
                break
        if stock_index < 0:
            self._warn_once("default_victim missing from its priority class")
            return None
        if len(class_members) > _binding.MAX_CANDIDATES:
            self.counters["unsupported_running_size"] += 1
            self._warn_once(
                f"priority class has {len(class_members)} running requests, "
                f"the ABI carries at most {_binding.MAX_CANDIDATES} scalar "
                "candidates; keeping the stock decision"
            )
            return None

        candidates = [self._build_candidate(request) for request in class_members]
        read_stats = _backing.read_stats()
        mean_ns_per_kib = read_stats.get("mean_ns_per_kib")
        disk_read_ns_per_kib: Optional[int]
        if mean_ns_per_kib is not None:
            disk_read_ns_per_kib = max(1, int(round(float(mean_ns_per_kib))))
        else:
            disk_read_ns_per_kib = None
            self.counters["read_pricing_missing_decisions"] += 1

        result = self.decider.choose(
            candidates,
            disk_read_ns_per_kib or 0,
            self.recompute_ns_per_token,
            stock_index,
        )
        self._decision_seq += 1
        record: dict[str, Any] = {
            "t": time.monotonic_ns(),
            "decision_seq": self._decision_seq,
            "mode": self.mode,
            "n_candidates": len(candidates),
            "stock_index": stock_index,
            "disk_read_ns_per_kib": disk_read_ns_per_kib,
            "read_sample_count": read_stats.get("sample_count"),
            "recompute_ns_per_token": self.recompute_ns_per_token,
            "candidates": [
                {
                    "request_id": request.request_id,
                    "cookie": candidate["cookie"],
                    "freeable_bytes": candidate["freeable_bytes"],
                    "computed_tokens": candidate["computed_tokens"],
                    "disk_backed_tokens": candidate["disk_backed_tokens"],
                    "disk_backed_bytes": candidate["disk_backed_bytes"],
                    "priority": candidate["priority"],
                    "flags": candidate["flags"],
                }
                for request, candidate in zip(class_members, candidates)
            ],
        }
        record.update(result)

        victim: Optional[Any] = None
        if result["status"] != 0:
            record["action"] = "stock_degenerate_input"
        elif result["route"] == _binding.ROUTE_STOCK:
            record["action"] = "stock"
        else:
            index = result["index"]
            in_range = 0 <= index < len(candidates)
            cookie_ok = in_range and (
                result["cookie"] == candidates[index]["cookie"]
            )
            if not cookie_ok:
                self._warn_once(
                    "decider returned an index/cookie pair that does not "
                    "match any candidate; keeping the stock decision"
                )
                record["action"] = "stock_wrong_cookie"
            elif result["route"] == _binding.ROUTE_DISK_PREFIX:
                victim = class_members[index]
                with self._state_lock:
                    self._pending_route[victim.request_id] = {
                        "route": _binding.ROUTE_DISK_PREFIX,
                        "route_name": result["route_name"],
                        "estimated_ns": result["estimated_ns"],
                        "decision_seq": self._decision_seq,
                    }
                record["action"] = "override_disk_prefix"
                record["victim_request_id"] = victim.request_id
            elif result["route"] == _binding.ROUTE_FULL_RECOMPUTE:
                victim = class_members[index]
                with self._state_lock:
                    self._pending_route[victim.request_id] = {
                        "route": _binding.ROUTE_FULL_RECOMPUTE,
                        "route_name": result["route_name"],
                        "estimated_ns": result["estimated_ns"],
                        "decision_seq": self._decision_seq,
                    }
                record["action"] = "override_full_recompute"
                record["victim_request_id"] = victim.request_id
            else:
                record["action"] = f"unhandled_route_{result['route']}"

        if victim is not None:
            self.counters["override_picks"] += 1
            _backing.mark_preempted(victim.request_id)
        else:
            self.counters["stock_kept"] += 1
        self.decisions.append(record)
        return victim


_BOOTSTRAP_STATE: Optional[_BootstrapState] = None
_POLICIES: list[KvReclaimVictimPolicy] = []
_BACKING_ACTIVATED = False
_BOOTSTRAP_LOCK = threading.Lock()
_DIAG_WRITE_LOCK = threading.Lock()


def _activate_backing_state(impl: Any) -> None:
    """Bind the backing-state registry to the real engine after post_init.

    ``LMCacheConnectorV1Impl.register_kv_caches`` runs the manager's
    post_init at its end, which is the point where the live
    ``LMCacheEngine.storage_manager`` (and therefore the wrapped
    GdsBackend) exists.  Activation happens after the original call
    completed, once per impl, and is idempotent.
    """
    global _BACKING_ACTIVATED
    if getattr(impl, "_kv_reclaim_backing_active", False):
        return
    engine = impl.lmcache_engine
    if engine is None:
        raise RuntimeError(
            "kv reclaim activation found no live LMCacheEngine after "
            "register_kv_caches/post_init"
        )
    storage_manager = engine.storage_manager
    if storage_manager is None:
        raise RuntimeError(
            "kv reclaim activation found no storage_manager on the live "
            "LMCacheEngine"
        )
    gds_backends = [
        backend
        for backend in storage_manager.storage_backends.values()
        if backend.__class__.__name__ == "GdsBackend"
    ]
    if len(gds_backends) != 1:
        raise RuntimeError(
            "kv reclaim activation requires exactly one GdsBackend in the "
            f"engine storage backends, found {len(gds_backends)}"
        )
    setattr(impl, "_kv_reclaim_backing_active", True)
    _backing.enable(engine=engine, gds_backend=gds_backends[0])
    _BACKING_ACTIVATED = True
    logger.info(
        "LMCache disk-backing state bound to engine for kv reclaim "
        "(instance=%s)",
        getattr(engine, "instance_id", "?"),
    )


def diagnostics_payload() -> dict[str, Any]:
    """Structured snapshot for the runner's teardown collection."""
    state = _BOOTSTRAP_STATE
    payload: dict[str, Any] = {
        "adapter": "lmcache_kv_reclaim_adapter",
        "pid": os.getpid(),
        "t": time.time_ns(),
        "enabled": state is not None and state.enabled,
        "mode": state.mode if state else None,
        "recompute_ns_per_token": (
            state.recompute_ns_per_token if state else None
        ),
        "backing_activated": _BACKING_ACTIVATED,
        "policies": [],
        "backing": {
            "read_stats": _backing.read_stats(),
            "describe": _backing.describe(),
        },
    }
    for policy in _POLICIES:
        with policy._state_lock:
            payload["policies"].append(
                {
                    "kind": "victim_policy",
                    "mode": policy.mode,
                    "recompute_ns_per_token": policy.recompute_ns_per_token,
                    "counters": dict(policy.counters),
                    "warnings": sorted(policy._warnings),
                    "decisions": list(policy.decisions),
                    "recovery": list(policy.recovery),
                    "pending_routes": {
                        request_id: dict(entry)
                        for request_id, entry in policy._pending_route.items()
                    },
                    "open_cookies": len(policy._cookies),
                }
            )
    return payload


def write_diagnostics(path: Optional[str] = None) -> Optional[str]:
    """Write the diagnostics JSON snapshot to ``path`` (default env).

    Called automatically at process exit when any policy or backing
    activation is installed; the runner may also call it directly.
    """
    payload = diagnostics_payload()
    # UniProc teardown guard: bootstrap runs in the API/frontend process
    # too, where the adapter is enabled but installs nothing (no
    # Scheduler, no backing).  A frontend exit must never overwrite an
    # EngineCore-produced diagnostics file with its empty snapshot, so
    # write only when there is actual adapter evidence.
    has_content = bool(payload["policies"]) or payload["backing_activated"]
    if not has_content:
        return None
    target = path
    if target is None:
        state = _BOOTSTRAP_STATE
        if state is None or state.diag_out is None:
            return None
        target = state.diag_out
    parent = os.path.dirname(os.path.abspath(target))
    os.makedirs(parent, exist_ok=True)
    tmp = f"{target}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, target)
    return target


def _atexit_dump() -> None:
    try:
        state = _BOOTSTRAP_STATE
        if state is None or not state.enabled or state.diag_out is None:
            return
        written = write_diagnostics()
        if written is not None:
            logger.info("kv reclaim diagnostics written to %s", written)
    except Exception as error:
        logging.getLogger(__name__).error(
            "kv reclaim diagnostics write failed: %s", error
        )


def adapter_state() -> dict[str, Any]:
    payload = diagnostics_payload()
    return {
        "enabled": payload["enabled"],
        "mode": payload["mode"],
        "recompute_ns_per_token": payload["recompute_ns_per_token"],
        "backing_activated": payload["backing_activated"],
        "policies_installed": len(payload["policies"]),
        "diag_out": _BOOTSTRAP_STATE.diag_out if _BOOTSTRAP_STATE else None,
    }


def bootstrap_from_env(
    environ: Mapping[str, str] = os.environ,
) -> Optional[dict[str, Any]]:
    """Install the kv reclaim class hooks when the opt-in env is set.

    - Hooks ``Scheduler.__init__`` (post-call): verifies the installed
      seam attribute and helper exist and installs ``KvReclaimVictimPolicy``
      (setting ``preempt_victim_callback`` plus the resume-route hooks).
    - Hooks ``LMCacheConnectorV1Impl.register_kv_caches`` (post-call):
      binds the backing-state registry to the live engine/GDS backend once
      ``_manager.post_init()`` has produced them.
    Returns the adapter state summary, or None when the opt-in env is
    absent.
    """
    global _BOOTSTRAP_STATE
    if not _env_bool(RECLAIM_ENV, environ):
        return None

    _require_lmcache_version()

    # Actual connector binding for this installed target:
    # LMCacheConnectorV1.__init__ (vllm
    # distributed/kv_transfer/kv_connector/v1/lmcache_connector.py:93-112)
    # reads extra_config "use_native" with default False, which the target
    # does not set, so connector._lmcache_engine is
    # lmcache.integration.vllm.vllm_v1_adapter.LMCacheConnectorV1Impl.
    # The use_native=True in-vllm module
    # (vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration)
    # is unused and not importable against installed LMCache 0.5.4 (its
    # multi_process_adapter imports CudaIPCWrapper from
    # lmcache.v1.multiprocess.custom_types, absent there); it is not
    # imported or hooked here.
    from vllm.v1.core.sched.scheduler import Scheduler
    from lmcache.integration.vllm.vllm_v1_adapter import (
        LMCacheConnectorV1Impl as LatestDevImpl,
    )

    already_installed = False
    with _BOOTSTRAP_LOCK:
        if _BOOTSTRAP_STATE is not None:
            already_installed = True
        else:
            state = _BootstrapState(environ)
            _BOOTSTRAP_STATE = state

    if already_installed:
        return adapter_state()

    with _BOOTSTRAP_LOCK:
        if not hasattr(Scheduler, "_select_preempt_victim"):
            raise RuntimeError(
                "Scheduler.preempt_victim_callback seam "
                "(gds-control/vllm-preemption-seam.patch) is not present "
                "in the installed vLLM"
            )
        if getattr(Scheduler, _SCHEDULER_HOOK_ATTR, None) is not None:
            raise RuntimeError("kv reclaim scheduler hook already installed")
        original_scheduler_init = Scheduler.__init__

        @functools.wraps(original_scheduler_init)
        def hooked_scheduler_init(
            instance: Any, *args: Any, **kwargs: Any
        ) -> None:
            original_scheduler_init(instance, *args, **kwargs)
            if getattr(instance, BOOTSTRAP_STATE_ATTR, None) is not None:
                return
            if not hasattr(instance, "preempt_victim_callback"):
                raise RuntimeError(
                    "scheduler instance lacks preempt_victim_callback; "
                    "the vllm preemption seam attribute is required"
                )
            state = _BOOTSTRAP_STATE
            assert state is not None
            policy = KvReclaimVictimPolicy(instance, state)
            setattr(instance, BOOTSTRAP_STATE_ATTR, policy)
            _POLICIES.append(policy)
            logger.info(
                "kv reclaim victim policy installed on Scheduler "
                "(mode=%s, recompute_ns_per_token=%s)",
                policy.mode,
                policy.recompute_ns_per_token,
            )

        Scheduler.__init__ = hooked_scheduler_init
        setattr(Scheduler, _SCHEDULER_HOOK_ATTR, hooked_scheduler_init)

        for impl_cls in (LatestDevImpl,):
            _assert_signature(
                impl_cls,
                "register_kv_caches",
                ("self", "kv_caches"),
            )
            if getattr(impl_cls, _REGISTER_HOOK_ATTR, None) is not None:
                continue
            original_register = impl_cls.register_kv_caches

            def hooked_register(
                self: Any,
                kv_caches: Any,
                _original=original_register,
            ) -> None:
                _original(self, kv_caches)
                _activate_backing_state(self)

            impl_cls.register_kv_caches = hooked_register
            setattr(impl_cls, _REGISTER_HOOK_ATTR, hooked_register)

    if DIAG_OUT_ENV in os.environ or environ.get(DIAG_OUT_ENV):
        atexit.register(_atexit_dump)
    return adapter_state()


# With gds-control on PYTHONPATH, importing this module early (sitecustomize)
# installs the hooks when the opt-in environment is present.  Absence keeps
# ordinary LMCache processes unchanged.
if _env_bool(RECLAIM_ENV, os.environ):
    bootstrap_from_env()
