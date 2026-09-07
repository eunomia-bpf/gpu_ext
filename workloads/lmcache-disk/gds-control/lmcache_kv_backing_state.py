"""In-process disk-backing state registry for LMCache GDS store/retrieve flow.

Opt-in module owned by the kv-reclaim-integration route task
(gds-control/kv-reclaim-integration-route-20260907.md, section 7).  It
collects, from real call sites of the installed stack only, the association
between a serving request and actual contiguous disk-backed KV chunk state,
and exposes thread-safe snapshots for scheduler-side victim selection (the
typed native/kernel-BPF selector lives in a separate task; root and the
serving adapter session integrate the APIs).  This registry is preparatory
metadata, not automatic offload.

Capture seams (installed LMCache 0.5.4 / vLLM 0.27.1 under
current-venv/lib/python3.12/site-packages):

- CacheEngine.store kwargs ``req_id`` (vllm_v1_adapter.py:1229-1238 when the
  adapter threads it through).  A thread-local store ctx opened around the
  original call records request association BEFORE submission callbacks can
  fire.
- gpu_connector.batched_from_gpu(memory_objs, starts, ends, **kwargs)
  (cache_engine.py:557-558): token extents.  NOTE: the CUDA store path of
  VLLMPagedMemGPUConnectorV3 is asynchronous (gpu_connectors.py:645-649
  synchronizes only for non-CUDA staging; this workload stages in CUDA), so
  the Python return of store() is NOT GPU-gather or disk-write completion.
  ``pending_write`` never means "copy complete".
- StorageManager.batched_put(keys, memory_objs, ...)
  (storage_manager.py:386-435): records keys BEFORE the original submission
  so a completion callback fired inside the submission is the last writer
  for its key.
- GdsBackend.submit_put_task(key, memory_obj, on_complete_callback=...)
  (gds_backend.py:591-613).  The chained callback fires on successful GDS
  write completion only (gds_backend.py:712-719) and is the sole
  write-completion evidence; the original callback is preserved, synchronous
  raises mark the key ``not_submitted`` and re-raise.  Additionally the
  exact key object is probed with backend.contains() BEFORE submission:
  when an installed adapter stack skips the write for an already-stored key
  (no callback will fire) that key must not wait forever as pending; it is
  marked preexisting with evidence "backend.contains at submit".
- Warm retrieve: the OUTER CacheEngine.retrieve returns only a boolean CPU
  ret_mask (cache_engine.py:780-810) and proves nothing; the wrapper does
  not touch it.  The real processed chunk tuples (key, memory_obj, start,
  end) exist only as the ``List[ProcessedChunk]`` of ``ProcessTokensInternalResult``
  (cache_engine.py:74,:76) returned by ``_process_tokens_internal`` /
  ``_async_process_tokens_internal`` (cache_engine.py:842-854; both inherit
  the retrieve ``**kwargs`` containing ``req_id``).  We observe those actual
  returns, composing on whatever bound method is installed at enable() time
  (another task's async consumed-event wrapper stays inside the observed
  call; implementations are never copied), and pass the return and all
  refcounts through untouched.  Consumed chunks alone do NOT prove GDS
  backing: keys without recorded backing evidence are probed once with the
  exact consumed key object via backend.contains() and otherwise stay
  ``unknown``; pending writes stay pending.  No token re-hashing, no
  invented metadata.
- Actual GDS read cost: backend ``_load_bytes_from_disk_with_memory(key,
  path, memory_obj)`` (gds_backend.py:847-896) carries the exact key and
  calls ``_load_gds(gds_path, file_offset, gpu_pointer, size_in_bytes,
  dev_offset) -> int`` (gds_backend.py:1035-1102) synchronously inside its
  frame.  A thread-local stashed by the keyed wrapper correlates each
  ``_load_gds`` call to a key.  ``_load_gds`` returns -1 on failure; a
  sample (perf_counter wall duration of the installed call stack, actual
  returned read bytes) is recorded ONLY for completed actual reads
  (ret != -1).  Read timing is never derived from write timing; with no
  sample, ``mean_ns_per_kib`` is None (no fabricated read estimate).  WRITE
  submit->completion intervals (completion callbacks, including adapter
  deferral and queue time) are exported separately and never mixed into
  read stats.

Backed-state classification sources (consumers must not relax):

- a recorded successful GDS write completion callback;
- a backend.contains() hit with the existing exact CacheEngineKey at
  submit, at retrieve association, or via classify_keys probing.

No other path re-classifies a key as backed.  A retrieved/consumed CPU-cache
object is not disk evidence by itself.  Already-backed keyed re-puts and
skipped writes update association/extent/size/rewrites only; backed state is
never reset to pending.  Pending/in-progress writes stay unproven until a
successful completion callback or a contains() hit.

Contract of the exact outputs (root + serving adapter session consume):

- request_summary(req_id) -> dict with keys: ``req_id``, ``known`` (False
  when the request is untracked; all later fields then default), ``request``
  (request record dict), ``coverage_known`` (True only when every chunk key
  linked to the request has a record with a classification other than
  unknown/not_submitted AND no backed chunk in the request lacks token
  extents — missing extents are explicit unknown because they could hide a
  gap inside the reported run), ``prefix_tokens`` (length of the contiguous
  DISK_BACKED/PREEXISTING token run starting at token 0, deduplicated
  extent run, stopped at the first gap), ``prefix_bytes`` (sum of whole
  object byte sizes, each counted once when it first extends the run;
  identical duplicates deduplicated, no fractional token proration — GDS
  restores whole backing objects; None when any run object's size is
  missing or the run is empty),
  ``pending_tokens`` (sum of pending extents), ``read_cost_samples_s``
  (list of {"duration_s","bytes","token_start","token_end"} dicts for
  records inside the prefix run that have an actual recorded GDS read;
  empty when none observed), ``write_cost_samples_s`` (same shape for
  recorded write submit->completion intervals), ``shared_read``
  (read_stats_summary() dict), ``counts`` {backed_chunks, pending_chunks,
  unknown_chunks, total_chunks (chunk lines, append-duplicated)},
  ``extents_missing_chunks``.  Sums outside the contiguous prefix run are
  never reported as coverage; no decode-tail is inferred from cached prompt
  chunks.  Freeable bytes/refs are NOT computed here: they come from the
  live scheduler/KVCacheManager block refs at the scheduling seam.
- read_stats() -> shared policy statistics over all completed actual GDS
  reads: sample_count, duration_sum_s, bytes_sum, mean_ns_per_kib (None
  until a sample exists), samples_dropped, and the bounded recent
  ``samples`` list of {"duration_s","bytes"} dicts (<=4096).
- classify_keys(keys) -> per-key record dicts, probing only with the exact
  passed key objects; unknown stays explicit ``unknown``.
- enable(engine, gds_backend) is idempotent per object and never holds the
  singleton lock across registry() or wrapping.
- finish_request(req_id) drops request links only; mark_preempted(req_id)
  flags the request record; key records never invalidate disk objects.

Process-local for the single-GPU UniProc target (parallel.py:955-956,
uniproc_executor.py:45-105); no IPC schema.  Install after
``lmcache_gds_backend_adapter`` bootstrap so wrappers compose on top of its
policies/callbacks/errors, and after the async-prefetch adapter if it wraps
the same engine methods.  Bounds: 100k keys / 5k requests, evict oldest
non-pending records (no CacheEngineKey comparisons), overflow counted in
describe().  Ordinary py_compile passes. Earlier stub exercises are not
engagement evidence; no serving run or new performance result is claimed.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Tuple


class KeyState(str, Enum):
    PENDING_WRITE = "pending_write"
    DISK_BACKED = "disk_backed"
    PREEXISTING = "preexisting"
    NOT_SUBMITTED = "not_submitted"
    UNKNOWN = "unknown"


BACKED_STATES = (KeyState.DISK_BACKED, KeyState.PREEXISTING)

_INSTALL_ATTR = "_lmcache_backing_state_wrapped"

_MAX_KEYS = 100_000
_MAX_REQUESTS = 5_000
_MAX_READ_SAMPLES = 4_096


@dataclass
class KeyRecord:
    key: Any
    state: KeyState
    req_id: Optional[str] = None
    store_seq: int = -1
    size_bytes: Optional[int] = None
    token_start: Optional[int] = None
    token_end: Optional[int] = None
    submit_t: Optional[float] = None
    complete_t: Optional[float] = None
    write_cost_s: Optional[float] = None
    read_cost_s: Optional[float] = None
    read_bytes: Optional[int] = None
    read_count: int = 0
    evidence: str = ""
    rewrites: int = 0

    def as_dict(self) -> Dict[str, Any]:
        out = {
            "state": self.state.value,
            "req_id": self.req_id,
            "store_seq": self.store_seq,
            "size_bytes": self.size_bytes,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "submit_t": self.submit_t,
            "complete_t": self.complete_t,
            # write submit->completion wall interval (callback fires); NOT a
            # read cost.  Read costs are separate fields below.
            "write_cost_s": self.write_cost_s,
            # actual recorded successful GDS read (only when correlated via
            # _load_bytes_from_disk_with_memory); None = no read observed.
            "read_cost_s": self.read_cost_s,
            "read_bytes": self.read_bytes,
            "read_count": self.read_count,
            "evidence": self.evidence,
            "rewrites": self.rewrites,
        }
        if self.token_start is not None and self.token_end is not None:
            out["token_extent"] = self.token_end - self.token_start
        return out


@dataclass
class RequestRecord:
    req_id: str
    created_t: float
    chunk_keys: List[Any] = field(default_factory=list)
    preempted: bool = False
    finished: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "req_id": self.req_id,
            "created_t": self.created_t,
            "chunks": len(self.chunk_keys),
            "preempted": self.preempted,
            "finished": self.finished,
        }


class BackingStateRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._keys: Dict[Any, KeyRecord] = {}
        self._requests: Dict[str, RequestRecord] = {}
        self._store_seq = 0
        self._overflow_count = 0
        self._describe: Dict[str, int] = {}
        self._probe_backends: List[Any] = []
        self._tl = threading.local()
        # Actual successful GDS read (=_load_gds ret != -1) observed samples:
        # (duration_s, bytes).  Bounded; oldest dropped beyond the cap.
        self._read_samples: Deque[Tuple[float, int]] = deque(maxlen=_MAX_READ_SAMPLES)
        self._read_samples_dropped = 0
        self._read_tot_s = 0.0
        self._read_tot_bytes = 0
        self._read_tot_count = 0
        # Thread-local set by the _load_bytes_from_disk_with_memory wrapper
        # so _load_gds wrapper can correlate its sample to the exact key.
        self._read_tl = threading.local()

    def current_ctx(self) -> Optional["_StoreCtx"]:
        return getattr(self._tl, "ctx", None)

    def current_read_key(self) -> Optional[Any]:
        return getattr(self._read_tl, "key", None)

    def set_current_read_key(self, key: Optional[Any]) -> None:
        self._read_tl.key = key

    # ---- capture side (called by wrappers) ----

    def bind_probe_backends_append(self, backend: Any) -> None:
        if backend is None:
            return
        with self._lock:
            if backend not in self._probe_backends:
                self._probe_backends.append(backend)

    def begin_store(self, req_id: Optional[str]) -> "_StoreCtx":
        with self._lock:
            self._store_seq += 1
            seq = self._store_seq
            if req_id is not None and req_id not in self._requests:
                self._evict_requests_locked()
                self._requests[req_id] = RequestRecord(
                    req_id=req_id, created_t=time.monotonic()
                )
        return _StoreCtx(self, req_id, seq)

    def note_extents(self, ctx: "_StoreCtx", memory_objs: Any, starts: Any, ends: Any) -> None:
        pairs = zip(list(memory_objs), list(starts), list(ends))
        for memory_obj, start, end in pairs:
            ctx.extents[id(memory_obj)] = (int(start), int(end))

    def note_batched_put(self, ctx: Optional["_StoreCtx"], keys: Any, memory_objs: Any) -> None:
        now = time.monotonic()
        req_id = ctx.req_id if ctx is not None else None
        seq = ctx.store_seq if ctx is not None else -1
        extents = ctx.extents if ctx is not None else {}
        keys_l = list(keys)
        objs_l = list(memory_objs)
        with self._lock:
            req = self._requests.get(req_id) if req_id is not None else None
            for key, memory_obj in zip(keys_l, objs_l):
                extent = extents.get(id(memory_obj))
                size = _safe_size(memory_obj)
                record = self._keys.get(key)
                # Existing backed/pending records are never reset to pending:
                # a re-put of an already-backed key gets no fresh completion
                # callback, and overwriting here could erase a completion that
                # already fired on this event loop.
                if record is not None:
                    record.rewrites += 1
                    if req_id is not None:
                        record.req_id = req_id
                        record.store_seq = seq
                    if extent is not None:
                        record.token_start, record.token_end = extent
                    if size is not None:
                        record.size_bytes = size
                    if req is not None:
                        req.chunk_keys.append(key)
                    continue
                self._keys[key] = KeyRecord(
                    key=key,
                    state=KeyState.PENDING_WRITE,
                    req_id=req_id,
                    store_seq=seq,
                    size_bytes=size,
                    token_start=(extent[0] if extent else None),
                    token_end=(extent[1] if extent else None),
                    submit_t=now,
                )
                if req is not None:
                    req.chunk_keys.append(key)
            self._bound_check_keys_locked()

    def note_submit_error(self, key: Any) -> None:
        now = time.monotonic()
        with self._lock:
            record = self._keys.get(key)
            if record is None:
                self._keys[key] = KeyRecord(
                    key=key, state=KeyState.NOT_SUBMITTED, submit_t=now,
                    evidence="submit_put_task raised",
                )
                return
            # Only downgrade an outstanding pending write; backed evidence
            # from a completed callback is never erased.
            if record.state is KeyState.PENDING_WRITE:
                record.state = KeyState.NOT_SUBMITTED
                record.evidence = "submit_put_task raised"

    def note_submit_success(self, key: Any) -> None:
        """GDS completion callback fired: successful disk write only
        (gds_backend.py:712-719).  This is the sole write-completion
        evidence; its submit->completion interval is WRITE cost.
        """
        now = time.monotonic()
        with self._lock:
            record = self._keys.get(key)
            if record is None:
                self._keys[key] = KeyRecord(
                    key=key,
                    state=KeyState.DISK_BACKED,
                    complete_t=now,
                    evidence="gds completion callback",
                )
                return
            if record.submit_t is not None:
                record.write_cost_s = now - record.submit_t
            record.complete_t = now
            record.state = KeyState.DISK_BACKED
            record.evidence = "gds completion callback"

    def note_read_sample(
        self, key_repr: Optional[str], duration_s: float, read_bytes: int
    ) -> None:
        """Actual completed GDS read (``_load_gds`` returned != -1).

        ``duration_s`` is perf_counter wall time of the call.  ``read_bytes``
        is the byte count ``_load_gds`` actually returned (its contract,
        gds_backend.py:1049-1054,1092) — short results are recorded with
        their actual returned bytes, never silently as the requested size.
        Correlated to a key only via the keyed load seam; keyless samples
        still feed the shared policy stats.
        """
        with self._lock:
            self._read_tot_s += duration_s
            self._read_tot_bytes += read_bytes
            self._read_tot_count += 1
            if len(self._read_samples) >= _MAX_READ_SAMPLES:
                self._read_samples_dropped += 1
            self._read_samples.append((duration_s, read_bytes))
            if key_repr is not None:
                record = self._keys.get(key_repr)
                if record is not None and isinstance(record, KeyRecord):
                    record.read_cost_s = duration_s
                    record.read_bytes = read_bytes
                    record.read_count += 1
            self._describe["read_samples"] = self._read_tot_count

    def note_preexisting_at_submit(self, key: Any, memory_obj: Any) -> None:
        """backend.contains(exact key) hit at submit time.

        Covers stores skipped for keys already on disk (the backend may
        return without ever firing the completion callback) and re-puts of
        backed objects: they must not wait forever as pending_write.
        Never resets backed evidence; never marks an in-flight write.
        """
        now = time.monotonic()
        size = _safe_size(memory_obj)
        ctx = self.current_ctx()
        req_id = ctx.req_id if ctx is not None else None
        with self._lock:
            record = self._keys.get(key)
            if record is None:
                record = KeyRecord(
                    key=key,
                    state=KeyState.PREEXISTING,
                    submit_t=now,
                    complete_t=now,
                    size_bytes=size,
                    evidence="backend.contains at submit",
                )
                self._keys[key] = record
            else:
                if record.state not in BACKED_STATES:
                    record.state = KeyState.PREEXISTING
                    record.evidence = "backend.contains at submit"
                    record.complete_t = now
                if size is not None and record.size_bytes is None:
                    record.size_bytes = size
            if req_id is not None:
                record.req_id = req_id
                req = self._requests.get(req_id)
                if req is not None and key not in req.chunk_keys:
                    req.chunk_keys.append(key)
            self._bound_check_keys_locked()

    def read_stats_summary(self) -> Dict[str, Any]:
        """Aggregate of actual completed GDS reads, no per-sample list.

        ``mean_ns_per_kib`` is None while no sample exists (no fabricated
        read estimate).  Aggregate sums use every captured sample, also
        ones evicted from the bounded ``samples`` deque of read_stats().
        """
        with self._lock:
            mean_ns_per_kib: Optional[float] = None
            if self._read_tot_bytes > 0:
                mean_ns_per_kib = (
                    self._read_tot_s / self._read_tot_bytes
                ) * 1024 * 1e9
            return {
                "sample_count": self._read_tot_count,
                "duration_sum_s": self._read_tot_s,
                "bytes_sum": self._read_tot_bytes,
                "mean_ns_per_kib": mean_ns_per_kib,
                "samples_dropped": self._read_samples_dropped,
            }

    def read_stats(self) -> Dict[str, Any]:
        """Shared actual-read stats for candidate policy consumers.

        ``samples`` is the bounded recent list of {"duration_s", "bytes"}
        dicts; see read_stats_summary() for the aggregate fields.
        """
        with self._lock:
            out = self.read_stats_summary()
            out["samples"] = [
                {"duration_s": d, "bytes": b} for d, b in self._read_samples
            ]
            return out

    def note_retrieve(self, req_id: Optional[str], chunks: Any) -> int:
        """Warm/read association from the engine's real processed chunks.

        ``chunks`` is the actual ``List[ProcessedChunk]`` returned by
        ``CacheEngine._process_tokens_internal`` /
        ``_async_process_tokens_internal``: ``List[(CacheEngineKey,
        MemoryObj, start, end)]`` (cache_engine.py:74, :842-849; both
        receive the retrieve ``**kwargs`` containing ``req_id``).  The
        outer ``retrieve`` returns only a boolean CPU ret_mask and is not
        observed.

        Consumed chunks do NOT alone prove GDS backing: an object may
        come from an in-flight/unverified write or a CPU tier.  For keys
        without recorded backing evidence the exact key object is probed
        once via backend ``contains()``; pending writes stay pending
        until the successful-write completion callback.
        """
        recorded = 0
        now = time.monotonic()
        with self._lock:
            req = None
            if req_id is not None:
                req = self._requests.get(req_id)
                if req is None:
                    self._evict_requests_locked()
                    req = RequestRecord(req_id=req_id, created_t=now)
                    self._requests[req_id] = req
            for chunk in chunks:
                try:
                    key, memory_obj, start, end = chunk  # type: ignore[misc]
                except (TypeError, ValueError):
                    continue
                if key is None or start is None or end is None:
                    continue
                size = _safe_size(memory_obj)
                record = self._keys.get(key)
                if record is None:
                    state, evidence = self._probe_locked(key)
                    record = KeyRecord(key=key, state=state, evidence=evidence)
                    self._keys[key] = record
                elif record.state not in BACKED_STATES and (
                    record.state is not KeyState.PENDING_WRITE
                ):
                    # unknown / not_submitted: probe with the exact key
                    # object the engine actually consumed; no re-hashing.
                    state, evidence = self._probe_locked(key)
                    if state is KeyState.PREEXISTING:
                        record.state = state
                        record.evidence = "retrieve consumed; backend.contains"
                if req_id is not None:
                    record.req_id = req_id
                if size is not None and record.size_bytes is None:
                    record.size_bytes = size
                try:
                    record.token_start = int(start)
                    record.token_end = int(end)
                except (TypeError, ValueError):
                    pass
                if req is not None:
                    req.chunk_keys.append(key)
                recorded += 1
            self._bound_check_keys_locked()
        self._describe["retrieve_chunks"] = (
            self._describe.get("retrieve_chunks", 0) + recorded
        )
        return recorded

    # ---- lifecycle ----

    def mark_preempted(self, req_id: str) -> None:
        with self._lock:
            req = self._requests.get(req_id)
            if req is not None:
                req.preempted = True

    def finish_request(self, req_id: str) -> None:
        with self._lock:
            self._requests.pop(req_id, None)

    # ---- snapshots ----

    def classify_keys(self, keys: Any) -> List[Dict[str, Any]]:
        with self._lock:
            results = []
            for key in list(keys):
                record = self._keys.get(key)
                if record is not None:
                    results.append(record.as_dict())
                    continue
                state, evidence = self._probe_locked(key)
                self._keys[key] = KeyRecord(key=key, state=state, evidence=evidence)
                results.append(self._keys[key].as_dict())
            self._bound_check_keys_locked()
        return results

    def request_summary(self, req_id: str) -> Dict[str, Any]:
        with self._lock:
            req = self._requests.get(req_id)
            base: Dict[str, Any] = {
                "req_id": req_id,
                "known": False,
                "coverage_known": False,
                "prefix_tokens": 0,
                "prefix_bytes": None,
                "pending_tokens": 0,
                "read_cost_samples_s": [],
                "write_cost_samples_s": [],
                "shared_read": self.read_stats_summary(),
                "counts": {"backed_chunks": 0, "pending_chunks": 0,
                           "unknown_chunks": 0, "total_chunks": 0},
                "extents_missing_chunks": 0,
            }
            if req is None:
                return base
            states: Dict[KeyState, int] = {s: 0 for s in KeyState}
            unknown = 0
            backed: List[KeyRecord] = []
            pending_tokens = 0
            pending_chunks = 0
            for key in req.chunk_keys:
                record = self._keys.get(key)
                if record is None:
                    unknown += 1
                    continue
                states[record.state] += 1
                if record.state is KeyState.UNKNOWN or (
                    record.state is KeyState.NOT_SUBMITTED
                ):
                    unknown += 1
                    continue
                if record.state in BACKED_STATES:
                    backed.append(record)
                elif record.state is KeyState.PENDING_WRITE:
                    pending_chunks += 1
                    if record.token_start is not None and record.token_end is not None:
                        pending_tokens += record.token_end - record.token_start
            # Contiguous disk-backed prefix from token 0 only.  Overlapping
            # or duplicate identical objects are deduplicated by the cursor;
            # bytes prorated to the newly covered token span so nothing
            # inside the run is doubled.  First gap ends the run: anything
            # behind it is NOT a contiguous loaded prefix.  No decode-tail
            # inference from stored prompt chunks.
            prefix_tokens = 0
            prefix_bytes: Optional[int] = 0
            read_samples: List[Dict[str, Any]] = []
            write_samples: List[Dict[str, Any]] = []
            cursor = 0
            seen: List[int] = []
            for record in sorted(
                (r for r in backed if r.token_start is not None
                 and r.token_end is not None),
                key=lambda r: (r.token_start, r.token_end),
            ):
                if id(record) in seen:
                    continue
                seen.append(id(record))
                start = record.token_start
                end = record.token_end
                if end <= start:
                    continue
                if start > cursor:
                    break  # gap: stop the contiguous run here
                if end <= cursor:
                    continue  # fully covered already; no double counting
                if record.size_bytes is None:
                    # explicit unknown: bytes of the run are not fully known
                    prefix_bytes = None
                elif prefix_bytes is not None:
                    # GDS restores whole backing objects; count each
                    # distinct object's whole byte size once when it first
                    # extends the run.  Identical duplicate objects are
                    # deduplicated (end <= cursor skip above); there is no
                    # fractional token proration that would underprice the
                    # real transferred bytes.
                    prefix_bytes += record.size_bytes
                cursor = end
                if record.read_cost_s is not None:
                    read_samples.append(
                        {"duration_s": record.read_cost_s,
                         "bytes": record.read_bytes,
                         "token_start": start,
                         "token_end": end}
                    )
                if record.write_cost_s is not None:
                    write_samples.append(
                        {"duration_s": record.write_cost_s,
                         "bytes": record.size_bytes,
                         "token_start": start,
                         "token_end": end}
                    )
            prefix_tokens = cursor
            extents_missing = sum(
                1 for r in backed if r.token_start is None or r.token_end is None
            )
            # Missing extents are explicit unknown to the selector: a backed
            # chunk without token placement could be hiding a gap inside the
            # reported run.  coverage_known also requires an exact known
            # state for every linked chunk key.
            coverage_known = unknown == 0 and extents_missing == 0
            return {
                "req_id": req_id,
                "known": True,
                "request": req.as_dict(),
                "coverage_known": coverage_known,
                "prefix_tokens": prefix_tokens,
                "prefix_bytes": (
                    prefix_bytes if prefix_bytes is not None
                    and prefix_tokens > 0 else None
                ),
                "pending_tokens": pending_tokens,
                "read_cost_samples_s": read_samples,
                "write_cost_samples_s": write_samples,
                "shared_read": self.read_stats_summary(),
                "counts": {
                    "backed_chunks": states[KeyState.DISK_BACKED] + states[KeyState.PREEXISTING],
                    "pending_chunks": pending_chunks,
                    "unknown_chunks": unknown,
                    "total_chunks": len(req.chunk_keys),
                },
                "extents_missing_chunks": extents_missing,
            }

    def iter_request_records(self, req_id: str) -> Iterator[Dict[str, Any]]:
        with self._lock:
            req = self._requests.get(req_id)
            if req is None:
                return iter(())
            records = [self._keys.get(key) for key in list(req.chunk_keys)]
        for record in records:
            if record is not None:
                yield record.as_dict()

    def describe(self) -> Dict[str, Any]:
        with self._lock:
            counts: Dict[str, int] = {}
            for record in self._keys.values():
                counts[record.state.value] = counts.get(record.state.value, 0) + 1
            out: Dict[str, Any] = {
                "keys": len(self._keys),
                "requests": len(self._requests),
                "state_counts": counts,
                "overflow_drops": self._overflow_count,
                "capture_stats": dict(self._describe),
            }
            out.update(self.read_stats_summary())
            return out

    # ---- internals ----

    def _probe_locked(self, key: Any) -> Tuple[KeyState, Optional[str]]:
        for backend in self._probe_backends:
            contains = getattr(backend, "contains", None)
            if contains is None:
                continue
            try:
                hit = bool(contains(key))
            except Exception:
                continue
            if hit:
                return KeyState.PREEXISTING, "backend.contains"
        return KeyState.UNKNOWN, "no evidence"

    def _bound_check_keys_locked(self) -> None:
        if len(self._keys) <= _MAX_KEYS:
            return
        # Evict the oldest non-pending records; dropped records simply
        # re-probe as preexisting/unknown later.  Disk identity untouched.
        items = list(self._keys.items())
        drop = min(len(items) - _MAX_KEYS, len(items))
        for idx in _evictable_index_locked(items)[:drop]:
            self._keys.pop(items[idx][0], None)
        self._overflow_count += drop

    def _evict_requests_locked(self) -> None:
        if len(self._requests) < _MAX_REQUESTS:
            return
        droppable = sorted(
            (rec.created_t, rid)
            for rid, rec in self._requests.items()
            if rec.finished or rec.preempted
        )
        if not droppable:
            self._overflow_count += 1
            return
        self._requests.pop(droppable[0][1], None)
        self._overflow_count += 1


def _safe_size(memory_obj: Any) -> Optional[int]:
    if memory_obj is None:
        return None
    for getter in ("get_size", "get_physical_size"):
        fn = getattr(memory_obj, getter, None)
        if fn is None:
            continue
        try:
            return int(fn())
        except Exception:
            return None
    return None


class _StoreCtx:
    __slots__ = ("registry", "req_id", "store_seq", "extents", "_prev")

    def __init__(self, registry: BackingStateRegistry, req_id: Optional[str], store_seq: int) -> None:
        self.registry = registry
        self.req_id = req_id
        self.store_seq = store_seq
        self.extents: Dict[int, Tuple[int, int]] = {}
        self._prev: Optional["_StoreCtx"] = None

    def __enter__(self) -> "_StoreCtx":
        self._prev = getattr(self.registry._tl, "ctx", None)
        self.registry._tl.ctx = self
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        self.registry._tl.ctx = self._prev
        self._prev = None
        self.extents.clear()
        return False


def _clean_req_id(value: Any) -> Optional[str]:
    """Accept only non-empty string req_ids carried by real call sites."""
    return value if isinstance(value, str) and value else None


def _evictable_index_locked(items: List[Tuple[Any, KeyRecord]]) -> List[int]:
    """Indices of evictable (non-pending) records, oldest store_seq first.

    Sort key avoids comparing CacheEngineKey objects: store_seq is
    monotonic; the item index is only a deterministic tie-breaker.
    """
    order = [
        (rec.store_seq, idx)
        for idx, (_key, rec) in enumerate(items)
        if rec.state is not KeyState.PENDING_WRITE
    ]
    order.sort()
    return [idx for _seq, idx in order]


class BackingCapture:
    """Installs wrappers on real objects; uninstall restores exact originals."""

    def __init__(self, registry: BackingStateRegistry) -> None:
        self._registry = registry
        self._undo: List[Callable[[], None]] = []

    @property
    def installed(self) -> bool:
        return bool(self._undo)

    def wrap_cache_engine(self, engine: Any) -> "BackingCapture":
        if getattr(engine, _INSTALL_ATTR, False):
            return self
        orig_store = engine.store
        registry = self._registry

        def store(*args: Any, **kwargs: Any) -> Any:
            req_id = _clean_req_id(kwargs.get("req_id"))
            with registry.begin_store(req_id):
                return orig_store(*args, **kwargs)

        setattr(engine, _INSTALL_ATTR, True)
        engine.store = store
        self._undo.append(lambda: _restore(engine, "store", orig_store))
        self._registry._describe["store_wrapped"] = (
            self._registry._describe.get("store_wrapped", 0) + 1
        )

        # Retrieve capture: the outer CacheEngine.retrieve returns a boolean
        # CPU ret_mask (cache_engine.py:780-810), NOT chunks; it cannot
        # prove disk backing and carries no chunk tuples.  The processed
        # chunk tuples only exist as the (chunks, tot_kv_size) returns of
        # _process_tokens_internal / _async_process_tokens_internal
        # (cache_engine.py:74,:76,:842-854), both called with the retrieve
        # **kwargs that contain req_id.  Observe those actual returns.
        # Any other wrapper already installed on them (e.g. the async
        # consumed-event marker owned elsewhere) stays inside the call we
        # observe: we compose on the installed bound method and copy no
        # implementation.
        for name in ("_process_tokens_internal", "_async_process_tokens_internal"):
            orig_method = getattr(engine, name, None)
            if orig_method is None or getattr(orig_method, _INSTALL_ATTR, False):
                continue
            observe = self._make_retrieve_observer(name, orig_method)
            setattr(engine, name, observe)
            self._undo.append(lambda name=name, orig_method=orig_method: _restore(engine, name, orig_method))
            self._registry._describe[name + "_observed"] = (
                self._registry._describe.get(name + "_observed", 0) + 1
            )

        return self._bind_connector_wraps(engine)

    def _make_retrieve_observer(
        self, name: str, orig_method: Callable[..., Any]
    ) -> Callable[..., Any]:
        registry = self._registry

        def observe(
            tokens: Any, mask: Any, ret_mask: Any, *args: Any, **kwargs: Any
        ) -> Any:
            # Call the installed original (it updates ret_mask and returns
            # (List[(key, memory_obj, start, end)], tot_kv_size)); return
            # value and every refcount are passed through untouched.
            result = orig_method(tokens, mask, ret_mask, *args, **kwargs)
            try:
                if isinstance(result, tuple) and len(result) == 2:
                    chunks, _tot = result
                    if chunks:
                        registry.note_retrieve(
                            _clean_req_id(kwargs.get("req_id")), chunks
                        )
            except Exception:
                pass
            return result

        setattr(observe, _INSTALL_ATTR, True)
        return observe

    def _bind_connector_wraps(self, engine: Any) -> "BackingCapture":
        connector = getattr(engine, "gpu_connector", None)
        if connector is None or getattr(connector, _INSTALL_ATTR, False):
            return self
        orig_bfg = connector.batched_from_gpu
        registry = self._registry

        def batched_from_gpu(memory_objs: Any, starts: Any, ends: Any, **kwargs: Any) -> Any:
            ctx = registry.current_ctx()
            if ctx is None:
                req_id = _clean_req_id(kwargs.get("req_id"))
                if req_id is None:
                    registry.note_extents(ctx, memory_objs, starts, ends)
                    return orig_bfg(memory_objs, starts, ends, **kwargs)
                with registry.begin_store(req_id) as top_ctx:
                    registry.note_extents(top_ctx, memory_objs, starts, ends)
                    return orig_bfg(memory_objs, starts, ends, **kwargs)
            registry.note_extents(ctx, memory_objs, starts, ends)
            return orig_bfg(memory_objs, starts, ends, **kwargs)

        setattr(connector, _INSTALL_ATTR, True)
        connector.batched_from_gpu = batched_from_gpu
        self._undo.append(lambda: _restore(connector, "batched_from_gpu", orig_bfg))
        self._registry._describe["batched_from_gpu_wrapped"] = (
            self._registry._describe.get("batched_from_gpu_wrapped", 0) + 1
        )

        storage_manager = getattr(engine, "storage_manager", None)
        if storage_manager is not None:
            self._wrap_storage_manager(storage_manager)
        return self

    def _wrap_storage_manager(self, storage_manager: Any) -> None:
        if getattr(storage_manager, _INSTALL_ATTR, False):
            return
        orig_bp = storage_manager.batched_put
        registry = self._registry

        def batched_put(keys: Any, memory_objs: Any, *args: Any, **kwargs: Any) -> Any:
            # Recorded BEFORE submission so a completion callback fired
            # inside the original submission is the last writer for its key.
            ctx = registry.current_ctx()
            registry.note_batched_put(ctx, keys, memory_objs)
            return orig_bp(keys, memory_objs, *args, **kwargs)

        setattr(storage_manager, _INSTALL_ATTR, True)
        storage_manager.batched_put = batched_put
        self._undo.append(lambda: _restore(storage_manager, "batched_put", orig_bp))
        self._registry._describe["batched_put_wrapped"] = (
            self._registry._describe.get("batched_put_wrapped", 0) + 1
        )

    def wrap_gds_backend(self, backend: Any) -> "BackingCapture":
        if getattr(backend, _INSTALL_ATTR, False):
            return self
        orig_submit = backend.submit_put_task
        registry = self._registry

        def submit_put_task(key: Any, memory_obj: Any, *args: Any, **kwargs: Any) -> Any:
            # Existing disk key at submit: the installed backend stack may
            # skip the write for a key it already contains and never fire
            # the completion callback; the exact key object is probed
            # BEFORE submission so such records do not stay pending and
            # the association is recorded before any callback can fire.
            try:
                if bool(backend.contains(key)):
                    registry.note_preexisting_at_submit(key, memory_obj)
            except Exception:
                pass
            original_cb: Optional[Callable[[Any], None]] = None
            if args:
                original_cb = args[0]
                args = ()
            else:
                original_cb = kwargs.pop("on_complete_callback", None)

            def chained(k: Any) -> None:
                # Fires on successful GDS write completion only
                # (gds_backend.py:712-719).  Its interval is WRITE cost;
                # read costs are captured separately at _load_gds.
                try:
                    registry.note_submit_success(k)
                except Exception:
                    pass
                if original_cb is not None:
                    original_cb(k)

            try:
                if args:
                    return orig_submit(key, memory_obj, chained)
                return orig_submit(key, memory_obj, on_complete_callback=chained)
            except Exception:
                try:
                    registry.note_submit_error(key)
                except Exception:
                    pass
                raise

        setattr(backend, _INSTALL_ATTR, True)
        backend.submit_put_task = submit_put_task
        registry.bind_probe_backends_append(backend)
        self._undo.append(lambda: _restore(backend, "submit_put_task", orig_submit))
        self._registry._describe["submit_put_task_wrapped"] = (
            self._registry._describe.get("submit_put_task_wrapped", 0) + 1
        )

        # Read-cost capture at the actual successful GDS read seam.
        # _load_bytes_from_disk_with_memory(key, path, memory_obj) carries
        # the exact key and calls _load_gds synchronously inside its frame
        # (gds_backend.py:847-896), so a thread-local stashed there
        # correlates each _load_gds call to a key.  _load_gds returns the
        # byte count and -1 on failure (:1035-1102).  Composed wrappers
        # (e.g. the backend adapter's admission) remain inside the call we
        # time; duration is the installed path's real wall time.
        orig_keyed_load = getattr(backend, "_load_bytes_from_disk_with_memory", None)
        if orig_keyed_load is not None and not getattr(
            orig_keyed_load, _INSTALL_ATTR, False
        ):
            def keyed_load(key: Any, path: Any, memory_obj: Any, *a: Any, **kw: Any) -> Any:
                registry.set_current_read_key(key)
                try:
                    return orig_keyed_load(key, path, memory_obj, *a, **kw)
                finally:
                    registry.set_current_read_key(None)

            setattr(keyed_load, _INSTALL_ATTR, True)
            backend._load_bytes_from_disk_with_memory = keyed_load
            self._undo.append(
                lambda: _restore(backend, "_load_bytes_from_disk_with_memory", orig_keyed_load)
            )
            self._registry._describe["keyed_load_observed"] = (
                self._registry._describe.get("keyed_load_observed", 0) + 1
            )

        orig_load_gds = getattr(backend, "_load_gds", None)
        if orig_load_gds is not None and not getattr(orig_load_gds, _INSTALL_ATTR, False):
            def observed_load_gds(
                gds_path: Any,
                file_offset: Any,
                gpu_pointer: Any,
                size_in_bytes: Any,
                dev_offset: Any,
            ) -> Any:
                t0 = time.perf_counter()
                try:
                    ret = orig_load_gds(
                        gds_path, file_offset, gpu_pointer, size_in_bytes, dev_offset
                    )
                finally:
                    duration = time.perf_counter() - t0
                # ret == -1 is the backend's failure contract; no sample.
                # _load_gds returns the actual byte count read (gds_backend
                # .py:1049-1054,1092); short results are still completed
                # reads by the return contract and are recorded with their
                # ACTUAL ret bytes, never silently the requested size.
                if ret is not None and ret != -1:
                    try:
                        observed = int(ret) if ret > 0 else 0
                        registry.note_read_sample(
                            registry.current_read_key(),
                            duration,
                            observed,
                        )
                    except Exception:
                        pass
                return ret

            setattr(observed_load_gds, _INSTALL_ATTR, True)
            backend._load_gds = observed_load_gds
            self._undo.append(lambda: _restore(backend, "_load_gds", orig_load_gds))
            self._registry._describe["load_gds_observed"] = (
                self._registry._describe.get("load_gds_observed", 0) + 1
            )
        return self

    def unwrap(self) -> None:
        while self._undo:
            undo = self._undo.pop()
            try:
                undo()
            except Exception:
                pass


def _restore(obj: Any, name: str, original: Any) -> None:
    try:
        setattr(obj, name, original)
        if hasattr(obj, _INSTALL_ATTR):
            try:
                delattr(obj, _INSTALL_ATTR)
            except AttributeError:
                pass
    except Exception:
        pass


_REGISTRY_SINGLETON: Optional[BackingStateRegistry] = None
_CAPTURE_SINGLETON: Optional[BackingCapture] = None
_SINGLETON_LOCK = threading.Lock()


def registry() -> BackingStateRegistry:
    global _REGISTRY_SINGLETON
    with _SINGLETON_LOCK:
        if _REGISTRY_SINGLETON is None:
            _REGISTRY_SINGLETON = BackingStateRegistry()
        return _REGISTRY_SINGLETON


def enable(engine: Any = None, gds_backend: Any = None) -> BackingCapture:
    """Opt-in activation; pass the live CacheEngine and (optionally) the
    adapter-wrapped GdsBackend.  Call once, after the GDS policy adapter
    bootstrap so wrappers compose on top of it.

    Safe to call repeatedly: wrapping is idempotent per object (install
    markers), and registry()/wrapping never run while _SINGLETON_LOCK is
    held by this function (registry() acquires the same lock).
    """
    # registry() acquires _SINGLETON_LOCK; resolve it BEFORE taking the
    # lock here so this thread cannot dead-lock against itself.
    reg = registry()
    global _CAPTURE_SINGLETON
    with _SINGLETON_LOCK:
        if _CAPTURE_SINGLETON is None:
            _CAPTURE_SINGLETON = BackingCapture(reg)
        capture = _CAPTURE_SINGLETON
    if engine is not None:
        capture.wrap_cache_engine(engine)
    if gds_backend is not None:
        capture.wrap_gds_backend(gds_backend)
    return capture


def finish_request(req_id: str) -> None:
    registry().finish_request(req_id)


def mark_preempted(req_id: str) -> None:
    registry().mark_preempted(req_id)


def request_summary(req_id: str) -> Dict[str, Any]:
    return registry().request_summary(req_id)


def read_stats() -> Dict[str, Any]:
    """Shared actual-GDS-read samples/aggregate for candidate policy use.

    Aggregate fields from every completed read (sample_count,
    duration_sum_s, bytes_sum, mean_ns_per_kib or None when no sample,
    samples_dropped) plus the bounded recent ``samples`` list of
    {"duration_s", "bytes"} dicts.  WRITE submit->completion intervals are
    exported separately (KeyRecord.write_cost_s /
    request_summary["write_cost_samples_s"]) and never mixed in here.
    """
    return registry().read_stats()


def classify_keys(keys: Any) -> List[Dict[str, Any]]:
    return registry().classify_keys(keys)


def iter_request_records(req_id: str) -> Iterator[Dict[str, Any]]:
    return registry().iter_request_records(req_id)


def describe() -> Dict[str, Any]:
    return registry().describe()
