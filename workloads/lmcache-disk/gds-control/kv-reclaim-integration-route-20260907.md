# Linking LMCache completed disk writes to vLLM KV block reclaim: source route note

Date: 2026-09-07. Source-only inspection of installed serving sources under
`gpu_ext/workloads/lmcache-disk/current-venv/lib/python3.12/site-packages/`.
Read-only; no code changed, no GPU commands, no new framework. This note
records discovered extension points and a shortest next code change. It is a
partial note: unresolved ownership links are listed explicitly in the used
status labels below, not resolved by assertion.

Status labels: [S] source-checked today against the two installed packages;
[SN] source fact recorded in `disk-uvm-source-boundary-20260907.md` (section
7) or `async-prefetch-followup-20260907.md` (serving stack in measurement
today, files owned by other tasks); [U] unresolved: the link exists in data
but no code captures it.

## 1. Serving endpoint under measurement (what must be preserved)

Real serving stack (Qwen3-30B-A3B-FP8 arms, per follow-up note): vLLM 0.27.1
`LMCacheConnectorV1` (`kv_role=kv_both`) `->` LMCache 0.5.4 engine `->`
StorageManager `->` GDS backend wrapped by workload adapters `->` cuFile
compatibility mode to disk. Policy modes fifo/native/bpf enter at two existing
adapters:

- `gds-control/lmcache_gds_backend_adapter.py`: per-key admission before
  `_save_gds` / `_load_gds` [SN, boundary §3].
- `gds-control/lmcache_gds_async_prefetch_adapter.py`: ordered prefix lookup
  + async cuFile retrieval + native/BPF submit/defer for speculative reads;
  `mark_demand` not connected to a scheduler feed [SN, follow-up].
- Opt-in install via `gds-control/bootstrap/sitecustomize.py`, imports both
  adapters' `bootstrap_from_env` [S, bootstrap/sitecustomize.py:3,8].

The assigned endpoint keeps this LMCache GDS disk path. Assignment boundary
(root, 2026-09-07): the scheduler-side optional victim-selection/callback seam
in vLLM `scheduler.schedule` (~:591/:615/:617) is owned by the separate
implementation task with `gds-control/vllm-preemption-seam.patch` and its own
note; this note concentrates on the LMCache side — real backed-state/cost
inputs and the worker-completion-to-scheduler-request-ownership join — and
must not duplicate that patch's development. The vLLM in-tree
`OffloadingConnector`/`kv_offload` tiering system was inspected and is
recorded only as source context (section 6); it does not use LMCache and is
not the route.

## 2. The KV-ownership chain that already exists in the LMCache connector [S]

All line numbers in this section are
`lmcache/integration/vllm/vllm_v1_adapter.py` unless noted. The scheduler-side
adapter already tracks the serving engine's actual KV block ownership
per request, per step:

1. `RequestTracker.allocated_block_ids: list[int]` (:111-143, field :123,
   comment "the block ids that has been allocated so far") — one tracker per
   live request.
2. Initialized from `NewRequestData.block_ids` group 0
   (`from_new_request` :173-207), extended each step in `update()` (:267)
   from `scheduler_output.scheduled_cached_reqs[].new_block_ids`
   (`build_connector_meta` :1711-1718, :1751, :1834-1841).
3. Per step, `ReqMeta.from_request_tracker` turns the block ids into
   `slot_mapping = offsets + block_ids * block_size` (:401-408) and ships it
   to the worker inside `LMCacheConnectorMetadata` (:440-451).
4. Save: worker `wait_for_save` (:1106-1262) gathers KV for the chunk-aligned
   token range straight from the registered paged KV tensors
   (`register_kv_caches` :754-765) at those slots and calls
   `lmcache_engine.store(token_ids, mask, kvcaches, slot_mapping, offset,
   ...)` (:1229-1238). The disk write is therefore sourced from engine-owned
   blocks at store time; chunks are steward copies (CacheEngine allocates
   separate MemoryObjs, copies the selected GPU KV ranges with
   `batched_from_gpu`, then calls `StorageManager.batched_put`,
   cache_engine.py:488-568 [SN, boundary §7]). Two distinct completion
   events follow from this structure:
   - source-block copy completion — a conceptual distinction, NOT an
     established synchronous point: cache_engine allocates the separately
     allocated staging MemoryObjs and issues the gathers through
     `batched_from_gpu` (cache_engine.py:488-568 [SN, boundary §7]). In the
     connector actually selected for the vLLM paged path,
     `VLLMPagedMemGPUConnectorV3.from_gpu` enqueues the D2H kernels on
     `store_stream` and calls `self.store_stream.synchronize()` only when
     the staging target is NOT CUDA (gpu_connectors.py:645-649);
     `batched_from_gpu` loops without any sync (:660-662). This workload's
     GDS staging is CUDA, so the Python return of `store()` alone is not
     proof the GPU gather finished. The actual source-copy-completion
     point a reclaim integration may depend on is a CUDA event/stream
     dependency on `store_stream` (event recorded after the gather
     kernels and awaited on the staging-consuming/disk-write side, or an
     equivalent poll); whether such a dependency already exists between
     the gather and the GDS write for CUDA staging is not established by
     this inspection [U] and is the first fact to pin from the installed
     GDS submit path before any reclaim timing is used. Conceptually,
     once source-copy completion is established by such a dependency, the
     original serving KV blocks lose their writer role only — active
     kernels/requests still require scheduler preemption/finish before
     blocks actually free (§3).
   - transport-buffer disk completion: the async GDS write finishing
     (backend-owned MemoryObj reference; scheduler `batched_put` drops its
     own reference at submission, storage_manager.py:386-435 [SN]). This
     event governs staging-buffer reuse and disk-key lookup availability,
     not engine block lifetime.
5. Restore: scheduler `get_num_new_matched_tokens` looks up LMCache
   (:1362-1534, `lookup_client.lookup` :1431-1435 — async lookup + prefetch
   per follow-up note); `update_state_after_alloc` marks `can_load`
   (:1537-1612); worker `start_load_kv` copies retrieved objects into the
   engine paged buffer at the allocated slots
   (`retrieve(..., kvcaches, slot_mapping, ...)` :862-870). Failed/partial
   loads are reported as invalid block ids (:877-898, :1346-1349) and vLLM
   recomputes them.

Fact: both directions already touch real engine KV blocks, not merely
temporary staging. What does not exist is a record tying one to the other.

## 3. Engine block lifetime vs disk objects [S]

- Request finish: `request_finished` returns `(False, ...)` (:1857-1939,
  return at :1939), so vLLM frees the request's engine blocks immediately;
  the connector does not defer freeing. In contrast, the base connector API
  allows returning True to hold blocks until `get_finished()` reports save
  completion (vllm base.py:17-22, :559-577); the in-tree OffloadingConnector
  uses that option, LMCache does not. Worker `get_finished` returns
  `(None, None)` (:1341-1344). Assistant note: this means an async disk write
  never extends engine block lifetime; retention is staging-side only.
- Preemption/recompute: on preemption the tracker resets
  `allocated_block_ids` to the post-preemption allocation and restores
  `token_ids` from `all_token_ids` so chunk keys match (:247-265, preempted
  resume flag :1761-1798). Lookup is idempotent and re-runs for resumed
  requests (:1382-1388).roll back after failed retrieve is truncated
  (:1803-1828). LMCache chunk keys are content hashes of token prefixes, so
  preempt/recompute does not invalidate disk objects; the restart path
  reloads them via the section 2 restore path.
- Abort: `sm.cancel_request` and async-lookup cancel on
  `FINISHED_ABORTED` (:1876-1913).

Actual connector synchronization in the installed path (no disk-write barrier
on compute, and none may be added by the recorder):

- `wait_for_save` runs at forward-context exit after the step's kernels; its
  synchronous span covers staging allocation, D2H kernel enqueue on
  `store_stream`, and backend submission — NOT the GPU gather itself, which
  remains in flight on `store_stream` for CUDA staging targets (§2.4).
  `save_spec.skip_leading_tokens` is updated after the call (:1255-1262).
  Decode/prefill of subsequent steps does not wait for disk — and, for CUDA
  staging, the code path shown today also does not wait for the gather.
  Correct reclaim integration must not rely on Python store-return as a
  copy-completion signal.
- `request_finished` always returns False (:1939): vLLM frees the finishing
  request's engine blocks without waiting for any pending GDS write. Worker
  `get_finished` returns `(None, None)` (:1341-1344): no save completion
  deferral either. The only block-level feedback to vLLM is invalid block ids
  from failed loads (:1346-1349).
- Lookup pinning is balanced in the same function
  (`lookup_unpin` :1118-1124, :1136, :1149); store-side staging is held by
  the backend's own reference until write completion, not by engine blocks.
- HBM budget nuance: engine block free returns blocks to vLLM's block-pool
  allocator; the staging MemoryObjs remain resident in LMCache's GPU
  allocator budget until disk completion. Releasing the original is
  allocator reuse, not an automatic net GPU physical memory return; net HBM
  occupancy includes engine pool plus in-flight staging during writes (both
  campaigns' 256 MiB GDS staging budget is HBM-resident during writes [SN,
  follow-up invocation]).

Consequence: "reclaim" of engine KV is entirely vLLM's (block pool LRU and
request lifetime); it is invisible to, and unnecessary for, the LMCache disk
object store. The missing piece is not reclaim correctness — it is recording
engine-ownership state so the existing per-key BPF decision inputs can know
it (for example, which chunks were written from blocks a live request still
holds, or which chunks were restored recently into engine residency).

## 4. Unresolved links [U] — nothing below exists in code today

1. No record links a completed disk write to the identity that wrote it: the
   GDS save-completion event carries the CacheEngineKey only. The data to
   build the link is present at decision time — worker sees
   `metadata.requests[]` with `token_ids` + `slot_mapping` (assert-length
   pairing :1161-1171), chunk boundaries are arithmetic on token ranges, and
   the GDS completion callback exists (gds_backend.py:713-718 [SN, boundary
   §7] — it follows data-write completion and does not await the separately
   scheduled metadata-file task).
2. No direct callback capture at the allocation seam: the v1 API hands the
   allocated `KVCacheBlocks` to `update_state_after_alloc`, but the LMCache
   wrappers forward only `num_external_tokens` (lmcache_connector.py:281-287,
   lmcache_connector_v1.py:171-177), so there is no capture point for the
   restored block ids at that callback. This is not absolute absence: §2.2-2.3
   shows the same ids still reach the scheduler-side `RequestTracker`
   one step at a time via `scheduled_cached_reqs.new_block_ids`, so restore
   block ownership is learnable per step; what is missing is a
   direct allocation callback and any worker-side destination-block capture.
3. No hook observes vLLM's own eviction/reuse of prefix-cached blocks; the
   connector sees preemption and finish only.
4. GDS has no disk-object eviction (pin/unpin return False, `remove` raises
   NotImplementedError; allocate warns GDS eviction unsupported;
   gds_backend.py:1104-1178 [SN, boundary §7]). Disk-side retain/reclaim has
   no implementation to link into yet; today admitted disk objects simply
   persist.
5. Prefetch decisions have no live use-time signal (`mark_demand` pending
   [SN, follow-up]).

## 5. Next integrated behavioral change (opt-in, real serving) [proposed, no code exists]

The deliverable endpoint is not a recorder: it is a serving run in which
LMCache-backed request/KV lifecycle state (real lookups, store submissions,
completions, preemption resets, finishes) changes native/BPF decisions that
are observable in behavior. Any key->chunk->block-id stitching is an internal
implementation step feeding those decisions, not an experimental endpoint,
and validation uses the real serving stack, not synthetic metadata or fake
completions.

Control seams for actual behavior, source status:

1. Store admission [SN, boundary §3]: the backend adapter decides per key
   immediately before `_save_gds` / `_load_gds`, and the existing policy ABI
   records SUBMIT_NOW / DEFER (<= 10 ms) / RECOMPUTE. Lifecycle input
   (request still running vs finished; blocks still engine-resident vs freed
   at copy-completion) can change store/defer/admit decisions through this
   existing seam. This is the cheapest real behavioral change and is
   reclaim-adjacent: admission decides what ever reaches disk.
2. Prefetch submit/defer and in-flight adoption [SN, follow-up]: the
   async-prefetch adapter already has native/BPF submit/defer for
   speculative reads with the standing requirement that a demand read adopt
   an existing in-flight read. Connecting the real lookup call
   (`lookup_client.lookup` -> storage manager, vllm_v1_adapter.py:1431-1435
   [S]) as the demand signal that flips deferred speculative reads to
   submit-now is a concrete decision change at an existing seam.
3. Restore extent [S]: the hit count returned from `get_num_new_matched_tokens`
   (vllm_v1_adapter.py:1521-1534) is how much vLLM restores from LMCache vs
   recomputes; the lookup path is a real,_SCHED scheduler-visible control
   point, usable for restore/recompute tradeoff arms without touching
   transport.
4. Missing seam, stated plainly: disk-object retain/reclaim eviction has no
   control point to change — GDS implements none (pin/unpin False, `remove`
   raises; §4.4). A behavioral reclamation change would first require an
   opt-in eviction path inside the workload adapter stack; it is not a small
   existing seam and nothing here claims it.

Bounded next implementation task: in the two existing adapters (new opt-in
module layered under the established `bootstrap/sitecustomize.py` env gate;
no edits to vLLM/LMCache sources or files currently in measurement), feed the
lifecycle events of §2 into (1) store admission and (2) prefetch demand/
in-flight adoption, so that mode-native and mode-bpf decisions differ from
the all-submit/fifo baseline on the same requests. Validate with a real
serving comparison in the existing campaign shape — four-arm style on the
Qwen stack with identical budgets and hints (max-num-seqs=2, 768 MiB KV,
256 MiB GDS staging; follow-up §"First four-arm serving campaign invocation"
[SN]): current admission baseline vs lifecycle-informed native vs
lifecycle-informed BPF, same request arrivals, reporting TTFT and whole-
response throughput with disk bytes and staging occupancy as in existing
records. Real KV must be disk-populated and consumed; no fake
RequestTracker/completion objects, no fake schedules.

Constraints inherited from earlier sections: the lifecycle input never
imposes a disk-write barrier on compute — original engine blocks remain
reclaimable at source-block copy completion (§2.4, §3) — and staging-buffer
reuse stays governed by disk completion only.

Explicitly not established: automatic engine-block reclaim control, disk
object eviction, engine protection during async writes, transparent
same-address paging. Section 7 of `disk-uvm-source-boundary-20260907.md`
remains the boundary text for those.

## 6. vLLM-native tiering source context (not the route)

Recorded because it shows where vLLM itself formalizes "completed write -> key
loadable -> block retire" (`Medium.STORAGE`, `LookupResult.HIT_PENDING`,
`OffloadingManager.prepare_store/complete_store`, GPU block ids carried in
`GPULoadStoreSpec`, fs tier with hash-named files and O_DIRECT, out-of-tree
spec opt-in via `OffloadingSpecFactory` `spec_module_path`): see
`vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`,
`vllm/v1/kv_offload/base.py`, `vllm/v1/kv_offload/tiering/`. It replaces the
LMCache transport and is excluded by assignment; it is useful only as a
semantic reference for retain/reclaim wording, not as an adapter target.

## 7. Registry module handoff addendum (repaired 2026-09-07, integration pending)

`gds-control/lmcache_kv_backing_state.py` is the in-process backing-state
registry of this route.  After root's repair assignment the delivered file
contradicted an earlier "fixes done" handoff (startup deadlock, retrieve
parsing that captured nothing, write intervals exported as read costs, fake
backing classifications, a broken bounds routine); all of that is now
actually repaired in the file. Ordinary `py_compile` passes. Earlier stub
exercises are not engagement evidence. No serving run, GPU run or new
performance result is claimed. This
registry is preparatory metadata for native/BPF victim selection — NOT
completed automatic offload.

Fixed defects (as of 2026-09-07, in the file):

1. enable() no longer calls registry() while holding `_SINGLETON_LOCK`
   (non-reentrant; the registry() call re-acquired it and deadlocked
   startup).  Wrapping is idempotent per object via install markers.
2. Warm retrieve capture observes the REAL returns of
   `CacheEngine._process_tokens_internal` / `_async_process_tokens_internal`
   (cache_engine.py:74,:76,:842-854): `List[ProcessedChunk]` entries are
   actual `(CacheEngineKey, MemoryObj, start, end)` tuples, and
   `**kwargs` carries `req_id`.  The outer `retrieve` boolean CPU ret_mask
   is never parsed (it contains no chunks and proves nothing); its wrapper
   was removed.  Observers compose on the installed bound methods (the
   async-prefetch task's consumed-event wrapper stays inside the observed
   call; no implementation is copied) and pass returns/refcounts through.
3. Read cost is captured at the ACTUAL read seam, separate from writes:
   `_load_bytes_from_disk_with_memory(key, path, memory_obj)`
   (gds_backend.py:847-896) correlates the exact key thread-locally;
   `_load_gds(...) -> int` (:1035-1102) is timed with perf_counter and
   sampled ONLY on completed reads (return != -1), recording the ACTUAL
   returned byte count (short results are not silently converted to the
   requested size).  Exported as shared stats `read_stats()`
   (sample_count, duration_sum_s, bytes_sum, mean_ns_per_kib — None until
   a sample exists, samples_dropped, bounded recent samples list) and, per
   request, `read_cost_samples_s` entries {"duration_s","bytes",
   "token_start","token_end"} for records inside the prefix run.  WRITE
   submit->completion intervals (the completion callbacks) are now
   `KeyRecord.write_cost_s` / `request_summary["write_cost_samples_s"]` and
   are never mixed into read stats; no fabricated read estimate exists.
4. Backed classification uses ONLY: recorded successful GDS write
   completion callbacks, or backend.contains() with the existing exact
   CacheEngineKey (probed at submit time, at retrieve association, and via
   classify_keys).  A consumed CPU-cache chunk is not disk evidence;
   pending/in-progress writes stay pending until one of those two
   evidence sources resolves.  The submit-time contains() probe prevents
   keys whose store was skipped for an existing disk copy (no callback
   will fire) from lingering pending forever.
5. `_bound_check_keys_locked` eviction no longer uses the dead `pending`
   comprehension and no longer sorts tuples that could compare
   unorderable CacheEngineKey objects (store_seq + item-index tie-break).
5b. request_summary: contiguous DISK_BACKED/PREEXISTING token run from 0
   only, deduplicated, stopped at the first gap; `prefix_bytes` counts
   WHOLE backing-object byte sizes once each when first extending the run
   (identical duplicates never double-count, no fractional token
   proration — GDS restores whole objects), and is None when any run
   object's size is missing or the run is empty; `coverage_known` is False
   when any linked chunk key is unknown/not_submitted OR any backed chunk
   misses token extents (explicit unknown); `extents_missing_chunks` is
   reported.  No decode-tail inference from cached prompt chunks.
6. All prior contract guarantees kept: association recorded before
   submission callbacks can fire; no GPU-gather claim from Python store
   return (CUDA staging path is asynchronous, gpu_connectors.py:645-649);
   backed state never reset to pending; request cleanup drops request
   links only; registry never releases/pins engine KV blocks.

Module contract for the serving adapter session (root reconciles APIs) —
call `enable(engine=<UUIDCacheEngine>, gds_backend=<adapter-wrapped
GdsBackend>)` once after the adapter bootstrap; then per request consume:

- `request_summary(req_id)`: exact fields documented in the module
  docstring — `known`, `request`, `coverage_known`, `prefix_tokens`,
  `prefix_bytes` (None = bytes deliberately unknown, e.g. sizes missing
  inside the run), `pending_tokens`, `read_cost_samples_s`,
  `write_cost_samples_s`, `shared_read`, `counts`, `extents_missing_chunks`.
  Treat `coverage_known=False` or `prefix_tokens=0` as not disk-backed.
- `read_stats()`: shared actual-read ns/KiB aggregate or duration+bytes
  samples — `mean_ns_per_kib` is None while no completed read has been
  observed: no fabricated read estimate.
- `finish_request(req_id)` on finish; `mark_preempted(req_id)` on the
  preemption seam; `classify_keys(keys)` for explicit-unknown on arbitrary
  exact keys.  `read_cost_s`/`read_bytes`/`read_count`/`write_cost_s` are
  per-key record fields via `classify_keys`/`iter_request_records`.

Freeable blocks/bytes and computed_tokens are NOT in this registry: they
come from live scheduler/KVCacheManager block refs at the scheduling seam.

Still true and unchanged (root, continuation; supersedes the numbered
list that was here before the repair):

1. Root integrates wiring: `enable(...)` after
   `lmcache_gds_backend_adapter` bootstrap; wrappers compose if the
   adapter already wrapped the same methods (registered before enable()).
2. Scheduler-side consumer (Qwen seam): merge `request_summary` fields
   with live KVCacheManager freeable blocks/bytes and computed_tokens.
3. Lifecycle hooks: `mark_preempted` on preemption, `finish_request` on
   request finished.
4. `enable()` is still not wired into bootstrap/sitecustomize.py by
   design; no serving run, policy rule, or victim-selection integration
   has happened yet — the endpoint remains real disk-backed victim
   selection measured against baseline serving.  The serving adapter
   session (ses_f82951573ffeR4a731y9xddsJU) owns the vLLM-side adapter and
   consumes the exact fields above; root reconciles the seam.
