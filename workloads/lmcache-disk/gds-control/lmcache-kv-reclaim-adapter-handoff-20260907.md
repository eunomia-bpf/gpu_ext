# Disk-aware KV reclaim serving adapter - handoff (2026-09-07)

Adapter for the disk-aware KV reclaim serving path: real LMCache disk
backing -> native/BPF victim decision -> upstream vLLM preemption +
recovery via the installed `Scheduler.preempt_victim_callback` seam.

Owner of this file: serving-adapter session (GLM). Runner
(`run_gds_kv_reclaim.py`) belongs to the runner session; the reclaim
ABI/native/BPF/driver belong to the Qwen27 session (binding commit
`5ad1fa25`); `lmcache_kv_backing_state.py` registry belongs to the
QwenNext session's implementer lineage.

## Artifacts

- `gds-control/lmcache_kv_reclaim_adapter.py` - the adapter.
- `gds-control/gds-control-bootstrap-sitecustomize.patch` - artifact of
  record for the six-line sitecustomize addition (already applied by
  root via apply_patch; verify with `patch -p1 -R --dry-run`).
- `gds-control/bootstrap/sitecustomize.py` - imports
  `lmcache_kv_reclaim_adapter.bootstrap_from_env` after the async
  bootstrap (root-applied; do not re-apply).

## Activation contract (fail-fast, no defaults invented)

Bootstrap runs at interpreter start via sitecustomize in every process
(API/frontend and EngineCore) when the runner exports:

- `LMCACHE_KV_RECLAIM=1` - master switch.
- `LMCACHE_KV_RECLAIM_MODE=native|bpf` - decider arm; bpf additionally
  requires `LMCACHE_KV_RECLAIM_UVM_DEVICE`.
- `LMCACHE_KV_RECLAIM_RECOMPUTE_NS_PER_TOKEN` - positive int measured
  proxy; the runner must pass the identical value to every campaign arm.
- `LMCACHE_KV_RECLAIM_DIAG_OUT` - optional diagnostics JSON path (see
  below).

Scheduler/connector hooks install lazily in the EngineCore process at
the first `Scheduler` construction; the frontend process installs
nothing and never writes diagnostics.

## Hook surfaces

1. `Scheduler.__init__` (class hook, post-original): requires the seam
   (`_select_preempt_victim`) and installs
   `scheduler.preempt_victim_callback = KvReclaimVictimPolicy.pick_victim`.
2. Scheduler connector impl: the actual binding for this installed
   target is `lmcache.integration.vllm.vllm_v1_adapter.LMCacheConnectorV1Impl`
   - `LMCacheConnectorV1.__init__` (installed vllm,
   `distributed/kv_transfer/kv_connector/v1/lmcache_connector.py:93-112`)
   reads `extra_config` `use_native` with default `False`, and the target's
   `kv_transfer_config` does not set it, so `connector._lmcache_engine` is
   that class. The `use_native=True` in-vllm variant
   (`vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.vllm_v1_adapter`)
   is unused and not importable against installed LMCache 0.5.4 (its
   `multi_process_adapter` imports `CudaIPCWrapper` from
   `lmcache.v1.multiprocess.custom_types`, which has no such symbol); it is
   not imported or hooked. The runtime impl check accepts only the
   lmcache-side module, resolved via `connector._lmcache_engine`; anything
   else fails fast.
3. Lookup client wraps (instance-level, call-site signatures):
   `lookup_cache(lookup_id)` and `lookup(token_ids, lookup_id,
   request_configs)` force a 0-hit only for pending FULL_RECOMPUTE
   routes; pending DISK_PREFIX routes pass through to the real lookup so
   the disk restore can be scheduled.
4. `impl.update_state_after_alloc(request, num_external_tokens)` -
   consumes the pending route after the original impl update.
5. `impl.request_finished(request, block_ids)` - drops cookie/pending
   route and calls `_backing.finish_request`.
6. `register_kv_caches` (class hook on the lmcache-side
   `LMCacheConnectorV1Impl` only) - after the real register +
   engine post-init, resolves the single GdsBackend from
   `engine.storage_manager.storage_backends` and calls
   `_backing.enable(engine, gds_backend)` once.

## Bootstrap import fix (applied 2026-09-07, post-handoff)

Every native/BPF arm failed at interpreter start:
`Error in sitecustomize; ImportError: cannot import name
'CudaIPCWrapper' from 'lmcache.v1.multiprocess.custom_types'`. Cause:
`bootstrap_from_env` eagerly imported the `use_native=True` in-vllm impl
module (`vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.vllm_v1_adapter`)
before installing hooks; that package's `__init__` imports
`multi_process_adapter`, which imports `CudaIPCWrapper` from
`lmcache.v1.multiprocess.custom_types` - absent in installed LMCache 0.5.4.
The runtime impl is never that module (`use_native` unset -> default False
-> lmcache-side impl), so the import was pure failure surface. Fix:
removed the eager import and class-hooked only
`lmcache.integration.vllm.vllm_v1_adapter.LMCacheConnectorV1Impl`; the
runtime impl check now accepts only that module. No stubs, shims,
vendored-library edits, error suppression, or new gates; the Scheduler
hook, backing activation, native/BPF selector, env contract, and all
policy logic are unchanged. All prior performance arms that used this
bootstrap are failed-bootstraps, not valid policy comparisons (all raw
retained, committed/pushed as correction `ae1d7e2f`). Post-fix check:
env-enabled import of the adapter reports `enabled=True` with no
`CudaIPCWrapper`/`NameError` (bootstrap only, not policy execution).

## Route semantics

- `ROUTE_STOCK (0)`: adapter returns None, stock victim kept.
- `ROUTE_FULL_RECOMPUTE (1)`: override pick; lookup bypass forces the
  impl's consistent zero-load resume (LoadSpec(0, 0, can_load=False)).
- `ROUTE_DISK_PREFIX (2)`: override pick; real lookup runs, vLLM
  schedules an external prefix load of the realized token count.
- Overrides respect the seam rule: candidates are only the stock
  victim's priority class (request.priority equality, lower = more
  important; clamped 0..MAX_PRIORITY - same-class, so no reorder risk).
- Degenerate decider status, wrong cookie, or ROUTE_STOCK falls back to
  the stock victim.

## Diagnostics contract (single JSON file)

Written by `write_diagnostics()` at atexit (and callable directly by the
runner). Rules:

- Path default `LMCACHE_KV_RECLAIM_DIAG_OUT`; pass `path=` to override.
- Diagnostic ownership: the file is written only when the process actually
  installed adapter evidence - a policy (`policies` non-empty) or a
  backing activation (`backing_activated`). This is unconditional on
  `enabled`: the frontend process (enabled but empty) never writes, so
  it cannot clobber the EngineCore's populated file if it exits later
  (UniProc target, shared DIAG_OUT path).
- Atomic write: `tmp<pid>` + `os.replace`.
- Content ~bounded: ≤512 decision/recovery records per policy (deque).

JSON schema (`policies[i]` fields exact):

- top level: `adapter`, `pid`, `t` (time_ns), `enabled`, `mode`,
  `recompute_ns_per_token`, `backing_activated`,
  `backing.read_stats`, `backing.describe`, `policies[]`.
- `policies[i]`: `kind=victim_policy`, `mode`,
  `recompute_ns_per_token`, `counters{}`, `warnings[]` (sorted, unique),
  `decisions[]`, `recovery[]`, `pending_routes{request_id->entry}`,
  `open_cookies` (int).
- counters: `seam_invocations`, `override_picks`, `stock_kept`,
  `unsupported_running_size`, `coverage_unknown_candidates`,
  `read_pricing_missing_decisions`, `forced_zero_lookups`,
  `route_consumed_full_recompute`, `route_consumed_disk_prefix`,
  `disk_route_recompute_fallback`, `requests_finished_dropped`.
- `decisions[i]`: `t`, `decision_seq`, `mode`, `n_candidates`,
  `stock_index`, `disk_read_ns_per_kib` (null until real GDS read
  samples exist), `read_sample_count`, `recompute_ns_per_token`,
  `candidates[{request_id, cookie, freeable_bytes, computed_tokens,
  disk_backed_tokens, disk_backed_bytes, priority, flags}]`, plus the
  decider result (`status`, `index`, `route`, `route_name`, `cookie`,
  `estimated_ns`) and `action`.
- `action` values: `stock_degenerate_input`, `stock`,
  `stock_wrong_cookie`, `override_disk_prefix`,
  `override_full_recompute`, `unhandled_route_<n>`.
- `recovery[i]`: `t`, `request_id`, `route`, `route_name`,
  `estimated_ns`, `decision_seq`, `num_external_tokens`, `outcome`.
- `outcome` values (admission stage at `update_state_after_alloc`, NOT
  transfer completion): `full_recompute_zero_admitted`,
  `full_recompute_zero_admitted_mismatch`, `disk_prefix_admitted`,
  `disk_prefix_zero_admitted`, `unknown_route`.

Evidence semantics for the runner's interpretation:

- `disk_prefix_admitted` means the scheduler granted an external-token
  load at admission; completed disk I/O is only evidenced by the backing
  registry's actual read stats (`backing.read_stats`: `sample_count`,
  `mean_ns_per_kib`; per-request `read_cost_samples_s` via
  `request_summary`), never by the recovery outcome alone.
- `disk_read_ns_per_kib` is null in every decision made before the
  first completed GDS read is observed; such decisions are counted in
  `read_pricing_missing_decisions` and the decider runs with pricing 0
  (the unknown disk rate receives the saturated high cost, not a free
  read; candidates without `FLAG_COVERAGE_KNOWN` have zero known-backed
  tokens).

## Registry fields consumed (read-only coordinates)

`_backing.enable(engine, gds_backend)` once from `register_kv_caches`;
per seam candidate: `request_summary(req_id)` -> `known`,
`coverage_known`, `prefix_tokens`, `prefix_bytes` (None if unknown);
pricing: `read_stats()` aggregate (`mean_ns_per_kib`, `sample_count`);
lifecycle: `mark_preempted(victim_req_id)` on override,
`finish_request(req_id)` on impl `request_finished`. Write intervals are
never mixed into read stats by the registry; per-object
`read_cost_samples_s` is ignored in favor of the aggregate.

## Byte accounting

`_derive_bytes_per_block` derives `sum(kv_cache_tensors.size) /
num_blocks` and requires exactly: one kv_cache group, every tensor
`shared_by` exactly one layer, `block_stride == 0` (non-packed), no
duplicate layer names, sum divisible by num_blocks. Anything else
raises at policy construction (no guessed bytes; never hardcodes a
bytes/token constant). `freeable_bytes` counts unique non-null pool
blocks with `ref_cnt == 1` over the victim request's block groups.

## Known limitations / unconnected seams

- Adapter does not touch the driver or BPF loader; attaching
  (`kv_reclaim_loader`, cmd83) stays root/Qwen27-side (BPF policy
  loaded unchanged per root).
- The route-window lookup bypass skips HitLimit/ChunkStatistics client
  accounting for forced zero lookups; short window (preempt -> resume
  schedule), documented trade-off.
- Full-recompute route relies on the impl's consistent zero-load
  behavior so `build_connector_meta`'s preempted-branch assert stays
  satisfied (spec exists with zeros).
- Single shared DIAG_OUT path (no per-PID files); safe for UniProc by
  the diagnostic ownership rule above. Missing diagnostics do not discard
  measured performance.
- `py_compile` + env-less import smoke only were run here; no fake
  tests, no GPU runs - the real performance runner is root's.

## Verification status

- `current-venv/bin/python -m py_compile gds-control/lmcache_kv_reclaim_adapter.py` OK.
- Patch artifact reverse-applies cleanly against the live sitecustomize
  (`patch -p1 -R --dry-run` rc=0), proving it matches the actual applied
  delta.
- After the bootstrap import fix: `py_compile` OK; the old in-vllm module
  import reproduces the `CudaIPCWrapper` ImportError against installed
  LMCache 0.5.4 while
  `lmcache.integration.vllm.vllm_v1_adapter.LMCacheConnectorV1Impl`
  imports cleanly; env-enabled adapter import reports `enabled=True`
  (bootstrap only, not policy execution).
