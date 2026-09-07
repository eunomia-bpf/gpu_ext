# GDS async prefetch adapter — handoff note (2026-09-07)

## Artifact

- Draft (TEMP, GLM-owned): `/tmp/opencode/startup-stage-timing/async-prefetch-draft.py`
- Installed by root in `ac97d02f`, including the existing sitecustomize entry:
  `workloads/lmcache-disk/gds-control/lmcache_gds_async_prefetch_adapter.py`
- Status: `py_compile` clean, pyflakes clean, imports under the current-venv
  (LMCache 0.5.4) with and without the opt-in env; `bootstrap_from_env()` with
  `LMCACHE_GDS_ASYNC_PREFETCH=1` passed its signature asserts against the
  installed LMCache and installed the chained `GdsBackend.__init__` hook.

## Wiring (serving)

1. Put `gds-control/` on PYTHONPATH and import
   `lmcache_gds_async_prefetch_adapter` BEFORE the engine constructs
   `GdsBackend` instances. Importing the module with the env set installs the
   class hook automatically; importing after construction requires an explicit
   `install_backend(backend)` per instance.
2. Envs (same hints across all comparison arms):
   - `LMCACHE_GDS_ASYNC_PREFETCH=1` — enables async seams (absent →
     process unchanged).
   - `LMCACHE_ENABLE_ASYNC_LOADING=True` — framework async lookup path.
   - `LMCACHE_GDS_POLICY_MODE=` `bpf` | `native` | `fifo` (fifo = eager
     prefetch arm; native default when unset).
   - Optional `LMCACHE_GDS_ASYNC_STAGING_BUDGET_BYTES` — caps staging below
     `gds_buffer_size` (default 256 MiB in this campaign).
3. A future server-side scheduler integration may call
   `adapter.mark_demand(key)` when a chunk becomes needed. The current HTTP
   runner supplies deadline hints only; it cannot call the backend object in
   the separate serving process. There is no live demand feed in this run.

## Deferred-path semantics (as fixed)

- `_start_read_locked` reuses another active handle or still-staged data, but
  a banked handle passed by `_deferred_read` transitions banked→reading and
  really starts; deferred reads unlink their banked bytes at that point.
- Wake events are consumed after observation; no hot re-decision spin.
- When the finite staging pool is full, deferred entries retry on 1 ms slots;
  speculative (non-demand) banked entries whose app-provided deadline has
  passed and still cannot stage are failed (`future → None`) instead of
  spinning forever.

## Known limitations (disclosed)

- Transfer-rate EWMA starts at zero; every policy sees eager submit until the
  first completed read supplies an estimate.
- `lmcache.prefetch_deadline_ns` is an app-provided host-monotonic hint, not a
  scheduler forecast. There is no live scheduler demand signal wired in; the
  only demand input is `mark_demand()` (optionally called) — absent here.
- The live allocator counter includes all allocated staging buffers,
  including writes and in-flight or unconsumed reads; it is not vLLM's KV
  residency counter.
- `batched_get_non_blocking` is a prefetch continuation, not demand.
- Registry purges run on subsequent lookup/continuation calls. A completed
  object that is never consumed can retain its allocation; purge only removes
  finished entries whose objects are absent, invalid or already released.
- Temporary fake-backend checks encountered fixture issues and were not
  completed. No further fake-test campaign was requested. Import/bootstrap
  success above does not establish disk I/O or serving performance; those
  measurements remain pending.
