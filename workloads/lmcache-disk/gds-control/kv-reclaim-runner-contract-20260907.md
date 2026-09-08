# LMCache disk-aware KV reclaim runner contract (2026-09-07)

Owner: this runner (`workloads/lmcache-disk/run_gds_kv_reclaim.py`) and this
note only.  Calibration lifecycle is owned by
`workloads/lmcache-disk/kv_reclaim_calibration.py` (separate session); reclaim adapter +
diagnostics payload by the serving-adapter session; selector/binding/loader by
the KvReclaim session; root drives GPU, locks, and commits.

## Experiment shape (real serving only)

- 3 arms, same GDS/cuFile demand transport and same FIFO per-IO policy in
  every arm: `stock` (reclaim opt-in off, stock vLLM victim), `native`
  (`LMCACHE_KV_RECLAIM_MODE=native`), `bpf` (identical policy algorithm via
  the kernel decider).  5 rotated blocks x 3 arms = 15 cells, fresh server
  process per cell, no retries, no result filtering, no desirable-ratio
  tuning.  Failed cells and missing diagnostics are preserved as records and
  never discard performance.
- Capacity: `--kv-cache-memory-bytes 402653184` (384 MiB), `--max-num-seqs 2`
  (argv from `lmcache_primitives.server_argv` keeps `--max-model-len 4096`
  and `--no-enable-prefix-caching`).  Actual KV pool is derived from the
  server log (`GPU KV cache size:` line scan), never assumed.
- Fixed prompts from `prompts.json`.  Warm prompts are truncations of the
  existing real warm token arrays: even prefix index = full 1536-token warm
  prefix, odd prefix index = 1024-token truncation; generation up to
  `--warm-output-tokens 1024`, identical stop config in every arm, actual
  generated token counts recorded from usage (never assumed 1024).  With two
  running sequences each reaching 1024 generated tokens, a 1536-token
  prompt paired with a 1024-token prompt needs 4608 tokens; two 1536-token
  prompts need 5120. These exceed the 4096-token pool observed in calibration,
  whereas two 1024-token prompts nominally fit. Bounded concurrent burst: 4 workers, 250 ms
  stagger; arrival order rotates by block and is identical across arms within
  a block.  Cold population: 8 sequential real disk writes (16 output tokens)
  with the existing non-gating store barrier; no async-prefetch opt-ins and
  no prefetch deadlines anywhere.

## Environment supplied to each cell server (subprocess env only)

Common (all arms): the `lmcache_disk` base env from
`lmcache_primitives.server_environment` with `LMCACHE_LOCAL_DISK` removed and
`PYTHONPATH` prepended by `gds-control/bootstrap` + `gds-control`, then
`LMCACHE_GDS_PATH=<cell cache dir>`, `LMCACHE_GDS_BUFFER_SIZE=256`,
`LMCACHE_USE_GDS=True`, `LMCACHE_GDS_BACKEND=cufile`,
`LMCACHE_GDS_POLICY_MODE=fifo`, `LMCACHE_EXTRA_CONFIG={"use_direct_io":
true}`.  Reclaim policy is configured independently of per-IO admission; the
admission env is byte-identical in all arms.

- `stock`: nothing extra (all `LMCACHE_KV_RECLAIM*` keys popped).
- `native`: `LMCACHE_KV_RECLAIM=1`, `LMCACHE_KV_RECLAIM_MODE=native`,
  `LMCACHE_KV_RECLAIM_RECOMPUTE_NS_PER_TOKEN=<measured>`,
  `LMCACHE_KV_RECLAIM_DIAG_OUT=<run_dir>/kv-reclaim-diagnostics.json`.
- `bpf`: same as native with mode `bpf` plus
  `LMCACHE_KV_RECLAIM_UVM_DEVICE=/dev/nvidia-uvm`.

The runner never imports `lmcache_kv_reclaim_adapter`, `kv_reclaim_binding`,
or vLLM; hooks engage inside the server process via the bootstrap
`sitecustomize` import.

## Calibration (one real phase, before any cell)

- Default: import `run_calibration` and call
  `run_calibration(run_dir=<root>/calibration, model_path, port,
  token_arrays=[warm 1536/1024 arrays], expected_driver,
  kv_cache_memory_bytes)`; the returned record
  (`recompute_ns_per_token`, `error`, `ready`, `requests`,
  `server_returncode`) is embedded verbatim in `campaign.json`.
- `--calibration-result PATH`: reuse an existing raw `calibration.json`
  (e.g. raw/kv-reclaim-recompute-calibration-575-20260907-01/calibration.json)
  without rerunning; the source path is kept in the campaign record.  No
  hash/check gate is applied to it.
- The price is the positive measured median; if missing, non-positive, the
  helper fails, or the file is unreadable, the campaign exits 2 before any
  cell with the real error retained.  A dummy price is never served to a cell
  and a launched server's env is never changed later.
- Semantics label carried in every cell record: end-to-end recompute proxy
  (TTFT per prompt token from real baseline serving), explicitly not pure
  GPU compute; one estimate shared identically by all 15 cells.

## Adapter diagnostics contract

- Per reclaim cell `LMCACHE_KV_RECLAIM_DIAG_OUT` points inside the cell run
  dir; after teardown the runner polls bounded (10 s) and reads the exit dump
  (`diagnostics_payload` JSON) verbatim into `adapter_diagnostics.payload`:
  `policies[].counters/decisions/recovery/pending_routes/open_cookies`,
  `backing.read_stats`, `backing.describe`, `backing_activated`.  Disk read
  rate evidence comes exclusively from `backing.read_stats` (registry
  observed reads inside serving); the runner computes no read-rate constant.
  The stock cell expects no dump; if one appears it is recorded.
- Requested from the adapter session here: keep the exit dump writing on
  clean EngineCore shutdown and keep `counters`, `decisions` (with
  `action`/`victim_request_id`), `recovery` (with `route`, `outcome`,
  `num_external_tokens`) populated even when empty lists, so the runner can
  distinguish "diagnostics arrived, zero reclaim events" from "diagnostics
  missing".  Missing/unreadable dumps are reported (`adapter_diagnostics`
  fields `arrived`, `error`, `waited_s`) and never hide numbers.

## Derived capacity logging

`kv_pool_log` records the raw log-scan results for `GPU KV cache size`,
`# GPU blocks`, `Maximum concurrency ...`, `KV cache memory: ... GiB` patterns
(`found`, typed `value`, all matches).  If vLLM phrasing changes, entries
report `found=false`; nothing is assumed or invented.  Preemption log lines
are counted with bounded samples, informational only.

## Explicit non-claims

No claim of automatic offload being complete; no claim of physical HBM
release from fixed-pool block return; no claim of hardware NVMe-GPU P2P (the
demand transport is the existing GDS/cuFile compatibility path).  Same-policy
bpf/native numbers separate mechanism from policy.

## Resume (`--resume`, only with an existing `--output` root)

- `--resume` without `--output` is a CLI error; without `--resume` a fresh run keeps refusing an existing root (`exist_ok=False`), so an old root is never reused implicitly. The plan is the current CLI plan (rotated block orders, per-block warm orders); cell directories are classified only under their planned names `block-XX/position-<i>-<arm>/`, so cells recorded under other names (for example older manual stock-sync/native-sync/bpf-sync directories) sit outside the plan and would run as missing planned cells: do not resume such roots whole. Each planned cell directory is classified from its own files only, via plain value equality; no hash/digest is read or recorded.
  - `result.json` present and matching `kind`, `schema`, `arm`, `block`, `position`, and ordinary knobs (`expected_driver_parameter`, `max_num_seqs`, `kv_cache_memory_bytes`, `gds_buffer_size_mib`, `warm_output_tokens_bound`, `warm_stagger_ms`, `warm_concurrency`, `gds_policy_mode`, `prompt_count`, `warm_prefix_tokens`, `warm_order`, `recompute_ns_per_token`): the record is reused verbatim through the existing `campaign["cells"]` + `median_summary` path and the cell is never rerun. Failed completed cells (e.g. `ready_error`, warm timeouts) are reused the same way: no retry, no filter.
  - A present but incompatible record is not adopted; its directory is left untouched and the mismatch fields are listed (`incompatible_fields` under `campaign["resume"]`).
  - A nonempty directory without `result.json` (live or interrupted cell) is left untouched and reported unfinished; no age, timeout, or liveness assumption is made.
- Adopted records and preserved completed failed attempts are appended to `raw.jsonl` once per (block, position, arm); existing lines are retained verbatim.  `campaign.json`/`summary.json` are rebuilt from the reused plus newly run records in the same planned order; per-cell directories and their files (result.json, server.log, cache/, diagnostics) are never rewritten.  The existing `campaign.json`, if any, is loaded before anything is written and stays untouched while classification runs: an unreadable or non-matching campaign.json aborts resume with the file preserved; on success the original timestamp is kept and old params are retained under `campaign["resume"]["previous_params"]`.
- Calibration on resume: an explicit `--calibration-result` always wins; without it an existing measured calibration is reused, in preference order `<root>/calibration/calibration.json`, then the previously recorded `reused_from` path, then an embedded prior calibration record with a positive measured price; if none of these exists, the one real calibration runs as in a fresh campaign.  The same measured price backs the per-record compatibility check.
- Records carrying the matching campaign identity and an actual `error` field but lacking the full knob fields (top-level exception results) are preserved verbatim as completed failed attempts under `campaign["resume"]["completed_failures"]`, pass through the existing cells/summary path as measured-failed (not compatible measured) cells, and are never rerun; any other sparse record stays untouched as unfinished.  For records whose server never reached the reclaim hooks, arm labels remain requested configurations, not executed policy comparisons.  Exit codes are unchanged; unfinished or not-adopted cells leave the campaign incomplete (exit 2), deferred stop still exits 3 between cells.

## Known unconnected seams (exact, API-level)

1. Diagnostics exit dump only survives clean EngineCore atexit; SIGKILL or
   teardown races lose it (runner records the absence; no alternate
   collection channel exists in the adapter).
2. Backing read stats populate only when
   `LMCacheConnectorV1Impl.register_kv_caches` post-activation bound the
   registry to the single `GdsBackend`; adapter fail-fast during scheduler
   init surfaces as `ready_error`, and `read_pricing_missing_decisions`
   counts decisions made with `mean_ns_per_kib=None` (decider receives 0
   read price; no runner-side constant is injected).
3. Recovery-route consumption evidence (`num_external_tokens`) exists only in
   adapter `recovery` records; the runner does not parse scheduler internals.
4. `--calibration-result` reuse assumes `kv_reclaim_calibration`'s raw JSON
   schema (`recompute_ns_per_token` positive); any other content aborts
   before cells with the real error.
5. Warm-burst HTTP loop is a minimal local variant only because the reused
   cold helper hardcodes the 16-token cold output setting; stop semantics
   are identical across arms and recorded per request.
6. The runner takes no file locks (root holds
   `/tmp/gpubpf-revision-gpu0.lock` and `/tmp/gpubpf-revision-struct-ops.lock`)
   and imposes no campaign wall-clock timeout; per-request urllib timeout
   (600 s) and the non-gating barrier timeout match the existing helpers.

## Output layout per campaign root

`campaign.json`, `summary.json`, `raw.jsonl` (one complete record per cell,
fsync'd after each), `block-XX/position-<i>-<arm>/{result.json, server.log,
cache/, kv-reclaim-diagnostics.json}`; calibration raw under `calibration/`
or the reused source path.  Exit codes: 0 all cells measured, 2 NOT STARTED /
incomplete or calibration failure, 3 deferred-stop early exit.
