# LMCache GDS mixed runner: write-io-workers implementation

One bounded follow-up for actual LMCache storage performance. Only
`workloads/lmcache-disk/run_gds_mixed_backend.py` changed; no vendored
LMCache, kernel, adapter, timing, policy, or driver change.

## What changed

- New opt-in `--write-io-workers N` (int, default 0). Validated
  non-negative like the other numeric options; no new gates or preflight.
- `start_event_loop(write_io_workers)`: with `N > 0` it installs
  `ThreadPoolExecutor(max_workers=N, thread_name_prefix="gds-mixed-write-io")`
  as the private loop's default executor via the supported
  `loop.set_default_executor(...)` before the backend is constructed or any
  scanning happens. `N = 0` leaves the loop's default executor exactly as
  before.
- `stop_event_loop(loop, thread, default_executor)`: when a custom default
  executor was installed, it is shut down first with the ordinary supported
  `loop.shutdown_default_executor()`, after the runner has already reaped
  every write future and joined every reader thread (pending operations
  finished). No new arbitrary timeout and no abandoning shutdown; then the
  existing `loop.stop()` + join. With `N = 0` this path is unchanged.
- Propagation: the value is passed verbatim into the fresh `--single-cell`
  child argv and recorded in the cell `result.json`, the campaign `params`,
  and the dry-run plan (with a scope label).

## Scope (named)

This limits loop-default-executor I/O work only: the worker threads the
private loop uses for every job dispatched without an explicit executor, in
particular the installed LMCache `GdsBackend._async_save_bytes_to_disk`
(`await asyncio.to_thread(self._save_gds, ...)`) and any other LMCache
`to_thread` jobs on that loop. It is not GPU driver scheduling, not a write
deferral policy, and not a new BPF algorithm; the FIFO/native/BPF decision
code, policy inputs, read arrival schedule, timing boundaries, and defaults
are all preserved.

## Checked (no GPU)

- `python3 -m py_compile` clean.
- `--help` / `--dry-run` expose the option and record it in metadata;
  negative values are rejected (exit 2) like the other validations.
- The `--single-cell` child argv carries `--write-io-workers <N>`.
- An exploratory in-process probe failed in its test script: an asyncio
  Future was called with the unsupported `result(timeout=5)` argument.
  It did not validate executor shutdown. Root is proceeding to the planned
  real storage campaign, not adding another synthetic check or gate.

## Usage

```
python3 workloads/lmcache-disk/run_gds_mixed_backend.py \
    --write-io-workers 4 --output ...
```

Default (no flag, `--write-io-workers 0`) is the control.

## Next (root)

Root runs the same real mixed-storage benchmark, default (0) vs 4 workers,
equally for FIFO/native/BPF, 200 ms budget, with the current reuse
optimization consistently enabled for BPF, and publishes the comparison.
Existing completed studies are not rerun.
