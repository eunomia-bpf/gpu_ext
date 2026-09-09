# Native constant-bank expansion: builds and launches, resume still fails

2026-09-09, RTX 5090 / CUDA 12.9 / NVIDIA 575.57.08. This is an
unfinished original-cuXtra execution attempt, not a performance result.
It does not replace the completed 15-cell shared-NVBit-actuator comparison.

The local implementation grows each extended kernel's `.nv.constant0`
section and relocates subsequent ELF/fatbin data. Root found and the
implementation task repaired the multi-kernel tail-span update. Previous
original-argument layouts and resolver changes remain. The portable change
is in `../../level2/native/xsched-native-meta-extend.patch`.

`build.log` records the first compile failure: unused `put16` and `put32`
helpers rejected by existing `-Werror`. Root removed only those two unused
functions, preserving compiler options. `build-fixed.log` records successful
compilation and installation into the isolated native HAL at 09:31:23 PDT.
The two source snapshots retain both versions. No system driver was replaced.

The first launcher attempt (`run.log`, `run-uv-default-cache.sh`) failed
before starting workers because the default uv cache was inaccessible.
`run.sh` uses a new temporary uv cache instead; its actual output is
`run-isolated-cache.log`. Both build and execution held the GPU and
struct-ops leases. The six-process workload retained four streams per
process, 50 tasks per stream, 340 blocks, 256 threads and 9,511,106 iterations.

## Actual execution

- All six workers log two constant-bank expansions, to `0x18a0`, and a
  total container growth of `0x2a20`.
- Both LC workers exit zero and emit 200 service records each. Each BE
  log retains 27 `launch ret=0` lines; these are accepted launch calls,
  not 27 proven completed kernels.
- The old first-launch 701 rejection is no longer observed. The BE logs
  show restore/resume (`type=2`) launches returning zero, followed by
  CUDA 700 reported at `CudaCommand::Synchronize` / `EventSynchronize`.
  All four BE workers exit with SIGSEGV during this failed execution.
- Three reader threads also hit UnicodeDecodeError on diagnostic bytes,
  so some later stderr is missing. This is a logging limitation, not an
  explanation of the GPU illegal access. No missing lines are reconstructed.
- The runner exits 1 without a completed cell result. Worker JSON,
  failure JSON and protocol remain under `cells/`.

The evidence moves the immediate problem from launch admission to execution
after resume. It does not identify the exact failing GPU instruction or
prove every other launch-buffer field correct. The next implementation
investigates the restore/resume path, rather than repeating the old resolver
or first-launch diagnostics. No partial LC timing is promoted to a matched
performance comparison. After cleanup, no worker/xserver remained and the
GPU reported 0% utilization with 1 MiB used; no driver recovery was needed.
