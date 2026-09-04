# Matched NVBit observability adapters

These adapters implement the three RQ4 observability tasks on the official
NVBit 1.8 core. They deliberately instrument only the exact mangled kernel in
`OBS_TARGET_SYMBOL` and do not enable related functions.

- `kernelretsnoop`: inject before every `EXIT`, check its execution predicate,
  emit one 32-byte `(global_x, global_y, global_z, timestamp)` record per
  actually exiting logical thread through NVBit's device-to-host channel, and
  validate nonzero `%globaltimer` timestamps, exact coordinate extent and
  multiplicities, selected launches, and complete channel framing on the host.
  These global coordinates deliberately match the compact gpubpf observable;
  they do not preserve the original CUDA block/thread decomposition.
- `threadhist`: inject before every `EXIT`, check its execution predicate, increment the full
  configured logical-thread array, and report its nonzero entries and total at
  context termination. Per-thread increments match gpubpf's non-atomic semantics.
- `launchlate`: bracket real `%globaltimer` reads with `CLOCK_MONOTONIC` before
  and after observation. Each selected host callback reserves one bounded
  managed record and passes its pointer through NVBit's per-launch argument;
  block/thread zero writes the matching GPU-entry timestamp. After device
  synchronization, the host interpolates the conservative offset interval
  between both calibration anchors and builds the authoritative histogram from
  those retained raw pairs. Capacity exhaustion, a missing device entry, or a
  wholly negative latency interval is a clock error; only an interval that
  overlaps zero or a bin boundary is uncertain. The validity gate requires
  exact selected/classified/uncertain/error accounting, no clock errors, at
  most 10% uncertainty, and endpoint drift no greater than 10,000 ppb. This is
  measured across an unconditional minimum one-second anchor span so that a
  sub-second workload does not turn the microsecond-scale endpoint bracket
  width into an unresolvable drift-rate bound. The longer span does not change
  the 10,000 ppb limit, histogram bins, or 10% uncertainty gate, and it does not
  guarantee that a genuinely drifting clock will pass. This is the closest
  native NVBit counterpart to gpubpf's exact host-stub uprobe plus
  device-entry probe; the different host hook locations remain explicit in the
  experiment plan.

Build against the pinned release extracted under `revision-rq4/deps`:

```bash
make -C nvbit_adapters/observability \
  CXX=g++ \
  NVBIT_ROOT="$PWD/deps/nvbit_release_x86_64" \
  ARCH=sm_120
```

The calibration arithmetic and fail-closed classification have a CPU-only
test: `make -C nvbit_adapters/observability CXX=g++ test`.

The revision runner sets `LD_PRELOAD`, `OBS_MODE`, `OBS_TARGET_SYMBOL`, and
`OBS_GPU_THREAD_COUNT`. A run is invalid unless the selected kernel launches
and the task-specific sample checks pass.

For code-path diagnosis, `OBS_TRACE_LAUNCHES=1` reports up to 256 distinct
mangled launch symbols and `OBS_TRACE_TARGET_FAMILY=1` reports distinct
`rope_norm` variants. These switches do not relax the exact-symbol engagement
gate and are disabled in experiment runs.

The 2026-09-03 [predicate repair](../nvbit-exit-predicate-repair.md) fixes the
custom adapter's previously omitted guard argument. Its sm_120 build passes;
new GPU measurements are pending. The old preflight's 901,120 NVBit events
must not be treated as an independently established correct exit count.

On driver 610.43.02, a pp=32 diagnostic loaded this adapter and completed the
real RTX 5090 llama.cpp workload, but received no launch callback. The official
NVBit 1.8 `instr_count` tool behaved the same way: initialization ran, while no
kernel record or termination total appeared. This isolates the current failure
to the unsupported NVBit/runtime path rather than this adapter's symbol filter.
It is diagnostic evidence only; the paper-facing comparison still requires the
documented 575.x stack.
