# Public TSC/PTIMER launch-clock recovery plan

Date: 2026-09-04  
Role: supporting gate for the RTX 5090 `launchlate` Table 1 row

## Admission and question

The retained `launchlate` run is invalid because its host launch timestamp and
device `%globaltimer` timestamp could not be related precisely enough.  The
versioned endpoint command repaired precision, but the currently loaded stock
575.57.08 module does not implement it, and changing modules is outside this
safe follow-up.  The strongest remaining source-native path is the stock
`NV2080_CTRL_CMD_TIMER_GET_GPU_CPU_TIME_CORRELATION_INFO` command with
`cpuClkId=TSC`: the 575 source pairs an x86 TSC midpoint with PTIMER and the
public ABI permits up to 16 returned pairs.

Hypothesis: on this single-socket host, with the process pinned to one CPU, the
stock RM TSC/PTIMER command yields monotonic pairs with a stable affine rate and
conservative per-sample uncertainty below 1.5 us; a separate ordered
RM--CUDA-`%globaltimer`--RM control confirms PTIMER/globaltimer identity.  If
both gates pass, a bpftime-private TSC helper can timestamp the existing exact
launch hook in that same calibrated domain without changing the NVIDIA module.

This adds evidence beyond the failed RAW midpoint control because it tests a
different documented RM clock selector and permits the host callback to use
the exact clock returned by RM.  A pass admits a repaired launchlate preflight;
a precision, rate, identity, or accounting failure closes this safe path and
leaves the row omitted.  It does not change histogram bins or reinterpret old
raw data.

## Controls and gates

1. Confirm Linux uses `tsc` as its clocksource and `/proc/cpuinfo` advertises
   `constant_tsc`, `nonstop_tsc`, `tsc_known_freq`, and `rdtscp`.
2. Pin the diagnostic and eventual workload to one online CPU.
3. Request 16 TSC/PTIMER samples per direct public RM call.  Retain every TSC
   midpoint and PTIMER value plus serialized userspace TSC brackets.
4. Reject missing samples, zeros, TSC/PTIMER regressions, CPU migration,
   malformed ordering, RM errors, or incomplete cleanup.
5. Use only source-justified conservative intervals.  For an interior sample,
   the complete RM zipper lies after the prior returned midpoint and before the
   following midpoint, and its selected shortest one of three CPU gaps is at
   most one third of that enclosing span.  Center this bound on the returned
   midpoint.  Edge samples are not precision evidence.  Add one 32 ns PTIMER
   allowance on both ends.
6. Require 200 accepted interior samples, zero rejected samples/regressions,
   median interval width at most 1,500 ns, and an affine endpoint rate whose
   residual intervals cover all retained calibration points.  The exact rate
   estimator and overflow-safe arithmetic are fixed in code before the real
   run and covered by CPU-only tests.
7. Run 200 ordered identity trials: RM TSC/PTIMER batch A, a one-thread CUDA
   kernel reading `%globaltimer`, then batch B.  Require the device value to be
   nonzero and between the bracketing PTIMER values, with all CUDA/RM resources
   released.

If these controls pass, revise the existing launchlate implementation to store
serialized TSC at the host launch hook and classify its `%globaltimer` entry
using start/end TSC/PTIMER calibration intervals.  Then run the unchanged
three-arm pp32 correctness preflight and, only if it passes all 220-launch
gates, the ten randomized pp512 blocks from seed 1797.  Prefill tokens/s remains
the paper-facing performance metric; the clock controls are correctness
evidence, not performance results.

## Commands and evidence

CPU-only tests and build:

```bash
make -C launch-clock-tsc test
```

Real controls run only while holding `/tmp/gpubpf-revision-gpu0.lock` and
`/tmp/gpubpf-revision-struct-ops.lock`.  Raw JSONL, stderr, environment
metadata, the analysis, and the result review go under `raw/` and this
directory.  No driver unload, module replacement, reboot, or service restart is
permitted.
