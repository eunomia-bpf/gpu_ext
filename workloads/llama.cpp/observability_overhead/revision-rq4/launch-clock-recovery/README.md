# Launch-clock preflight-575-07 recovery audit

This audit is a repair note for the `launchlate` correctness arms. It does not
promote the failed preflight to a result, change any raw file, or relax the
frozen histogram bins, 10% uncertain-sample limit, or 10,000 ppb drift limit.

## What failed

The gpubpf arm retained and paired all 220 host/device samples with zero queue
or clock errors. Its start and end calibration brackets were 5,261 ns and
5,220 ns wide. Only 23 samples could be certified into one histogram bin; 197
were uncertain. Under the classifier's existing semantics, each uncertain
sample has a possible latency interval that either overlaps zero or crosses a
histogram boundary. The log does not retain the individual final intervals, so
it cannot distinguish those two causes after the run. The 106.381 s endpoint
span did bound the endpoint offset change at 1,590 ppb. This arm fails the
precision gate, not accounting or engagement.

The NVBit arm retained all 220 pairs, classified 219, marked one uncertain,
and reported zero clock errors. Its start/end brackets were 4,621 ns and 4,784
ns wide. Those offset intervals overlap: an actual zero endpoint change is
compatible with the data. However, their conservative difference is
[-5,092 ns, +4,313 ns]. Dividing that resolution envelope by the short
0.325411 s anchor span produces the reported 15,648 ppb upper bound, above the
unchanged 10,000 ppb limit. This arm fails the drift-resolution gate; the log
does not establish actual excessive drift.

Histogram classification is intentionally range based. A sample is counted
only when every latency consistent with its calibration interval lies in the
same frozen bin. `uncertain` therefore means finite but insufficiently precise,
whereas a wholly negative interval, malformed pair, overflow, or invalid
calibration is a clock error. Reclassifying uncertain samples by an interval
midpoint would silently invent precision and is not an acceptable repair.

## Minimal implemented repair

NVBit now waits, with an absolute `CLOCK_MONOTONIC` deadline, until its two
calibration anchors are at least one second apart. The runner independently
rejects shorter spans. This addresses only the known sub-second rate-bound
resolution problem. It does not change the rate threshold and does not force a
pass if the endpoint intervals or their centers still imply more than
10,000 ppb. The primary llama-bench metric is its reported prefill throughput,
so the termination-side wait is outside that metric.

An unchanged NVBit rerun is superseded by this repair. A repaired NVBit-only
sanity rerun is worthwhile, but a complete Table 1 preflight is not yet
worthwhile because the gpubpf precision blocker remains.

## CUPTI is not a documented raw-device clock source

The installed CUDA 12.9 CUPTI API documentation states that
`cuptiGetTimestamp` corresponds to the *normalized* start/end values in CUPTI
activity records. The same official page states that Linux defaults to
`clock_gettime(CLOCK_REALTIME)` and that timestamps captured on the GPU are
linearly interpolated into CPU timestamps during activity post-processing:

- `/usr/local/cuda-12.9/extras/CUPTI/doc/html/api/group__CUPTI__ACTIVITY__API.html`
  (`cuptiActivityRegisterTimestampCallback` and `cuptiGetTimestamp`)
- `/usr/local/cuda-12.9/targets/x86_64-linux/include/cupti_activity.h`

Neither contract says that `cuptiGetTimestamp` returns raw PTIMER or the value
read by PTX `%globaltimer`. Therefore a narrow `CLOCK_MONOTONIC` bracket around
that call cannot be used as the production GPU/host calibration, and the check
`cupti_before <= raw_globaltimer <= cupti_after` is not an API-level proof even
if it happens to pass on one driver/GPU.

`cupti_globaltimer_sanity.cu` is a compile-tested, diagnostic-only probe for a
future GPU maintenance window. It records narrow host brackets around CUPTI
calls and checks whether a kernel's raw `%globaltimer` observation falls
between surrounding CUPTI values. Passing every trial would show a local
empirical relationship only; it must not automatically enable CUPTI
calibration or satisfy the paper gate. A failing trial decisively rejects the
assumption on that stack.

Compile without executing the probe:

```bash
/usr/local/cuda-12.9/bin/nvcc -std=c++14 -arch=sm_120 \
  -Xcompiler=-Wall,-Wextra \
  -I/usr/local/cuda-12.9/targets/x86_64-linux/include \
  launch-clock-recovery/cupti_globaltimer_sanity.cu \
  -L/usr/local/cuda-12.9/targets/x86_64-linux/lib \
  -Xlinker=-rpath,/usr/local/cuda-12.9/targets/x86_64-linux/lib \
  -lcupti -o /tmp/cupti_globaltimer_sanity
```

## Principled gpubpf path

The 575 open driver exposes
`NV2080_CTRL_CMD_TIMER_GET_GPU_CPU_TIME_CORRELATION_INFO`. Its public control
header says the returned GPU value is PTIMER/global timer at 32 ns resolution,
and its `PLATFORM_API` implementation uses a `c G c G c G c` zipper and the
closest CPU pair. On this Linux implementation that CPU source ultimately uses
`ktime_get_raw_ts64`, i.e. `CLOCK_MONOTONIC_RAW`:

- `gpu_ext-kernel-575/src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080tmr.h`
- `gpu_ext-kernel-575/src/nvidia/src/kernel/gpu/subdevice/subdevice_ctrl_timer_kernel.c`
- `gpu_ext-kernel-575/kernel-open/nvidia/os-interface.c`

This is the correct clock relationship to pursue, but the current ABI returns
only the midpoint pair, not the closest CPU-pair width needed for the existing
conservative interval classifier. It also requires an RM client/subdevice
control path not present in the current launchlate loader. A correct production
repair should expose the chosen CPU bracket width (or both CPU endpoints), map
`CLOCK_MONOTONIC_RAW` to `CLOCK_MONOTONIC` with an explicit CPU-only bracket,
and then retain the same interval classifier. Treating the midpoint as exact
would recreate the problem in a less visible form.

Finally, the current two-anchor interpolation assumes the relative host/device
clock offset evolves affinely between anchors. The endpoint drift check bounds
the observed endpoint change; it does not independently rule out an arbitrary
interior excursion. Any paper description should name that clock-model
assumption, or a later implementation should add periodic correlation anchors.

## CPU-only verification performed

- NVBit adapter build for `sm_120`: passed.
- `clock_domain_test`: passed, including minimum-deadline and overflow cases.
- `python3 -m unittest -v test_offline.py`: 41/41 passed, including rejection
  of a self-consistent 0.5 s record whose reported drift is below the limit.
- `python3 -m py_compile run_revision_rq4.py test_offline.py`: passed.
- CUPTI diagnostic compile for `sm_120`: passed; the binary was not executed.
- Re-evaluating the untouched preflight logs with the repaired runner preserves
  the intended split: gpubpf clock model passes but uncertainty fails; NVBit
  uncertainty passes but its old short-window clock model fails.
