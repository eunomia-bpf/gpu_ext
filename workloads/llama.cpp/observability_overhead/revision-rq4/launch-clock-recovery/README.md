# Launch-clock recovery audit

The three-anchor experiment definition is frozen in
[`launchlate-frozen-plan-v2.md`](launchlate-frozen-plan-v2.md). It retains the
unchanged experiment requirements from
[`launchlate-frozen-plan.md`](launchlate-frozen-plan.md), replaces only the
two-anchor clock gate with a pre-execution three-anchor validation, and keeps
attempt 07 failed under its original 10,000 ppb rule. Calibration controls
remain separate from the paired 10-block result, and the same-stack
PTIMER/`%globaltimer` identity canary still precedes the 220-launch rerun.

Attempts 08 through 11 remain retained failures. Attempt 09 passed its fresh
preflight but found two narrow RM endpoint regimes: a 781 ns median bracket
after candidate activation and a 2,174 ns median bracket after the GPU returned
to idle clocks. The exact evidence boundary is in
[`results-launchlate-575-09-clock-state-failed-20260904.md`](results-launchlate-575-09-clock-state-failed-20260904.md).
Attempt 10 requested an exact 2400 / 14001 MHz lock, but the observation after
its completed preflight was 2392 / 14001 MHz. The exact gate stopped before the
full child; the conservative failure record is in
[`results-launchlate-575-10-clock-lock-failed-20260904.md`](results-launchlate-575-10-clock-lock-failed-20260904.md).
Attempt 11 requested 2392 / 14001 MHz, but its first post-lock observation was
2400 / 14001 MHz, so the exact gate stopped before the probe or either child.
The retained record is in
[`results-launchlate-575-11-clock-bin-failed-20260904.md`](results-launchlate-575-11-clock-bin-failed-20260904.md).
The attempt-12 retry in
[`launchlate-attempt12-frozen-clock-state-plan.md`](launchlate-attempt12-frozen-clock-state-plan.md)
keeps one fixed 2392 / 14001 MHz request but admits only P0 observations with
memory exactly 14001 MHz and SM in the explicitly enumerated set
`{2392, 2400}`. This is an enumerated-bin set, not a tolerance. Fresh controls,
clock reset, and the unchanged 1.5 us gate remain mandatory.

The runnable campaign is the launch-only three-arm matrix; it never mixes the
old cross-clock records into a result. From the `revision-rq4` directory, first
run the CPU-only checks:

```bash
python3 -m unittest -v test_offline.py test_analyze_revision_rq4.py
python3 -m py_compile run_revision_rq4.py analyze_revision_rq4.py
```

Then, during an authorized GPU window, use a runtime freshly built from the
recorded bpftime source and run preflight followed by the independently gated
full campaign:

```bash
cmake -S /home/yunwei37/workspace/gpu/bpftime-table1-575 \
  -B /home/yunwei37/workspace/gpu/bpftime-table1-575/build-launchlate-575 \
  -DCMAKE_BUILD_TYPE=Debug -DENABLE_EBPF_VERIFIER=ON \
  -DBPFTIME_ENABLE_CUDA_ATTACH=ON -DBPFTIME_LLVM_JIT=ON \
  -DBPFTIME_CUDA_ROOT=/usr/local/cuda-12.9
cmake --build /home/yunwei37/workspace/gpu/bpftime-table1-575/build-launchlate-575 -j8

python3 -B run_revision_rq4.py --phase preflight --tools launchlate \
  --output-dir raw/launchlate-575-preflight \
  --bpftime-root /home/yunwei37/workspace/gpu/bpftime-table1-575 \
  --bpftime-build-dir /home/yunwei37/workspace/gpu/bpftime-table1-575/build-launchlate-575 \
  --gpu-thread-count 22528

python3 -B run_revision_rq4.py --phase full --tools launchlate \
  --preflight-dir "$PWD/raw/launchlate-575-preflight" \
  --output-dir raw/launchlate-575-full \
  --bpftime-root /home/yunwei37/workspace/gpu/bpftime-table1-575 \
  --bpftime-build-dir /home/yunwei37/workspace/gpu/bpftime-table1-575/build-launchlate-575 \
  --gpu-thread-count 22528

python3 -B analyze_revision_rq4.py raw/launchlate-575-preflight
python3 -B analyze_revision_rq4.py raw/launchlate-575-full
```

The matrix is exactly baseline, gpubpf launchlate, and NVBit launchlate: one
pp=32 preflight block and ten randomized pp=512 blocks. The runner performs
both 200-sample clock controls before the 220-launch correctness cells. The
analyzer reopens the raw control, process, cleanup, safety, correctness,
engagement, and throughput evidence; a stored `valid` flag alone cannot pass.

## CPU-only readiness check (2026-09-04)

The final runtime directory `build-launchlate-575` was freshly configured from
bpftime revision `a86d789` with the eBPF verifier, CUDA attachment, LLVM JIT,
and `BPFTIME_CUDA_ROOT=/usr/local/cuda-12.9` enabled. The two targets required
by this experiment, `bpftime-agent` and `bpftime-syscall-server`, built
successfully. A separate all-target `cmake --build ... -j8` currently stops in
vendored Catch2 because that dependency is missing `<cstdint>` and a resulting
object file; this unrelated all-target failure is not a failure of either
required runtime target.
The private RAW-clock helper passed the runtime `Test helpers` case (five
assertions), and the launchlate loader CPU self-test passed.  The gpubpf loader,
the NVBit adapter for `sm_120`, its clock-domain test, and both clock-control
self-tests also built and passed without executing a GPU campaign.

The runner/analyzer suite passes 86 CPU-only tests, and the endpoint lifecycle
suite passes 24 more. The launch-only dry runs
produce exactly three preflight timing cells and 30 full timing cells over
`baseline`, `gpubpf_launchlate`, and `nvbit_launchlate`.  Independent OpenCode
review session `ses_f92e26cfaffeu90Lyc00BpJNHV` returned PASS with no confirmed
blocker; its request and disposition are retained in
`../opencode-launchlate-rm-review-02/`.  Real 200-sample controls, the 220-launch
correctness cells, and throughput blocks remain intentionally pending a GPU
window.

`rm_globaltimer_identity.cu` implements that second control.  Each trial
orders an endpoint-v1 RM/PTIMER sample, one device `%globaltimer` read, and a
second RM/PTIMER sample, then requires both host and device ordering plus
containment.  Its JSONL is calibration evidence only.  Build and exercise the
CPU-only arithmetic paths with `make test`; a real control run requires the
shared GPU lease and is:

```bash
./rm_globaltimer_identity --samples 200 \
  > /new/raw/directory/rm-globaltimer-identity.jsonl \
  2> /new/raw/directory/rm-globaltimer-identity.stderr
```

## Public RM/PTIMER correlation diagnostic

`rm_ptimer_correlation_sanity.c` is the first executable admission gate for
the source-backed RM design in `../launchlate-rm-correlation-design.md`.  It
uses the exact public 575 userspace layouts to allocate a private RM root,
device 0, and subdevice 0 while holding `/dev/nvidia0` open for the driver's
GPU-accessibility check. The device allocation uses the new private root as
its client-share handle, following the driver's own RM utility path. It then
repeats control `0x20800406` with the
`PLATFORM_API|CPU` clock source.  Each call is bracketed by
`CLOCK_MONOTONIC_RAW` and emitted as one JSON line.  The summary passes only
when every requested call is structurally valid and the median conservative
bracket, including the 32 ns PTIMER allowance on each side, is at most 1.5 us.
The 1.5 us threshold is a Phase-0 admission heuristic: it requires at least a
threefold improvement over the roughly 5.2 us failed calibration, but it is
not a launch-latency correctness threshold.  The summary also rejects any
CPU-midpoint or PTIMER regression and is emitted only after RM cleanup.

Build without running the GPU-facing control:

```bash
make -C launch-clock-recovery rm_ptimer_correlation_sanity
```

Run only while holding the shared GPU lease and retain stdout and stderr as
raw evidence:

```bash
./launch-clock-recovery/rm_ptimer_correlation_sanity --samples 200 \
  > /new/raw/directory/rm-correlation.jsonl \
  2> /new/raw/directory/rm-correlation.stderr
```

The default `xfer` control transport preserves the first canary path. Because
the 32-byte `NVOS54` payload fits the normal size-encoded NVIDIA ioctl ABI, a
fresh diagnostic may remove the extra forwarding layer without changing the
RM command or interval proof:

```bash
./launch-clock-recovery/rm_ptimer_correlation_sanity --samples 200 \
  --control-transport direct
```

Both transports are named in every sample and in the final summary. The
direct path does not weaken the `W/3`, PTIMER allowance, monotonicity, cleanup,
or precision gates.

The diagnostic also has an opt-in path for the versioned 575 driver extension
`0x20800408`, which returns the zipper's selected CPU timestamps rather than
only their midpoint:

```bash
./launch-clock-recovery/rm_ptimer_correlation_sanity --samples 200 \
  --control-transport direct --correlation-command endpoints-v1
```

This option requires a module built with the matching extension; the default
remains the stock public `0x20800406` command. For `endpoints-v1`, the offset
interval uses the returned `[cpuBeforeNs, cpuAfterNs]` pair directly and adds
one 32 ns PTIMER allowance at each end. The userspace outer timestamps remain
an independent containment and syscall-duration check. Every record names the
selected command, and public-command records leave the exact-endpoint fields
at zero rather than presenting inferred bounds as driver-returned values.

A pass proves only that the public RM control works and supplies materially
narrower conservative offset brackets on this exact stack.  It does not yet
prove that RM PTIMER and device `%globaltimer` track identically, repair the
bpftime host clock, satisfy the 220-launch uncertainty gate, or authorize a
timing campaign.  Those remain later gates; failures must be retained.

This audit is a repair note for the `launchlate` correctness arms. It does not
promote the failed preflight to a result, change any raw file, or relax the
frozen histogram bins, 10% uncertain-sample limit, or 10,000 ppb drift limit.

## Historical cross-clock run: what failed

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

## Earlier partial repair (superseded)

This section records the intermediate NVBit-only change that predated the
endpoint-v1 repair above. It is retained as failure history, not as the current
runnable design.

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

## Design rationale for the implemented gpubpf path

The 575 open driver exposes
`NV2080_CTRL_CMD_TIMER_GET_GPU_CPU_TIME_CORRELATION_INFO`. Its public control
header says the returned GPU value is PTIMER/global timer at 32 ns resolution,
and its `PLATFORM_API` implementation uses a `c G c G c G c` zipper and the
closest CPU pair. On this Linux implementation that CPU source ultimately uses
`ktime_get_raw_ts64`, i.e. `CLOCK_MONOTONIC_RAW`:

- `gpu_ext-kernel-575/src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080tmr.h`
- `gpu_ext-kernel-575/src/nvidia/src/kernel/gpu/subdevice/subdevice_ctrl_timer_kernel.c`
- `gpu_ext-kernel-575/kernel-open/nvidia/os-interface.c`

The public ABI returns only the midpoint pair, not the selected CPU endpoints.
The new diagnostic conservatively bounds the hidden selected gap by one third
of a userspace `CLOCK_MONOTONIC_RAW` bracket around the whole ioctl, then adds
one 32 ns PTIMER period at each end. This `W/3` construction must first pass on
the target stack; the midpoint is never treated as exact. If it is sufficiently
narrow, the production repair can add a distinct bpftime host helper for
`CLOCK_MONOTONIC_RAW` and keep launch timestamps and RM anchors in that domain
end to end. The standard `bpf_ktime_get_ns` helper must remain unchanged.

The public-data bracket stayed too wide in both retained transports, so the
separate versioned driver control exposes the selected CPU endpoints while
leaving the stock command unchanged. Its source build and CPU-only probe tests
pass; the runner requires a fresh endpoint canary to pass the frozen precision
and cleanup gates before any launchlate correctness or timing cell.

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
- RM/PTIMER diagnostic builds with `-Werror`; its checked-arithmetic self-test
  passes. The GPU-facing RM control has not yet been run in this note.
- Re-evaluating the untouched preflight logs with the repaired runner preserves
  the intended split: gpubpf clock model passes but uncertainty fails; NVBit
  uncertainty passes but its old short-window clock model fails.
