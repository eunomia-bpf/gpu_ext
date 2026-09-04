# Launchlate RM/PTIMER correlation design

## Status and scope

This note records a source-backed design for replacing gpubpf `launchlate`'s
wide CUDA-kernel clock calibration with NVIDIA RM/PTIMER correlation on the
RTX 5090 / 575.57.08 stack. It is not an implementation or an experimental
result. No RM correlation call, derived bracket, repaired 220-launch
correctness cell, or timing cell has run yet.

The intended order is: implement a calibration-only diagnostic, admit one
correctness run only if that diagnostic passes, and admit performance timing
only after the unchanged correctness gates pass. Historical failed preflights
must not be reclassified.

## Current timestamp path

The host launch event and calibration currently use the following path:

1. `example/gpu/launchlate/launchlate.bpf.c::uprobe_cuda_launch` exact-filters
   the selected `cudaLaunchKernel@plt` argument and stores
   `bpf_ktime_get_ns()` in the bounded host/device FIFO.
2. In this bpftime tree, `BPF_FUNC_ktime_get_ns` dispatches to
   `runtime/src/bpf_helper.cpp::bpftime_ktime_get_ns`, implemented with
   `std::chrono::steady_clock`.
3. On the installed target `libstdc++.so.6`, read-only disassembly shows
   `steady_clock::now()` calling `clock_gettime` with clock ID 1, so it is
   `CLOCK_MONOTONIC` on this stack. This is a target implementation fact, not
   a portable C++ guarantee.
4. `launchlate.c::sample_gpu_clock` explicitly brackets a synchronized CUDA
   kernel `%globaltimer` read with two `CLOCK_MONOTONIC` reads. Four warmups
   and 32 trials feed `consider_calibration_sample`, which retains the
   narrowest `[G-host_after, G-host_before]` offset interval.
5. `launchlate.bpf.c::cuda__probe` records `bpf_get_globaltimer()`. GPU helper
   502 reaches
   `attach/nv_attach_impl/trampoline/default_trampoline.cu::read_globaltimer`,
   whose PTX is `mov.u64 ..., %globaltimer`.
6. Start and end calibration intervals feed
   `affine_offset_interval`/`classify_affine_sample`. The classifier counts a
   latency only if its entire interval is nonnegative and lies in one frozen
   histogram bin; otherwise a finite interval is `uncertain`.

`run_revision_rq4.py`'s Python `time.monotonic_ns()` is used only to create a
private shared-memory name. It is not the launch measurement timestamp.

## Existing NVIDIA 575 correlation control

The open 575.57.08 driver exposes
`NV2080_CTRL_CMD_TIMER_GET_GPU_CPU_TIME_CORRELATION_INFO` (`0x20800406`) in
`ctrl2080tmr.h`. The public result is a `{cpuTime, gpuTime}` pair; the header
identifies `gpuTime` as PTIMER/global timer with 32 ns resolution.

For `PLATFORM_API` source and `CPU` processor,
`subdeviceCtrlCmdTimerGetGpuCpuTimeCorrelationInfo_IMPL` performs exactly
`c-G-c-G-c-G-c`: three PTIMER-low reads between four CPU reads. It selects the
smallest adjacent CPU gap and returns the intervening GPU read plus
`floor((c_i+c_{i+1})/2)` in `samples[0]`. This branch writes only
`samples[0]`, irrespective of a larger requested sample count, so callers
should request `sampleCount=1` and repeat calls themselves. The public ABI
does not return the selected CPU endpoints or their width.

The CPU path is
`osGetPerformanceCounter -> os_get_current_tick_hr -> ktime_get_raw_ts64`, so
its domain is `CLOCK_MONOTONIC_RAW`, not the current launch event's
`CLOCK_MONOTONIC` domain.

The proposed userspace object/control sequence is:

1. Open `/dev/nvidiactl`.
2. Issue `NV_ESC_RM_ALLOC` with the exact 575 `NVOS64` layout to allocate a
   private `NV01_ROOT_CLIENT`, using `hObjectNew=0` and validating the returned
   nonzero handle.
3. Under that client, allocate GPU 0 `NV01_DEVICE_0` with
   `NV0080_ALLOC_PARAMETERS`, then subdevice 0 `NV20_SUBDEVICE_0` with
   `NV2080_ALLOC_PARAMETERS`.
4. Issue `NV_ESC_RM_CONTROL` on the subdevice for command `0x20800406`, with
   `PLATFORM_API | CPU` and `sampleCount=1`.
5. Free the root client and close the fd on every exit path.

The outer xfer ioctl is already demonstrated by
`gpu_ext/extension/gpu_preempt.h::gp_rm_control`, but that helper accepts
known CUDA-owned handles and provides neither RM allocation nor timer
correlation. The installed CUDA/NVIDIA userspace libraries expose no dynamic
`nvRmApiAlloc`/`nvRmApiControl` entry point, so the implementation should use
the version-pinned ioctl ABI with compile-time layout checks.

The kernel call chain is
`nvidia_ioctl -> rm_ioctl -> RmIoctl -> Nv04ControlWithSecInfo ->` generated
subdevice dispatch `-> subdeviceCtrlCmdTimerGetGpuCpuTimeCorrelationInfo_IMPL`.
The generated method has parameter-size metadata and no additional access
right, but actual allocation permission and command success remain runtime
questions.

## Conservative public-data bracket

For each synchronous control call, read `CLOCK_MONOTONIC_RAW` immediately
before and after the ioctl. Let those outer values be `B` and `A`, the returned
CPU midpoint be `M`, and the returned PTIMER value be `G`.

All four internal CPU samples satisfy

```text
B <= c0 <= c1 <= c2 <= c3 <= A .
```

Let `W=A-B`. The three adjacent gaps sum to `c3-c0 <= W`, and the driver
chooses their minimum `d`. Therefore

```text
D = floor(W/3),  d <= D.
```

For the chosen pair `(c,c')`, the returned integer midpoint is
`M=c+floor(d/2)`. The PTIMER-low read occurs after `c` and before `c'`, giving
the conservative CPU-time bracket

```text
L = max(B, M - floor(D/2))
U = min(A, M + ceil(D/2)).
```

The corresponding GPU-minus-RAW interval is

```text
[G-U, G-L].
```

Because the public contract states 32 ns resolution but does not specify the
rounding convention, the implementation should conservatively expand the
interval outward by one period:

```text
[G-U-32 ns, G-L+32 ns].
```

This is a quantization allowance, not GPU scheduling delay. It assumes the RM
PTIMER value and device `%globaltimer` are observations of the same reported
counter domain; that relationship must be checked on the target stack.

All subtraction/addition must use checked signed or wider arithmetic. Reject
`A<B`, `M` outside `[B,A]`, zero/invalid results, overflow, unexpected RM
status, and an excessive outer duration. A strict userspace limit such as
`W<10 ms` is far below the approximately 4.295 s low-32-bit wrap interval and
also prevents treating a heavily preempted ioctl as a useful anchor. This
outer check avoids relying unconditionally on the driver's high-before /
high-after equality test.

## Recommended common host clock

The preferred repair is a new bpftime host-only helper that directly returns
`CLOCK_MONOTONIC_RAW`. `uprobe_cuda_launch` should use that helper, and the RM
calibration anchors and retained host samples should remain in RAW throughout.
The existing standard `bpf_ktime_get_ns` emulation must retain its current
semantics for other programs.

This removes a third clock and avoids an unsound shortcut: one independent
`RAW-MONO-RAW` zipper bounds `RAW-MONO` only at its middle read, not
automatically at the later RM sample. Keeping MONOTONIC would require bridge
samples immediately around every RM call plus an explicit, padded model for
relative RAW/MONOTONIC evolution. RAW end-to-end is the smaller proof burden.

Start and end RM anchors can continue to use the existing affine interval
classifier. That classifier's assumption remains limited: endpoint evidence
does not rule out arbitrary interior offset excursions. Periodic anchors
would strengthen the model but are not required for this minimum repair and
must not be implied by the result.

## Minimum implementation inventory

- `bpftime-table1-575/runtime/src/bpf_helper.cpp`: implement and register a
  distinct `CLOCK_MONOTONIC_RAW` helper without changing helper 5.
- `bpftime-table1-575/example/gpu/launchlate/launchlate.bpf.c`: use the RAW
  helper for the exact host launch event; preserve FIFO and error accounting.
- `bpftime-table1-575/example/gpu/launchlate/launchlate.c`: replace the CUDA
  calibration kernel path with the private RM client, checked `W/3` bracket,
  32 ns allowance, RAW anchors, detailed diagnostics, and fail-closed cleanup;
  preserve start/end drift and affine classification.
- `bpftime-table1-575/example/gpu/launchlate/Makefile`: add any new local RM
  helper source/header dependency. Do not depend on an ambient sibling source
  tree; keep the narrow 575 ABI definitions explicit and layout-checked.
- `run_revision_rq4.py`: update the source-schema marker, parse and recompute
  the new RM/bracket diagnostics, and pass the expected launch count to both
  launchlate correctness validators.
- `analyze_revision_rq4.py`: mirror the method, bracket, and correctness
  checks rather than trusting runner booleans.
- `test_offline.py`, `test_analyze_revision_rq4.py`, and the launchlate CPU
  self-test: cover odd/even gaps, malformed midpoint, overflow/underflow,
  excessive outer duration, missing diagnostics, and the backend-specific
  220-launch rules.

No driver source belongs in the primary patch.

## Exact 220-launch semantics

The correctness oracle concerns 220 selected launches, not necessarily 220
classified histogram entries:

- gpubpf: require
  `host_launches == host_enqueued == device_entries == matched_samples ==
  sample_count == 220`, plus `classified + uncertain == 220`.
- NVBit: its `sample_count` is the classified count. Require
  `selected_launches == stored_pairs == device_entries ==
  process_selected_launches == 220`, plus `sample_count + uncertain == 220`.
- Preserve the existing 10% uncertainty gate: at most 22 uncertain and at
  least 198 classified. Do not silently require all 220 to classify, and do
  not hard-code 220 in timed llama-bench cells.

## Current evidence and admission sequence

Retained preflight 575-08 establishes the existing problem, not this repair:

- gpubpf retained and paired all 220 launches with zero queue or clock errors,
  but classified only 20 and marked 200 uncertain. Its start/end interval
  widths were 5,243/5,235 ns and endpoint drift bound was 2,433 ppb.
- NVBit retained all 220 pairs, classified 214, marked six uncertain, and had
  a 6,792 ppb endpoint drift bound.

Before another correctness or timing campaign, run a calibration-only
diagnostic that records every ioctl/RM status and every `B,A,M,G,D,L,U`, checks
GPU selection and same-counter behavior, rejects malformed/slow calls, and
reports the retained derived widths across repeated trials. This gate may
authorize one correctness-only 220-launch rerun only if the control path is
fully successful and the public-data brackets are materially narrower than
the current roughly 5.2 us brackets. It cannot itself claim that the 10%
classification gate passes. No timing cell is admitted until the subsequent
correctness run satisfies all unchanged pairing, clock, drift, uncertainty,
and exact-count checks.

## Driver boundary and fallback

No new driver and no kfunc are required if all of the following hold on the
target: private RM allocation/control succeeds, command `0x20800406` returns
the expected CPU/RAW and PTIMER values for GPU 0, the outer-derived bracket is
sufficiently narrow, and the repaired correctness cell passes unchanged
quality gates. Only bpftime/launchlate userspace components need rebuilding.

If the narrowest valid public-data bracket remains too wide, the clean
fallback is a new versioned RM control that returns the chosen
`c_before`, `c_after`, and intervening `G` directly. That fallback requires a
new `nvidia.ko` build, an explicit staged module, reload, and a fresh
calibration/correctness preflight. It should not silently change the existing
public command's ABI, and a kfunc is the wrong interface for this userspace
calibrator.

All full 575 staged `nvidia.ko` variants inspected already contain the stock
correlation implementation and RAW clock symbol; the prefetch-only stage
contains only `nvidia-uvm.ko`. No stage contains a launchlate userspace wrapper
or an endpoint-returning correlation extension.

## Independent review

Strict deny-all OpenCode review session
`ses_f95116c55ffepUo1Z7ZdwN5ydi` returned **READY WITH CONDITIONS**. Its
conditions agree with this note's gates: runtime-prove the RM ioctl path, keep
all interval arithmetic signed and checked, avoid an unbounded RAW/MONOTONIC
conversion, apply the 32 ns allowance conservatively, and enforce the two
backend-specific 220-launch semantics.
