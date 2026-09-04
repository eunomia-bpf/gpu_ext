# Frozen plan: RTX 5090 launch-latency observability

Status: frozen before implementation and before any new GPU execution on
2026-09-04.  Historical `launchlate` runs used a CPU `CLOCK_MONOTONIC` launch
timestamp and a device `%globaltimer` entry timestamp.  They are calibration
diagnostics only and are not eligible performance results.

## Question and hypothesis

This experiment completes the third RTX 5090 Table 1 observability row.  It
tests whether gpubpf and matched NVBit can measure the same selected-kernel
host-submission-to-device-entry latency with complete engagement while their
prefill-throughput overhead is compared against the same uninstrumented
baseline.

The predeclared directional hypothesis is that engaged gpubpf has lower
prefill-throughput degradation than matched NVBit.  Every valid outcome is
reported, including equality, a loss, or an inconclusive run; there is no
post-hoc equivalence margin.

## Clock contract and minimum repair

The launch event is read from `CLOCK_MONOTONIC_RAW`.  Start and end RM anchors
come from the versioned NVIDIA 575 control `0x20800408`, which returns the
selected `[cpuBeforeNs, gpuTimeNs, cpuAfterNs]` zipper endpoints.  The GPU
endpoint is the driver's PTIMER/global-timer value.  Each interval is expanded
by one 32 ns PTIMER period on both sides.  The existing affine interval
classifier converts each retained RAW launch timestamp to a conservative
PTIMER interval and compares that interval with the device `%globaltimer`
entry timestamp.  Thus both latency endpoints are evaluated in the PTIMER
domain; no midpoint classification or negative-latency clamp is allowed.

The bpftime repair adds a host-only RAW helper under a new private helper ID.
Standard helper 5 keeps its existing semantics.  The gpubpf host callback uses
the RAW helper; its device callback keeps `%globaltimer`.  NVBit uses
`CLOCK_MONOTONIC_RAW` in its native launch callback and the same endpoint ABI
and interval arithmetic.  Neither implementation may fall back to the old
CUDA-kernel/CLOCK_MONOTONIC calibration.

## Calibration controls (not performance results)

Before launch correctness or timing, a fresh raw directory must contain both
of these passing controls on the same boot and driver as the campaign:

1. Endpoint precision: 200/200 valid direct `endpoints-v1` calls, zero CPU or
   PTIMER regressions, complete RM cleanup, and median conservative bracket at
   most 1,500 ns.  Every outer and selected endpoint is retained.
2. Counter identity: 200/200 trials ordered as RM anchor A, one CUDA kernel
   `%globaltimer` read, RM anchor B.  Each device value must be nonzero and lie
   between the two PTIMER values; both RM endpoint intervals must be valid,
   RAW and PTIMER values must not regress, and all CUDA/RM resources must be
   released.  This is an empirical stack check used together with the 575
   source contract that names PTIMER as the global timer.  It is not a latency
   or overhead observation.

Any missing record, malformed integer, unexpected method/driver/boot, failed
call, ordering violation, or cleanup failure rejects the control.  CUPTI
timestamps and the old public midpoint-only control cannot satisfy either
gate.

## Correctness gate

The existing deterministic `llama-cli` oracle runs once for each of
`baseline`, `gpubpf_launchlate`, and `nvbit_launchlate`.  All three must match
the same nonempty normalized output.  Baseline establishes workload
correctness only and does not claim a launch-latency distribution.

The instrumented arms must target the same selected kernel and satisfy:

- gpubpf:
  `host_launches == host_enqueued == device_entries == matched_samples ==
  sample_count == 220`;
- NVBit:
  `selected_launches == stored_pairs == device_entries ==
  process_selected_launches == 220`, and
  `sample_count + uncertain_samples == 220`;
- both: classified plus uncertain is 220, classified is at least 198,
  uncertain is at most 22, every histogram/count identity is exact, both RM
  anchors and the drift bound are valid, and every overflow, underflow,
  capture, queue-update, clock, and cleanup error is zero.

The different host hook points remain disclosed: gpubpf observes the exact
target ELF launch PLT stub; NVBit observes its native CUDA driver launch
callback.  A failed correctness cell stops the campaign before timing.

## Performance design

The experimental unit is one randomized block containing exactly one valid
cell of each arm: `baseline`, `gpubpf_launchlate`, and `nvbit_launchlate`.
There are exactly 10 blocks, with order independently shuffled per block from
seed 1797.  Every cell uses the same llama.cpp binary, TinyLlama model,
pp=512, tg=0, CUDA graphs disabled, CPU affinity, GPU power/clock state, target
symbol, and benchmark arguments.  No warmup policy differs by arm.

Primary metric: llama-bench prefill tokens/s.  Derived values are paired
per-block degradation versus baseline and the paired effect
`NVBit degradation - gpubpf degradation`.  Report all 10 raw triples,
geometric means, the median paired effect, and a fixed-seed paired bootstrap
95% interval.  Launch histograms are engagement/correctness evidence, not the
performance endpoint.  Calibration-control runtime is excluded from
llama-bench's reported prefill throughput but remains visible in raw logs.

No partial block, extra replacement cell, invalid retry, or historical run may
enter the result.  A failed cell may be rerun only as a new attempt in the same
block; all attempts remain recorded, and the first valid attempt is selected
by the frozen rule.  Ten complete blocks are required.

## Evidence, replay, and safety gates

The runner creates a new output directory, writes the frozen parameters and
schedule before execution, retains stdout/stderr plus structured records for
every control/correctness/performance attempt, and checkpoints after each
cell.  It uses the existing process, private shared-memory, safety, telemetry,
power-state, target-symbol, map-shape, exact-output, and owned-cleanup gates.
Ambient injection is rejected.  It never deletes a private segment unless
ownership checks pass and never kills an unowned process.

The independent analyzer reopens every named raw log and recomputes the clock,
220-launch, output, cardinality, schedule, pairing, safety, cleanup, and metric
gates from raw fields.  Missing, duplicated, non-finite, inconsistent, or
unparseable evidence fails closed.  Runner booleans are never accepted as the
sole evidence.  Preflight uses one pp=32 block after both calibration controls;
the full run requires that separately complete preflight and uses 10 pp=512
blocks.

The result boundary is explicit: a pass supports one RTX 5090 functional and
overhead comparison for this native launch-latency policy.  It does not turn
the calibration controls into performance measurements, prove arbitrary
interior clock stability, or establish device-verifier overhead.
