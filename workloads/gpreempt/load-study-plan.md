# GPreempt contention study — frozen scope

Prepared 2026-09-03 UTC. This is a new load study, not a replacement for the
completed original config-A comparison in
[results-575-host-mapped-20260903.md](results-575-host-mapped-20260903.md).
Independent XSched runtime work is paused while this study is completed.

## Question and hypothesis

Paper RQ3: “Does \sys improve tail latency, throughput, and resource fairness
compared to user-space and global policies in multi-tenant settings?”

Bounded hypothesis: as background contention increases, BPF implementing the
same GPreempt decisions preserves the original-C policy's foreground protection
and background progress. Compare both against native CUDA stream priorities;
measure any latency/throughput tradeoff. A null result is useful and retained.
This is not a test that BPF must outperform the original algorithm, nor a formal
equivalence test or reproduction of the original paper's hardware results.

## Fixed design

- RTX 5090, Linux 6.15.11-061511-generic, NVIDIA 575.57.08 with the already
  loaded scheduling port at source revision 849ea75d, fixed 400 W. No driver
  installation, replacement, or reboot. All arms use this same driver.
- Existing seeded FP32 TVM VGG19 foreground and ResNet152 background models,
  real batch one, CUDA graphs, original 200 us preprocessing, warmup and
  calibration. Exported testing models are not pretrained-accuracy evidence.
- Three arms: original native single-context/prioritized-stream client,
  original-C GPreempt, and actual BPF/JIT GPreempt. The policy arms keep the same
  two-context executor, host-mapped flags, blocking kernels, 100 us early hint,
  and LC/BE timeslice requests of 1,000,000/1 us. Host-mapped flags are an
  explicit compatibility transport, not original GDRCopy.
- Three prespecified scenarios, all with foreground 100 requests/s: background
  100 requests/s, 200 requests/s, and closed-loop continuous supply. No model,
  batch, kernel repetition, or policy change between arms. Continuous supply
  removes a request-rate cap but does not by itself prove full GPU saturation.
- New periodic_fifo load: fixed arrival phase zero, one common monotonic start
  and 60-second half-open measurement window for both roles. Unlike the old
  newest-only generator, expired arrivals remain FIFO backlog. Do not start a
  new request at or after the deadline; allow an already-started request to
  finish and record it separately. Do not phase-shift arrivals based on each
  arm's standalone calibration.
- Five complete paired blocks per scenario. Each block contains all three
  arms in seeded randomized order; scenario order is balanced/interleaved.
  Record the seed and full execution order before the first timed cell.
  GPU cells run serially under both existing GPU and struct-ops leases, with
  no competing compute jobs or compilation during timing.

## Measurements and validity

Primary foreground response is scheduled arrival to GPU-synchronized,
numerically verified output ready, including FIFO waiting. It excludes a
network/server front end. Retain scheduled, actual start, and verified finish
timestamps/request IDs, plus the old six-stage service metric as auxiliary.
Do not compare the new response p99 directly with the old service p99.

Periodic arrivals are deterministic: 6,000 foreground requests, and 6,000 or
12,000 background requests per window. Started IDs must be a consecutive FIFO
prefix. Report, per role:

    offered = completed_in_window + completed_after_window
              + started_unfinished + never_started_backlog

Numerical or CUDA errors invalidate a cell, not count as useful completions.
For a successful serial worker, started_unfinished is zero after cleanup and
at most one completion is after the window. Continuous supply has no periodic
offered denominator or missed-arrival rate; report these as not applicable.
Background goodput counts only verified completions inside the 60-second
window, divided by 60 seconds, never all recorded completions divided by 60.

Report completed-request response p99 with completion coverage. If offered
requests are unfinished, label that p99 conditional; do not claim an all-offered
p99 improvement from a selected surviving subset. Show unfinished/backlog
counts alongside latency and goodput, including adverse overload cases.

Every warmup/calibration/timed output must pass the existing full-output
numerical checks. All three arms must fail on CUDA graph/launch/sync errors.
Every BPF cell must prove real JIT decisions and both owned context controls;
original-C must not engage BPF, native must not engage GPreempt. Reuse existing
transport-cleanup, telemetry, new-Xid, GPU-idle, and struct-ops cleanup checks.
Retain failed attempts and reasons; never silently retry or drop slow cells.

## Execution and reporting

Keep old build/ninja binaries and run_three_way.py unchanged. New instrumentation
is opt-in, with replayable patch, CPU boundary tests, and build/load-study.
One independent plan review and implementation review precede a real three-arm
continuous-load preflight (excluded from final estimates). Freeze after a valid
preflight, then run all 45 cells; about 45 minutes of timing plus setup/cleanup.
Stop on correctness/engagement/safety failure, diagnose and retain evidence;
if code changes, do not combine different implementations as one frozen run.

Recompute metrics from raw timestamps and logs in an independent final audit.
Report per-cell points and five-block medians for all three scenarios, paired
BPF/original and policy/native geometric-mean ratios, and percentile 95%
bootstrap intervals resampling whole blocks (10,000 draws, seed 20260903).
These paired-effect intervals are not uncertainty bars on arm medians.
With only five blocks, describe uncertainty without claiming
formal equivalence. Produce standalone foreground-response/background-goodput
plots; the later 2x2 XSched/GPreempt figure can reuse them with distinct units.
Commit and push scoped code, plan, raw evidence, audit, results, and plots;
leave unrelated working-tree changes and the paper submodule untouched.

Independent read-only plan review: passed on 2026-09-03; the review's sole
nonblocking clarification, freezing the paired estimator and bootstrap, is
incorporated above. No additional scenarios were requested or admitted.
