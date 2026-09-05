# Experiment Plan: RQ3 slack-aware preemption with a BE progress floor

## Research Question

- RQ exactly as written in the paper: “Does \sys improve tail latency,
  throughput, and resource fairness compared to user-space and global policies
  in multi-tenant settings?”
- Specific uncertainty tested here: can the existing GPreempt actuator be
  controlled by a request-slack policy that avoids unnecessary LC preemptions
  and guarantees a minimum observed BE progress opportunity, while retaining
  the fixed policy's LC SLO attainment near the measured load knee?  Can the
  same policy execute as native C and host-JIT BPF without changing its
  decisions or material performance tradeoff?
- Why the answer matters: the completed fixed-GPreempt sweep protects LC tails
  by reducing BE goodput to 44% and 23% of native at 500 and 625 requests/s.
  A matched adaptive C/BPF result directly separates a policy improvement from
  gpubpf's mechanism cost and exercises more state than a static priority port.

## Paper-Value Admission

- Planned role: decisive evidence for the policy-versus-mechanism discussion;
  supporting evidence for RQ3 rather than a new headline result.
- Largest credible paper story this experiment could unlock: gpubpf can express
  and run a stateful composition of deadline urgency, hysteresis, and measured
  BE progress on the same real actuator as GPreempt, recovering useful BE work
  without silently changing the LC protection objective.
- Strongest reviewer reject argument or load-bearing uncertainty addressed:
  existing ports could be dismissed as literal, stateless replicas that do not
  demonstrate an interesting policy enabled by the mechanism, while the fixed
  GPreempt result exposes severe BE starvation.
- Independent evidence added beyond existing runs and published results: the
  experiment uses complete request arrival timestamps, an isolated-load SLO,
  actual verified BE completions, threshold crossings, and matched C/BPF
  decision records under a bursty public arrival trace.  None is present in the
  fixed periodic GPreempt sweep.
- Why the result is not tautological, already settled, or dominated: the
  progress floor can harm LC SLO attainment and the urgency gate can fail to
  reclaim BE work.  Either outcome changes the paper interpretation.  Published
  UrgenGo, Hummingbird, and UniBoost use different actuators/workloads and do
  not provide this matched implementation comparison on the current hardware.
- Paper decision if positive: present it as an agent-designed cross-policy
  composition and attribute the BE/LC tradeoff to policy while using C/BPF as
  the mechanism-cost comparison.
- Paper decision if contradictory, mixed, or inconclusive: retain the result as
  a boundary showing that the available request-level signals or GPreempt's
  blocking actuator cannot recover BE progress under this workload; do not
  claim a new policy win.
- Best alternative experiment and why this one has higher decision value: a
  denser fixed-GPreempt knee sweep would only localize an already established
  tradeoff.  This experiment instead tests whether a new stateful policy changes
  that tradeoff and whether BPF faithfully carries it.

## Expected And Alternative Outcomes

- Current expected answer: adaptive C and BPF will issue fewer noncritical
  blocks than fixed GPreempt, improve BE goodput, and retain LC SLO attainment
  within one percentage point of fixed at both loads.
- Strongest competing explanation: once a VGG request reaches the userspace
  hook, the remaining service time and GPreempt actuation delay dominate;
  suppressing even one preemption may create an LC backlog that overwhelms the
  progress floor's benefit.
- Result that would contradict the expectation: either adaptive arm fails the
  one-percentage-point LC guard, does not improve BE goodput, or C/BPF decisions
  diverge under identical policy inputs.

## Published Precedent And Real Assets

- Closest published protocols:
  - GPreempt (ATC 2025) supplies the original two-context blocking-kernel and
    timeslice actuator.
  - Hummingbird (2026 preprint) defines the isolated-execution P99 SLO and
    opportunistic BE bubble-harvesting objective.
  - UrgenGo (MobiCom 2025) defines task urgency from deadline laxity and remaining
    GPU work.
  - UniBoost (ICML 2026) motivates soft priority changes and starvation guards,
    but for multi-request LLM serving and KV-aware preemption.
- Official system/model/data/benchmark/tool and version: pinned GPreempt
  `249ee3e`; existing TVM VGG19 LC and ResNet152 BE graphs; public BurstGPT
  revision recorded by `workloads/hummingbird/arrivals-burstgpt.json`; RTX 5090,
  Linux 6.15.11, NVIDIA 575.57.08 and the already staged scheduling port.
- What is reused: GPreempt contexts, 1,000,000/1 us LC/BE timeslices, host-mapped
  stop flag, blocking kernels, CUDA graphs, numerical oracle, common FIFO
  window, telemetry, ownership checks, leases, and raw-request accounting.
- Necessary deviations or custom glue: expose each FIFO request's scheduled and
  started steady-clock timestamps to the existing host policy callback; accept
  a trace-offset FIFO schedule; add state/counters to the host bridge; and add a
  separate runner/analyzer.  No driver, model, kernel, or timeslice change is
  allowed.

## Comparison

- Proposed system or method: slack-aware dual-threshold GPreempt with a verified
  BE-completion floor, in native C and host-uBPF JIT implementations.
- Main baselines and the competing position each represents:
  - native CUDA stream priorities/no GPreempt policy: work-conserving current
    practice;
  - fixed original-C GPreempt: strongest matched foreground-protection policy;
  - adaptive original-C: identical algorithm without the BPF JIT, the strongest
    alternative mechanism for policy-versus-mechanism attribution.
- Why each main baseline needs a matched run instead of citation alone: all four
  arms need the same BurstGPT-derived arrivals, isolated SLO, models, transport,
  driver, and 60-second window.  Existing results use periodic arrivals or a
  different Hummingbird executor.
- Controls or ablations, labeled separately: isolated native LC calibration at
  each offered load defines the SLO; it is not a performance baseline and has
  no BE task.
- Conclusion if each main baseline matches or wins: native winning bounds the
  value of any preemption; fixed winning rejects adaptive admission; adaptive C
  winning over adaptive BPF exposes mechanism overhead or a decision mismatch.
- Information, tuning, and compute fairness: adaptive C/BPF receive the same
  scheduled time, current time, derived SLO, and verified BE completion count.
  Fixed GPreempt receives no extra decision input.  All colocated arms use the
  same models, arrival offsets, timeslices, preprocessing and GPU budget.
- Split or leakage rule when relevant: SLO values are derived only from a
  separate isolated calibration.  No formal cell can tune thresholds, rates,
  correctness gates, or the interpretation rule.

## Workloads And Metrics

- Real workloads or tasks: continuous closed-loop ResNet152 BE plus bursty VGG19
  LC arrivals derived from the first chronological successful rows of the
  public BurstGPT trace.  The same rank/tie-preserving affine scaling supplies
  exactly 34,500 and 37,500 arrivals over 60 seconds (mean 575 and 625 rps), two
  points below/at the previously observed periodic knee boundary.
- Frozen adaptive policy:
  - SLO deadline = scheduled arrival + the median isolated-LC P99 for that load.
  - Enter pressure when remaining deadline time is at most 1,550,000 ns; clear
    it only at or above 1,700,000 ns.
  - A pressure-state request may block BE only after at least one numerically
    verified BE completion and 200,000 ns without a block since the previous
    release.  Remaining time at or below 1,400,000 ns is critical and overrides
    the progress floor.  These constants come from the existing VGG service
    distribution and are frozen before calibration/preflight.
  - Safe requests still install the existing 100 us hint; the decision is
    reevaluated there.  Every actual block is released after enqueueing LC work.
- Primary metrics: all-offered LC SLO attainment and BE verified in-window
  goodput.  Full arrival-to-verified-response P99 is co-primary only with 100%
  completion coverage; otherwise it is explicitly conditional with backlog.
- Secondary mechanism metrics: adaptive/fixed block count, critical/threshold
  blocks, safe/floor skips, pressure transitions, observed BE completions,
  native/BPF decision agreement, actual JIT calls, and bridge/context activity.
- Correctness check or ground truth: the existing full-output FP32 oracle for
  every warmup, calibration, timed, and drained request; exact FIFO prefix and
  offered/completed/backlog accounting; no CUDA error; exact context ownership,
  timeslice binding, transport cleanup, Xid, telemetry and lease checks.
- Repetitions, seeds, and uncertainty: three complete paired blocks, with all
  four arms in seeded Latin order per load and load order reversed by block.
  Report cell points, medians, and 95% percentile intervals from 10,000
  whole-block bootstrap draws using seed 20260904.  Do not claim formal
  equivalence from three blocks.
- Cost estimate when material: six isolated 60-second calibration cells, one
  four-arm 10-second preflight at 625 rps, then 24 formal 60-second cells; about
  35 minutes of timed GPU work plus initialization and cooldown.

## Planned Runs

| Run group | Role | Workload | System/method | Repetitions | Decision consequence |
|---|---|---|---|---:|---|
| calibration | control | BurstGPT-derived VGG at 575/625 rps, no BE | native CUDA | 3/load | Freeze isolated P99 SLOs; failure stops the comparison |
| preflight | dependency | 625 rps bursty VGG + continuous ResNet152 | all four arms | 1 | Establish real correctness, engagement, trace and decision path only |
| main | baseline | both loads + continuous BE | native/no policy | 3/load | Work-conserving reference |
| main | baseline | both loads + continuous BE | fixed original-C GPreempt | 3/load | Fixed-protection reference |
| main | proposed | both loads + continuous BE | adaptive original C | 3/load | Isolate adaptive policy effect |
| main | proposed | both loads + continuous BE | identical host-JIT BPF | 3/load | Isolate mechanism cost/fidelity |

## Execution

- Authoritative workflow: the separate `run_adaptive_study.py` invokes the
  existing prepared GPreempt binaries and loader through the same owned-process,
  telemetry, lease and cleanup path as `run_load_study.py`; it never changes the
  driver or services.
- Real preflight case: one complete 10-second four-arm block at 625 rps using
  the first ten seconds of the frozen 60-second arrival offsets and calibration
  SLO.  It is excluded from estimates.
- Full completion rule: all six calibration cells, all four preflight cells and
  all 24 formal cells pass correctness, arrival/accounting, engagement and
  cleanup.  Failed attempts are retained and never pooled.
- Raw-result path: `raw/adaptive-slack-{calibration,preflight,full}-575-01/`.
- Checkpoint or recovery approach: each cell is an immutable subdirectory and
  the runner appends completed-cell metadata only after validation.  Resume may
  skip accepted cells with the exact frozen plan but may not retry or replace a
  failed formal cell.

## Interpretation

- Positive result: at both loads, adaptive BPF/fixed BE-goodput interval lower
  bounds exceed one, adaptive BPF minus fixed LC SLO-attainment interval lower
  bounds are at least -1 percentage point, all LC requests complete, and
  adaptive BPF/C decision counts match exactly.  Attribute BE recovery to the
  policy, not BPF.
- Negative or contradictory result: any LC guard failure, absent BE recovery,
  incomplete coverage, or decision divergence rejects the joint claim while
  retaining valid component results.
- Mixed or inconclusive result: report load-specific effects and intervals; do
  not pool loads or promote one favorable point.
- Target paper figure or table: a two-load Pareto panel (LC SLO attainment versus
  BE goodput) plus a compact C/BPF matched-policy row, only after independent
  result review.  This task does not edit the paper.

## Reproducibility Notes

- Software and data versions: record ordinary Git revisions, explicit binary
  paths/sizes, driver/kernel/GPU identity and BurstGPT source revision; never use
  file hashes or digests.
- Config and seed notes: thresholds, rates, trace transformation, Latin orders,
  bootstrap seed and SLO derivation are fixed above before GPU execution.
- Known deviations: host-mapped flags replace unavailable GDRCopy; the workload
  is two batch-one TVM DNNs rather than autonomous-driving chains, LLM serving,
  or Hummingbird's split-kernel executor.  The progress floor observes verified
  BE request completion and a preemption-free interval, not continuous SM-level
  progress.  This is a cross-policy composition, not a full reproduction or a
  claim that deadline urgency, hysteresis, or starvation protection is novel.
