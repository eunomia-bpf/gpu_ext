# Experiment Plan: RQ4 device-verifier admission scaling

Status: frozen after read-only plan review; no timing data have been collected.

## Research Question

- RQ exactly as written in the paper: **RQ4 (Overhead): What is the overhead of
  gpubpf's core mechanisms and observability capabilities?**
- Specific uncertainty tested here: how the one-time latency of the current
  `verify_gpu_program` entry point changes with accepted instruction count and
  with a dense, warp-uniform forward-branch control-flow graph (CFG).
- Why the answer matters: A1 measures only two real 13- and 60-instruction
  policies. Reviewers B/F asked for verifier depth; those two points cannot
  distinguish a small-program result from a scaling limitation.

## Paper-Value Admission

- Planned role: **supporting**.
- Largest credible paper story this experiment could unlock: a controlled bound
  on one-time GPU-eBPF admission cost over programs up to 4,096 instructions,
  with a separate estimate of sensitivity to dense uniform control flow.
- Strongest reviewer reject argument or load-bearing uncertainty addressed:
  the SIMT verifier is described and tested on rejection cases, but the paper
  does not show whether its real admission path remains usable beyond two small
  policies.
- Independent evidence added beyond existing runs and published results: A1's
  13/60-instruction real objects anchor ecological relevance; this experiment
  independently controls instruction count and CFG density while invoking the
  same public API directly.
- Why the result is not tautological, already settled, or dominated: acceptance
  tests establish safety decisions, not latency scaling. Source inspection
  reveals multiple passes but does not establish their observed cost.
- Paper decision if positive: add one compact RQ4 verifier-scaling panel/table
  and bound the claim to accepted synthetic programs on this CPU/build.
- Paper decision if contradictory, mixed, or inconclusive: retain the data and
  state the measured scale/CFG boundary; do not generalize A1 to larger
  policies.
- Best alternative experiment and why this one has higher decision value:
  timing more existing policy objects would improve ecological breadth but
  would confound length, helpers, maps, and CFG. The controlled families answer
  the reviewer-facing scaling question causally, while A1 already supplies two
  real-policy anchors.

## Expected And Alternative Outcomes

- Current expected answer: admission latency grows approximately linearly with
  instruction count over the measured range; dense uniform diamonds are more
  expensive than straight-line code but remain accepted and finite.
- Strongest competing explanation: PREVAIL path refinement or the SIMT
  worklist/join implementation makes branch-dense programs superlinear even
  though every branch is legal and has a one-instruction forward displacement.
- Result that would contradict the expectation: any constructed safe arm is
  rejected or times out, or the lower endpoint of the 95% block-bootstrap
  interval for a family's log-log Theil--Sen exponent exceeds 1.25. The 1.25
  threshold is an empirical definition of "approximately linear," not an
  asymptotic complexity proof.

## Published Precedent And Real Assets

- Closest published protocol: there is no standard paper benchmark for this
  GPU-specific verifier. We use the direct public verifier entry point and
  report absolute admission latency, as in the existing A1 experiment.
- Official system/tool and version: the current bpftime
  `bpftime::verifier::gpu::verify_gpu_program` implementation at the recorded
  Git revision, built independently in `Release` mode with its PREVAIL,
  uniformity, and SIMT passes enabled.
- What is reused: the production verifier implementation, GPU helper model,
  eBPF ISA definition, and accepted warp-uniform helper 510
  (`bpf_get_warp_id`).
- Necessary deviations or custom glue: one deterministic C++ program
  constructor/timing probe and Python scheduling/analyzing scripts. They do not
  replace or mock the verifier.

### Source-confirmed bounds

- `verify_gpu_program` itself imposes no policy-facing fixed maximum; it rejects
  only a count beyond `std::vector<ebpf_inst>::max_size()` before allocation.
- bpftime's execution VM defines `EBPF_MAX_INSTS` as 65,536. That is the
  source-level runtime bound, not a bound established by this experiment.
- An eBPF branch displacement is signed 16-bit. Every constructed diamond uses
  displacement `+1`, so no arm approaches that encoding bound.
- The measured set is fixed at 16, 64, 256, 1,024, and 4,096 instructions. The
  maximum is 68 times A1's 60-instruction policy and keeps this experiment
  focused on policy-scale verifier work rather than allocator stress. We do
  **not** claim to measure the 65,536-instruction runtime boundary.

## Comparison

- Proposed system or method: the current, unmodified three-stage GPU verifier
  API (PREVAIL, uniformity analysis, SIMT checks).
- Main baselines: none. This is an absolute mechanism-scaling measurement, not
  a comparison with a competing verifier.
- Controls or ablations, labeled separately:
  - `linear`: one uniform helper call and identical three-instruction prefix,
    then ALU-immediate instructions and one exit; zero conditional branches.
  - `diamonds`: the same prefix/exit and exact instruction count, with the body
    filled by repeated warp-uniform `JEQ +1`/ALU pairs. Conditional-branch
    counts are 6, 30, 126, 510, and 2,046.
- Conclusion if the control matches or wins: a near-one diamonds/linear ratio
  means instruction traversal, rather than CFG joins, dominates this range; a
  growing ratio localizes overhead to branch-rich verification without making
  a soundness claim.
- Information, tuning, and compute fairness: both families use the same helper,
  section, exact length, build, CPU, fresh-process path, block schedule, and
  repetition count. Program construction and shape checks occur before the
  timed interval.
- Split or leakage rule: no tuning occurs after preflight. Preflight output is
  stored separately and never included in the full-run statistics.

## Workloads And Metrics

- Real workload/task: direct verification of deterministic, accepted eBPF
  instruction arrays through the production API. This is a verifier
  microbenchmark, not GPU execution.
- Primary metrics:
  - median `CLOCK_MONOTONIC_RAW` nanoseconds per family/size and a fixed-seed
    95% block-bootstrap interval;
  - paired per-block diamonds/linear latency ratio at each size, with a 95%
    block-bootstrap interval;
  - log-log Theil--Sen scaling exponent per family, with a 95%
    block-bootstrap interval.
- Diagnostics: `CLOCK_PROCESS_CPUTIME_ID`, minor/major faults, voluntary and
  involuntary context-switch deltas, and CPU-before/after identifiers.
- Correctness check or ground truth: before timing, the probe independently
  checks exact length, opcode classes, helper/exit/conditional counts, every
  branch target, and the fixed family formula. The API must return no error;
  affinity must remain CPU 23. The analyzer reconstructs the frozen schedule
  and expected structural counts without importing runner code.
- Repetitions, seeds, and uncertainty: 20 complete randomized blocks; each
  block contains every one of the 10 family/size arms exactly once. Seed 1797
  fixes the schedule. Each cell is one call in a fresh process. Bootstrap seed
  1797 and 20,000 resamples are fixed.
- Cost estimate: 200 measured verifier calls plus 10 discarded warmup calls;
  each child has a 60-second timeout. A timeout stops the run and is retained;
  there are no retries or optional stopping.

## Planned Runs

| Run group | Role | Workload | System/method | Repetitions | Decision consequence |
|---|---|---|---|---:|---|
| preflight | dependency | 16-linear and 4,096-diamonds | real API, isolated Release build | 1 each | establish end-to-end acceptance and output path only |
| full linear | control | 16/64/256/1,024/4,096 instructions | real API, no conditional branches | 20 blocks | instruction-count scaling |
| full diamonds | CFG control | same exact lengths | real API, dense uniform diamonds | 20 blocks | CFG sensitivity at matched length |

## Execution

- Authoritative workflow: configure and build `bpftime-verifier` from
  `/home/yunwei37/workspace/gpu/bpftime-table1-575` in the isolated directory
  `/home/yunwei37/workspace/gpu/bpftime-device-verifier-scaling-build`; build
  and invoke the probe linked to that library. The existing
  `build-table1-575-strict` tree and its DSOs are never read as build inputs or
  modified.
- Timing isolation: formal cells run sequentially with an exact CPU-23 affinity
  mask. CPU model, online state, cpufreq driver/governor, kernel, compiler,
  build type, executable metadata, and source Git revision are retained. The
  workflow does not open a GPU device and sets `CUDA_VISIBLE_DEVICES` empty.
- Real preflight case: one fresh 16-linear call and one fresh 4,096-diamonds
  call through the complete runner/probe/raw-output path, stored under
  `raw/preflight-01/`.
- Full command (only after the orchestrator releases it):

  ```bash
  env CUDA_VISIBLE_DEVICES= taskset -c 23 \
    python3 run_verifier_scaling.py \
      --probe /home/yunwei37/workspace/gpu/bpftime-device-verifier-scaling-build/verifier_scaling_probe \
      --bpftime-root /home/yunwei37/workspace/gpu/bpftime-table1-575 \
      --output-dir raw/scaling-575-01
  ```

- Full completion rule: exactly 200 successful measured cells, 20 complete
  blocks, the frozen schedule/cardinality, all structural gates, all API
  acceptances, no timeout, no extra/missing cell, and a complete independent
  replay. Any failure leaves the run invalid or incomplete; it is never
  silently dropped.
- Raw-result path: `raw/scaling-575-01/`; independent analysis is written there
  as `analysis.json`.
- Checkpoint or recovery approach: the runner writes each raw cell and updates
  `result.json` after that cell. Resume is deliberately unsupported because the
  CPU noise regime may change; a failed full run remains retained and a new
  attempt starts in a new directory.

## Interpretation

- Positive result: all arms are accepted and both families' upper 95% exponent
  endpoint is at most 1.25. Report absolute latency and CFG ratios as supporting
  evidence for approximately linear scaling over 16--4,096 instructions.
- Negative or contradictory result: a safe arm is rejected/times out, or a
  family's lower 95% exponent endpoint exceeds 1.25. Report the exact boundary
  and do not generalize beyond smaller programs.
- Mixed or inconclusive result: an exponent interval crosses 1.25, diagnostics
  show material scheduling interference, or families differ in classification.
  Report descriptive measurements without a scaling claim.
- Target paper figure or table: one small RQ4 log-scale line panel or compact
  table, only if the completed run passes independent review.
- Claim exclusions: this does not prove verifier soundness, execution safety,
  GPU-side overhead, attach/JIT/bootstrap latency, cross-vendor portability, or
  behavior at 65,536 instructions.

## Reproducibility Notes

- Software/data versions: record bpftime Git revision, compiler/CMake versions,
  Release configuration, CPU/kernel/cpufreq metadata, and the exact probe/runner
  command. Git revisions are bookkeeping; no content hashes or checksums are
  generated or used.
- Config and seed notes: schedule and bootstrap seeds are both 1797; CPU 23;
  20 blocks; 60-second per-cell timeout.
- Known deviations: synthetic accepted programs isolate verifier scaling but do
  not represent the helper/map mix of production policies. A1 remains the
  separate real-policy measurement and is not pooled with this matrix.
