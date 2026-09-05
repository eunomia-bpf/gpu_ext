# Experiment Plan: RQ1 outcome-pressure adaptive expert prefetch

Status: proposed; parameters and held-out cohort are frozen before real
preflight.  This experiment does not edit the paper.

## Research Question

- RQ exactly as written in the paper: “How much performance gain can gpubpf's
  programmable memory and scheduling policies provide on oversubscribed
  single-tenant workloads?”
- Specific uncertainty tested here: whether a byte-budget governor driven by
  completed speculative-copy outcomes and live demand pressure can preserve
  useful overlap while avoiding the large unused-copy churn measured for the
  existing MoE-Infinity policy, and whether host BPF executes the same governor
  with similar end-to-end performance as native C.
- Why the answer matters: the prior controlled GPT-OSS-120B campaign found only
  about 0.5% throughput benefit from unbounded prediction while 60.82% of
  completed speculative logical bytes were evicted before first use.  Reviewer
  guidance explicitly asks for new policies enabled by the mechanism and for
  policy benefit to be separated from mechanism cost.

## Paper-Value Admission

- Planned role: supporting, with potential to become a reviewer-facing new-policy
  result.
- Largest credible paper story: gpubpf can express a closed-loop policy that
  combines full-stack copy outcomes and current executor pressure, rather than
  merely reproducing a static selector, while retaining native-policy performance.
- Load-bearing uncertainty: unused-copy reduction may not improve throughput;
  a simple fixed cap may be sufficient; or demand-only may remain best.
- Independent evidence: the experiment adds temporal A→B→A adaptation, exact
  outstanding-byte accounting, live mandatory-demand pressure, a fixed-budget
  discriminator, and matched native/BPF implementations.  None is present in
  the completed unbounded on/off factorial.
- Non-tautology: the primary metric is full serving throughput, not the
  controller's own byte objective.  The fixed cap and demand-only arms can win.
- Positive decision: report a new outcome/pressure-guided policy, separately
  identifying its policy effect and BPF mechanism cost.
- Contradictory/mixed decision: retain a boundary result—fixed admission or no
  speculation is preferable on this workload—and do not advertise adaptation.
- Best alternative: a SpecMD Least-Stale eviction port has high direct-overlap
  risk and needs a new victim ABI; this admission governor reuses the already
  audited predictor/executor and directly attacks measured churn with less
  integration risk.

## Expected And Alternative Outcomes

- Explicit hypothesis: on the frozen real GPT-OSS-120B A→B→A workload, the
  adaptive host-BPF governor improves drain-inclusive output throughput over
  the original unbounded native policy while reducing evicted-unused completed
  speculative logical bytes; adaptive BPF and adaptive native C make identical
  same-snapshot decisions.
- Strongest competing explanation: one background copy worker and frequent
  prediction replacement already bound physical execution, so byte admission
  only removes copies that were harmlessly overlapped; alternatively a fixed
  byte cap captures all benefit and feedback adds none.
- Contradiction: adaptive BPF does not improve paired throughput over unbounded
  native, increases evicted-unused bytes, fails decision parity, or loses to the
  fixed cap without a distinct A→B→A response.

## Published Precedent And Real Assets

- Closest protocols: MoE-Infinity v3 activation-aware prediction and the
  completed matched GPT-OSS-120B prefetch factorial; FineMoE's probability-set
  selection; SpecMD's pass-aware eviction; APEX's confidence-controlled extra
  experts.
- Official assets: the existing MoE-Infinity source revision, GPT-OSS-120B
  checkpoint/store, 575.57.08/6.15.11 stack, paper-v3 predictor, strict common
  cache executor, exact SSE frontend, and host-uBPF/JIT bridge.
- Reused: prediction order, EAMC history, scored eviction, copy worker, cache
  budget, demand execution, output oracle, telemetry, and safety gates.
- Necessary glue: one bounded integer snapshot ABI, a native-C and bytecode
  implementation of the same governor, exact outstanding/background-discard
  accounting, per-request governor history, and a five-arm runner/auditor.
- Scope boundary: this is not FineMoE, SpecMD, or APEX.  It does not alter the
  probability threshold, predictor, victim policy, or mandatory expert routing;
  it only caps an already-ranked speculative prefix.  FineMoE's Qwen-specific
  semantic/probability predictor is not relabeled as a GPT-OSS baseline.

## Comparison

Main baselines:

1. `demand-only`: strongest null—same predictor work, executor, scored native
   eviction and cache budget, but zero speculative admission.
2. `unbounded-native`: strongest matched published-policy implementation—the
   existing MoE-Infinity v3 native predictor/ranker with unrestricted positive
   candidate publication.

Controls and mechanism ablations:

- `fixed-native`: the same live demand/outstanding headroom rule at a fixed
  512 MiB byte budget; distinguishes feedback from bounded admission.
- `adaptive-native`: native-C outcome/pressure governor; isolates BPF mechanism.
- `adaptive-bpf`: identical integer snapshot and decision rule through actual
  host-uBPF JIT, with no fallback; proposed arm.

All five share predictor, ranking order, executor, eviction, 0.75 sparse-cache
budget, temporary-overload disabled, model, request sequence and CPU affinity.
The governor may select only a prefix of the original rank.  Mandatory demand
always bypasses the governor.  If demand-only wins, speculation is not useful
here; if unbounded wins, throttling destroys useful overlap; if fixed wins,
adaptation adds no demonstrated value; if adaptive-native wins but BPF loses,
the mechanism cost matters.

## Frozen Governor

All byte arithmetic is unsigned 64-bit with checked/saturating operations.  The
budget parameters are frozen from the previous campaign's observed
13,246,464-byte expert payload—not from this experiment's held-out outcomes:

- initial/fixed budget: 512 MiB;
- minimum reopening budget: 64 MiB;
- maximum budget: 2 GiB;
- maximum adjustment quantum: 128 MiB.

At each completed request, after speculative drain, the controller receives
newly realized first-use bytes, newly evicted-unused bytes, currently censored
resident-unused bytes, and new demand-wait time.  It moves the current budget
toward higher or lower speculation according to the signed useful-minus-unused
fraction, with the adjustment continuously damped by the censored fraction.
Any observed demand wait forces a decrease of at least one quantum.  With no
resolved outcome it holds the budget.  The 64 MiB floor is the bounded reopening
probe and avoids feedback lockout; censored residency is never counted as
failure.

At every ranked-prefix publication, a second governor event subtracts exact
currently outstanding speculative bytes and exact current mandatory-demand
fetch bytes from the current budget, then admits only the largest whole-expert
prefix fitting the remaining bytes.  The fixed control uses this same live
headroom rule but never changes its 512 MiB budget.  This is adaptive byte
admission, not a renamed probability threshold.

## Workloads And Metrics

- Real workload: one RTX 5090, GPT-OSS-120B, batch/concurrency one, 512 input and
  64 generated tokens per request, real expert offload and 0.75 cache budget.
- Formal sequence per cell: two frozen group-A requests, two disjoint group-B
  requests, then the same two group-A requests again.  Policy/cache/history are
  preserved across A→B→A and reset only between arms.  The cohort is selected
  from the existing ShareGPT source with seed 20260904, excludes every source
  row in `prompts.json`, and is frozen before preflight.  It is held out from
  controller parameter selection; it is not claimed to represent a natural
  production distribution.
- Primary metric: 384 verified output tokens divided by the full six-request
  wall interval including every request-boundary and final speculative drain.
- Secondary application metrics: request E2E, first-visible-text TTFT, total
  demand-prefetch/cache wait time, and A1/A2/B1/B2/A1'/A2' E2E.
- Mechanism metrics: proposed/admitted/dropped candidate count and logical
  bytes; queued/in-flight outstanding speculative bytes; live demand pending
  count/bytes; completed first-use, evicted-unused and censored-resident bytes;
  copy starts/completions; budget/decision trace; native/BPF same-snapshot
  mismatches and JIT calls.
- Correctness: every arm must emit exactly the frozen demand-only reference SSE
  text, 64 token frames plus DONE per request, with no abort/drop.  Copy counts
  and bytes must conserve after drain; outstanding speculative and demand bytes
  must return to zero.  Logical bytes are not PCIe traffic.
- Repetitions: five seeded randomized complete paired blocks; whole-block
  geometric ratios and 10,000-draw paired bootstrap 95% intervals.  Intervals
  crossing one are inconclusive, not equivalence.

## Planned Runs

| Run group | Role | Workload | Arms | Repetitions | Decision consequence |
|---|---|---|---|---:|---|
| CPU | dependency | ABI/rule boundary cases and queue concurrency | native C/BPF oracle | deterministic | Exact parity and conservation only. |
| preflight | dependency | Real GPT-OSS-120B A→B→A | all five | one cell each | Real output/copy/BPF/pressure path works; no directional gate. |
| full | supporting | Frozen held-out A→B→A | all five | five paired blocks | Tests the explicit hypothesis. |

## Execution

- Authoritative workflow: `run_adaptive_prefetch.py`; it extends the existing
  exact-SSE, safety, telemetry and lease machinery rather than replacing the
  inference engine.
- Real preflight: all five arms, one complete A→B→A cell each, using the same
  built extension and runtime inventory as formal timing.  At most three real
  preflight attempts; every failure remains retained.
- Full completion: 25 valid fresh cells, 150 verified requests, 9,600 output
  tokens, five whole paired blocks, exact mechanism engagement, clean teardown,
  and independent raw review.
- Raw paths: `raw/adaptive-prefetch-575/preflight-*` and
  `raw/adaptive-prefetch-575/full-*`.
- Recovery: retain every attempt and rerun an entire failed paired block; never
  pool a partial block.

## Interpretation

- Positive: adaptive BPF improves paired drain-inclusive throughput over
  unbounded native, reduces evicted-unused logical bytes, shows a budget response
  across A→B→A, and its same-policy native comparison is acceptably small and
  directionally disclosed.
- Negative: preserve and report fixed/unbounded/demand-only winner and the BPF
  mechanism cost; do not tune or weaken the hypothesis after seeing formal data.
- Mixed/inconclusive: report waste and throughput separately and avoid causal or
  superiority wording.
- Target: one policy-versus-mechanism panel plus a small A→B→A budget/outcome
  trace, only if result review accepts the run.

## Reproducibility Notes

- No file hashes, checksums, fingerprints or digests are generated or used.
  Ordinary Git revisions, explicit file metadata, exact outputs, tests and real
  engagement provide evidence.
- The fixed cohort, schedule, parameters and oracle are immutable after the
  first real preflight starts.  Correctness or executability repairs rerun all
  affected cells and are recorded as deviations.
- OpenCode review is read-only/deny-all and advisory; a timeout is no verdict.
