# MoE predictive-prefetch factorial on RTX 5090

Completed on 2026-09-04 UTC. The campaign has **five valid paired blocks,
20 cells, 120 measured requests, and 7,680 exactly verified output tokens**.
Independent raw audit and a separate read-only OpenCode review both accept the
result. Two failed block-2 attempts remain retained and are wholly excluded.

## 1. Question and result

This experiment follows up the older three-arm MoE-Infinity comparison by
holding the executor, 0.75 cache budget, strict no-overload behavior, model,
prompts, and eviction algorithm fixed while independently toggling:

- native versus userspace-uBPF/JIT execution of the same policy; and
- predictive prefetch off versus on.

Predictive prefetch changes cache behavior substantially but changes throughput
only slightly. With native policy execution, prefetch improves paired throughput
by **0.447%**, with a five-block bootstrap interval of **[0.015%, 0.882%]**.
With BPF execution, the point estimate is **+0.495%**, but its interval
**[-0.058%, 1.051%]** crosses zero. The BPF mechanism costs about **0.555%** in
the controlled prefetch-off comparison; with prefetch on, the BPF/native interval
crosses zero.

This supports the reviewer-facing claim that gpubpf can execute this existing
policy with performance close to the native implementation. It is not a new
headline speedup or a formal equivalence result.

## 2. Fixed protocol and eligibility

- Hardware/software: one RTX 5090, NVIDIA 575.57.08, Linux 6.15.11, and a
  fixed 400 W limit.
- Workload: GPT-OSS-120B, one active request, six frozen prompts per cell,
  512 input and 64 output tokens per request.
- Design: five seeded randomized complete blocks; every block contains
  `native-prefetch-off`, `native-prefetch-on`, `bpf-prefetch-off`, and
  `bpf-prefetch-on` in a randomized order.
- Primary metric: 384 verified output tokens divided by the full six-request
  wall window, including the final prefetch drain.
- Every cell starts a fresh server and cache. Model loading and one excluded
  warm-up request are outside the timing window.
- The same 0.75 cache budget applies to all arms. The temporary prefill
  overload slot is disabled, removing the major executor confound in the older
  baseline comparison.

The required four-arm real [preflight](raw/prefetch-ablation-575/preflight-02/)
passed exact output, engagement, conservation, telemetry, and cleanup gates.
The authoritative [full campaign](raw/prefetch-ablation-575/full-01/) records
the fixed schedule and runtime inventory in its manifest and the producer
statistics in `analysis.json`.

## 3. Correctness, engagement, and excluded attempts

An independent invocation of `audit_prefetch_ablation.audit_block` re-read all
five accepted blocks. It verified the exact launch/environment for each arm,
all raw SSE frames and golden text, 64 output tokens per request, timing-window
arithmetic, telemetry, counter conservation, clean server exit, and before/after
safety state. All 20 cells passed; all servers exited with code zero and no
cleanup error.

Within the measured windows, the two BPF arms each made **69,120 real JIT rank
calls and 69,120 real JIT EAMC-match calls** across their five cells. The BPF
prefetch-on arm also made 136,849 JIT eviction calls. Native arms record zero
BPF calls. Prefetch-off arms record zero speculative copies, hits, or waste;
prefetch-on arms record about 93,000 completed speculative copies.
Temporary-slot activity is zero in every arm.

The full timing run and required factorial preflight both set shadow
verification off to avoid timing duplicate native calculations. Their zero
mismatch fields therefore do not establish per-decision parity. A separate
earlier prediction-protected [three-arm enhanced canary](raw/paper-v3-575/preflight-protected-849ea75d/README.md)
exercised same-snapshot native/BPF selector checking and observed zero
mismatches. Exact same-golden SSE outputs independently establish correctness
for every factorial cell.

Two failed attempts are preserved without reuse or pooling:

- `block-02-attempt-01`: the second cell exceeded the predeclared 900-second
  cold-start readiness timeout;
- `block-02-attempt-02`: the no-build-contention gate detected an unrelated
  `cc1` process during the third cell and aborted the block.

The successful `block-02-attempt-03` reran all four arms from fresh state.

## 4. Performance and traffic

Throughput is the median of five cell throughputs; TTFT is the median of five
per-cell medians. TTFT measures the first nonempty visible text and need not be
the first model token. Demand hit rates are ratios of summed hit/access counters
across the five valid cells. Paired estimates are geometric means of
within-block throughput ratios with percentile bootstrap intervals that
resample whole blocks. Higher throughput is better.

| Arm | Median output token/s | Median first-visible-text TTFT, ms | Decode demand hit rate | Prefill demand hit rate |
|---|---:|---:|---:|---:|
| Native, prefetch off | 11.2594 | 1,550.39 | 93.88% | 36.53% |
| Native, prefetch on | 11.3069 | 1,487.89 | 99.34% | 54.45% |
| BPF, prefetch off | 11.1784 | 1,551.07 | 93.87% | 36.56% |
| BPF, prefetch on | 11.2434 | 1,499.41 | 99.36% | 54.94% |

| Paired comparison | Throughput ratio [95% block-bootstrap interval] | Interpretation |
|---|---:|---|
| Native prefetch on / off | 1.004473 [1.000149, 1.008816] | Small positive interval |
| BPF prefetch on / off | 1.004952 [0.999424, 1.010510] | Inconclusive |
| BPF off / native off | 0.994450 [0.989641, 0.998592] | About 0.56% mechanism cost |
| BPF on / native on | 0.994924 [0.982730, 1.006492] | Inconclusive |

Native prefetch raises prefill and decode demand-hit rates by 17.92 and 5.47
percentage points and reduces demand-copy bytes by **41.72%**. The BPF arm
shows nearly the same directional changes. However, native prefetch also
performs 1.234 TB of completed speculative H2D copies. Demand plus prefetch
bytes are therefore **80.97% higher** than with prefetch off. Of those
prefetched bytes, 38.63% receive a first-use hit, 60.82% are evicted before
first use, and 0.55% remain unused at the final drain.

These byte counters are logical expert-payload H2D bytes, count repeated
transfers repeatedly, and are not PCIe hardware-analyzer measurements.

These counters explain why a much higher hit rate need not produce a large
throughput gain: saved demand misses coexist with substantial speculative copy
and eviction churn. They do not by themselves identify the critical-path
compute or PCIe bottleneck, and observed hit bytes are not a counterfactual
measure of bytes saved.

## 5. Verdict and scope

**Valid supporting result.** For this single RTX 5090, model, memory budget,
prompt set, and concurrency level, predictive prefetch materially changes
residency and traffic but changes throughput by only about 0.5%. The BPF port
has a measurable roughly 0.56% cost without prefetch and no resolved difference
from native when prefetch is enabled. This is strong evidence of policy
expressibility with small mechanism cost, not evidence that the policy is
universally beneficial or that native and BPF are formally equivalent.

The independent OpenCode audit used model
`opencode/ling-3.0-flash-fin-free` in read-only mode with snapshots, sharing,
shell, network, editing, and delegation disabled. Its corrected session is
`ses_f94b5570fffe0gaU4OCJcVmaWj`.
