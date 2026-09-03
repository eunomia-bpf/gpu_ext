# Independent audit: Hummingbird full 575 study

Audited 2026-09-03, after campaign exit. **Complete: 50/50 cells accepted;
no pending, rejected, unexpected, or discarded cells. The throughput-recovery
hypothesis was not supported.** Raw evidence:
[`raw/idle-study-575-01/`](raw/idle-study-575-01/).

## What was checked

The existing [read-only analyzer](analyze_study.py) was rerun on CPU17. A second
stdlib-only calculation independently decoded every request from `client.log`,
reconstructed the randomized matrix, checked `completed-cells.json` against
individual results, and recomputed the key paired estimates without importing
either experiment analyzer. Results agreed, apart from floating-point rounding.
No GPU execution, build, threshold change, raw rewrite, or sample selection was
performed. The analyzer's safety validators were reused, not independently
reimplemented.

- Matrix: five blocks × periodic/BurstGPT × native/fixed/equal-timeslice/C/BPF;
  all 60-second windows occur in the recorded serial order. Safety snapshots
  span 09:51:40–10:53:23 UTC. All cells record driver 575.57.08, the 400 W cap,
  unchanged inventories of nine runtime artifacts, and passing raw telemetry
  and pre/post safety checks.
- Requests: **768,248 timed outputs checked**, plus 11,000 untimed checks;
  every started request has an ordered arrival/start/verified-completion record.
  Full-output checker counts match every request; recorded maximum absolute
  error is zero for both models (`atol=1e-6`, `rtol=1e-4`). This reconciles actual
  runtime checks, rather than rerunning inference or reconstructing unsaved
  output tensors.
- LC: 300,000 offered, 299,683 completed inside the window, five verified late,
  and 312 never started. **All 317 misses belong to BurstGPT equal-timeslice
  control.** Every other LC cell has 100% window completion. All 50 BE cells
  have one verified late request, excluded from window goodput.
- Engagement: C records 1,930,854,357 decisions and zero JIT calls; BPF records
  1,907,838,828 decisions, all through the actual host JIT. Each idle cell has
  real split, whole, small-output and large-interval launches; exact total
  CTA/copy and HP/BE request counts agree. LP in-flight maximum is one.
  Both arms use the same selected profile: input-small disabled, output-small
  enabled; its contents match the completed qualification's selected profile.

## Paired results and negative findings

Ratios are five-block geometric means; brackets are the frozen 10,000-draw
whole-block bootstrap 95% intervals, seed 20260903. SLO differences are mean
percentage points over **all offered LC requests**, with a frozen 1.811879 ms
threshold. LC p99 is scheduled-arrival-to-verified-output, not service-only time.

| Scenario / comparison | BE goodput ratio ↑ | LC response-p99 ratio ↓ | SLO difference, pp ↑ |
| --- | --- | --- | --- |
| Periodic, BPF/C | 0.99428 [0.98598, 0.99960] | 0.82526 [0.54485, 1.02067] | −1.743 [−3.483, −0.713] |
| BurstGPT, BPF/C | 0.99417 [0.95256, 1.03643] | 1.02046 [0.96018, 1.09275] | −0.233 [−1.150, +0.897] |
| Periodic, BPF/fixed | 0.80685 [0.79792, 0.81333] | 0.76151 [0.53661, 1.02922] | +1.467 [−3.723, +9.447] |
| BurstGPT, BPF/fixed | 0.79693 [0.77145, 0.81358] | 0.74287 [0.48002, 1.07612] | −0.060 [−2.627, +4.017] |

BPF therefore loses approximately 19%–20% BE goodput against fixed GPreempt;
the C port also loses throughput (C/fixed ratios 0.81149 and 0.80160). Neither
port meets the predeclared throughput-recovery win criterion. BPF/C has a small
periodic BE loss and lower SLO attainment; BurstGPT estimates remain uncertain.
These results establish neither equivalence nor negligible overhead.

Keep all tail observations: periodic C block 1 has p99 5.297834 ms versus
BPF 1.909046 ms; the other four BPF/C block ratios exceed one. This explains
why the aggregate p99 ratio and medians appear inconsistent; it does not justify
discarding that block or claiming BPF is faster.

Against equal-timeslice control, periodic BPF substantially improves LC
protection (p99 ratio 0.28645 [0.28311, 0.28982]; SLO +89.417 pp
[86.160, 92.040]) while losing BE throughput (ratio 0.76375
[0.75849, 0.76708]). BurstGPT control completes only 29,683/30,000 LC requests
inside its windows and its median 1,994.245 ms p99 is **conditional on started
requests**. Do not turn that conditional ratio into a headline tail-speedup
or a retained-protection win. Its all-offered coverage/SLO deficits remain real
negative evidence for the ungated control.

## Execution domain and claim boundary

This is a **paper-described scheduling-component port**, not the original
Hummingbird artifact or its published hardware results. BPF decisions run in
the CPU/host ubpf JIT, with no C fallback on BPF load/JIT failure; the shared
trusted CUDA executor performs launches. This is neither kernel `struct_ops`
BPF nor GPU-injected BPF. C/BPF share source-level grid splitting, equal
1,000,000 µs context timeslices, and an event-query-before-next-launch fence.
Fixed GPreempt has different protection/timeslice machinery, so idle/fixed is
a scheduling-package comparison, not pure interpreter overhead.

Declared deviations remain: source transformation instead of generic PTX
rewriting, a stronger completion fence and bounded backlog, only qualified
output-small bubbles, no predictor, multi-GPU offloading, automatic framework
pattern discovery, or multi-LP fairness evaluation. The measured throughput
loss does not identify which deviation caused it; that requires a separate
ablation, not an inferred explanation.
