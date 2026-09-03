# Independent read-only audit of the fixed-bound ablation

2026-09-03: the independent `eb_shadow_bridge` reviewer returned **valid,
supporting ablation; no blocking finding**, after reviewing all 40 full cells
and eight separate preflight cells on CPU 17. This is an additional audit of
the existing campaign, not another experiment or another OpenCode review.

The reviewer ran the existing analyzer and a separate, read-only stdlib check
of raw client logs, CSV telemetry and JSON records. The latter script/output
remain in the agent tool conversation and were **not saved as a second
repository analyzer**. This document transcribes its handoff, not a raw
execution transcript. The preserved, reproducible analysis entry is:

```sh
taskset -c 17 python3 -B workloads/hummingbird/pipeline/analyze_study.py \
  workloads/hummingbird/pipeline/raw/full-575-01
```

The full campaign has 240,000/240,000 LC requests completed within their
windows. All 344,236 BE requests were verified, with 344,196 within the windows
and 40 normally completing after their boundaries. The 584,236 timed output
checks each cover 1,000 float32 values, with finite-value and tolerance checks;
the saved maximum absolute errors are all zero. These are deterministic
seed-0 exported-model correctness checks, not pretrained-model accuracy.

The event accounting independently reconciles 106,089,548 issued and retired
events, zero final outstanding, 55,927,238 overlapping launches, and exactly
3,839,761,392 actual JIT decisions for BPF / zero for C. Every depth-2 cell
reaches two outstanding events; every depth-1 cell reaches one. Raw CTA,
no-op and request-lifecycle accounting agrees with the recorded results.

The reviewer also checked all 14 current runtime paths/sizes/mtimes against
48 cells, the 12 model files and shared model specification, and the full
telemetry. The formal window has 13,601 telemetry samples, maximum interval
0.222 seconds; all pre/post gates are clean on 575.57.08 at 400 W. No boot ID
was recorded by this campaign, so this is not a recorded boot-identity check.

The raw-derived medians and 10,000-draw paired confidence intervals agree
with the [result report](results-575-20260903.md): increasing the fixed bound
raises BE goodput about 14.9–15.0% in both C and BPF. However, BurstGPT LC SLO
attainment falls 0.440 percentage points for C and 0.560 for BPF; unchanged
foreground protection is not established. BPF/C BE goodput at BurstGPT depth 1
is −0.242% (95% interval −0.348% to −0.135%). Other BPF/C BE intervals include
zero change; that is not an equivalence proof.

This is a same-profile, shared-executor **fixed-bound ablation**. Outstanding
host events are not hardware queue occupancy or measured preemption latency.
The host-uBPF path is not a SIMT-verifier test. No fixed/GPreempt arm exists in
this new campaign, so it cannot be pooled with the older study to claim the
entire earlier 19–20% gap is closed. No new policy or full-system SOTA victory
is established by this audit.
