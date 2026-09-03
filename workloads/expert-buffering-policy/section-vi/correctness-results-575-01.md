# Section VI real GPU correctness: three arms pass

2026-09-03: FIFO, the Section VI native policy, and its identical host-uBPF
implementation all pass the same Qwen/FineMoE executor's real GPU preflight.
Root independently reaudited every saved launch, cleanup, telemetry record
and all **27 retained logits arrays** against the unchanged original-HF
reference: **65,636,352 float32 values match exactly**, at the frozen 0.0
tolerance. All 432 generated tokens across the three one-warmup/eight-evaluation
cohorts also match. This is correctness evidence, not performance measurement.

Actual BPF-first/native-after shadow validation covers all **20,182 real JIT
decisions**, including warmup, with zero mismatches. The BPF output determines
the actual action; the native check does not replace it. Native and BPF final
decision, admission, eviction, layer and residency counts agree exactly.
All three arms exercise all 24 layers, real whole-expert eviction and completed
copies at K=16. Each retains 6,643,777,536 sparse-expert bytes within the unchanged
16,834,658,304-byte strict pool. No speculative copying or copy/compute errors
are observed, and all owned processes/telemetry are cleaned up.

Evaluation-only accounting excludes warmup:

| Arm | Decisions | Actual JIT calls | Completed demand copies / evictions | Logical copied bytes |
| --- | ---: | ---: | ---: | ---: |
| FIFO | 17,959 | 0 | 13,304 | 230,179,209,216 |
| Section VI native | 17,959 | 0 | 11,595 | 200,610,938,880 |
| Section VI BPF | 17,959 | 17,959 | 11,595 | 200,610,938,880 |

The byte column counts completed whole-expert copy payloads, **not measured
PCIe traffic**. Fewer copied bytes are not a throughput result; this preflight
retains numerical arrays and the BPF shadow, so its timing is not reported.
FIFO is the same-K cache comparator used for this Section VI policy study,
not original UVM or FineMoE's trajectory-prefetch default. Both paper-policy
implementations share the same sequential executor, actual current-batch
gating, byte budget and whole-expert actions. This is a policy port, not the
original paper's distributed system or original-model throughput reproduction.

## Evidence and next admission

The complete [raw campaign](raw/575-section-vi-correctness-01/campaign.json)
contains worker logs, telemetry, launch/result records, all per-request
numerical records and complete arrays: 71 files, 268,387,869 bytes. Its three
arms ran from 16:14:58 to 16:18:23 UTC. The independent
[root audit](raw/575-section-vi-correctness-01/root-audit.json) re-runs
`validate_preflight`, including `audit_retained_logits`, and checks the current
94-entry runtime inventory, original model/reference file inventories and
actual checkpoint decoding configuration. No original golden was regenerated
or tolerance relaxed. Both native and BPF pass the real post-conversion
configuration gate noted in the [OpenCode review](opencode-review.md).

GPU and struct-ops leases were held on driver 575.57.08. GDM and persistenced
were stopped for the campaign and restored afterwards; the
[coordinator log](../../../docs/experiment/revision-safety/eb-section-vi-correctness-01/coordinator.log)
records each arm and service restoration. The untimed preflight overlapped
source/documentation work and scoped Git publication, not another GPU workload.
Only this new non-cache raw directory was reassigned from root to the repository
user for publication; content/mtime and original model/cache files were unchanged.

The frozen same-worker `full` path is now admitted with this exact runtime,
K=16 and preflight: five randomized complete FIFO/native/BPF blocks, seed
20260903. It uses the direct real selector, no shadow, no logits capture, and
retains exact generated-token checks. **The 15-cell performance matrix is
not completed by this report.** No performance-dependent tuning was performed.
