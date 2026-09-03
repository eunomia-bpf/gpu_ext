# Section VI preflight 02: repaired runtime passes all three arms

2026-09-03: after the owned-process cleanup repair, the fresh FIFO/native/BPF
preflight completed with exit 0. Root independently reloaded and compared all
**27 complete float32 arrays / 65,636,352 values**, with exact equality to the
unchanged original-HF reference. All **432 generated tokens** match. Actual
BPF-first shadow validation performed **20,182 JIT decisions**, zero mismatches.
Native and BPF final decision, admission, eviction and residency counts agree.

All three arms again exercise 24 layers, real whole-expert copying/eviction,
K=16 and the unchanged 16,834,658,304-byte strict pool. The interpretation and
same-executor policy-port boundary are unchanged from
[preflight 01](correctness-results-575-01.md). This is not a full original
distributed-system reproduction or a performance result.

The [root audit](raw/575-section-vi-correctness-02/root-audit.json) re-runs all
saved launch, numerical, engagement, telemetry and cleanup gates, verifies the
current **94-entry runtime inventory**, model files and original reference,
and reconciles all three actual worker results. The
[complete new campaign](raw/575-section-vi-correctness-02/campaign.json) has
**71 files / 268,404,193 bytes**, including all arrays, logs and root audit.
The [coordinator log](../../../docs/experiment/revision-safety/eb-section-vi-correctness-02/coordinator.log)
records all three passes and both services restored to active. No GPU compute
client remains. Only this new non-cache directory's ownership changed for
publication; the original model, golden and cache were not modified.

This untimed preflight ran without the formal CPU-affinity guard and overlapped
light source work, scoped Git publication, and the separate one-second
[CPU-only affinity canary](../../../docs/experiment/revision-safety/eb-affinity-canary-01/results.md).
Its timing is therefore neither reported nor pooled with performance cells.
It admits full attempt 02 at the same fixed parameters with the reviewed
external guard: five fresh randomized FIFO/native/BPF blocks, no shadow or
logit capture, and exact generated-token checks. The entire
[interrupted full attempt 01](full-01-abandoned.md) remains excluded.
