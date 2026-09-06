> **2026-09-06: 时钟公平性门槛已由用户指示废除；本文档中的 clock-state / P0 / 精确时钟对匹配要求不再适用。仅以性能数据为准。**

# Endpoint lifecycle attempt 07: retained drift-gate failure

Date: 2026-09-04
GPU / driver: NVIDIA GeForce RTX 5090 / 575.57.08
Result directory: `raw/rm-correlation-575-07-endpoint-lifecycle`

## Outcome

The PATH repair worked and the fixed launchlate preflight reached all three
correctness arms. The baseline produced the exact 47-byte oracle. The gpubpf
launchlate loader retained 220/220 matched samples, detached its probes,
completed both RM cleanups and all accounting, and removed its private shared
segment. It then deliberately returned `ERANGE` (process status 34) because its
recomputed RAW/PTIMER drift bound was 20,990 ppb, above the unchanged 10,000 ppb
gate. The paired NVBit arm also retained 220 samples but independently measured
21,912 ppb and failed the same gate. No performance cell or full campaign ran.

The runner mislabeled the gpubpf semantic-gate exit as `private probe did not
exit cleanly`, discarding its structured result from `result.json`; the raw
probe and process evidence remained intact. This is a diagnostic defect, not
evidence that the loader or shared-memory cleanup failed. It also does not make
the correctness cell valid: both instrumented arms violate the frozen drift
bound.

The outer lifecycle correctly removed the endpoint candidate and restored the
exact admitted `gpreempt-849ea75d-6.15.11` module set, parameters, nodes, 400 W
limit, services, and labels on the same boot. It reports no recovery error,
interrupt, Xid, abnormal kernel message, UVM reference, or `struct_ops` residue.

## Repair boundary

The runner now recognizes status 34 only when the full raw log independently
recomputes an above-limit drift rate and proves both RM cleanups, detach,
lossless sample accounting, and pairing. Such a record proceeds to normal raw
parsing and remains invalid under the unchanged 10,000 ppb gate. Any missing
marker, malformed arithmetic, incomplete cleanup/accounting, different tool,
or different nonzero status remains a hard private-probe error.

This repair improves failure attribution and evidence retention. It does not
relax the clock model, rescue attempt 07, or justify another GPU retry without
a separately predeclared clock-design change.
