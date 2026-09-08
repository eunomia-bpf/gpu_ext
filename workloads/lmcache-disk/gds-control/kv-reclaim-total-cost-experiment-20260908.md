# Total-recovery-cost reclaim ablation

Status: implementation assigned; no new performance cells launched.

This follows the completed [five-block grace comparison](../results-575-gds-kv-reclaim-grace-20260908.md).
It addresses Q1/storage offload and the shepherd's policy-versus-mechanism
question without adding another paper reproduction or editing the paper.

## One uncertainty

Does ranking by total recovery cost, rather than recovery cost per freeable
byte, reduce the repeated disk restoration seen with mixed-length requests?
The old policy frees more bytes for its estimated cost but can repeatedly
evict a long request. This ablation tests the denominator's effect; it does
not assume a complete causal explanation or claim algorithmic novelty.

Change only the ranking: choose the unique minimum estimated recovery ns
among the same eligible candidates. Preserve the same min(disk-prefix read
plus unsaved-tail recompute, full recompute) estimator, priority semantics,
tie-to-stock rule, backing metadata and recovery routes in native and BPF.
The default per-byte algorithm and its binaries/results remain intact.
Separate native/BPF binaries compile the shared selector in total-cost mode.
No driver ABI change or driver reload is needed: the existing record helper
does not enforce a specific cost formula.

The current interface does not expose the allocation shortfall. Therefore
absolute cost can choose a victim that frees too little and require another
preemption. More restores, poorer throughput, or worse TTFT would contradict
the proposed improvement; no policy advantage is assumed in advance.

## Matched comparison and execution

Use four arms in each of five arrival-rotation blocks:

1. Stock victim policy, common grace-patched runtime.
2. Original native cost-per-byte policy, concurrent control for the policy change.
3. Native total-cost policy.
4. BPF total-cost policy, same algorithm as arm 3.

These twenty new cells test the changed policy with fresh matched controls;
they do not overwrite or repeat a completed output directory. Repeating the
old BPF cost-per-byte arm is unnecessary: its original native/BPF comparison
is complete. Retain its old numbers, but do not treat historical measurements
as newly paired total-cost controls.

Rotate the four execution arms by block index; use the existing
`warm_arrival_order(block, 8)` identically within a block. Reuse the same
RTX 5090/575, Qwen3-30B-A3B-FP8, LMCache/cuFile disk backend, 384 MiB KV
pool, 256 MiB GDS buffer, two running sequences, four HTTP workers, 250 ms
spacing, 1536/1024 alternating prompts, 1024 output-token bound and 62502
ns/token recompute price. Keep async scheduling and the common grace patch.

Root invokes `run_gds_kv_reclaim.run_cell` with the existing GPU and
struct-ops locks. The native total-cost arm selects the separately built
library through its recorded environment; the existing BPF loader attaches
the separately built total-cost object. Preserve the old attached-policy
command for restoration. Use runner `a1011e64` consistently in all new arms;
its completion-progress writes occur inside the measured warm interval.
No completed cell is rerun, no cold recalibration is repeated, and no
additional clock test, preflight, audit, accuracy requirement or timeout is
added. Full execution means terminal records for all twenty cells, including
adverse outcomes, not obtaining a desired ranking.

Planned output root: `raw/gds-kv-total-cost-575-20260908-01/`. Record the
policy label and library/object choice alongside each raw result so the two
native implementations cannot be conflated. Actual build commands and
artifact paths will be recorded when implementation finishes. Expected GPU
cost is roughly one hour, not an imposed timeout.

## Interpretation

Primary metrics remain completed-output token/s and per-cell median TTFT.
Pair native total against native per-byte for the policy change, BPF total
against native total for mechanism cost, and both total-cost arms against
stock for practical benefit. Report all block values and paired dispersion.
Existing restore-batch/payload and rollback log lines help interpret the
change but are not physical SSD byte counters or a separate experiment.

A gain supports this ranking change on the measured pressure workload;
a loss or mixed result motivates allocation-deficit/re-admission feedback,
not a reduced workload or deletion of unfavorable data. Neither result
establishes cross-process physical HBM reclamation or transparent disk-UVM
paging. The independent UVM backing-route analysis remains a separate
implementation question, not a substitute performance claim.

## Alternative considered during implementation

The local analysis suggested subtracting `disk_backed_bytes` from
`freeable_bytes` before ranking, as an approximation to capacity remaining
after a future restore. That is not the selected ablation. The existing ABI
deliberately distinguishes whole-object transfer bytes from exclusive,
actually freeable KV bytes, and a full-recompute route need not read that
backing object at all. Future restored occupancy also differs from capacity
released now. Unconditionally subtracting these quantities would introduce
another unvalidated model rather than isolate the original denominator.
The total-cost comparison stays unchanged; a later re-admission-aware policy
would need to define the relevant allocation deficit and occupancy explicitly.

Source/runtime preparation: the existing loader accepts a BPF object path
as its first argument. Before the new build, the old loaded-policy process
is PID 1783868, running `kv_reclaim_loader kv_reclaim_policy.bpf.o` via
absolute paths under this directory. The original BPF object is 42544 bytes,
native library 28568 bytes, and loader 1517248 bytes. These sizes are an
inventory, not a content identity check. Root will recheck the exact live
process while holding the existing locks before any policy switch; the
new build must not overwrite these original experiment assets.
