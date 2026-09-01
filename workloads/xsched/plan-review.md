# Independent plan review

## Round 1 — 2026-08-31

Reviewer: `/root/xsched_plan_review` (fresh independent agent).

Verdict: **approve with required repairs**.

Required repairs were: retain the paper's one-kernel/~80-ms unit and 400 LC samples; define a host-submission-to-actual-device-start oracle that includes XSched interception; use a common BE-throughput interval; implement a recorded two-phase BE-then-LC arrival; use XSched's official role-specific `16/8` and `4/2` launch settings; require one gpubpf-capable driver stack for every configuration; replace XQueue-state inference with passive evidence of actual Level-1 suspend/resume and verify HPF, queue count, priority, and level; freeze exact commands/cleanup; and predeclare a BE non-inferiority margin with paired bootstrap analysis.

Nonblocking notes: explicitly unset `XSCHED_CUDA_LV3_IMPL`, do not claim fairness without a fairness metric, and record the previously built upstream example artifact.

## Author response

All required repairs are incorporated in `plan.md`, `priority_workload.cu`,
`run_xsched_rq3.py`, and the hashed `xsched-engagement.patch`. The primary
question remains tail latency plus BE throughput; no fairness result will be
claimed from this experiment. Runtime remains gated on an idle RTX 5090 and
the exact custom 575.57.08 gpubpf driver/BTF stack.

## Round 2 — 2026-08-31

Verdict: **approve with required repairs**.

The reviewer confirmed that the paper-scale sample count, BE denominator,
official XSched role settings, stack/idle admission, randomized blocks,
scoped cleanup, and two-phase release were repaired. Remaining required
repairs were: restore device-start rather than completion latency as primary;
make entry the minimum over all CTAs and barrier the final block; restore the
paper's 1-second LC timeslice; ensure no pre-measurement kernel can satisfy
engagement; make the patch deterministically applicable and hash runtime
libraries; rebuild gpubpf loaders from source; replace per-process single-core
pinning with an equal adequate mask; and encode the full statistical decision
rule. It also recommended unique-handle and actual threshold/batch gates and a
frozen isolated checksum.

## Round 2 author response

All eight required repairs and all three nonblocking recommendations are now
implemented. Workers report `ready` without launching a kernel, so XSched and
gpubpf engagement counters cannot be satisfied before the recorded release.

## Round 3 — 2026-08-31 (final)

Verdict: **reject; proposal closed without GPU execution**.

The final reviewer found two remaining blocking defects. First, the primary
latency subtracts a CUPTI `cuptiGetTimestamp()` value from an inline PTX
`%globaltimer` value without proving that those clocks have the same epoch. An
ordering check detects only one offset direction and cannot rule out a positive
constant offset. Second, the encoded result categories overlap: because the
`negative` test precedes `mixed`, a clear latency improvement plus a clear BE
throughput regression is mislabeled negative rather than a tradeoff.

The reviewer also noted stale plan prose that still describes the submitted
RQ3 metric as submission-to-completion. All other round-2 repairs were verified,
including CTA timing/barrier mechanics, timeslices, post-release engagement,
24 unique XSched queues, exact patch/build hashes, equal CPU affinity, and
frozen checksums.

The research-experiment-design protocol permits at most three review rounds.
Therefore this proposal is closed and `run` is not authorized. Any future
clock-domain calibration and mutually exclusive interpretation logic are
preparation for a distinct proposal, not retroactive approval of this one.

## Offline no-fingerprint repair — 2026-08-31

The repository-wide no-fingerprint rule supersedes the earlier review requests
for file fingerprints and aggregate output checks. The inactive experiment was
repaired offline without reopening its scientific review or authorizing a GPU
run: admission now compares the exact small reviewed XSched diff and records
ordinary file metadata; the CUDA harness recomputes the 32 lane-specific
floating-point recurrences on the host and checks every copied output value.
The Python adapter compiles, the sm_120 harness rebuilds, and read-only admission
correctly remains blocked by the loaded 610.43.02 driver and missing core NVIDIA
BTF hooks. The two round-3 scientific blockers remain unchanged.
