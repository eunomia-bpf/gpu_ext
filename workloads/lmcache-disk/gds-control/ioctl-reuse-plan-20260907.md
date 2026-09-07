# LMCache: same-policy decision-path allocation reuse

## Question and scope

RQ2 (Mechanism Cost): What does the gpubpf mechanism cost, in hook and
observability overhead and in executing the same policy through gpubpf?

This supporting optimization tests whether reusing the scalar ioctl buffer
and ctypes view reduces Python-side BPF decision overhead without changing
the policy, decision count, driver ABI, or shared storage executor. The
completed 200 ms write-budget study already answers the policy-parameter
question; none of its 25 cells is repeated or replaced.

The existing admission diagnostic measured native/BPF decider medians of
1.827/17.962 us under concurrent storage traffic. Those intervals also include
scheduling and locking. They are not a prediction of isolated call cost.
Hundreds-of-milliseconds storage tails may remain unchanged even if allocation
reuse helps. A direct measurement of the new path is missing from old data.

## Comparison and interpretation

Use the existing native algorithm, legacy BPF, and opt-in reusable-buffer BPF.
Every BPF admission still invokes the same live command-82 policy exactly once.
Keep GIL release, locks, policy inputs and error behavior; no native shortcut
for demand reads. Legacy remains the default and all adverse results remain.

First measure six rotated blocks of 1,000 decisions per arm, with 100 warmup
calls per arm. All arms receive the same deterministic request sequence and
IDs. Retain per-call nanoseconds, returned actions, errors, aggregate call-loop
throughput and exact commands in a new raw directory. Compare paired block
medians and ranges, not historical instrumented timings. There is no numerical
acceptance threshold, extra preflight, clock study, timeout, or retry campaign.

This is an actual driver-decision benchmark, not a storage-transfer benchmark.
If results show negligible or inconsistent savings, record that limit and do
not rerun the completed storage matrix just to seek favorable numbers. If the
change plausibly affects the application path, follow with a separately
recorded new/legacy/native 200 ms comparison using the existing real LMCache
mixed-storage runner. Report read tails together with write throughput;
do not infer an end-to-end gain from an isolated-call improvement.

Larger driver ABI changes or policy tuning are alternatives, but would no
longer isolate this same-policy implementation change. This small experiment
tests an actionable local cost first without a module reload.

## Ownership and execution

Root plans, reviews, runs measurements and publishes results. Local OpenCode:
Qwen 27B owns the opt-in adapter change and focused development tests; GLM
reviews the plan and existing bottleneck evidence; Qwen Next owns the small
measurement script. At most three sessions, with no artificial short timeout.
No worker changes existing results, third-party code or paper text.

Runtime: RTX 5090 host, Linux 6.15.11, NVIDIA 575.57.08, existing attached
storage policy. The loader is already live (PID 996479 at inspection).
An attempted duplicate attach returned `File exists` and changed no policy;
the system `bpftool struct_ops show` crashed, so it is not used for this task.
No module reload or replacement is needed.

The script and exact invocation will be recorded with the measured result.
Raw destination: `raw/gds-ioctl-reuse-575-20260907/`, preserving all outputs.
