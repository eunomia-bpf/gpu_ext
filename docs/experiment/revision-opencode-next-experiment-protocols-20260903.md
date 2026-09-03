# OpenCode follow-up experiment protocols — 2026-09-03

Status: **read-only design work; no GPU cell, module operation, or performance
measurement was run.** These protocols do not change any existing result.

Two OpenCode 1.18.27 sessions reviewed direct attachments in parallel with
`snapshot:false`, sharing disabled, and write, shell, network, task, and
external-directory access denied:

- `ses_f9666b24affeK61h3p52Hv95Su`: POD setup/steady-state decomposition.
- `ses_f9666b222ffepDmFo6FdcgFsin`: stale-state/thrashing sensitivity.

The separately reviewed scheduler-init protocol and its two OpenCode event
records are in
[`revision-safety/sched-init-native-candidate-575-01/`](revision-safety/sched-init-native-candidate-575-01/).
All three are candidate protocols, not experiment evidence.

## 1. POD device-BPF setup versus recurring cost

The existing 250-cell POD study establishes operator correctness, real
device-BPF participation, and steady operator cost, but its process wall time
combines Python/import work, loader setup, PTX extraction and compilation,
attachment, first launch, diagnostics, and the numerical scan. It therefore
cannot identify the source of the retained roughly 296-second BPF process wall
time or call that value an attachment cost.

The smallest useful follow-up is one frozen operator shape, five interleaved
fresh-process blocks, and three matched arms (`pod_inline`, `pod_cuda`, and
`pod_bpf`): 15 cells total. Monotonic phase markers should delimit process
start, imports, loader ready, immediately before the first diagnostic launch,
first synchronization, warmup completion, and steady samples. The existing
hard output comparison, device engine 2, exactly-once CTA accounting, bridge
launch counts, shared-memory opt-in, driver, telemetry, private-segment, and
cleanup gates remain mandatory. Each of the six lazily attached kernels must
identify its own first launch.

This can separate the bounded deployment phases from recurring cost for the
tested POD adapter. It cannot establish strict-verifier admission, arbitrary
binary attachment cost, a constant total trampoline cost, or full POD serving
system performance. Existing batch-size points vary useful work and grid
shape together; they may support only a bounded trend over the recorded
operator shapes. A causal trampoline-scaling law requires a future controlled
grid sweep and must not be inferred from those points alone.

## 2. Stale cross-layer state and thrashing

The causal comparison must not be fresh native state versus stale BPF state.
The mechanism comparison gives the native implementation and BPF the same
timestamped snapshots at the same delay; a separate freshness comparison
holds the implementation fixed and changes only snapshot delay.

A minimal real workload alternates dense and sparse phases over the same
managed allocation and records a host-side phase truth timeline. Candidate
delays are fresh, 100 ms, and 1 s, subject to a CPU preflight showing that they
produce distinguishable decision ages. Required outputs are phase time,
end-to-end time, faults, migrated and prefetched bytes, discarded prefetches,
decision age, wrong-phase decisions, and full numerical correctness. Both
high- and low-rate actions must engage; decision, helper, migration, drop,
cleanup, exclusivity, and safety counters must close.

Two paired questions are sufficient: native versus BPF at the same delay
(mechanism cost), and fresh versus delayed observations within one
implementation (information cost). A delay that does not increase wrong-phase
decisions or migrations is a valid negative result, not a reason to change the
workload after seeing results.

This protocol is not immediately runnable. The repository has a fresh BPF
policy and useful lifecycle/monitor patterns, but lacks a shared delayed
snapshot producer, a native same-algorithm consumer of those snapshots, the
phase-switching harness, and matched engagement diagnostics. Those interfaces
must be implemented and CPU-reviewed before reserving GPU time.

## Priority after review

1. Finish the already committed Table 1 collector integration and its real
   correctness preflight; this closes an explicit reviewer commitment.
2. Implement and run the scheduler-init constructor commit/reject experiment;
   it closes remaining Q2 native-transition evidence.
3. Run the 15-cell POD phase decomposition; it is small and directly improves
   the trampoline/deployment-cost discussion.
4. Implement stale-state shared-snapshot infrastructure last because it has
   the largest new mechanism surface.

LMCache remains paused by user direction. Completed MoE-Infinity, XSched,
GPreempt, Expert Buffering, FineMoE, Hummingbird, and POD comparison campaigns
are not rerun merely to seek a more favorable result.
