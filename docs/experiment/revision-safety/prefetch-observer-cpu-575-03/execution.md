# Address-free prefetch observer CPU preparation 575-03

Result: **eighteen synthetic tests and the independent BPF/loader build pass. No
BPF program or driver module was loaded, and no Q2 live control ran.**

## What changed

The obsolete six-observer design was reduced to three tracing links:

1. wrapper entry creates one frame keyed by the full `pid_tgid`;
2. wrapper exit records the actual action, request, and legal bounds;
3. the new diagnostic program requires exactly one SELECTED and one FINISHED
   event before deleting the frame.

The struct_ops policy is the fourth program in the object and is attached only
for BYPASS and invalid99. The fixture no longer stores a bitmap-tree, output,
stack, or other kernel address in a BPF map. It observes the driver's actual
region-validation result, chosen effect, returned region, native completion,
and native loop-body count. All checks are per decision before aggregate
counters are accepted.

The runner now verifies the exact 88-byte/14-field tagged context, the
`void(const context *)` BTF chain, phase enum values, three unique live tracing
links, and program run counts. It also requires CPU 17 for the loader, includes
the nonzero policy link in cleanup checks, and starts a third continuous monitor
that records compute-process PIDs. Its lifecycle gate requires empty before
target start, target-only at pause and attachment-ready, target-only after
release, and empty after both target exit and loader detachment. Samples from
the first to last marker may not have a gap above one second, and any sampled
foreign PID is fatal. Each query records its monotonic start and finish: a gate
accepts only a query started after it, and duration plus idle/start/finish
cadence are each bounded to one second. All cross-process ordering uses the
shared system monotonic clock; wall time is retained only for inspection. This is
a bounded sampled-exclusivity assumption; it cannot rule out a foreign client
whose whole lifetime falls between two samples.

## CPU evidence

`taskset -c 17 python3 -B extension/revision-prefetch/test_offline.py` exits 0;
the final [`tests-14.log`](tests-14.log) reports 18 tests passing. Earlier
successful iterations and both pre-test parse failures are retained rather
than overwritten. The tests reject missing or duplicate
phases, wrong program counts, fallback without native completion/iteration,
BYPASS traversal/output, observer errors, wrong BTF layout/prototype/enum,
legacy range observation, persisted pointer fields, foreign compute clients,
sampling holes/missing tail coverage, and incomplete owned cleanup.
They also pin the post-release timestamp after the target pipe is successfully
flushed and closed, pin the tail after loader detachment, ignore wall-clock
steps for cross-process ordering, clear `complete` on every body exception, and
make link disappearance independent of subsequent monitor/safety failures.
They additionally reject a query that starts before a lifecycle gate but
finishes after it, reject a near-two-second cadence composed of individually
bounded query and idle intervals, and prove the installed signal handler queues
cancellation without throwing inside child registration or cleanup.

The first attempt to resolve integer base types from a base-less split-module
BTF dump correctly exposed a checker limitation and is retained in
[`interface-check-attempt01.md`](interface-check-attempt01.md). The narrowed
runtime boundary then passed against the freshly built module; the latest
check is [`interface-check-03.log`](interface-check-03.log).

`taskset -c 17 make -C extension/revision-prefetch -j1` exits 0. The complete
[`build-02.log`](build-02.log) records BPF compilation, object and skeleton
generation, and a warning-free loader build; [`build-03.log`](build-03.log)
confirms the objects remain current after the lifecycle-gate fix. The preceding name-collision
failure is retained separately in [`build-attempt01.md`](build-attempt01.md).
The generated object's [`frame BTF`](object-frame-btf.txt) contains only the
four request/bounds scalars, action, and eight state/result fields; its
[`program inventory`](object-programs.txt) contains the three observers plus
the fixed struct_ops policy.

An intentionally one-second, SIGINT-terminated monitor smoke produced multiple
empty compute-process samples and a final record with zero internal sampling
errors in [`compute-monitor-smoke.jsonl`](compute-monitor-smoke.jsonl). GNU
`timeout` itself returned 124 after sending SIGINT; the monitor emitted its
final record before exit. A second smoke used the runner's exact process-group
SIGINT behavior; it exited 0 and emitted zero internal errors in
[`compute-monitor-killpg-smoke.jsonl`](compute-monitor-killpg-smoke.jsonl).
These check only the monitor's sampling and signal path, not a live target or
foreign-client gate.

After moving lifecycle ordering to the monotonic clock, the same exact
process-group SIGINT path exited 0 with eight ordered empty samples, both clock
fields, and zero internal errors in
[`compute-monitor-monotonic-smoke.jsonl`](compute-monitor-monotonic-smoke.jsonl).

The final producer smoke records query start/finish spans and exits 0 with zero
internal errors in
[`compute-monitor-query-span-smoke.jsonl`](compute-monitor-query-span-smoke.jsonl).

Independent code review and the final no-tools OpenCode review both return
READY under the stated sampled-exclusivity limitation. The retained review
trajectories are [`independent-review.md`](independent-review.md) and
[`opencode-monitor-review-01/outcome.md`](opencode-monitor-review-01/outcome.md).

## Remaining admission

CPU preparation does not prove that the loaded module exposes the new hook or
that fentry accepts the program. A separate module-lifecycle coordinator must
stage revision `0c109956`, stop only the required display services, replace
only the zero-reference UVM module, prove live BTF/fentry admission, run the
three fixed cells into a fresh directory, then restore the previously loaded
module and services before marking the campaign complete.
