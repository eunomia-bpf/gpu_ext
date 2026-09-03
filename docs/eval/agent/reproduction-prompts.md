# Current policy-port reproduction prompts

Authored for this revision on 2026-09-03. These are **new reusable prompts**,
not recovered Claude conversations, not the prompts behind the historical
59-policy/974-run study, and not evidence that a new agent run has completed.
The [original-corpus inventory](../../experiment/revision-artifact-inventory.md)
records that separate release gap. Model/version, tools, initial source revision
and actual resulting interaction logs must be recorded by anyone using these
prompts; the text alone does not reproduce a trajectory.

## Port an existing policy

```text
Implement one existing GPU resource-management policy using the current gpubpf
interfaces. First read its official paper/PDF and available author source, plus
the local workload's implementation and experiment plan. Name the exact policy
component, observations, actions, state and execution domain being reproduced.
Distinguish a component port from the original full system. Preserve the paper
and source revision; explain unavailable hardware, datasets or APIs explicitly.

Compare a no-policy/default baseline, the original native policy, and actual BPF
execution of that same policy on a common workload and executor wherever possible.
Document every unavoidable executor or transport difference. Numerical outputs,
actual BPF decision counts, and applied actions must pass before timing. Loading
a program, a ready message or host-only replay is not device execution evidence.
Do not silently call the native selector as a BPF fallback.

Reuse the workload's frozen model, inputs, budgets, randomization, numerical
tolerances and complete paired-block protocol. Retain failed/interrupted attempts
and every valid adverse result. Do not tune acceptance thresholds after seeing
performance. Report raw cell values and paired uncertainty; a confidence interval
crossing zero does not prove equivalence. Separate policy benefit from mechanism
cost and state whether GPU/kernel verification was actually enabled and enforced.

Follow workspace instructions. Preserve unrelated changes and the shared runtime.
Use the existing GPU/struct-ops leases, exclusive GPU timing, private owned state,
bounded process cleanup and before/after safety checks. Do not compile or load
another model during a timed campaign. Do not reboot, replace drivers or terminate
unrelated processes as an implicit part of a reproduction command.

Record commands, ordinary source revisions, explicit file inventories and sizes,
build inspection, correctness, engagement, raw results and limitations. Never
generate or use content hashes/checksums/digests as evidence or resume gates.
Make scoped code/documentation commits and push them to the designated project
branch when that publication is authorized. Do not claim unpushed or ignored
records are public artifacts.
```

Use the matching existing implementation rather than rebuilding a second harness:

| Policy | Current implementation and protocol |
| --- | --- |
| Expert prediction / residency | [MoE-Infinity scope](../../../workloads/moe-infinity/activation-aware-port.md) |
| Highest-priority-first queue admission | [XSched comparison](../../../workloads/xsched/performance-full-575-20260903.md) |
| Foreground-protection scheduling | [GPreempt load plan](../../../workloads/gpreempt/load-study-plan.md) |
| Dynamic expert-prefetch sets | [FineMoE results and scope](../../../workloads/finemoe/results-performance.md) |
| Idle-interval work admission | [Hummingbird plan](../../../workloads/hummingbird/plan.md) |
| In-kernel task selection | [POD-Attention plan](../../../workloads/pod-attention/plan.md) |

## Audit a completed comparison

```text
Independently audit the named comparison without running GPU work or modifying
its raw records. Recompute each reported metric from request-level timestamps,
counts and outputs, then recompute complete-block paired effects and uncertainty.
Verify exact workload/configuration/order, source and driver scope, every missing
or failed attempt, actual policy engagement, telemetry and owned teardown. Explain
which validators are reused rather than independently reimplemented. Do not infer
full-system reproduction, superiority, equivalence or safety from a source mapping,
build success, summary status flag or lack of a crash. If evidence is absent,
report the specific missing check instead of manufacturing or reconstructing it.
```

Inference input files named `prompts.json` are workload data, not instructions to
the policy-generating agent. Neither those files nor these newly authored prompts
replace the missing original study interactions.
