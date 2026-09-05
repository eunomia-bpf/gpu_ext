# Launch-latency attempt 09: retained full-control failure

Date: 2026-09-04
GPU / driver: NVIDIA GeForce RTX 5090 / 575.57.08
Lifecycle directory: `raw/rm-correlation-575-09-endpoint-lifecycle`

## Outcome

Attempt 09 is **not a paper-facing performance result**. The lifecycle probe
and the fresh preflight passed. The preflight had 200/200 accepted direct
endpoint samples with a 781 ns median bracket, both clock controls passed, all
three correctness arms passed, and its single timing block passed independent
analysis. The full child then ran its own fresh controls before correctness or
timing, as required by the frozen campaign path. Its endpoint control returned
status 2 because the median bracket was 2,174 ns, above the unchanged 1,500 ns
admission threshold. The globaltimer control, full correctness cells, and all
full timing cells therefore did not run. Lifecycle rollback completed; no
attempt-09 value is reclassified or copied into a later attempt.

This was not a rejected-sample or scheduling-outlier failure. Both endpoint
runs used `taskset -c 8-15`, the same boot, driver, candidate module, direct
transport, endpoint-v1 command, and 200 requested samples. Each accepted
200/200 with zero CPU-midpoint or PTIMER regressions. Their distributions were
two narrow, disjoint regimes:

| Control | RM selected gap | Conservative bracket | Outer ioctl width | GPU telemetry around control |
|---|---:|---:|---:|---|
| lifecycle probe | not separately retained here | median 760 ns | retained in lifecycle record | candidate validation reported 2400 / 14001 MHz |
| preflight | 715--721 ns | 779--785 ns; median 781 ns | median 7,902 ns | safety before: 2400 / 14001 MHz; after: 23 / 405 MHz |
| full | 2,006--2,301 ns | 2,070--2,365 ns; median 2,174 ns | median 13,950 ns | sampled telemetry and safety after: 22 / 405 MHz |

The preflight completed approximately four minutes after the lifecycle probe,
and the full control began immediately after that child. No module replacement,
server replacement, driver change, CPU-affinity change, or boot occurred
between the two child campaigns. The material recorded difference is GPU power
state: candidate validation and the preflight control's safety-before snapshot
observed active clocks, although its safety-after snapshot had already returned
to 23 / 405 MHz. The full control's sampled telemetry and safety-after snapshot
observed idle P8 clocks after the preflight workload had cleaned up. Reloading
the original stack during rollback again reported 2400 / 14001 MHz. The evidence
therefore supports a power-state-dependent RM endpoint service cost as the
leading explanation. It
does not establish which firmware or driver subcomponent causes that cost, so
the result must not be described as a proven causal mechanism.

The direct control opens the NVIDIA RM device and issues the endpoint-v1 ioctl
itself. It runs before a fresh child starts any bpftime agent or timing cell, so
no persistent bpftime server/agent transition is present on the failing path.

## Why full must retain its own controls

The full child is a fresh 10-block campaign, not a continuation of the one-block
preflight. The runner consequently executes its two controls in every fresh
launchlate campaign before that campaign's correctness cells. The frozen plans
also require those controls before correctness. Reusing only the earlier
preflight control would hide exactly the state dependence exposed here.

Controls need not be inserted next to every timing block. Each instrumented
timing arm already records independent start, measurement-end, and held-out
validation-end anchors, and every anchor has its own 1,500 ns bracket gate.
Adding extra controls between blocks would change the interleaving and could
itself perturb GPU state without strengthening the per-arm clock proof.
