# Experiment plan: stale cross-layer state and UVM thrashing

Status: **CPU/source preparation only. The shared snapshot/native-consumer
source and workload-truth-FD coordinator are implemented, but live execution
is still blocked on module installation, attach/diagnostic integration, and a
controlled seven-cell preflight described in `README.md`.**

## Question and hypothesis

This supporting experiment tests whether fixed publication delay in a
cross-layer phase observation makes a legal adaptive prefetch policy select
the previous dense/sparse phase often enough to increase GPU faults or
migration traffic and reduce verified-work throughput. The competing outcome
is that the workload and driver absorb 100 ms or 1 s staleness with no material
increase. Either outcome is retained.

It adds evidence about the stale-state limitation discussed in Reviewer D and
`docs/paper/tex/discussion.tex`. It does not test transition-validator safety,
generic BPF overhead, trampoline scaling, Table 1, or a serving system.

## Workload

`stale_state_workload.cu` freezes one 40 GiB managed allocation, 64 KiB logical
regions, a sparse stride of 32 regions, a 256 MiB dense launch span, a 1.2 s
unmeasured sparse bootstrap, and six alternating two-second measured phases.
The host is the phase-truth authority. It records monotonic start/end events
and the fixed scheduled offset of every phase. Each GPU launch is synchronized,
and every returned word is compared to its expected value. Any CUDA failure,
phase-schedule overrun, mismatch, missing phase, or incomplete output fails the
cell.

The bootstrap lets all delayed arms receive an actual sparse snapshot before
measurement. During measured phases the policy is deliberately simple:

- an observed dense snapshot requests the callback's full legal maximum
  prefetch region;
- an observed sparse snapshot requests the legal empty region, discarding that
  callback's speculative prefetch;
- a missing, malformed, future-dated, or torn snapshot is an error, never a
  guessed action.

Both actions must occur in every policy row. This is a phase-sensitivity
workload, not an application-performance claim.

## Comparisons

Each of three seeded paired blocks contains exactly seven fresh-process cells:

| Cell family | Delay | Role |
| --- | ---: | --- |
| driver-default UVM | none | contextual control |
| native same algorithm | 0, 100, 1000 ms | mechanism and freshness comparisons |
| BPF same algorithm | 0, 100, 1000 ms | mechanism and freshness comparisons |

Mechanism cost pairs native and BPF only at identical delay within a block.
Information cost pairs 100 ms or 1 s against fresh only within the same
implementation and block. No fresh-native versus stale-BPF contrast is
reported. The shared producer must publish the same phase record format and
fixed delay to either implementation; actual source, publication, status
observation, and decision timestamps are retained.

One excluded complete seven-cell preflight must precede the three formal
blocks. Its purpose is only to establish the real path, both policy actions,
monitor coverage, numerical correctness, clean lifecycle, and distinguishable
decision ages. The frozen workload is not changed in response to the observed
performance direction.

## Metrics and validity

Primary descriptive outcomes are GPU faults per second, migrated bytes per
second, prefetched bytes, verified words per second, measured phase time, and
end-to-end time. Decision age and wrong-phase fraction are the causal
explanatory metrics. Also retain prefetch-migration count, thrashing and
eviction events, and actual sparse-policy discarded-prefetch decisions.

A cell is valid only when:

- all configured values and all per-launch returned words match exactly;
- phase truth and snapshot publications are complete and monotonic;
- every policy decision names a published snapshot, occurs no earlier than
  publication, and joins to exactly one host-truth interval;
- policy final counters equal the retained per-decision records, dense-prefetch
  and sparse-discard actions are both nonzero, and missing/invalid snapshot,
  request-error, effect-error, and record-drop counters are zero; callback,
  snapshot-read, decision-request, effect-request, diagnostic, and
  effect-record totals close exactly;
- UVM Tools observes nonzero GPU faults and migrations, with zero dropped GPU
  fault and migration events;
- continuous GPU, compute-client, and kernel monitors cover the cell; no
  foreign compute client, new kernel anomaly, surviving owned process, attached
  policy, UVM reference, or new BPF object remains.

The analyzer recomputes decision age, wrong phase, action counts, and throughput
from raw timestamps/events. It does not accept a policy's self-declared
wrong-phase count or invent fault/migration/prefetch/discard values.

## Interpretation

Within each paired block report native/BPF ratios at each delay and delayed/fresh
ratios within each implementation. Three blocks bound only this frozen
workload and RTX 5090/575 stack; they are not a population-wide confidence
claim. Increased wrong-phase decisions accompanied by more faults/migration
and lower throughput supports sensitivity to stale state. No increase is a
valid negative boundary. Numerical, engagement, monitor, or fairness failure
makes the affected comparison invalid rather than favorable to either arm.

No live command is authorized until the driver-owned shared snapshot and
matched native/BPF diagnostic interface is installed and passes a controlled
load/attach preflight. A new private-map-only BPF
policy or a userspace prefetch substitute is insufficient.
