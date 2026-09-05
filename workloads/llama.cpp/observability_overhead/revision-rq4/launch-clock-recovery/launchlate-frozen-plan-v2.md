# Frozen plan v2: three-anchor launch-latency clock validation

Status: frozen on 2026-09-04 after retained attempt 07 and before any attempt
08 execution. This plan changes the clock-model admission test for future
attempts only. It does not reclassify, reuse, or rescue attempt 07; that attempt
remains invalid under the original 10,000 ppb drift gate.

All workload, correctness, engagement, performance, scheduling, safety,
cleanup, and reporting requirements in `launchlate-frozen-plan.md` remain
unchanged except where this document explicitly replaces the two-anchor clock
admission rule.

## Why the clock gate changes

Attempt 07 established complete 220-launch accounting and narrow RM endpoint
brackets, but the two arms reported conservative start-to-end slope bounds of
20,990 and 21,912 ppb. Those values failed the original frozen gate. They do
not distinguish a stable affine host/device clock-rate difference, which the
classifier corrects by interpolation, from a non-affine clock excursion, which
would invalidate that interpolation.

For attempt 08 and later, slope is therefore measured and reported but not
capped. The fixed 10,000 ppb line remains in the raw output as a diagnostic
reference, and `clock_drift_bounded` must still agree exactly with that
diagnostic comparison. It is not an admission field. This is a replacement of
the model check, not a higher numerical threshold.

## Three anchors and sample classification

Each instrumented arm records three independently bracketed endpoint-v1
RAW/PTIMER anchors, each selected from exactly 32 successful trials:

1. `start`, before any selected launch;
2. `measurement_end`, after every selected launch and after gpubpf probes have
   detached or the NVBit context has synchronized; and
3. `validation_end`, captured only after at least 1,000,000,000 ns of
   `CLOCK_MONOTONIC_RAW` time has elapsed since `measurement_end`.

Only `start` and `measurement_end` classify launch samples. `validation_end`
must never change a sample's latency interval or histogram bin. It is held out
as an admission check.

Every anchor must have 32/32 accepted samples, zero rejected samples, complete
RM cleanup, a zero RM status, and a conservative bracket in `(0, 1,500]` ns.
The analyzer independently reconstructs every offset interval and host anchor
from the emitted selected RM endpoints.

## Existential affine interval-overlap gate

Let the three host times be `t0 < t1 < t2` and their independently measured
offset intervals be `I0`, `I1`, and `I2`. The analyzer interpolates the lower
endpoints of `I0` and `I2` at `t1` with floor division and the upper endpoints
with ceiling division, producing the conservative predicted interval `P1`.
Admission requires:

- `t2 - t1 >= 1,000,000,000 ns`;
- all three anchor-quality gates above;
- the emitted prediction equals the analyzer's exact recomputation;
- `P1` intersects `I1`, including a single-point intersection; and
- the emitted overlap bounds and held-out-pass marker equal the analyzer's
  exact recomputation.

This gate establishes that at least one affine line is consistent with the
three interval-valued observations. It does not prove unique slope, arbitrary
interior stability, or behavior outside `[t0, t2]`. The result boundary will
state that limitation. The existing classified/uncertain interval rule and
10% uncertainty limit remain unchanged.

## Attempt boundary

No field from attempt 07 may satisfy this v2 gate because that attempt did not
record a third anchor. The runner and independent analyzer must fail closed on
a missing, duplicate, malformed, unordered, wide, dirty, or non-overlapping
anchor; a missing held-out marker; a validation span below one second; or any
inconsistent slope diagnostic. Attempt 08 must use a fresh raw directory and
must pass both unchanged calibration controls before its correctness cells.
