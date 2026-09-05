# Launch-latency attempt 13: frozen telemetry-fairness plan

Status: frozen on 2026-09-05 after retaining attempt 12 and before any
attempt-13 live lifecycle or GPU execution.

## Single scientific change

Attempts 10--12 show that a successful minimum-equals-maximum clock request
does not make one boundary SM-frequency observation stable: adjacent supported
bins were observed at later gates. Attempt 13 therefore does not treat an
outer boundary frequency as proof of performance fairness.

The lifecycle keeps the fixed high-clock requests, P0, and memory exactly
14001 MHz. Before applying either clock lock it records the complete `nvidia-smi`
memory/graphics support inventory. Every active outer observation must be P0,
must have memory 14001 MHz, and must be an exact pair in that inventory. No
tolerance, range, rounding, nearest-bin rule, or frequency normalization is
permitted.

Before any child calibration, correctness, or timing cell, the child runner
separately records its supported-clock inventory. For each randomized timing
block, baseline, gpubpf launchlate, and NVBit launchlate must each satisfy:

- every telemetry sample reports P0;
- the existing no-throttle, no-external-GPU-use, and safety gates pass;
- the exact observed `(SM MHz, memory MHz)` set is nonempty;
- all observed pairs occur in the pre-recorded support inventory; and
- the three arms' exact observed-pair sets are identical.

Failure invalidates all three cells in that block. The independent analyzer
reopens each raw telemetry CSV and re-evaluates these predicates; stored valid
flags or stored summaries cannot establish the gate. This is a stricter
within-block comparability requirement, not a relaxation of the latency or
result gate.

## Unchanged experiment and fresh paths

The direct endpoint-v1 transport, 1,500 ns median-bracket threshold, 200
samples per control, three arms, correctness comparison, exact 220-launch
accounting, start/measurement-end/validation-end RM anchors, held-out affine
validation, ten randomized pp=512 blocks, primary paired overhead metric, raw
closure, and rollback are unchanged.

The only admitted attempt-13 paths are:

- lifecycle output: `raw/rm-correlation-575-13-endpoint-lifecycle`
- child preflight:
  `raw/rm-correlation-575-13-endpoint-lifecycle/launchlate-preflight`
- child full: `raw/rm-correlation-575-13-endpoint-lifecycle/launchlate-full`
- lifecycle stage:
  `/opt/gpubpf/modules/575.57.08/launchlate-endpoint-stage-575-13`

All must be fresh. Attempt-12 artifacts remain retained in place and must not
be resumed or copied into attempt 13. This plan authorizes no live action by
itself.
