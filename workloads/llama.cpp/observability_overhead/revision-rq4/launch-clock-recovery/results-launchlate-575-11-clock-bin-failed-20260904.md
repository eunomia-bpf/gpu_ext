# Launch-latency attempt 11: retained exact-clock failure

Date: 2026-09-04
GPU / driver: NVIDIA GeForce RTX 5090 / 575.57.08
Lifecycle directory: `raw/rm-correlation-575-11-endpoint-lifecycle`

## Outcome

Attempt 11 is **not a paper-facing performance result**. The lifecycle issued
the fixed 2392 MHz SM and 14001 MHz memory lock commands successfully. Its first
required observation immediately after those commands reported
2400 / 14001 MHz. The exact 2392 / 14001 MHz gate failed before the endpoint
probe and before any preflight or full child began. No workload result exists
for this attempt.

Rollback completed. Attempt 11 remains failed and may not be resumed,
reclassified, or used as performance evidence.

## Evidence boundary for the next retry

Attempt 10 requested 2400 / 14001 MHz and was later observed at
2392 / 14001 MHz. Attempt 11 requested 2392 / 14001 MHz and was immediately
observed at 2400 / 14001 MHz. Both SM values are explicit adjacent entries for
the 14001 MHz memory clock in the device's supported-clock inventory. Together,
the retained attempts establish that driver telemetry can toggle between these
two enumerated bins under a minimum-equals-maximum lock request. They do not
justify a numeric tolerance, a continuous accepted range, or acceptance of any
other clock or P-state.

Attempt 12 therefore keeps a single fixed request but uses the smallest stable
state gate supported by the retained observations: P0, memory exactly
14001 MHz, and SM exactly one of `{2392, 2400}`.
