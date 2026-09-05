# Launch-latency attempt 12: retained outer-boundary failure

Date: 2026-09-04
GPU / driver: NVIDIA GeForce RTX 5090 / 575.57.08
Lifecycle directory: `raw/rm-correlation-575-12-endpoint-lifecycle`

## Outcome

Attempt 12 is **not a paper-facing performance result**. The endpoint probe
passed and the fresh launch-only child preflight returned successfully. Its
independent analysis passed both 200-sample calibration controls, all three
correctness arms including the instrumented 220-launch accounting, and its one
randomized pp=32 timing block. This is retained dependency/preflight evidence
only.

The outer observation immediately after that child reported P0 and
2385 / 14001 MHz. Because attempt 12 admitted only the explicitly enumerated SM
set `{2392, 2400}` at 14001 MHz, the lifecycle stopped before the full child.
Rollback completed without a retained recovery error. Attempt 12 remains a
failed lifecycle and may not be resumed, reclassified, or used as a paper
result.

## Evidence motivating attempt 13

The child timing telemetry for baseline, gpubpf launchlate, and NVBit
launchlate all contained the same observed 2385 MHz SM bin. The raw telemetry
also showed the device moving between supported memory states during a cell;
attempt 12 did not record P-state in every telemetry sample. Thus neither a
wider handwritten SM set nor an outer boundary observation proves that the
three performance arms ran in an identical clock state.

Attempt 13 instead separates lifecycle validity from performance fairness.
The lifecycle requires P0, memory exactly 14001 MHz, and an exact observed
memory/SM pair present in the pre-lock supported-clock inventory. The child
records P-state and exact clock pairs throughout each timing arm. A randomized
block is usable only when all three arms have nonempty, identical exact pair
sets, all samples are P0, every pair is in the independently pre-recorded
support inventory, and the existing throttle, external-use, and safety gates
pass. This is exact state matching, not a tolerance or frequency correction.
