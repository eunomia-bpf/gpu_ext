# Launch-latency attempt 10: retained exact-clock failure

Date: 2026-09-04
GPU / driver: NVIDIA GeForce RTX 5090 / 575.57.08
Lifecycle directory: `raw/rm-correlation-575-10-endpoint-lifecycle`

## Outcome

Attempt 10 is **not a paper-facing performance result**. The lifecycle
requested an exact 2400 MHz SM / 14001 MHz memory lock, then ran the endpoint
probe and fresh preflight. After the preflight, the outer lifecycle observation
reported 2392 / 14001 MHz rather than the exact requested pair. The fail-closed
clock gate stopped the attempt before the full child. No attempt-10 preflight
value is promoted, copied, or reclassified as a result.

The 2392 / 14001 MHz pair is explicitly present in the device's
`nvidia-smi --query-supported-clocks=memory,graphics` inventory. This record
does not infer why a successful 2400 MHz lock request was later observed at
2392 MHz. It establishes only that the requested and observed SM clocks differed
at the required post-preflight gate.

The retained preflight is diagnostic evidence only. Its two 200-sample clock
controls passed, as did its correctness and single-block timing path, but those
facts cannot bypass the failed outer exact-clock condition or authorize the
ten-block full campaign.

Recovery reset the clock constraints, restored the stock UVM stack and
services, and returned the GPU to its default idle clocks. No `dmesg` error was
observed during recovery.

## Fresh retry boundary

Attempt 11 changes only the exact fixed-clock pair and fresh paths. It requires
2392 MHz SM and 14001 MHz memory with no tolerance. The 1,500 ns endpoint gate,
200 samples per control, command and transport, configs, anchors, randomized
matrix, correctness, engagement, safety, raw-closure, and rollback gates remain
unchanged. Attempt 10 remains failed and may not be resumed.
