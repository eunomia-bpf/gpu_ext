> **2026-09-06: 时钟公平性门槛已由用户指示废除；本文档中的 clock-state / P0 / 精确时钟对匹配要求不再适用。仅以性能数据为准。**

# Launch-latency attempt 11: frozen exact-clock retry plan

Status: frozen on 2026-09-04 after retaining attempt 10 and before any
attempt-11 live lifecycle or GPU execution.

## Single admitted change

Attempt 10 requested an exact 2400 MHz SM / 14001 MHz memory lock. Its command,
endpoint probe, and fresh preflight ran, but the required observation after the
preflight reported 2392 / 14001 MHz. The exact gate correctly stopped before
the full child. The device's supported-clock inventory explicitly enumerates
2392 / 14001 MHz.

Attempt 11 therefore changes the target pair to exactly 2392 MHz SM and
14001 MHz memory. The lifecycle must issue, in order:

1. `nvidia-smi -i 0 --lock-gpu-clocks=2392,2392`
2. `nvidia-smi -i 0 --lock-memory-clocks=14001,14001`

The pre-mutation support query must contain the exact memory/SM pair
`14001, 2392`. Every required post-lock observation before the probe, before
and after the preflight child, and before and after the full child must report
exactly 2392 MHz SM and 14001 MHz memory. There is no tolerance. A command
failure or any observation mismatch is fatal before further experimental work.

The locks remain in effect across the endpoint probe and both child campaigns.
On success, child failure, timeout, or interruption, cleanup resets memory then
SM clocks before module rollback. Both reset commands are attempted even if one
fails, and an unsuccessful reset receives the existing single final retry
before rollback.

## Fresh paths and unchanged experiment

The only admitted attempt-11 paths are:

- lifecycle output: `raw/rm-correlation-575-11-endpoint-lifecycle`
- child preflight:
  `raw/rm-correlation-575-11-endpoint-lifecycle/launchlate-preflight`
- child full:
  `raw/rm-correlation-575-11-endpoint-lifecycle/launchlate-full`
- lifecycle stage:
  `/opt/gpubpf/modules/575.57.08/launchlate-endpoint-stage-575-11`

These paths were absent when this plan was frozen. Attempt-10 artifacts must
not be moved, copied, resumed, or reclassified.

The direct endpoint-v1 command, `taskset -c 8-15`, 1,500 ns median-bracket
threshold, 200 samples per control, three configs, correctness arms, exact
engagement, start/measurement-end/validation-end anchors, held-out affine
validation, randomized ten-block full matrix, safety gates, raw closure, and
rollback are unchanged. The preflight and full children retain their own fresh
controls. This plan authorizes no live action by itself.
