> **2026-09-06: 时钟公平性门槛已由用户指示废除；本文档中的 clock-state / P0 / 精确时钟对匹配要求不再适用。仅以性能数据为准。**

# Launch-latency attempt 12: frozen enumerated-bin retry plan

Status: frozen on 2026-09-04 after retaining attempt 11 and before any
attempt-12 live lifecycle or GPU execution.

## Single admitted change

Attempt 10 requested 2400 / 14001 MHz and was later observed at
2392 / 14001 MHz. Attempt 11 requested 2392 / 14001 MHz and its first post-lock
observation was 2400 / 14001 MHz. Both pairs are explicitly enumerated by the
device's supported-clock inventory. Attempt 11 stopped before its endpoint
probe or either child, and rollback completed.

Attempt 12 retains one fixed requested pair and the existing command order:

1. `nvidia-smi -i 0 --lock-gpu-clocks=2392,2392`
2. `nvidia-smi -i 0 --lock-memory-clocks=14001,14001`

The pre-mutation support query must enumerate both memory/SM pairs
`14001, 2392` and `14001, 2400`. Every active observation after the lock, before
the probe, and before and after each child must satisfy all three conditions:

- P-state is exactly P0;
- memory clock is exactly 14001 MHz; and
- SM clock is exactly one of the enumerated bins `{2392, 2400}`.

This is a finite enumerated-bin set, not a tolerance, range, rounding rule, or
nearest-bin policy. An observation of any other SM clock, memory clock, or
P-state is fatal before further experimental work.

The lock remains in effect across the endpoint probe and both child campaigns.
On success, child failure, timeout, or interruption, cleanup resets memory then
SM clocks before module rollback. Both reset commands are attempted even if one
fails, and an unsuccessful reset receives the existing single final retry.

## Fresh paths and unchanged experiment

The only admitted attempt-12 paths are:

- lifecycle output: `raw/rm-correlation-575-12-endpoint-lifecycle`
- child preflight:
  `raw/rm-correlation-575-12-endpoint-lifecycle/launchlate-preflight`
- child full: `raw/rm-correlation-575-12-endpoint-lifecycle/launchlate-full`
- lifecycle stage:
  `/opt/gpubpf/modules/575.57.08/launchlate-endpoint-stage-575-12`

These paths were absent when this plan was frozen. Attempt-11 artifacts must
not be moved, copied, resumed, or reclassified.

The direct endpoint-v1 command, `taskset -c 8-15`, 1,500 ns median-bracket
threshold, 200 samples per control, three configs, correctness arms, exact
engagement, start/measurement-end/validation-end anchors, held-out affine
validation, randomized ten-block full matrix, safety gates, raw closure, and
rollback are unchanged. The preflight and full children retain their own fresh
controls. This plan authorizes no live action by itself.
