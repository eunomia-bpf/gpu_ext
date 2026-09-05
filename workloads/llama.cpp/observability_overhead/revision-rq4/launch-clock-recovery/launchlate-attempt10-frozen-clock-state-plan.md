# Launch-latency attempt 10: frozen fixed-clock retry plan

Status: independently reviewed and frozen on 2026-09-04 after retained attempt
09 and before lifecycle implementation or attempt-10 execution.

## Evidence-driven change

Attempt 09 observed a 781 ns passing endpoint bracket at active GPU clocks and
a 2,174 ns failing bracket at idle clocks. Attempt 10 should remove that
recorded power-state confound without relaxing the 1,500 ns gate or reusing any
attempt-09 result.

After loading and validating the candidate modules, the lifecycle must issue,
in order:

1. `nvidia-smi -i 0 --lock-gpu-clocks=2400,2400`
2. `nvidia-smi -i 0 --lock-memory-clocks=14001,14001`

The RTX 5090 reports the 2400 / 14001 MHz pair in its supported-clock inventory.
These are also the clocks in the passing attempt-09 preflight safety-before
snapshot. The lifecycle must record a clock observation before and after the
lock and before and after each child. Each post-lock observation must report
exactly 2400 MHz SM and 14001 MHz memory clocks. A command failure or observation
mismatch is fatal before further experimental work.

The lock must remain in effect, unchanged and recorded, through both the
preflight and full children so baseline, gpubpf, and NVBit run under the same
condition. On success, child failure, timeout, or interruption, cleanup must
issue the inverse operations in reverse order:

1. `nvidia-smi -i 0 --reset-memory-clocks`
2. `nvidia-smi -i 0 --reset-gpu-clocks`

Both resets are attempted even if the first fails. Reset starts before candidate
module removal. Any reset failure is retained as a recovery error, makes the
attempt invalid, and triggers one final reset retry before module rollback. The
original module, service, and label rollback remains mandatory regardless of
clock-reset outcome.

This is preferable to an unrecorded warm-up or a concurrent keepalive: it is an
explicit, symmetric benchmark condition and adds no competing GPU work. It is
also preferable to reusing the preflight controls: the full campaign remains
independently fail-closed. The controls stay once per child campaign, before
correctness, because every timing arm already carries its own three-anchor
quality and held-out affine-validation gates.

## Unchanged experiment and fresh paths

Prospective fresh paths are:

- lifecycle output: `raw/rm-correlation-575-10-endpoint-lifecycle`
- child preflight: `raw/rm-correlation-575-10-endpoint-lifecycle/launchlate-preflight`
- child full: `raw/rm-correlation-575-10-endpoint-lifecycle/launchlate-full`
- lifecycle stage:
  `/opt/gpubpf/modules/575.57.08/launchlate-endpoint-stage-575-10`

These paths were absent when this plan was frozen. The commands, 200
samples per control, direct endpoint-v1 transport, `taskset -c 8-15`, 1,500 ns
median-bracket threshold, three correctness arms, randomized 10-block full
matrix, exact engagement, three-anchor affine validation, safety gates, and raw
closure remain unchanged. Attempt 09 remains failed.

Before any live attempt, offline tests must require clock-lock setup before the
preflight child, persistence across the full child, and reset on every success,
failure, timeout, and interruption path. A CPU-only lifecycle dry-run must show
the new fixed paths and clock actions without changing live state. Attempt 10
may begin only once and must retain any failure rather than retrying a partial
directory.
