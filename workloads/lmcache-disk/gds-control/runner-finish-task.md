# Narrow local-model implementation task

## Execution update — completed implementation and real measurement

The two fixes below were committed as `3c278471`. Root immediately ran the
updated runner for five rotations of the 64-read/96-write workload; all 15
children returned zero, with 960 reads and 1440 writes completed. The report
is `../results-575-gds-mixed-scheduled-20260907.md`. These completed measurements
must not be repeated. Remaining source work for this task is limited to
concrete defects, if any, discovered while finishing the scoped changes.
The separate pending-demand-feedback idea is not part of this runner task.
The root closed the implementation session after these fixes, the 33 scoped
tests and all 15 real measurements were complete. This was task completion,
not cancellation for silence or a short timeout. Additional subprocess smoke
attempts are not experiment evidence. A separate local-model session now owns
the opt-in feedback variant described in `next-policy-experiment.md`.

## Assigned scope

Read workspace AGENTS.md and edit only `../run_gds_mixed_backend.py`, plus
minimal updates to its existing test file if those changes require them.
Do not use nested agents, inspect unrelated papers, expand tests, run GPU
experiments, or commit. The root has already completed two 15-measurement
real cuFile comparisons; do not repeat or overwrite them.

Implement two concrete runner fixes:

1. Run each measurement in a fresh subprocess by default. The current
   backend.close leaves CUDA pools retained across measurements and the
   seven-pool OOM is documented in mixed-runner-followup.md. Add an internal
   single-measurement CLI route carrying configuration, repetition, position,
   output directory and the same workload arguments. The parent invokes the
   current Python executable, waits without an artificial timeout, records
   each child's result/exit, and continues the rotated order without retries.
   Preserve all per-request output and record failures rather than inventing
   missing results. Process startup remains outside request timing.

2. Keep existing `offer_s` as actual dispatch for backward compatibility and
   add `scheduled_offer_s` plus explicit dispatch timing. Initialize all write
   buffers and create waiting reader threads before releasing a common start
   event; derive scheduled arrivals from that shared monotonic start and the
   configured intervals. Record both dispatch-to-completion and
   scheduled-offer-to-completion p50/p99 with unambiguous names. The existing
   result field read_end_to_end currently denotes dispatch-to-completion and
   must not silently change meaning. Use wall time only for wall labels.

The same FIFO/native/BPF executor and policy remain unchanged. Do not alter
the driver, installed LMCache, BPF policy, old raw files or paper. No clock,
preflight, correctness or admission gates. A syntax check and narrowly relevant
existing checks suffice; return the implementation promptly. The next task,
after these fixes, is live pending-demand feedback rather than additional
fixed-delay testing; do not implement that broader feature in this task.
