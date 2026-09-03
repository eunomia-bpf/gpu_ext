# Preserve affinity changes not owned by this guard

An independent pre-launch review found a cleanup ownership edge case: an
initial thread could be externally pinned to CPU 17 before the guard's first
successful change. The old failure path would mistakenly widen that thread's
mask even though the guard had not changed it. **No real affinity run had
occurred before this repair.**

The guard now records each initial `(tid, start_ticks)` only after its own
successful affinity call. It restores only these owned initial threads and
same-process newly created threads inheriting the restriction; unowned initial
threads are preserved. If no change succeeded, it does not restore or launch
the child. External changes still fail admission, rather than being overridden.

The [new test log](affinity-guard.log) retains successful tests covering both
partial ownership and an external pin before the first change. Root reviewed
the diff and independently reran all three test methods successfully. These
are synthetic tests, not a real affinity or GPU experiment.

Numeric TID identity checks and affinity syscalls are separate operations:
the pre/post checks detect changes but are not an atomic PID-reuse guarantee.
The guard's cleanup covers its own child process group; the existing GPU
controller separately verifies its worker, telemetry and post-safety state.
The outer guard is not part of the frozen 94-file GPU runtime inventory.
