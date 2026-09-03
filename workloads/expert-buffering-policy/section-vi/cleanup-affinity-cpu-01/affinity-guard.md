# Scoped CPU-affinity coordination

The [interrupted full attempt](../full-01-abandoned.md) motivates temporarily
restricting one identified workspace OpenCode process, not stopping it or
altering its session/configuration. `affinity_guard.py` accepts its exact PID,
previously inspected process start ticks, and CPU 17. It verifies command name,
workspace cwd and uniform original thread masks, and durably records those
identities/masks before changing affinity.

The direct child command runs in its own session. The guard checks all target
thread affinities once per second and records check count and UTC boundaries.
On failure/first signal, it allows the owned child 30 seconds for cooperative
cleanup before the existing bounded helper. Repeated signals cannot interrupt
that cleanup or restoration. **No signal is sent to the external OpenCode.**

Restoration rechecks process/thread start times, restores only masks still
equal to the guard's CPU 17 restriction, and preserves independent changes.
A bounded rescan covers new threads; restoration failure is recorded and makes
the guard fail. This controls CPU-core contention, not total machine isolation.
The worker's internal pinned CPUs 1–5 and main/Torch CPUs 8–11 exclude CPU 17.
The child coordinator must explicitly retain CPUs **8–17**; pinning the guard
itself to CPU 17 must not inadvertently restrict that coordinator.

[First tests](affinity-guard.log) and [rerun](affinity-guard-rerun.log) each pass
three synthetic cases: exact identity/cwd, restoration/new threads/external
changes/PID reuse, and signal/cleanup/restoration ordering with the original
record persisted before the first change. Root read both source files and
independently reran all three tests on CPU 17 successfully. These tests mock
affinity and signals; they do not claim a real restriction or GPU run.

The wrapper is external coordination and does not alter the frozen selector,
worker or offloader. The separately repaired, inventoried cleanup helper still
requires fresh preflight 02 before full attempt 02. Service stop/restoration
belongs to the root's outer launcher, not the direct child command.
