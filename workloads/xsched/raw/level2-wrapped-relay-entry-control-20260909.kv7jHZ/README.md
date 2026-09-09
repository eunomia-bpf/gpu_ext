# Wrapped-metadata / window-relay original-entry control

Run: 2026-09-09 00:11:45–00:11:47 PDT. This is a launch-failure
diagnostic, not a performance sample or completed Level-2 reproduction.

The installed native HAL is the unchanged candidate measured in
`../level2-native-wrapped-meta-20260908.NmlhJN/`; main commit `6e5250e0`
retains that candidate's source snapshot and build/run records. No new build
was performed for this control. Both GPU and struct-ops leases were held.

One BE process used four queues, 50 tasks per queue, 9,511,106 recurrence
iterations, 340 blocks and 256 threads. Its command was:

```
workloads/xsched/level2-build/.output/service-mismatch-20260908.sHSYtE/priority_workload be 1 4 50 9511106 340 256 1 0
```

The native HAL library directory was
`workloads/xsched/level2-build/.output/hal-native-20260908.i8na51/install/lib`.
The environment selected GLB scheduling, auto XQueues at Level 2, priority 0,
threshold 4, batch size 2, the sm_120 port, tool actuator off and service-only
output. Both `XG_NATIVE_META_EXTEND=1` and
`XG_NATIVE_ORIGINAL_ENTRY_CONTROL=1` were set. A dedicated `xserver HPF 50000`
was stopped after the worker exited. Input was the existing `GO` record.

The large window-argument relay and wrapped-image metadata extension were
retained; only the conditional pre-launch entry-point redirection was
disabled. Other binary preparation, register/barrier adjustment and debugger
parameter setup remained. This differs from the earlier successful
original-entry control, which preceded the window-relay/metadata changes.

Observed: both kernel parameter-region records were extended to `0x1520`,
and the owned wrapper/nested-image sizes were `0x20`/`0x4818` bytes. The first
queued launch still returned CUDA 701 (too many resources requested for
launch); later queue launches returned 4. The process exited 139 during
error handling and produced no completed service result. Raw worker bytes,
including malformed error text, are preserved in `worker.log`.

Conclusion: redirecting execution into the guardian prefix is not necessary
for this candidate's launch rejection. The common launch setup remains to
be repaired; this control does not distinguish parameter layout from the
other retained resource modifications. It does not prove a particular fix.

The final observation was 0% GPU utilization and 1 MiB used (P0 at the
immediate observation). No driver reload or GPU recovery was performed.
See `lifecycle.log` and `xserver.log`; no old result is replaced.
