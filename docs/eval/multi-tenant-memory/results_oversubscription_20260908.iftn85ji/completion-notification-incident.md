# Early completion notification and lock recovery

The notification watched outer flock PID 2663710. That process exited while
the driver lifecycle shell (2663713, reparented to PID 1) and sweep runner
2664070 continued. The notification was premature: only 298/300 cells were
complete at inspection. The two original flock leases had been released,
while the GPU still ran the two K-Means tenants. Driver restoration had not
started. The external parent exit cause was not established.

The experiment Codex was immediately told not to use the GPU or reload
modules. Replacement lock holder PID 3073539 acquired both original lock
files without contention before the final memory-only cell began. It retains
both leases until completion and driver/service/storage-loader restoration
are independently confirmed. No measurement was stopped or restarted.
The notifier source now watches the lifecycle shell, not the outer flock.

Lock-holder log: /tmp/gpubpf-sweep-recovery-locks.log.
The root agent will append the final recovery outcome after inspection.

## Final outcome

All 300 cells completed with successful tenant/tool exits. The ratio 1.8
guard stopped the original loop as intended. Automatic restoration failed
because nvidia_uvm still had a reference immediately after the final cell.
A subsequent inspection found refcount zero and no device-file clients.
The manual retry restored installed core 575.57.08, the saved original UVM,
gdm and nvidia-persistenced, and storage loaders 3080911/3081084. Both loaders
remain alive and report attached; their BPF program/map descriptors exist.
The retry log records RESTORATION_OK=1 at 08:23:49 PDT.

The local bpftool struct_ops listing crashed with exit 139 during read-only
inspection. Restoration was checked separately through the successful reload
commands, loaded version, services, loader logs and live BPF descriptors.
No GPU performance test was launched as a restoration check.
