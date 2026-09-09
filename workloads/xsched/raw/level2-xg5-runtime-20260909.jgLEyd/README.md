# Device-consumed XSched arguments: runtime attempt

The existing native_port workload is run once with the local GLM XG5 device
probe and host drain. This is a diagnostic for the unfinished Level-2 path,
not a throughput comparison or an additional correctness campaign.
No completed baseline, Level-1 or prior performance cell is repeated.

The device tool was built and linked in yiBj3W. Root connected the existing
drain through the manager and inherited queue synchronization; the local
model also added four bounded publication drains. There is no additional
CUDA synchronization. Marks retain only the latest changed value in each
of the first 63 slots, not a full device event trace.

The first HAL build failed because root/model concurrently supplied the
same passthrough. Root removed its duplicate, leaving the model's definition.
build-hal-retry.log records successful build and installation. source/ is
the initial snapshot; source-rebuild/ is the retry snapshot. The build logs
and snapshots are retained, not performance evidence.

Both shared GPU/struct-ops leases cover build and execution. run.sh restores
the previously installed tool binary on exit. Large binaries remain local.
No driver reload, manuscript edit, clock calibration or new gate is added.
## Result

Runtime ended at approximately 06:23:22 PDT with runner exit 1. Both LC
workers exit 0. BE1, BE2 and BE4 exit 2 with missing/repeated work; BE3 is
interrupted by the runner's cleanup after the failure (return -2), not a
completed worker. This is not a successful Level-2 performance result.

In BE1, host records publish distinct slots 1, 2, 3, 4, 103, 152 and later
slots, with corresponding task and command indices; submissions return zero.
The retained device marks instead repeatedly update slot 1 with kernel_idx 1,
as the device entry ordinal advances through 200. That slot changes from
guardian type 1 to resume type 2 at reactivation; subsequent entries continue
to report type 2/kernel_idx 1. BE4 analogously keeps reporting slot 3 through
ordinal 200. BE1 reports 680 completed blocks for stream 0 command 1 where
the geometry is 340, and zero for several following commands.

These observations point to stale per-launch context delivery or its tool
argument ABI, rather than a lack of host publications. They do not yet
identify the exact broken API call. The probe samples one CTA leader and
drains only slots 1--63 with overwrites possible: it is not an exhaustive
trace of all launches, slots or threads. The local model has the raw evidence
for the next parameter-delivery repair; no unchanged cell should be rerun.

The original tool file was restored and compared byte-for-byte to the local
backup. After workers exited, GPU utilization remained 100% at 1 MiB, with
only nvidia-persistenced holding the device nodes. Under both leases, root
restarted only that service at 06:24:42--44. The GPU subsequently returned to
0%, 1 MiB, P8 and 12.69 W. GDM and persistence remained active. Commands and
immediate recovery observations are in recover-persistence.sh and
persistence-recovery.log. No module reload was performed.

source-rebuild/ is the successful build input. The live isolated source was
subsequently edited by the model at 06:22:53, removing the remaining manager
passthrough after the successful build had already ended at 06:22:02.
That later source is not the binary used in this run. The snapshots preserve
the measured implementation independently of ongoing local development.
