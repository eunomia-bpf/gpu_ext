# Old LMCache worktree cache deletion

Status: completed, including retirement of the old worktree. The 149 listed cache directories and the three obsolete
build executables were deleted after preservation push `fdbd1c12` on
`revision/lmcache-gds-control` and collection push `294bda3e` on master.
All 890 pre-existing non-cache raw files remain; the cleanup record adds one
file. The worktree shrank from about 322 GiB to 4.3 GiB, and workspace
available space rose from about 25 GiB to 342 GiB.

Removed build products (not published as binary artifacts):

- `workloads/lmcache-disk/gds-control/gds_policy`: 1517272 bytes.
- `workloads/lmcache-disk/gds-control/gds_executor`: 1057656 bytes.
- `workloads/lmcache-disk/gds-control/ioctl_probe`: 25272 bytes.

The source Makefile remains in Git for rebuilding. Cache deletion is permanent,
not a move to another archive. After the cache cleanup, the remaining 4.3 GiB
worktree was also removed with `git worktree remove --force`. Its source and
historical results remain recoverable from the pushed branch
`revision/lmcache-gds-control` at `4c469d2f`; the collected results and this
record are also on master. No large build product or cache payload was committed.

Before removing the worktree, the active 575 driver's 122 MiB Git repository
metadata was relocated from the old worktree's administration directory to
`/home/yunwei37/workspace/gpu/gpu_ext/.git/driver-repositories/nvidia-575.git`.
The active `/home/yunwei37/workspace/gpu/gpu_ext-kernel-575-gds/.git` pointer
now uses that independent location. The common repository is bare, with the
active driver checkout registered as its linked worktree. No driver source,
build product, loaded module, or running OpenCode session was removed or stopped.

Post-removal checks: the old worktree path is absent; the main repository's
worktree list contains only its main checkout; the driver remains on
`revision/gpu-storage-decision-575` at `ff68a1d4`. Its two modified headers
(`uvm_forward_decl.h`, `uvm_va_range.h`) and untracked `uvm_disk_backing.h`
remain present. Before removal, the old checkout and all initialized submodules
had no uncommitted source changes; their revisions were present in remote
branches. Its only remaining untracked entries were `current-venv` and `deps`
symlinks, whose targets in the main workspace were not deleted.

User explicitly requested deletion, not archival, of regenerable cache and large
build products. Scope is the inactive gpu_ext-lmcache-gds-control worktree only.
The active 575 driver source worktree remains untouched; only its Git metadata
location was changed to remove the dependency on the retired worktree.
The SASS exporter and Table1 still have live source/build dependencies, so those
worktrees are not cleanup targets.

Classification:

- Preserve source and historical Python prototype on revision/lmcache-gds-control.
- Preserve all 890 non-cache raw files; 201 previously untracked result/log files total 7571602 bytes and were committed and pushed before deletion. No large product was added to Git.
- Delete only the 149 explicitly listed cache directories below, totaling 340726042624 allocated bytes (317.326 GiB before deletion).
- No tracked file is inside a target cache directory. Cache payloads/sidecars are regenerable transport data, not performance measurements. Deleted contents cannot be recovered from Git; rerunning their producer is necessary.
- Old compiled gds_policy, gds_executor and ioctl_probe may be removed after their sizes are recorded. Active loaders use the separate gpu_ext main path, not these binaries.

## Exact cache targets

Paths are relative to /home/yunwei37/workspace/gpu/gpu_ext-lmcache-gds-control.

| Path | Allocated bytes |
| --- | ---: |
| `workloads/lmcache-disk/raw/gds-mixed-live-feedback-575-20260907-five-block/block-02/position-1-gds_fifo/cache` | 4027904000 |
| `workloads/lmcache-disk/raw/gds-mixed-live-feedback-575-20260907-five-block/block-02/position-0-gds_bpf/cache` | 4027899904 |
| `workloads/lmcache-disk/raw/gds-mixed-live-feedback-575-20260907-five-block/block-02/position-2-gds_native/cache` | 4027904000 |
| `workloads/lmcache-disk/raw/gds-mixed-live-feedback-575-20260907-five-block/block-00/position-0-gds_fifo/cache` | 4027936768 |
| `workloads/lmcache-disk/raw/gds-mixed-live-feedback-575-20260907-five-block/block-00/position-1-gds_native/cache` | 4027887616 |
| `workloads/lmcache-disk/raw/gds-mixed-live-feedback-575-20260907-five-block/block-00/position-2-gds_bpf/cache` | 4027904000 |
| `workloads/lmcache-disk/raw/gds-mixed-live-feedback-575-20260907-five-block/block-01/position-0-gds_native/cache` | 4027895808 |
| `workloads/lmcache-disk/raw/gds-mixed-live-feedback-575-20260907-five-block/block-01/position-1-gds_bpf/cache` | 4027895808 |
| `workloads/lmcache-disk/raw/gds-mixed-live-feedback-575-20260907-five-block/block-01/position-2-gds_fifo/cache` | 4027891712 |
| `workloads/lmcache-disk/raw/gds-mixed-live-feedback-575-20260907-five-block/block-04/position-0-gds_native/cache` | 4027883520 |
| `workloads/lmcache-disk/raw/gds-mixed-live-feedback-575-20260907-five-block/block-04/position-1-gds_bpf/cache` | 4027908096 |
| `workloads/lmcache-disk/raw/gds-mixed-live-feedback-575-20260907-five-block/block-04/position-2-gds_fifo/cache` | 4027912192 |
| `workloads/lmcache-disk/raw/gds-mixed-live-feedback-575-20260907-five-block/block-03/position-0-gds_fifo/cache` | 4027904000 |
| `workloads/lmcache-disk/raw/gds-mixed-live-feedback-575-20260907-five-block/block-03/position-1-gds_native/cache` | 4027899904 |
| `workloads/lmcache-disk/raw/gds-mixed-live-feedback-575-20260907-five-block/block-03/position-2-gds_bpf/cache` | 4027904000 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-five-block-formal/block-02/position-0-gds_fifo/cache` | 1208688640 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-five-block-formal/block-02/position-1-gds_native/cache` | 1208676352 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-five-block-formal/block-02/position-4-lmcache_cpu/cache` | 4096 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-five-block-formal/block-02/position-2-gds_bpf/cache` | 1208659968 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-five-block-formal/block-02/position-3-recompute/cache` | 4096 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-five-block-formal/block-00/position-1-lmcache_cpu/cache` | 4096 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-five-block-formal/block-00/position-3-gds_native/cache` | 1208688640 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-five-block-formal/block-00/position-0-recompute/cache` | 4096 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-five-block-formal/block-00/position-2-gds_fifo/cache` | 1208692736 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-five-block-formal/block-00/position-4-gds_bpf/cache` | 1208684544 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-five-block-formal/block-01/position-1-gds_fifo/cache` | 1208684544 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-five-block-formal/block-01/position-2-gds_native/cache` | 1208688640 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-five-block-formal/block-01/position-4-recompute/cache` | 4096 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-five-block-formal/block-01/position-0-lmcache_cpu/cache` | 4096 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-five-block-formal/block-01/position-3-gds_bpf/cache` | 1208688640 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-five-block-formal/block-04/position-3-gds_fifo/cache` | 1208676352 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-five-block-formal/block-04/position-4-gds_native/cache` | 1208692736 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-five-block-formal/block-04/position-2-lmcache_cpu/cache` | 4096 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-five-block-formal/block-04/position-0-gds_bpf/cache` | 1208680448 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-five-block-formal/block-04/position-1-recompute/cache` | 4096 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-five-block-formal/block-03/position-2-recompute/cache` | 4096 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-five-block-formal/block-03/position-4-gds_fifo/cache` | 1208705024 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-five-block-formal/block-03/position-0-gds_native/cache` | 1208700928 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-five-block-formal/block-03/position-1-gds_bpf/cache` | 1208664064 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-five-block-formal/block-03/position-3-lmcache_cpu/cache` | 4096 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-block1-rerun/block-00/position-1-lmcache_cpu/cache` | 4096 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-block1-rerun/block-00/position-0-recompute/cache` | 4096 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-block1-rerun/block-00/position-2-gds_fifo/cache` | 4096 |
| `workloads/lmcache-disk/raw/gds-policy-ablation-575-20260906-five-block/block-02/position-1-gds_full_native/cache` | 1208684544 |
| `workloads/lmcache-disk/raw/gds-policy-ablation-575-20260906-five-block/block-02/position-3-gds_fifo/cache` | 1208696832 |
| `workloads/lmcache-disk/raw/gds-policy-ablation-575-20260906-five-block/block-02/position-0-gds_defer_native/cache` | 1208696832 |
| `workloads/lmcache-disk/raw/gds-policy-ablation-575-20260906-five-block/block-02/position-2-gds_full_bpf/cache` | 1208684544 |
| `workloads/lmcache-disk/raw/gds-policy-ablation-575-20260906-five-block/block-02/position-4-gds_bpf_floor/cache` | 1208659968 |
| `workloads/lmcache-disk/raw/gds-policy-ablation-575-20260906-five-block/block-00/position-2-gds_defer_native/cache` | 1208692736 |
| `workloads/lmcache-disk/raw/gds-policy-ablation-575-20260906-five-block/block-00/position-0-gds_fifo/cache` | 1208700928 |
| `workloads/lmcache-disk/raw/gds-policy-ablation-575-20260906-five-block/block-00/position-1-gds_bpf_floor/cache` | 1208688640 |
| `workloads/lmcache-disk/raw/gds-policy-ablation-575-20260906-five-block/block-00/position-4-gds_full_bpf/cache` | 1208668160 |
| `workloads/lmcache-disk/raw/gds-policy-ablation-575-20260906-five-block/block-00/position-3-gds_full_native/cache` | 1208672256 |
| `workloads/lmcache-disk/raw/gds-policy-ablation-575-20260906-five-block/block-01/position-3-gds_full_bpf/cache` | 1208688640 |
| `workloads/lmcache-disk/raw/gds-policy-ablation-575-20260906-five-block/block-01/position-4-gds_fifo/cache` | 1208684544 |
| `workloads/lmcache-disk/raw/gds-policy-ablation-575-20260906-five-block/block-01/position-2-gds_full_native/cache` | 1208680448 |
| `workloads/lmcache-disk/raw/gds-policy-ablation-575-20260906-five-block/block-01/position-1-gds_defer_native/cache` | 1208700928 |
| `workloads/lmcache-disk/raw/gds-policy-ablation-575-20260906-five-block/block-01/position-0-gds_bpf_floor/cache` | 1208692736 |
| `workloads/lmcache-disk/raw/gds-policy-ablation-575-20260906-five-block/block-04/position-3-gds_defer_native/cache` | 1208713216 |
| `workloads/lmcache-disk/raw/gds-policy-ablation-575-20260906-five-block/block-04/position-2-gds_bpf_floor/cache` | 1208684544 |
| `workloads/lmcache-disk/raw/gds-policy-ablation-575-20260906-five-block/block-04/position-1-gds_fifo/cache` | 1208680448 |
| `workloads/lmcache-disk/raw/gds-policy-ablation-575-20260906-five-block/block-04/position-0-gds_full_bpf/cache` | 1208676352 |
| `workloads/lmcache-disk/raw/gds-policy-ablation-575-20260906-five-block/block-04/position-4-gds_full_native/cache` | 1208680448 |
| `workloads/lmcache-disk/raw/gds-policy-ablation-575-20260906-five-block/block-03/position-3-gds_bpf_floor/cache` | 1208692736 |
| `workloads/lmcache-disk/raw/gds-policy-ablation-575-20260906-five-block/block-03/position-0-gds_full_native/cache` | 1208684544 |
| `workloads/lmcache-disk/raw/gds-policy-ablation-575-20260906-five-block/block-03/position-1-gds_full_bpf/cache` | 1208700928 |
| `workloads/lmcache-disk/raw/gds-policy-ablation-575-20260906-five-block/block-03/position-4-gds_defer_native/cache` | 1208680448 |
| `workloads/lmcache-disk/raw/gds-policy-ablation-575-20260906-five-block/block-03/position-2-gds_fifo/cache` | 1208684544 |
| `workloads/lmcache-disk/raw/gds-mixed-backend-575-20260906-02/block-00/position-0-gds_fifo/cache` | 4027928576 |
| `workloads/lmcache-disk/raw/gds-mixed-backend-575-20260906-02/block-00/position-1-gds_native/cache` | 4027899904 |
| `workloads/lmcache-disk/raw/gds-mixed-backend-575-20260906-02/block-00/position-2-gds_bpf/cache` | 4027891712 |
| `workloads/lmcache-disk/raw/gds-mixed-burst-fresh-575-20260906-five-block/block-4/position-0-gds_native/block-00/position-0-gds_native/cache` | 4027920384 |
| `workloads/lmcache-disk/raw/gds-mixed-burst-fresh-575-20260906-five-block/block-4/position-1-gds_bpf/block-00/position-0-gds_bpf/cache` | 4027924480 |
| `workloads/lmcache-disk/raw/gds-mixed-burst-fresh-575-20260906-five-block/block-4/position-2-gds_fifo/block-00/position-0-gds_fifo/cache` | 4027895808 |
| `workloads/lmcache-disk/raw/gds-mixed-burst-fresh-575-20260906-five-block/block-1/position-0-gds_native/block-00/position-0-gds_native/cache` | 4027908096 |
| `workloads/lmcache-disk/raw/gds-mixed-burst-fresh-575-20260906-five-block/block-1/position-1-gds_bpf/block-00/position-0-gds_bpf/cache` | 4027895808 |
| `workloads/lmcache-disk/raw/gds-mixed-burst-fresh-575-20260906-five-block/block-1/position-2-gds_fifo/block-00/position-0-gds_fifo/cache` | 4027904000 |
| `workloads/lmcache-disk/raw/gds-mixed-burst-fresh-575-20260906-five-block/block-0/position-0-gds_fifo/block-00/position-0-gds_fifo/cache` | 4027965440 |
| `workloads/lmcache-disk/raw/gds-mixed-burst-fresh-575-20260906-five-block/block-0/position-1-gds_native/block-00/position-0-gds_native/cache` | 4027940864 |
| `workloads/lmcache-disk/raw/gds-mixed-burst-fresh-575-20260906-five-block/block-0/position-2-gds_bpf/block-00/position-0-gds_bpf/cache` | 4027920384 |
| `workloads/lmcache-disk/raw/gds-mixed-burst-fresh-575-20260906-five-block/block-2/position-1-gds_fifo/block-00/position-0-gds_fifo/cache` | 4027928576 |
| `workloads/lmcache-disk/raw/gds-mixed-burst-fresh-575-20260906-five-block/block-2/position-0-gds_bpf/block-00/position-0-gds_bpf/cache` | 4027912192 |
| `workloads/lmcache-disk/raw/gds-mixed-burst-fresh-575-20260906-five-block/block-2/position-2-gds_native/block-00/position-0-gds_native/cache` | 4027928576 |
| `workloads/lmcache-disk/raw/gds-mixed-burst-fresh-575-20260906-five-block/block-3/position-0-gds_fifo/block-00/position-0-gds_fifo/cache` | 4027924480 |
| `workloads/lmcache-disk/raw/gds-mixed-burst-fresh-575-20260906-five-block/block-3/position-1-gds_native/block-00/position-0-gds_native/cache` | 4027949056 |
| `workloads/lmcache-disk/raw/gds-mixed-burst-fresh-575-20260906-five-block/block-3/position-2-gds_bpf/block-00/position-0-gds_bpf/cache` | 4027957248 |
| `workloads/lmcache-disk/raw/gds-mixed-backend-dispatch-575-20260906-five-block/block-02/position-1-gds_fifo/cache` | 4096 |
| `workloads/lmcache-disk/raw/gds-mixed-backend-dispatch-575-20260906-five-block/block-02/position-0-gds_bpf/cache` | 4027940864 |
| `workloads/lmcache-disk/raw/gds-mixed-backend-dispatch-575-20260906-five-block/block-02/position-2-gds_native/cache` | 4096 |
| `workloads/lmcache-disk/raw/gds-mixed-backend-dispatch-575-20260906-five-block/block-00/position-0-gds_fifo/cache` | 4027953152 |
| `workloads/lmcache-disk/raw/gds-mixed-backend-dispatch-575-20260906-five-block/block-00/position-1-gds_native/cache` | 4027891712 |
| `workloads/lmcache-disk/raw/gds-mixed-backend-dispatch-575-20260906-five-block/block-00/position-2-gds_bpf/cache` | 4027891712 |
| `workloads/lmcache-disk/raw/gds-mixed-backend-dispatch-575-20260906-five-block/block-01/position-0-gds_native/cache` | 4027904000 |
| `workloads/lmcache-disk/raw/gds-mixed-backend-dispatch-575-20260906-five-block/block-01/position-1-gds_bpf/cache` | 4027891712 |
| `workloads/lmcache-disk/raw/gds-mixed-backend-dispatch-575-20260906-five-block/block-01/position-2-gds_fifo/cache` | 4027904000 |
| `workloads/lmcache-disk/raw/gds-mixed-backend-dispatch-575-20260906-five-block/block-04/position-0-gds_native/cache` | 4096 |
| `workloads/lmcache-disk/raw/gds-mixed-backend-dispatch-575-20260906-five-block/block-04/position-1-gds_bpf/cache` | 4096 |
| `workloads/lmcache-disk/raw/gds-mixed-backend-dispatch-575-20260906-five-block/block-04/position-2-gds_fifo/cache` | 4096 |
| `workloads/lmcache-disk/raw/gds-mixed-backend-dispatch-575-20260906-five-block/block-03/position-0-gds_fifo/cache` | 4096 |
| `workloads/lmcache-disk/raw/gds-mixed-backend-dispatch-575-20260906-five-block/block-03/position-1-gds_native/cache` | 4096 |
| `workloads/lmcache-disk/raw/gds-mixed-backend-dispatch-575-20260906-five-block/block-03/position-2-gds_bpf/cache` | 4096 |
| `workloads/lmcache-disk/raw/gds-mixed-scheduled-575-20260907-five-block/block-02/position-1-gds_fifo/cache` | 4027891712 |
| `workloads/lmcache-disk/raw/gds-mixed-scheduled-575-20260907-five-block/block-02/position-0-gds_bpf/cache` | 4027916288 |
| `workloads/lmcache-disk/raw/gds-mixed-scheduled-575-20260907-five-block/block-02/position-2-gds_native/cache` | 4027912192 |
| `workloads/lmcache-disk/raw/gds-mixed-scheduled-575-20260907-five-block/block-00/position-0-gds_fifo/cache` | 4027977728 |
| `workloads/lmcache-disk/raw/gds-mixed-scheduled-575-20260907-five-block/block-00/position-1-gds_native/cache` | 4027904000 |
| `workloads/lmcache-disk/raw/gds-mixed-scheduled-575-20260907-five-block/block-00/position-2-gds_bpf/cache` | 4027899904 |
| `workloads/lmcache-disk/raw/gds-mixed-scheduled-575-20260907-five-block/block-01/position-0-gds_native/cache` | 4027895808 |
| `workloads/lmcache-disk/raw/gds-mixed-scheduled-575-20260907-five-block/block-01/position-1-gds_bpf/cache` | 4027887616 |
| `workloads/lmcache-disk/raw/gds-mixed-scheduled-575-20260907-five-block/block-01/position-2-gds_fifo/cache` | 4027891712 |
| `workloads/lmcache-disk/raw/gds-mixed-scheduled-575-20260907-five-block/block-04/position-0-gds_native/cache` | 4027887616 |
| `workloads/lmcache-disk/raw/gds-mixed-scheduled-575-20260907-five-block/block-04/position-1-gds_bpf/cache` | 4027912192 |
| `workloads/lmcache-disk/raw/gds-mixed-scheduled-575-20260907-five-block/block-04/position-2-gds_fifo/cache` | 4027920384 |
| `workloads/lmcache-disk/raw/gds-mixed-scheduled-575-20260907-five-block/block-03/position-0-gds_fifo/cache` | 4027899904 |
| `workloads/lmcache-disk/raw/gds-mixed-scheduled-575-20260907-five-block/block-03/position-1-gds_native/cache` | 4027932672 |
| `workloads/lmcache-disk/raw/gds-mixed-scheduled-575-20260907-five-block/block-03/position-2-gds_bpf/cache` | 4027924480 |
| `workloads/lmcache-disk/raw/gds-mixed-backend-575-20260906-01/block-00/position-0-gds_fifo/cache` | 251752448 |
| `workloads/lmcache-disk/raw/gds-mixed-backend-575-20260906-01/block-00/position-1-gds_native/cache` | 251752448 |
| `workloads/lmcache-disk/raw/gds-mixed-backend-575-20260906-01/block-00/position-2-gds_bpf/cache` | 251756544 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-block1-formal/block-00/position-1-lmcache_cpu/cache` | 4096 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-block1-formal/block-00/position-3-gds_native/cache` | 1208672256 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-block1-formal/block-00/position-0-recompute/cache` | 4096 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-block1-formal/block-00/position-2-gds_fifo/cache` | 1208668160 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-block1-formal/block-00/position-4-gds_bpf/cache` | 1208684544 |
| `workloads/lmcache-disk/raw/gds-policy-ablation-575-20260906-block1/block-00/position-2-gds_defer_native/cache` | 1208680448 |
| `workloads/lmcache-disk/raw/gds-policy-ablation-575-20260906-block1/block-00/position-0-gds_fifo/cache` | 1208655872 |
| `workloads/lmcache-disk/raw/gds-policy-ablation-575-20260906-block1/block-00/position-1-gds_bpf_floor/cache` | 1208659968 |
| `workloads/lmcache-disk/raw/gds-policy-ablation-575-20260906-block1/block-00/position-4-gds_full_bpf/cache` | 1208705024 |
| `workloads/lmcache-disk/raw/gds-policy-ablation-575-20260906-block1/block-00/position-3-gds_full_native/cache` | 1208668160 |
| `workloads/lmcache-disk/raw/gds-mixed-fresh-process-575-20260906-five-block/block-4/position-0-gds_native/block-00/position-0-gds_native/cache` | 4027895808 |
| `workloads/lmcache-disk/raw/gds-mixed-fresh-process-575-20260906-five-block/block-4/position-1-gds_bpf/block-00/position-0-gds_bpf/cache` | 4027944960 |
| `workloads/lmcache-disk/raw/gds-mixed-fresh-process-575-20260906-five-block/block-4/position-2-gds_fifo/block-00/position-0-gds_fifo/cache` | 4027912192 |
| `workloads/lmcache-disk/raw/gds-mixed-fresh-process-575-20260906-five-block/block-1/position-0-gds_native/block-00/position-0-gds_native/cache` | 4027887616 |
| `workloads/lmcache-disk/raw/gds-mixed-fresh-process-575-20260906-five-block/block-1/position-1-gds_bpf/block-00/position-0-gds_bpf/cache` | 4027887616 |
| `workloads/lmcache-disk/raw/gds-mixed-fresh-process-575-20260906-five-block/block-1/position-2-gds_fifo/block-00/position-0-gds_fifo/cache` | 4027904000 |
| `workloads/lmcache-disk/raw/gds-mixed-fresh-process-575-20260906-five-block/block-0/position-0-gds_fifo/block-00/position-0-gds_fifo/cache` | 4027969536 |
| `workloads/lmcache-disk/raw/gds-mixed-fresh-process-575-20260906-five-block/block-0/position-1-gds_native/block-00/position-0-gds_native/cache` | 4027924480 |
| `workloads/lmcache-disk/raw/gds-mixed-fresh-process-575-20260906-five-block/block-0/position-2-gds_bpf/block-00/position-0-gds_bpf/cache` | 4027895808 |
| `workloads/lmcache-disk/raw/gds-mixed-fresh-process-575-20260906-five-block/block-2/position-1-gds_fifo/block-00/position-0-gds_fifo/cache` | 4027887616 |
| `workloads/lmcache-disk/raw/gds-mixed-fresh-process-575-20260906-five-block/block-2/position-0-gds_bpf/block-00/position-0-gds_bpf/cache` | 4027912192 |
| `workloads/lmcache-disk/raw/gds-mixed-fresh-process-575-20260906-five-block/block-2/position-2-gds_native/block-00/position-0-gds_native/cache` | 4027912192 |
| `workloads/lmcache-disk/raw/gds-mixed-fresh-process-575-20260906-five-block/block-3/position-0-gds_fifo/block-00/position-0-gds_fifo/cache` | 4027924480 |
| `workloads/lmcache-disk/raw/gds-mixed-fresh-process-575-20260906-five-block/block-3/position-1-gds_native/block-00/position-0-gds_native/cache` | 4027891712 |
| `workloads/lmcache-disk/raw/gds-mixed-fresh-process-575-20260906-five-block/block-3/position-2-gds_bpf/block-00/position-0-gds_bpf/cache` | 4027920384 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-block1/block-00/position-1-lmcache_cpu/cache` | 4096 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-block1/block-00/position-3-gds_native/cache` | 4096 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-block1/block-00/position-0-recompute/cache` | 4096 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-block1/block-00/position-2-gds_fifo/cache` | 4096 |
| `workloads/lmcache-disk/raw/gds-five-arm-575-20260906-block1/block-00/position-4-gds_bpf/cache` | 4096 |
