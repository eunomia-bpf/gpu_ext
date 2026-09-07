# Admission diagnostic temporary-cache cleanup

After all 15 children exited zero and their results/logs were committed and
pushed in `37299d27`, the following newly generated cache directories were
removed to free space for the next LMCache experiment. Each directory occupied
4,027,842,560 apparent bytes (`du -sb`), including its directory metadata;
15 directories total about 56.27 GiB. These were regenerable synthetic KV
payloads, not timing results, source data, reports or raw request records.
No tracked file was inside these cache directories. All 15 result files,
campaign/summary, admission records, runner/cuFile logs and analysis remain.
Deleted payloads are not archived; the recorded runner command regenerates them.

- `workloads/lmcache-disk/raw/gds-admission-timing-575-20260907-five-block/block-00/position-0-gds_fifo/cache`
- `workloads/lmcache-disk/raw/gds-admission-timing-575-20260907-five-block/block-00/position-1-gds_native/cache`
- `workloads/lmcache-disk/raw/gds-admission-timing-575-20260907-five-block/block-00/position-2-gds_bpf/cache`
- `workloads/lmcache-disk/raw/gds-admission-timing-575-20260907-five-block/block-01/position-0-gds_native/cache`
- `workloads/lmcache-disk/raw/gds-admission-timing-575-20260907-five-block/block-01/position-1-gds_bpf/cache`
- `workloads/lmcache-disk/raw/gds-admission-timing-575-20260907-five-block/block-01/position-2-gds_fifo/cache`
- `workloads/lmcache-disk/raw/gds-admission-timing-575-20260907-five-block/block-02/position-0-gds_bpf/cache`
- `workloads/lmcache-disk/raw/gds-admission-timing-575-20260907-five-block/block-02/position-1-gds_fifo/cache`
- `workloads/lmcache-disk/raw/gds-admission-timing-575-20260907-five-block/block-02/position-2-gds_native/cache`
- `workloads/lmcache-disk/raw/gds-admission-timing-575-20260907-five-block/block-03/position-0-gds_fifo/cache`
- `workloads/lmcache-disk/raw/gds-admission-timing-575-20260907-five-block/block-03/position-1-gds_native/cache`
- `workloads/lmcache-disk/raw/gds-admission-timing-575-20260907-five-block/block-03/position-2-gds_bpf/cache`
- `workloads/lmcache-disk/raw/gds-admission-timing-575-20260907-five-block/block-04/position-0-gds_native/cache`
- `workloads/lmcache-disk/raw/gds-admission-timing-575-20260907-five-block/block-04/position-1-gds_bpf/cache`
- `workloads/lmcache-disk/raw/gds-admission-timing-575-20260907-five-block/block-04/position-2-gds_fifo/cache`
