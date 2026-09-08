# Completed cache payload archive

After both the main campaign and the missing-control process exited zero,
root moved the seventeen remaining completed-cell cache directories to the
existing recoverable archive below. The first four completed block-0 cache
directories had been archived during the earlier ENOSPC recovery. The failed
cold-population directory remains in place.

No result, configuration, server log, or progress record was moved or deleted.
No cache payload was deleted. These are regenerable LMCache backing files, not
new experiment measurements. Workspace available space increased from about
5.2 GiB to 25 GiB; archive filesystem has about 56 GiB available afterward.

Original paths below are relative to workloads/lmcache-disk. Each now resides
under `/var/tmp/lmcache-retired-cache-20260908.9ypd5z/` with that same relative
path. Moving it back restores the original placement, when workspace capacity
permits and no active experiment owns that path.

- `raw/gds-kv-total-cost-575-20260908-01/block-01/position-0-native_ratio-after-enospc/cache`
- `raw/gds-kv-total-cost-575-20260908-01/block-01/position-1-native_total/cache`
- `raw/gds-kv-total-cost-575-20260908-01/block-01/position-2-bpf_total/cache`
- `raw/gds-kv-total-cost-575-20260908-01/block-01/position-3-stock/cache`
- `raw/gds-kv-total-cost-575-20260908-01/block-02/position-0-native_total/cache`
- `raw/gds-kv-total-cost-575-20260908-01/block-02/position-1-bpf_total/cache`
- `raw/gds-kv-total-cost-575-20260908-01/block-02/position-2-stock/cache`
- `raw/gds-kv-total-cost-575-20260908-01/block-02/position-3-native_ratio/cache`
- `raw/gds-kv-total-cost-575-20260908-01/block-03/position-0-bpf_total/cache`
- `raw/gds-kv-total-cost-575-20260908-01/block-03/position-1-stock/cache`
- `raw/gds-kv-total-cost-575-20260908-01/block-03/position-2-native_ratio/cache`
- `raw/gds-kv-total-cost-575-20260908-01/block-03/position-3-native_total/cache`
- `raw/gds-kv-total-cost-575-20260908-01/block-04/position-0-stock/cache`
- `raw/gds-kv-total-cost-575-20260908-01/block-04/position-1-native_ratio/cache`
- `raw/gds-kv-total-cost-575-20260908-01/block-04/position-2-native_total/cache`
- `raw/gds-kv-total-cost-575-20260908-01/block-04/position-3-bpf_total/cache`
- `raw/gds-kv-total-cost-575-20260908-01/block-01/position-4-native_ratio-restored/cache`
