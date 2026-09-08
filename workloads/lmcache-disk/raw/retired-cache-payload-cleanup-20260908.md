# Retired LMCache payload cleanup

User requested disk cleanup at 2026-09-07T21:56:25.351789-07:00

Only regenerable *.kvcache.safetensors payloads in the retired cache directories listed below are removed. Metadata files are retained. Original result.json, server.log, CSV, model weights, source, paper, and current experiment caches are untouched. No open descriptors or memory mappings referenced this retired archive before cleanup. Historical notes that these payloads remain recoverable are superseded by this cleanup. Recreate payloads by rerunning cache population if needed.

| Cache directory | Payload files | Bytes |
| --- | ---: | ---: |
| raw/gds-kv-reclaim-grace-575-20260908-01/block-01/position-0-bpf/cache | 48 | 1208156160 |
| raw/gds-kv-reclaim-grace-575-20260908-01/block-01/position-1-stock/cache | 48 | 1208156160 |
| raw/gds-kv-reclaim-grace-575-20260908-01/block-01/position-2-native/cache | 48 | 1208156160 |
| raw/gds-kv-reclaim-grace-575-20260908-01/block-02/position-0-stock/cache | 48 | 1208156160 |
| raw/gds-kv-reclaim-grace-575-20260908-01/block-02/position-1-native/cache | 48 | 1208156160 |
| raw/gds-kv-reclaim-grace-575-20260908-01/block-02/position-2-bpf/cache | 48 | 1208156160 |
| raw/gds-kv-reclaim-grace-575-20260908-01/block-03/position-0-native/cache | 48 | 1208156160 |
| raw/gds-kv-reclaim-grace-575-20260908-01/block-03/position-1-bpf/cache | 48 | 1208156160 |
| raw/gds-kv-reclaim-grace-575-20260908-01/block-03/position-2-stock/cache | 48 | 1208156160 |
| raw/gds-kv-reclaim-grace-575-20260908-01/block-04/position-0-bpf/cache | 48 | 1208156160 |
| raw/gds-kv-reclaim-grace-575-20260908-01/block-04/position-1-stock/cache | 48 | 1208156160 |
| raw/gds-kv-reclaim-grace-575-20260908-01/block-04/position-2-native/cache | 48 | 1208156160 |
| raw/gds-kv-reclaim-grace-575-20260908-01/bpf-async-absolute/cache | 48 | 1208156160 |
| raw/gds-kv-reclaim-grace-575-20260908-01/native-async/cache | 48 | 1208156160 |
| raw/gds-kv-reclaim-grace-575-20260908-01/stock-async-absolute/cache | 48 | 1208156160 |
| raw/gds-kv-total-cost-575-20260908-01/block-00/position-0-stock/cache | 48 | 1208156160 |
| raw/gds-kv-total-cost-575-20260908-01/block-00/position-1-native_ratio/cache | 48 | 1208156160 |
| raw/gds-kv-total-cost-575-20260908-01/block-00/position-2-native_total/cache | 48 | 1208156160 |
| raw/gds-kv-total-cost-575-20260908-01/block-00/position-3-bpf_total/cache | 48 | 1208156160 |
| raw/gds-kv-total-cost-575-20260908-01/block-01/position-0-native_ratio-after-enospc/cache | 48 | 1208156160 |
| raw/gds-kv-total-cost-575-20260908-01/block-01/position-1-native_total/cache | 48 | 1208156160 |
| raw/gds-kv-total-cost-575-20260908-01/block-01/position-2-bpf_total/cache | 48 | 1208156160 |
| raw/gds-kv-total-cost-575-20260908-01/block-01/position-3-stock/cache | 48 | 1208156160 |
| raw/gds-kv-total-cost-575-20260908-01/block-01/position-4-native_ratio-restored/cache | 48 | 1208156160 |
| raw/gds-kv-total-cost-575-20260908-01/block-02/position-0-native_total/cache | 48 | 1208156160 |
| raw/gds-kv-total-cost-575-20260908-01/block-02/position-1-bpf_total/cache | 48 | 1208156160 |
| raw/gds-kv-total-cost-575-20260908-01/block-02/position-2-stock/cache | 48 | 1208156160 |
| raw/gds-kv-total-cost-575-20260908-01/block-02/position-3-native_ratio/cache | 48 | 1208156160 |
| raw/gds-kv-total-cost-575-20260908-01/block-03/position-0-bpf_total/cache | 48 | 1208156160 |
| raw/gds-kv-total-cost-575-20260908-01/block-03/position-1-stock/cache | 48 | 1208156160 |
| raw/gds-kv-total-cost-575-20260908-01/block-03/position-2-native_ratio/cache | 48 | 1208156160 |
| raw/gds-kv-total-cost-575-20260908-01/block-03/position-3-native_total/cache | 48 | 1208156160 |
| raw/gds-kv-total-cost-575-20260908-01/block-04/position-0-stock/cache | 48 | 1208156160 |
| raw/gds-kv-total-cost-575-20260908-01/block-04/position-1-native_ratio/cache | 48 | 1208156160 |
| raw/gds-kv-total-cost-575-20260908-01/block-04/position-2-native_total/cache | 48 | 1208156160 |
| raw/gds-kv-total-cost-575-20260908-01/block-04/position-3-bpf_total/cache | 48 | 1208156160 |

Completed: removed 1728 payload files, 43493621760 logical bytes (40.51 GiB).
