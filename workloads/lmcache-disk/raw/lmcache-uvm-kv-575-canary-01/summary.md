# LMCache/UVM-KV three-arm performance-only comparison

- Kind: `lmcache_uvm_kv_perf`
- Timestamp: `20260905T220545`
- Driver parameter: `575.57.08`
- Blocks: `1` (rotating base order ['recompute', 'lmcache_disk', 'lmcache_disk_uvm_kv'])
- Prompts: `/home/yunwei37/workspace/gpu/gpu_ext-lmcache-kv-page-governor/workloads/lmcache-disk/prompts.json` (8 prefixes x 1536 cached tokens, 16 output tokens)

No correctness, engagement, admission, retry, or filtering gates ran; nonzero server return codes are preserved per cell.

## Cell metrics

| Block | Position | Arm | ready | warm TTFT median (ms) | warm requests/s | warm out tok/s | warm ok/failed | server returncode |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 0 | 0 | recompute | true | 67.8793 | 1.9091 | 30.5452 | 8/0 | 0 |
| 0 | 1 | lmcache_disk | true | 96.3010 | 1.7874 | 28.5992 | 8/0 | 0 |
| 0 | 2 | lmcache_disk_uvm_kv | true | 97.5865 | 1.8166 | 29.0650 | 8/0 | 0 |

## Per-arm values across attempted cells

| Arm | cells | warm TTFT medians (ms) | warm requests/s | warm out tok/s |
|---|---:|---|---|---|
| recompute | 1 | 67.8793 | 1.9091 | 30.5452 |
| lmcache_disk | 1 | 96.3010 | 1.7874 | 28.5992 |
| lmcache_disk_uvm_kv | 1 | 97.5865 | 1.8166 | 29.0650 |
