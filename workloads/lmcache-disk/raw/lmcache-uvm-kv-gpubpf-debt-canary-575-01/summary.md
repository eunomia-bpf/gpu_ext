# LMCache/UVM-KV selected-arm performance-only comparison

- Kind: `lmcache_uvm_kv_perf`
- Timestamp: `20260906T002502`
- Driver parameter: `575.57.08`
- Blocks: `1` (rotating base order ['lmcache_disk_uvm_kv_gpubpf_debt'])
- Prompts: `/home/yunwei37/workspace/gpu/gpu_ext-lmcache-kv-page-governor/workloads/lmcache-disk/prompts.json` (8 prefixes x 1536 cached tokens, 16 output tokens)

No correctness, engagement, admission, retry, or filtering gates ran; nonzero server return codes are preserved per cell.

## Cell metrics

| Block | Position | Arm | ready | warm TTFT median (ms) | warm requests/s | warm out tok/s | warm ok/failed | server returncode |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 0 | 0 | lmcache_disk_uvm_kv_gpubpf_debt | true | 95.6518 | 1.8509 | 29.6146 | 8/0 | 0 |

## Per-arm values across attempted cells

| Arm | cells | warm TTFT medians (ms) | warm requests/s | warm out tok/s |
|---|---:|---|---|---|
| lmcache_disk_uvm_kv_gpubpf_debt | 1 | 95.6518 | 1.8509 | 29.6146 |
