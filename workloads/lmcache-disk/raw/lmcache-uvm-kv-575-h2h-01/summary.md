# LMCache/UVM-KV three-arm performance-only comparison

- Kind: `lmcache_uvm_kv_perf`
- Timestamp: `20260905T221127`
- Driver parameter: `575.57.08`
- Blocks: `5` (rotating base order ['recompute', 'lmcache_disk', 'lmcache_disk_uvm_kv'])
- Prompts: `/home/yunwei37/workspace/gpu/gpu_ext-lmcache-kv-page-governor/workloads/lmcache-disk/prompts.json` (8 prefixes x 1536 cached tokens, 16 output tokens)

No correctness, engagement, admission, retry, or filtering gates ran; nonzero server return codes are preserved per cell.

## Cell metrics

| Block | Position | Arm | ready | warm TTFT median (ms) | warm requests/s | warm out tok/s | warm ok/failed | server returncode |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 0 | 0 | recompute | true | 66.8981 | 1.9529 | 31.2463 | 8/0 | 0 |
| 0 | 1 | lmcache_disk | true | 94.8120 | 1.8170 | 29.0723 | 8/0 | 0 |
| 0 | 2 | lmcache_disk_uvm_kv | true | 93.4996 | 1.8032 | 28.8512 | 8/0 | 0 |
| 1 | 0 | lmcache_disk | true | 96.9046 | 1.7846 | 28.5533 | 8/0 | 0 |
| 1 | 1 | lmcache_disk_uvm_kv | true | 95.2314 | 1.8157 | 29.0509 | 8/0 | 0 |
| 1 | 2 | recompute | true | 65.7045 | 1.9679 | 31.4865 | 8/0 | 0 |
| 2 | 0 | lmcache_disk_uvm_kv | true | 95.4827 | 1.8285 | 29.2557 | 8/0 | 0 |
| 2 | 1 | recompute | true | 67.3918 | 1.9377 | 31.0036 | 8/0 | 0 |
| 2 | 2 | lmcache_disk | true | 92.7859 | 1.8485 | 29.5760 | 8/0 | 0 |
| 3 | 0 | recompute | true | 66.0786 | 1.9468 | 31.1481 | 8/0 | 0 |
| 3 | 1 | lmcache_disk | true | 90.1368 | 1.8164 | 29.0623 | 8/0 | 0 |
| 3 | 2 | lmcache_disk_uvm_kv | true | 96.4481 | 1.7954 | 28.7269 | 8/0 | 0 |
| 4 | 0 | lmcache_disk | true | 97.4630 | 1.8006 | 28.8100 | 8/0 | 0 |
| 4 | 1 | lmcache_disk_uvm_kv | true | 91.1625 | 1.8238 | 29.1808 | 8/0 | 0 |
| 4 | 2 | recompute | true | 65.9746 | 1.9566 | 31.3052 | 8/0 | 0 |

## Per-arm values across attempted cells

| Arm | cells | warm TTFT medians (ms) | warm requests/s | warm out tok/s |
|---|---:|---|---|---|
| recompute | 5 | 66.8981, 65.7045, 67.3918, 66.0786, 65.9746 | 1.9529, 1.9679, 1.9377, 1.9468, 1.9566 | 31.2463, 31.4865, 31.0036, 31.1481, 31.3052 |
| lmcache_disk | 5 | 94.8120, 96.9046, 92.7859, 90.1368, 97.4630 | 1.8170, 1.7846, 1.8485, 1.8164, 1.8006 | 29.0723, 28.5533, 29.5760, 29.0623, 28.8100 |
| lmcache_disk_uvm_kv | 5 | 93.4996, 95.2314, 95.4827, 96.4481, 91.1625 | 1.8032, 1.8157, 1.8285, 1.7954, 1.8238 | 28.8512, 29.0509, 29.2557, 28.7269, 29.1808 |
