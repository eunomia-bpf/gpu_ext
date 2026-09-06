# LMCache three-arm performance-only comparison

- Kind: `lmcache_perf_only`
- Timestamp: `20260905T173648`
- Driver parameter: `575.57.08`
- Blocks: `5` (rotated from schedule.json seed 2709)
- Prompts: `/home/yunwei37/workspace/gpu/gpu_ext-lmcache-perf/workloads/lmcache-disk/prompts.json` (8 prefixes x 1536 cached tokens, 16 output tokens)

No correctness, engagement, admission, retry, or filtering gates ran; nonzero server return codes are preserved per cell.

## Cell metrics

| Block | Position | Arm | ready | warm TTFT median (ms) | warm requests/s | warm out tok/s | warm ok/failed | server returncode |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 0 | 0 | lmcache_cpu | true | 73.5090 | 1.8808 | 30.0928 | 8/0 | 0 |
| 0 | 1 | recompute | true | 66.1587 | 1.9360 | 30.9753 | 8/0 | 0 |
| 0 | 2 | lmcache_disk | true | 95.2517 | 1.8279 | 29.2461 | 8/0 | 0 |
| 1 | 0 | recompute | true | 65.9496 | 1.9697 | 31.5145 | 8/0 | 0 |
| 1 | 1 | lmcache_disk | true | 95.6620 | 1.8074 | 28.9180 | 8/0 | 0 |
| 1 | 2 | lmcache_cpu | true | 72.3823 | 1.9178 | 30.6849 | 8/0 | 0 |
| 2 | 0 | lmcache_disk | true | 94.5592 | 1.8583 | 29.7327 | 8/0 | 0 |
| 2 | 1 | lmcache_cpu | true | 73.5325 | 1.8958 | 30.3331 | 8/0 | 0 |
| 2 | 2 | recompute | true | 67.1434 | 1.9077 | 30.5236 | 8/0 | 0 |
| 3 | 0 | recompute | true | 68.0176 | 1.9162 | 30.6597 | 8/0 | 0 |
| 3 | 1 | lmcache_disk | true | 95.1466 | 1.8293 | 29.2681 | 8/0 | 0 |
| 3 | 2 | lmcache_cpu | true | 73.4025 | 1.8869 | 30.1909 | 8/0 | 0 |
| 4 | 0 | lmcache_cpu | true | 73.9401 | 1.8802 | 30.0826 | 8/0 | 0 |
| 4 | 1 | recompute | true | 67.8896 | 1.9068 | 30.5084 | 8/0 | 0 |
| 4 | 2 | lmcache_disk | true | 92.3595 | 1.8551 | 29.6810 | 8/0 | 0 |

## Per-arm values across attempted cells

| Arm | cells | warm TTFT medians (ms) | warm requests/s | warm out tok/s |
|---|---:|---|---|---|
| recompute | 5 | 66.1587, 65.9496, 67.1434, 68.0176, 67.8896 | 1.9360, 1.9697, 1.9077, 1.9162, 1.9068 | n/a |
| lmcache_cpu | 5 | 73.5090, 72.3823, 73.5325, 73.4025, 73.9401 | 1.8808, 1.9178, 1.8958, 1.8869, 1.8802 | n/a |
| lmcache_disk | 5 | 95.2517, 95.6620, 94.5592, 95.1466, 92.3595 | 1.8279, 1.8074, 1.8583, 1.8293, 1.8551 | n/a |
