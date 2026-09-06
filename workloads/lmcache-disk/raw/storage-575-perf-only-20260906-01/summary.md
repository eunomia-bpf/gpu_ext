# LMCache three-arm performance-only comparison

- Kind: `lmcache_perf_only`
- Timestamp: `20260906T065024`
- Driver parameter: `575.57.08`
- Blocks: `5` (rotated from schedule.json seed 2709)
- Prompts: `/home/yunwei37/workspace/gpu/gpu_ext-lmcache-kv-page-governor/workloads/lmcache-disk/prompts.json` (8 prefixes x 1536 cached tokens, 16 output tokens)

No correctness, engagement, admission, retry, or filtering gates ran; nonzero server return codes are preserved per cell.

## Cell metrics

| Block | Position | Arm | ready | warm TTFT median (ms) | warm requests/s | warm out tok/s | warm ok/failed | server returncode |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 0 | 0 | lmcache_cpu | true | 73.7229 | 1.8645 | 29.8322 | 8/0 | 0 |
| 0 | 1 | recompute | true | 67.1691 | 1.9151 | 30.6422 | 8/0 | 0 |
| 0 | 2 | lmcache_disk | true | 97.3907 | 1.7858 | 28.5723 | 8/0 | 0 |
| 1 | 0 | recompute | true | 67.3472 | 1.9098 | 30.5570 | 8/0 | 0 |
| 1 | 1 | lmcache_disk | true | 95.4246 | 1.8507 | 29.6113 | 8/0 | 0 |
| 1 | 2 | lmcache_cpu | true | 72.3479 | 1.9106 | 30.5697 | 8/0 | 0 |
| 2 | 0 | lmcache_disk | true | 97.5615 | 1.7819 | 28.5110 | 8/0 | 0 |
| 2 | 1 | lmcache_cpu | true | 71.6462 | 1.9260 | 30.8158 | 8/0 | 0 |
| 2 | 2 | recompute | true | 66.3993 | 1.9509 | 31.2142 | 8/0 | 0 |
| 3 | 0 | recompute | true | 66.4902 | 1.9548 | 31.2761 | 8/0 | 0 |
| 3 | 1 | lmcache_disk | true | 95.2890 | 1.8304 | 29.2866 | 8/0 | 0 |
| 3 | 2 | lmcache_cpu | true | 73.0826 | 1.8714 | 29.9420 | 8/0 | 0 |
| 4 | 0 | lmcache_cpu | true | 72.6468 | 1.8823 | 30.1168 | 8/0 | 0 |
| 4 | 1 | recompute | true | 69.0891 | 1.8671 | 29.8733 | 8/0 | 0 |
| 4 | 2 | lmcache_disk | true | 96.3280 | 1.7840 | 28.5441 | 8/0 | 0 |

## Per-arm values across attempted cells

| Arm | cells | warm TTFT medians (ms) | warm requests/s | warm out tok/s |
|---|---:|---|---|---|
| recompute | 5 | 67.1691, 67.3472, 66.3993, 66.4902, 69.0891 | 1.9151, 1.9098, 1.9509, 1.9548, 1.8671 | 30.6422, 30.5570, 31.2142, 31.2761, 29.8733 |
| lmcache_cpu | 5 | 73.7229, 72.3479, 71.6462, 73.0826, 72.6468 | 1.8645, 1.9106, 1.9260, 1.8714, 1.8823 | 29.8322, 30.5697, 30.8158, 29.9420, 30.1168 |
| lmcache_disk | 5 | 97.3907, 95.4246, 97.5615, 95.2890, 96.3280 | 1.7858, 1.8507, 1.7819, 1.8304, 1.7840 | 28.5723, 29.6113, 28.5110, 29.2866, 28.5441 |
