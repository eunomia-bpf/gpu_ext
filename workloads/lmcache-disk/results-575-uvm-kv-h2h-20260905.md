# LMCache/UVM-KV three-arm head-to-head — tracked result (2026-09-05)

Source: `raw/lmcache-uvm-kv-575-h2h-01/campaign.json` and
`raw/lmcache-uvm-kv-575-h2h-01/summary.md` only (kind `lmcache_uvm_kv_perf`,
timestamp `20260905T221127`, driver parameter `575.57.08`, Qwen3-30B-A3B-FP8).

Design: 5 rotated blocks, 15 cells total, 8 warm requests per cell
(8 prefixes x 1,536 cached tokens, 16 output tokens). All 15 cells were
ready with server return code 0 and 8/8 warm requests OK. No correctness,
engagement, admission, retry, or filtering gates ran.

## Arm medians (5 cells per arm)

| arm | TTFT median (ms) | out tok/s |
|---|---:|---:|
| recompute | 66.0786 | 31.2463 |
| lmcache_disk (native) | 94.8120 | 29.0623 |
| lmcache_disk_uvm_kv | 95.2314 | 29.0509 |

## Paired-median changes (median of per-block paired ratios)

| comparison | TTFT | out tok/s |
|---|---:|---:|
| lmcache_disk vs recompute | +41.726% | -6.958% |
| lmcache_disk_uvm_kv vs lmcache_disk | -1.384% | -0.760% |

## Interpretation

The UVM-KV arm matches the native disk arm within run variability: its
paired TTFT delta is -1.384% and throughput delta is -0.760%, both inside
the spread of the per-block paired ratios (TTFT ratios ranging roughly
0.935-1.070, throughput ratios roughly 0.988-1.017). This is a parity
result, not an improvement.

No gpubpf policy is attached in this three-arm result. The next, fourth
arm is UVM-KV+gpubpf semantic eviction-pressure/migration-debt policy;
the exact per-KV region identity (which KV regions the policy governs)
remains open.
