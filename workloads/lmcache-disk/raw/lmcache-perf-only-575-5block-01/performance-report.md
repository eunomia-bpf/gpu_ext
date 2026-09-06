# LMCache 575 five-block storage performance report

Campaign `lmcache-perf-only-575-5block-01` (timestamp `20260905T173648`), run by
`run_perf_only.py` and recorded in `campaign.json`. This is the promised local-disk
comparison: recompute vs `lmcache_cpu` vs `lmcache_disk`, five complete rotated
blocks, performance only.

## Workload

- Model: `Qwen/Qwen3-30B-A3B-FP8` (revision `d206ba732169f29bb77fbf80fc2c4b81d4d30782`).
- Expected driver parameter: `575.57.08`; port `18080`.
- Schedule: seed `2709`, attempts 0-4; each of the 3 arms runs exactly once per block,
  5 blocks, 15 cells, one attempt per cell, no retry, no result filtering, no
  inter-cell GPU-idle wait, and no correctness, engagement, admission, or retry gates.
- Prompts (`prompts.json`): 8 fixed single-sequence prefixes of 1536 cached tokens
  each with 16 output tokens. Each cell sends the 8 cold requests, waits on the
  non-gating 120 s store barrier, then sends the 8 warm requests; only the warm
  phase (8 sequential requests) is scored.
- Outcome: all 15 cells ready, 8/8 warm requests succeeded in every cell
  (`complete_cells: 15`, 0 failures, server return code 0 in every cell).

## All 15 cells

| Block | Position | Arm | warm TTFT median (ms) | warm requests/s | warm out tok/s |
|---:|---:|---|---:|---:|---:|
| 0 | 0 | lmcache_cpu | 73.509018 | 1.880802806 | 30.092844897 |
| 0 | 1 | recompute | 66.158734 | 1.935957991 | 30.975327861 |
| 0 | 2 | lmcache_disk | 95.251708 | 1.827880262 | 29.246084187 |
| 1 | 0 | recompute | 65.949641 | 1.969657175 | 31.514514795 |
| 1 | 1 | lmcache_disk | 95.662044 | 1.807376742 | 28.918027873 |
| 1 | 2 | lmcache_cpu | 72.382291 | 1.917807999 | 30.684927994 |
| 2 | 0 | lmcache_disk | 94.559179 | 1.858291304 | 29.732660863 |
| 2 | 1 | lmcache_cpu | 73.532471 | 1.895820472 | 30.333127552 |
| 2 | 2 | recompute | 67.143417 | 1.907727674 | 30.523642778 |
| 3 | 0 | recompute | 68.017603 | 1.916228968 | 30.659663487 |
| 3 | 1 | lmcache_disk | 95.146604 | 1.829255512 | 29.268088186 |
| 3 | 2 | lmcache_cpu | 73.402549 | 1.886932590 | 30.190921441 |
| 4 | 0 | lmcache_cpu | 73.940074 | 1.880163231 | 30.082611688 |
| 4 | 1 | recompute | 67.889584 | 1.906773075 | 30.508369200 |
| 4 | 2 | lmcache_disk | 92.359540 | 1.855065164 | 29.681042616 |

## Recomputed medians across the 15 cells

Medians recomputed here from the per-cell values in `campaign.json` (the stored
`arm_summary.warm_output_tokens_per_s` series was null due to a presentation-only
key mismatch in `run_perf_only.py`, since fixed).

| Arm | median requests/s | median out tok/s | median of cell TTFT medians (ms) |
|---|---:|---:|---:|
| recompute | 1.916228968 | 30.659663487 | 67.143417 |
| lmcache_cpu | 1.886932590 | 30.190921441 | 73.509018 |
| lmcache_disk | 1.829255512 | 29.268088186 | 95.146604 |

Paired (per-block) request-rate difference versus recompute, median:
`lmcache_cpu` -1.5289%, `lmcache_disk` -4.5388%.

## Conclusion

This is the promised local-disk comparison: on this short single-sequence workload
(eight 1536-token prefixes, 16 output tokens each) neither local storage tier beats
recompute. `lmcache_disk` is the slowest arm on every metric (lowest request and
output-token rates, highest TTFT), and `lmcache_cpu` trails recompute as well; the
cached-prefix benefit does not offset the retrieval cost here.
