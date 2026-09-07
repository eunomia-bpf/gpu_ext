# Existing five-arm serving data: where the elapsed-time difference occurs

Analysis date: 2026-09-07. No new GPU cells or changes to old results.
Source: `../raw/gds-five-arm-575-20260906-five-block-formal/raw.jsonl`,
25 cells, five rotated blocks, eight sequential warm requests per cell.
See `../results-575-lmcache-gds-five-arm-20260906.md` for the original
throughput, median-request TTFT, setup, and all-submit interpretation.

## Decomposition

For each warm request, split recorded `e2e_ms` into `ttft_ms` and
`e2e_ms - ttft_ms`. For each cell, subtract the sum of request E2E times
from `warm_phase.elapsed_s * 1000` to obtain the interval outside request
timers. `lmcache_primitives.py:streamed_completion` starts timing before
opening HTTP and ends after consuming the response stream. Therefore the
post-first-token interval includes later generation, streaming, client
processing, and response completion; it is NOT pure GPU decode time.
`run_perf_only.py` times the complete sequential warm loop.

The following entries are medians across five cells of each cell's
eight-request arithmetic mean, except the last column (total per cell).
These TTFT means are intentionally different from the original report's
medians of per-cell median-request TTFT. Independently selected medians
need not add exactly.

| Arm | first token, ms/request | after first token, ms/request | E2E, ms/request | outside request timers, ms/cell |
|---|---:|---:|---:|---:|
| recompute | 66.622 | 449.215 | 515.837 | 1.043 |
| LMCache CPU | 75.477 | 461.961 | 537.438 | 1.033 |
| GDS FIFO | 78.380 | 349.187 | 427.567 | 1.036 |
| GDS native | 78.055 | 348.346 | 426.876 | 1.034 |
| GDS BPF | 76.804 | 340.017 | 416.821 | 1.022 |

Same-block differences versus recompute, computed on the sum of all eight
warm requests before taking the median across five blocks:

| Arm | first-token sum difference, ms | post-first-token sum difference, ms | post-first-token relative difference | observed range |
|---|---:|---:|---:|---|
| GDS FIFO | +94.065 | -800.221 | -22.267% | [-25.189%, -18.522%] |
| GDS native | +86.376 | -780.141 | -21.434% | [-24.565%, -20.350%] |
| GDS BPF | +94.873 | -782.830 | -22.255% | [-25.529%, -20.440%] |

The GDS advantage in the whole warm interval occurs after the first token,
not in first-token latency or the roughly 1 ms/cell outside-request interval.
All three GDS arms exhibit it in every block. This narrows the location of
the effect but does not identify its cause: there are no per-token device
timings, memory-migration attribution, or matched response-token intervention
in this analysis. Do not call it a disk-read speedup, a measured GPU-decode
speedup, or a BPF policy benefit. The original mixed BPF/native comparison
and all-submit limitation remain unchanged.

## Recompute the per-cell components

Run from the repository root; this reads the existing records only:

```sh
jq -s 'map([.requests[] | select(.phase == "warm")] as $w |
  {block, config, n: ($w | length),
   first_ms: ([$w[].ttft_ms] | add),
   after_ms: ([$w[] | .e2e_ms - .ttft_ms] | add),
   e2e_ms: ([$w[].e2e_ms] | add),
   outside_ms: (.warm_phase.elapsed_s * 1000 - ([$w[].e2e_ms] | add))})' \
  workloads/lmcache-disk/raw/gds-five-arm-575-20260906-five-block-formal/raw.jsonl
```

Divide each cell's sums by `n` and take each arm's median for the first table.
For the second, join each GDS cell to recompute by `block`, subtract sums,
and take medians. Relative differences use `(GDS / recompute - 1) * 100`
on the post-first-token sums; the range is the five observed pairs, not a
confidence interval.

## Consequence for the unfinished experiment

Continue the real asynchronous-prefetch serving comparison. Its intended
effect is to move disk retrieval before consumption and improve request
waiting; report TTFT and whole-response throughput separately. An overall
throughput increase alone cannot establish that prefetch shortened waiting.
Do not rerun the completed all-submit campaign to reproduce these sums.
