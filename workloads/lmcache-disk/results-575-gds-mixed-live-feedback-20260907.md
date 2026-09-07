# LMCache live-demand feedback: five completed blocks

All 15 fresh-process measurements completed with exit status zero: 960 demand
reads and 1440 background writes, with no recorded request or cleanup errors.
The workload retains 64 reads, 96 writes, 24 MiB objects, a 4096 MiB GPU pool,
and common-start scheduled read/write arrivals spaced by 2/4 ms. Source:
development `b207c878`, main `c666f2ff`; adapter main `f22a085f`.

## Performance

Medians of five measurements per arm; latency is scheduled arrival to read
completion, not serving TTFT or output-token throughput.

| Metric | FIFO | Native feedback | BPF feedback |
| --- | ---: | ---: | ---: |
| Read p50, ms | 549.539 | 246.446 | 394.861 |
| Read p99, ms | 1070.424 | 473.147 | 806.274 |
| Completed-write throughput, MiB/s | 1404.801 | 1513.061 | 1355.812 |
| Total storage throughput, MiB/s | 2341.335 | 2521.769 | 2259.686 |

Paired changes use `100 * (numerator / denominator - 1)` within each block;
the median below is not a ratio of the descriptive medians above.

| Read p99 comparison | Blocks 0, 1, 2, 3, 4 (%) | Median (%) |
| --- | --- | ---: |
| Native / FIFO | -8.555, -47.232, -43.716, -60.491, -48.161 | -47.232 |
| BPF / FIFO | +9.970, -38.821, +17.688, -26.716, -35.926 | -26.716 |
| BPF / native | +20.258, +15.938, +109.097, +85.485, +23.601 | +23.601 |

Native improves read p99 in all five FIFO pairs; BPF improves it in three.
BPF is slower than native in every pair, with p99 increases of 15.938% to
109.097%. This campaign therefore does not support low-overhead matching of
the native feedback executor. BPF/FIFO completed-write throughput changes
have median -3.374% (range -37.832% to +26.358%); BPF/native has median
-15.138% (range -51.780% to +7.053%). These ranges are not confidence intervals.
With 64 reads per cell, nearest-rank p99 is that cell's maximum read latency.

## Actual policy behavior and the next implementation question

All three arms use the same live pending-demand-read provider and write
recording path. FIFO submits immediately. Native and BPF share the executor,
deferring safe background writes while demand reads remain pending, with a
fresh decision before each wait of at most 1 ms and a cumulative 10 ms budget.
The counter tracks actual demand-read calls through completion, not measured
HBM pressure. LMCache still performs the actual disk transfer.

Across five cells, native records 4014 deferrals and BPF 3966, compared with
zero for FIFO. Each native/BPF cell records 95 of its 96 final write submissions
with the delay budget exhausted. Live feedback is implemented, but in this
workload almost all deferred writes still wait until their budget expires.
Repeated re-evaluation adds work without often releasing writes early.
That observation motivates an event-driven wakeup/budget-expiry executor;
it does not isolate how much of the measured latency gap comes from ioctl
calls, Python scheduling, storage variability, or feedback recording.

The preceding [fixed-delay campaign](results-575-gds-mixed-scheduled-20260907.md)
is retained separately. Its FIFO p99 was 298.220 ms, versus 1070.424 ms here;
these campaigns are not interleaved controls and cannot establish a causal
live-feedback-versus-fixed-delay speedup. No cell was discarded or retried.
The transport remains real cuFile compatibility-mode I/O with direct I/O
enabled, not demonstrated hardware NVMe-to-GPU P2P.

## Artifacts and command

The [raw campaign](raw/gds-mixed-live-feedback-575-20260907-five-block/)
retains `campaign.json`, `summary.json`, `raw.jsonl`, all 15 per-cell
`result.json` files, and `paired-analysis.json`. Records include request
timestamps and outcomes, live feedback inputs/actions, and cleanup results.
Regenerable KV cache payloads remain local and are not committed as results.

```sh
workloads/lmcache-disk/current-venv/bin/python \
  workloads/lmcache-disk/run_gds_mixed_backend.py \
  --policy-variant live-feedback --blocks 5 --reads 64 --writes 96 \
  --gds-buffer-size-mib 4096 \
  --output workloads/lmcache-disk/raw/gds-mixed-live-feedback-575-20260907-five-block
```

This campaign is complete; do not repeat its cells. Future changed executors
must use separate output directories and preserve this adverse native/BPF result.
