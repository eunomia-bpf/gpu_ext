# LMCache mixed storage with scheduled arrivals

The updated runner completed five cyclic rotations and all 15 fresh-process
measurements: 960 demand reads and 1440 background writes, with every child
returning zero and no request or cleanup errors. GPU allocation returned to
15 MiB after collection. Each measurement uses 64 reads, 96 writes, 24 MiB
objects and a 4096 MiB pool. Requested read/write spacing is 2/4 ms.
The source revision is `3c278471` on the LMCache development branch.

The reader threads are created before a common start event is released.
`scheduled_offer_s` derives from that start and the requested spacing;
`offer_s` retains its previous actual-dispatch meaning. This fixes the
previous mismatch between advertised offered-arrival latency and dispatch
latency without discarding any prior data. Process isolation is now part of
the runner, rather than an external invocation workaround.

## Results

Medians across five measurements per configuration:

| Metric | FIFO | Native policy | BPF policy |
| --- | ---: | ---: | ---: |
| Scheduled-arrival read p50, ms | 44.878 | 181.313 | 159.258 |
| Scheduled-arrival read p99, ms | 298.220 | 265.297 | 247.931 |
| Dispatch-to-completion read p99, ms | 298.160 | 262.860 | 247.872 |
| Completed write throughput, MiB/s | 5186.445 | 5076.371 | 4966.127 |
| Total storage throughput, MiB/s | 8644.076 | 8460.619 | 8276.878 |

The per-measurement p99 uses nearest rank and is the maximum of its 64 reads.
Same-repetition changes use 100 * (numerator/denominator - 1):

| Scheduled-arrival p99 comparison | Repetitions 0, 1, 2, 3, 4 (%) | Median (%) |
| --- | --- | ---: |
| Native / FIFO | +0.729, -17.646, -20.928, -11.040, -6.125 | -11.040 |
| BPF / FIFO | -7.963, -12.440, -20.502, -25.171, -4.453 | -12.440 |
| BPF / native | -8.629, +6.322, +0.538, -15.884, +1.781 | +0.538 |

BPF/FIFO completed-write-throughput changes have median -1.673%, range
[-7.762%, +2.094%]. BPF/native changes have median +0.674%, range
[-5.481%, +2.580%]. These are paired medians, not ratios of the arm medians
in the first table, and the ranges are not confidence intervals.

Under this common-start workload, the BPF policy lowers observed read p99
in all five pairs, while native lowers it in four. This is a tail-versus-median
and bandwidth tradeoff: the scheduled-arrival p50 medians worsen substantially.
The BPF/native p99 changes range from -15.884% to +6.322%; this does not provide
a tight mechanism-overhead bound or establish equivalence. The earlier
[spaced dispatch-only](results-575-gds-mixed-fresh-process-20260906.md) and
[burst](results-575-gds-mixed-burst-20260906.md) results remain separate,
including their larger variability and adverse observations.

## Policy, transport and collection

Every FIFO measurement submits all 160 requests immediately. Every native
and BPF measurement submits 64 demand reads and defers 96 background writes
once by 10 ms, using the same executor and controlled pressure 801 permille /
slack 10 ms inputs. This is actual storage-policy behavior, not live HBM
telemetry, repeated pending-demand feedback, recomputation or write coalescing.
The installed LMCache 0.5.4 GdsBackend performs real cuFile compatibility-mode
I/O with direct I/O enabled; hardware NVMe-to-GPU P2P is not established.
These are storage-request latencies, not vLLM TTFT or output token/s.

Command from the LMCache development worktree:

```sh
workloads/lmcache-disk/current-venv/bin/python workloads/lmcache-disk/run_gds_mixed_backend.py --blocks 5 --reads 64 --writes 96 --gds-buffer-size-mib 4096 --output workloads/lmcache-disk/raw/gds-mixed-scheduled-575-20260907-five-block
```

The [campaign](raw/gds-mixed-scheduled-575-20260907-five-block/campaign.json),
raw.jsonl, summary.json and paired-analysis.json retain all 15 measurements
and every request's timestamps and result. No retry or result exclusion was
used. The measurements ran immediately after the implementation compiled,
without waiting for a separate test checklist. The next implementation
question is live pending-demand feedback; the fixed-delay path is measured.
