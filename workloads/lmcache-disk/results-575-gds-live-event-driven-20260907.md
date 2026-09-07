# LMCache event-driven admission: five completed paired blocks

All 15 fresh-process cells completed with exit status zero: 960 demand reads
and 1,440 background writes, with no recorded request or cleanup errors.
Source is main `c2ecedcb` (development `6e850606`). The loaded GDS-enabled
575 driver was not replaced during the run. Cell starts span 2026-09-06
23:38:39--23:39:36 PDT on the RTX 5090.

## Performance

Medians of five cells per arm. Read latency includes scheduled-arrival
queueing through completion; this is storage traffic, not vLLM TTFT.

| Metric | FIFO | Native event-driven | BPF event-driven |
| --- | ---: | ---: | ---: |
| Read p50, ms | 136.842 | 257.354 | 158.944 |
| Read p99, ms | 279.361 | 431.303 | 255.874 |
| Write throughput, MiB/s | 5085.276 | 1529.832 | 5122.665 |
| Total bandwidth, MiB/s | 8475.460 | 2549.721 | 8537.775 |

Paired changes are `100 * (numerator / denominator - 1)` within each block,
not ratios of the descriptive medians above.

| Read-p99 comparison | Blocks 0, 1, 2, 3, 4 (%) | Median (%) |
| --- | --- | ---: |
| Native / FIFO | +13.218, -7.983, +593.710, +134.344, -57.602 | +13.218 |
| BPF / FIFO | +2.876, -8.407, -8.958, -18.295, -5.045 | -8.407 |
| BPF / native | -9.135, -0.461, -86.876, -65.135, +123.960 | -9.135 |

BPF improves p99 in four of five FIFO pairs and four of five native pairs.
However, the +123.960% native-relative regression and the wide observed
range mean this does not establish stable superiority or a tight overhead
bound. BPF/FIFO paired write-throughput change has median +3.972%, range
-4.821% to +40.120%; BPF/native has median +16.196%, range -4.984% to
+503.981%. BPF/FIFO paired p50 change has median +1.814%. These ranges are
observations across five pairs, not confidence intervals. With 64 reads,
nearest-rank p99 is the maximum read latency in a cell.

## Change and interpretation

The new opt-in `live-event-driven` executor waits on an asynchronous future
until pending demand reads reach zero or the remaining cumulative 10 ms write
budget expires. It then invokes the actual native/BPF decider again; Python
does not bypass the BPF decision to submit a write. Native and BPF use the
same request metadata, executor, real disk path and recording. FIFO submits
immediately. The old fixed-delay and <=1 ms polling variants remain unchanged.

Native records 318/322/350/350/350 total decisions; BPF records
306/336/308/350/350. The completed historical polling campaign recorded
932--993 native and 940--967 BPF decisions per cell. Repeated decisions are
substantially reduced, but these separate campaigns are not contemporaneous
controls for a causal executor speedup. Write-budget exhaustion counts are
77/78/95/95/95 for native and 70/86/72/95/95 for BPF, out of 96 writes each.

Storage performance again changes substantially over the run: the first
eight cells write at roughly 4,952--5,565 MiB/s, followed by native block 2
at 848 MiB/s; later cells range from 1,247 to 2,059 MiB/s. This is an observed
sequence, not proof of a particular SSD-cache, contention, or policy cause.
It limits attribution of the large native/BPF differences. Fewer decisions
is established here; a robust performance improvement remains unproven.
All adverse cells and the preceding polling and GIL-ablation results remain.

## Artifacts and command

The [raw directory](raw/gds-mixed-live-event-driven-575-20260907-five-block/)
contains all 15 `result.json` files, `campaign.json`, `raw.jsonl`, `summary.json`,
`paired-analysis.json`, the full runner log, and the shared cuFile log.
Analysis uses each result's recorded metrics grouped by `block` and `config`.
Regenerable KV payloads are retained locally, not committed as performance
records. No measured cell was discarded or retried.

```sh
env LMCACHE_GDS_IOCTL_KEEP_GIL=0 workloads/lmcache-disk/current-venv/bin/python \
  workloads/lmcache-disk/run_gds_mixed_backend.py \
  --policy-variant live-event-driven --blocks 5 --reads 64 --writes 96 \
  --gds-buffer-size-mib 4096 \
  --output workloads/lmcache-disk/raw/gds-mixed-live-event-driven-575-20260907-five-block
```

Each cell uses 24 MiB objects, common-start read/write spacing of 2/4 ms,
and a 4096 MiB GPU staging pool. LMCache 0.5.4 performs real cuFile
compatibility-mode I/O with direct I/O enabled, not demonstrated hardware
NVMe-to-GPU P2P. This completed campaign must not be repeated; any follow-up
must test a distinct question in a new output directory.
