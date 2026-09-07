# LMCache policy-ioctl GIL handoff: five completed paired blocks

All 20 fresh-process measurements completed with exit status zero: 1,280
demand reads and 1,920 background writes, with no recorded request or cleanup
errors. This is a four-arm mechanism ablation on the existing live-feedback
polling executor, not the separately pending event-driven implementation.
Source: main `2d27777d`; invocation correction documented in `8cae0ff3`.

## Performance

Medians of five cells per arm. Latency is scheduled arrival to read completion;
throughput is completed storage traffic, not vLLM output tokens.

| Metric | FIFO | Native | BPF, release GIL | BPF, keep GIL |
| --- | ---: | ---: | ---: | ---: |
| Read p50, ms | 233.568 | 173.898 | 138.650 | 220.626 |
| Read p99, ms | 399.491 | 444.176 | 310.374 | 406.131 |
| Write throughput, MiB/s | 2548.850 | 1839.248 | 1369.770 | 3485.265 |
| Total storage throughput, MiB/s | 4248.083 | 3065.414 | 2282.950 | 5808.775 |

The mechanism effect is computed **within each block**, as
`100 * (numerator / denominator - 1)`, not by dividing these arm medians.

| Paired read-p99 comparison | Blocks 0, 1, 2, 3, 4 (%) | Median (%) |
| --- | --- | ---: |
| BPF keep / BPF release | +4.923, -28.106, +513.574, -50.739, -42.630 | -28.106 |
| BPF keep / native | +12.663, +7.477, +33.448, -52.179, +1.103 | +7.477 |
| BPF release / native | +7.377, +49.493, -78.251, -2.924, +76.231 | +7.377 |
| Native / FIFO | -0.742, -9.883, -16.663, +112.590, -17.907 | -9.883 |
| BPF keep / FIFO | +11.826, -3.145, +11.212, +1.662, -17.001 | +1.662 |

Keeping the GIL improves p99 against ordinary BPF in three pairs but worsens
it in two, including a 513.574% increase. Its p99 is worse than native in
four pairs. This does **not** establish a reliable optimization or tight
mechanism-cost bound. The constructor option remains default-off.

BPF-keep/BPF-release paired write-throughput changes have median -0.094%,
range -16.460% to +169.537%; BPF-keep/native median is +0.358%, range
-16.893% to +108.781%. Thus the much larger ratio of descriptive throughput
medians is not an appropriate paired improvement claim. All ranges above
describe these five pairs, not confidence intervals. With 64 reads, the
nearest-rank p99 is the maximum read latency in that cell.

## What changed, and what the data cannot isolate

`LMCACHE_GDS_IOCTL_KEEP_GIL=1` selects `ctypes.PyDLL` instead of `ctypes.CDLL`
for the same libc ioctl. Every BPF request still enters the same command-82
driver/BPF path. The request ABI, native/BPF decision algorithm, live pending
read counter, write delay budget, polling executor, actual I/O and recording
are unchanged. This tests Python GIL handoff behavior, not a new policy or
an optimization that skips BPF execution.

The sequential cell record shows substantial variation beyond this option:
the first eight cells complete writes at 5027--5240 MiB/s, while the four
cells of block 2 run at
1074--1292 MiB/s. Subsequent cells remain variable. This pattern makes a
stable GIL-only explanation doubtful; storage state, system contention and
arrival-sensitive overlap are possible explanations, not measured causes.
A post-run NVMe snapshot reports 318 K and zero critical warnings, media
errors and thermal-transition counters; it is not a during-run trace and
does not identify the cause of the throughput change.

Read `submitted_s` equals dispatch `offer_s` before entering the adapter, so
dispatch-to-completion includes decision locks, Python scheduling and I/O.
There is no separate measured read-admission or pure disk-transfer interval.
Write `submitted_s` marks save-coroutine entry, not proof of physical device
dispatch. The transport is real cuFile compatibility-mode direct I/O, not
demonstrated hardware NVMe-to-GPU P2P. The next executor experiment remains
event-driven wakeup; this result does not pre-judge it.

## Artifacts and execution

The [raw directory](raw/gds-mixed-gil-handoff-575-20260907-five-block-absolute/)
contains all 20 `result.json` files and runner logs, plus `cell-summary.json`
and `paired-analysis.json`. The analysis uses recorded per-cell metrics;
original per-request timestamps and feedback decisions are retained.
Each cell uses 64 reads, 96 writes, 24 MiB objects, 2/4 ms scheduled spacing,
a 4096 MiB GPU pool and a fresh process. Run interval: 2026-09-06
22:25:30--22:26:52 PDT for cell starts, RTX 5090 / driver 575.57.08.

The arm order is `[fifo, native, bpf_release, bpf_keep]` rotated left by
`block % 4` for blocks 0--4. Both BPF arms pass `--config gds_bpf`; their
directory names and environment setting distinguish them. The ordinary BPF,
native and FIFO cells set `LMCACHE_GDS_IOCTL_KEEP_GIL=0`; BPF-keep sets it to
one. The exact per-cell invocation is:

```sh
env LMCACHE_GDS_IOCTL_KEEP_GIL=0 workloads/lmcache-disk/current-venv/bin/python \
  workloads/lmcache-disk/run_gds_mixed_backend.py \
  --single-cell --policy-variant live-feedback --reads 64 --writes 96 \
  --gds-buffer-size-mib 4096 --config gds_fifo --block 0 --position 0 \
  --cell-dir /home/yunwei37/workspace/gpu/gpu_ext/workloads/lmcache-disk/raw/gds-mixed-gil-handoff-575-20260907-five-block-absolute/block-00/position-00-fifo
```

Use the specified arm/config/environment and distinct absolute directory for
each cell; completed directories must not be reused. The preceding
[relative-path attempts](raw/gds-mixed-gil-handoff-575-20260907-five-block/)
all failed backend filesystem detection before I/O and remain separately
retained (20 attempts, zero performance samples). No completed cell was
discarded or retried. Both older live-feedback and fixed-delay reports remain
unchanged, including their unfavorable results.
