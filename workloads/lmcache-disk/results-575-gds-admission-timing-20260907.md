# LMCache per-admission timing: completed diagnostic

All 15 fresh-process cells completed with child exit 0: 960 reads and 1,440
writes. This diagnostic uses source `b18c342e`, the saved GDS-enabled
575.57.08 module restored after the separate stale-state campaign, and the
same attached GDS policy. It adds per-admission timers to the event-driven
10 ms executor; its end-to-end numbers do not replace uninstrumented results.

## What the timing establishes

Values below are medians of the five per-cell medians, in microseconds.

| Demand-read stage | FIFO | Native | BPF |
| --- | ---: | ---: | ---: |
| Request provider | 8.176 | 7.908 | 8.166 |
| Decision lock wait | 0.232 | 0.234 | 0.288 |
| Decider call | 1.532 | 1.827 | 17.962 |
| Total admission | 11.968 | 11.973 | 30.140 |

BPF adds a measurable Python-to-driver decision cost. Its maximum observed
read lock wait is 5,884.749 us and maximum total read admission is
5,914.727 us; native maxima are 1.004 and 78.735 us. BPF's maximum decider
call is 3,008.087 us for reads and 3,140.357 us for writes. Lock wait includes
thread scheduling, and the decider interval includes its internal lock,
packing, ioctl and output construction, not just BPF instructions.

The BPF median admission is tens of microseconds, whereas scheduled-arrival
read p99 is hundreds of milliseconds. These observations do not show
admission as the sole cause of the end-to-end variability. They also do not
prove that eliminating admission cannot affect downstream queueing. Stage
percentiles and maxima must not be added as if they describe the same request.

## Full performance record

FIFO/native/BPF read-p99 medians are 490.331/708.473/336.172 ms, and write
throughput medians are 2003.653/1201.882/4662.309 MiB/s. These instrumented
numbers are descriptive, not evidence of stable superiority. Within-block
BPF/native p99 changes are +29.021%, +12.231%, -78.178%, -49.199%, -6.097%;
BPF/FIFO changes are +24.600%, +9.854%, -79.854%, -0.744%, -20.490%.
Both improve in three of five pairs, with adverse pairs retained.

| Block | Arm | Read p99, ms | Read decider median, us | Read admission median, us |
| --- | --- | ---: | ---: | ---: |
| 0 | gds_fifo | 269.800 | 1.442 | 11.111 |
| 0 | gds_native | 260.556 | 1.827 | 11.973 |
| 0 | gds_bpf | 336.172 | 17.390 | 29.686 |
| 1 | gds_native | 231.507 | 1.743 | 11.305 |
| 1 | gds_bpf | 259.823 | 16.929 | 28.782 |
| 1 | gds_fifo | 236.517 | 1.532 | 12.270 |
| 2 | gds_bpf | 251.889 | 19.023 | 32.218 |
| 2 | gds_fifo | 1250.304 | 1.595 | 13.963 |
| 2 | gds_native | 1154.314 | 1.706 | 11.287 |
| 3 | gds_fifo | 490.331 | 1.377 | 11.702 |
| 3 | gds_native | 958.020 | 1.885 | 12.650 |
| 3 | gds_bpf | 486.684 | 17.962 | 30.922 |
| 4 | gds_native | 708.473 | 1.916 | 12.180 |
| 4 | gds_bpf | 665.280 | 18.032 | 30.140 |
| 4 | gds_fifo | 836.721 | 1.645 | 11.968 |

## Next improvement

The completed historical event-driven run shows most writes exhausting the
10 ms cumulative delay budget while demand reads remain active. A bounded,
longer write budget is therefore the next policy-parameter experiment:
compare 10 ms and 200 ms on both native and BPF, retaining FIFO and measuring
read latency together with write throughput. This is a new policy parameter,
not an unchanged-algorithm implementation speedup; a read gain purchased by
lower write throughput must be explicit. Each actual admission must still
invoke its native/BPF decider. No BPF decision is replaced by a Python rule.
A local Qwen 27B OpenCode session is implementing only the runner parameter.

## Reproduction and artifacts

Run the command in [the experiment plan](gds-control/admission-timing-experiment-20260907.md).
The [raw directory](raw/gds-admission-timing-575-20260907-five-block/) retains
all 15 result files, 2,400 request records, admission records, campaign,
summary, full runner and cuFile logs, and `admission-analysis.json`.
Analysis groups admission records by cell and kind, computes median,
nearest-rank p99, maximum and sum per stage, then takes the median of the
five cell medians. Paired changes are 100*(BPF/control-1) within each block.
No cell is discarded or repeated. With 64 reads, nearest-rank p99 is the
maximum read latency.

Traffic is unchanged: 24 MiB objects, 64 reads and 96 writes, 2/4 ms arrivals,
4096 MiB GPU staging pool, real cuFile compatibility-mode direct I/O.
This is a storage-request study, not vLLM TTFT or demonstrated hardware P2P.

