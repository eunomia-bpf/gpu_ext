# LMCache I/O concurrency: BPF improves, but the optimum is not shared

All 30 new fresh-process cells completed: five rotated blocks of six arms,
1,920 reads and 2,880 writes, every child exit zero and no recorded cleanup
errors. Source `a78c9d0f` adds only a loop-default-executor worker option;
the BPF decision, native algorithm and 200 ms write budget are unchanged.
All BPF arms enable allocation reuse from `b0eeb67c`.

## Performance

Medians across five cells per arm. Read latency includes scheduled-arrival
queueing through completion. Write throughput includes deferred completion.

| Metric | FIFO default | Native default | BPF default | FIFO 4 | Native 4 | BPF 4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Read p50, ms | 167.316 | 74.089 | 91.095 | 114.818 | 86.463 | 74.745 |
| Read p99, ms | 255.884 | 84.945 | 103.328 | 127.825 | 97.095 | 83.961 |
| Write throughput, MiB/s | 5133.003 | 5630.851 | 5613.128 | 5375.929 | 5674.237 | 5753.616 |
| Total bandwidth, MiB/s | 8555.006 | 9384.751 | 9355.214 | 8959.882 | 9457.062 | 9589.360 |

Paired changes use 100*(numerator/denominator-1) within each block, not
ratios of the table medians. Negative p99 changes and positive throughput
changes are improvements. Ranges below are observations, not confidence
intervals.

| Comparison | Read-p99 changes, blocks 0--4 (%) | Paired median | Write-throughput paired median |
| --- | --- | ---: | ---: |
| BPF 4 / BPF default | -29.541, -45.661, +1.610, +0.900, -50.101 | -29.541% | +6.359% |
| Native 4 / Native default | -11.274, +14.304, +46.225, +83.688, -50.484 | +14.304% | -0.009% |
| FIFO 4 / FIFO default | -55.676, -42.429, -56.895, -49.044, -45.654 | -49.044% | +6.953% |
| BPF default / Native default | +17.619, +21.011, +87.657, +56.692, -5.233 | +21.011% | -3.192% |
| BPF 4 / Native 4 | -6.597, -42.473, +30.401, -13.929, -4.500 | -6.597% | +1.399% |
| Native 4 / FIFO 4 | -18.075, -26.155, -16.913, -25.186, -57.764 | -25.186% | +4.964% |
| BPF 4 / FIFO 4 | -23.479, -57.519, +8.346, -35.607, -59.664 | -35.607% | +7.227% |

BPF 4/default improves both read p99 and write throughput in three of five
blocks. The two adverse read pairs are +1.610% and +0.900%; corresponding
write-throughput changes are -0.936% and -0.305%. Its complete throughput
range is -0.936% to +11.217%. FIFO read p99 improves in all five pairs,
but FIFO write throughput has one -0.033% pair. Native read p99 regresses
in three pairs. Therefore four workers is useful for this measured BPF
configuration, not a universal executor improvement or a general optimum.

## Attribution and the stronger native control

The installed LMCache 0.5.4 save coroutine dispatches `_save_gds` through
`asyncio.to_thread`. The option installs a standard four-worker default
executor on its private loop; other `to_thread` work on that loop shares
the same limit. Blocking reads retain their existing 64-thread arrival path.
The option is not a GPU driver policy, an exclusive hardware write queue,
or a new BPF algorithm. The observed effect is a shared executor-concurrency
change, not evidence that BPF instructions accelerate disk transfers.

Do not present only BPF 4 versus Native 4 as a win: native's default setting
has the stronger descriptive read-p99 median, 84.945 ms, versus 97.095 ms
at four workers. BPF 4 is 83.961 ms. Comparing those separately selected
configurations is descriptive and does not isolate BPF mechanism overhead
or establish equivalence. Even the same-four-worker BPF/native paired p99
range is -42.473% to +30.401%, with one adverse pair; it does not establish
stable superiority. FIFO 4 is also a stronger transport control than FIFO
default, and remains visible rather than attributing its generic gains to BPF.

The old default-worker BPF/native disadvantage is present in this new
campaign too (+21.011% paired p99 median, four adverse pairs). It is not
deleted or pooled away. All prior fifteen-cell campaigns, the complete
25-cell write-budget result and the isolated ioctl-reuse result remain
unchanged. This new comparison does not identify the internal SSD, GIL or
driver-lock cause of the observed variation. Keep the worker option opt-in;
do not change the default from this mixed result or search further worker
counts against these same observations.

## Reproduction and retention

[The plan](gds-control/write-workers-plan-20260907.md) fixes the comparison.
[Raw output](raw/gds-write-workers-575-20260907-five-block/) contains the
exact 30 commands/order in `run-plan.json`, each complete `result.json`,
per-request and feedback records, full per-cell runner logs, exit records,
the shared cuFile log, cache-removal records and `paired-analysis.json`.
Root invoked the existing single-cell runner directly while the local model
continued preparing a reusable launcher. No completed cell is rerun when
that launcher becomes available.

Traffic remains 24 MiB objects, 64 reads/96 writes, 2/4 ms scheduled arrivals,
4096 MiB GPU staging pool, and live-event-driven admission with a 200 ms
cumulative budget. GIL retention and decision-stage instrumentation are off.
The original default worker count follows Python 3.12's
`min(32, (os.cpu_count() or 1)+4)`; four is the only alternative measured.
Five rotations do not cover all six order positions per arm. With 64 reads,
the existing nearest-rank p99 is the maximum, not a long serving trace's p99.

This is real cuFile compatibility-mode direct I/O on RTX 5090, Linux 6.15.11,
NVIDIA 575.57.08: storage-request performance, not vLLM TTFT or demonstrated
hardware NVMe-to-GPU P2P. The driver and attached policy were unchanged.
Successful cell-owned generated caches totaling 120,835,276,800 apparent
bytes were removed only after the respective process exited. They can be
regenerated with the recorded workload commands; all measurement and log
files remain. Cleanup is outside timing and identical for every arm, but
can affect later SSD state. No measured cell was discarded or retried.
