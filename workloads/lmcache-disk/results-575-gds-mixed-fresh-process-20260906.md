# Mixed LMCache cuFile traffic: five fresh-process repetitions

All 15 measurements completed, totaling 960 demand reads and 1,440 background
writes; no request or cleanup errors were recorded. Each measurement starts a
fresh Python/CUDA process and a 4,096 MiB pool, avoiding the retained-pool OOM
in the preceding multi-measurement process. Each uses 64 reads and 96 writes
of 24 MiB, with identical 2 ms read and 4 ms write spacing. Startup, initial
read-object population, and write-buffer allocation are outside request timing.

## Performance

Medians across five measurements per configuration:

| Metric | FIFO | Native policy | BPF policy |
| --- | ---: | ---: | ---: |
| Read call-to-completion p50, ms | 34.935 | 110.774 | 89.304 |
| Read call-to-completion p99, ms | 225.177 | 208.939 | 218.392 |
| Completed write throughput, MiB/s | 5634.827 | 5553.767 | 5694.257 |
| Total storage throughput, MiB/s | 8697.202 | 8510.863 | 8686.379 |

The per-measurement p99 uses nearest rank; with 64 reads it is the maximum.
The existing raw field named `offer_s` records actual dispatch, so these are
call-to-completion durations, not scheduled-arrival queueing latency.

Same-repetition percentage changes use 100 * (numerator/denominator - 1):

| Read p99 comparison | Repetitions 0, 1, 2, 3, 4 (%) | Median (%) |
| --- | --- | ---: |
| Native / FIFO | -23.110, -0.150, +18.895, +4.903, -26.530 | -0.150 |
| BPF / FIFO | -19.889, -3.014, +10.426, +1.241, +30.458 | +1.241 |
| BPF / native | +4.190, -2.868, -7.124, -3.490, +77.566 | -2.868 |

BPF/native paired write-throughput changes have median +2.868%, range
[-12.916%, +7.232%]. The large final BPF read-tail observation is retained.
The lower ratio of arm medians does not imply consistent paired improvement:
neither native nor BPF improves read p99 in all repetitions, and the p50
medians worsen. These measurements do not establish a stable policy benefit
or a tight bound on BPF overhead.

## What executed

Every FIFO measurement submitted all 160 operations immediately. Every native
and BPF measurement submitted 64 demand reads and deferred 96 background
writes. Both policy implementations use the same executor and explicit
pressure 801 permille / slack 10 ms inputs. This demonstrates actual
write-deferral behavior while separating policy selection from its BPF
implementation; the pressure is controlled input, not live HBM telemetry.
No recomputation or write coalescing is claimed.

The backend uses real cuFile compatibility-mode disk I/O with direct I/O
enabled. It does not establish hardware NVMe-to-GPU P2P. This backend traffic
experiment measures storage calls, not model output token/s, and is separate
from the completed end-to-end five-arm and policy-input ablation experiments.

## Records and continuation

[Collection record](raw/gds-mixed-fresh-process-575-20260906-five-block/README.md)
describes the source revision, exact invocation and cyclic order. The
[summary](raw/gds-mixed-fresh-process-575-20260906-five-block/summary.json)
retains every per-repetition metric and paired difference; each explicitly
named measurement directory contains its raw.jsonl, campaign.json and
summary.json. No run was retried or omitted. Earlier successful pilots and
the OOM-interrupted run remain separate and unchanged.

The local-model follow-up is to record scheduled offer separately from actual
dispatch and incorporate per-measurement process isolation in the runner.
The next performance comparison should address the observed variability and
whether a single 10 ms delay is useful during a longer mixed-traffic interval,
without treating these five repetitions as a confirmed policy win.
