# LMCache total-recovery-cost policy comparison

Status: running, first block partially complete. This is not the completed
five-block result. Implementation `4381f660`; raw outputs and the exact
invocation are under `raw/gds-kv-total-cost-575-20260908-01/`.
See the [unchanged experiment plan](gds-control/kv-reclaim-total-cost-experiment-20260908.md).

The old native/BPF cost-per-byte comparison remains in the
[completed grace report](results-575-gds-kv-reclaim-grace-20260908.md).
These are fresh controls for a changed algorithm, not replacements for
those historical measurements. No policy advantage or novelty is presumed.

## Completed cells

| Block | Policy | Output token/s | Median TTFT, ms | Restore batches | Logged restore MiB | Restore ops | Rollback warnings |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | stock | 70.9041546753 | 25745.2044670 | 15 | 1632 | 68 | 7 |
| 0 | native cost/byte | 62.4583563017 | 32331.0981690 | 43 | 6000 | 250 | 35 |
| 0 | native total cost | 68.9647015543 | 26453.8161885 | 15 | 1632 | 68 | 7 |

All three completed cells finish eight warm requests, with 8192 completed
output tokens each and zero warm failures. The BPF total-cost cell and
remaining blocks are still running; no paired BPF mechanism effect is
available yet.

In this first block, native total cost improves throughput by 10.4171%
against the original ratio rule but remains below stock. Its restoration
counts return to the stock counts. This is evidence consistent with the
denominator contributing to repeated restoration on this arrival order,
not proof that total cost wins on all orders or that every victim decision
matches stock. The final comparison retains all five blocks.

## Measurement boundary

The shared runtime includes the prior deferred-free grace repair. All
four arms use real LMCache/cuFile compatibility-mode storage and FIFO
per-I/O admission. Only victim ranking changes between native variants;
the native total and BPF total versions compile the same shared algorithm.

Throughput is completed warm output tokens divided by full warm-phase
elapsed time. TTFT is HTTP-send to first-token, not scheduled-arrival
latency; four client workers mean later submissions can wait for a worker.
The completion-progress writes in runner `a1011e64` are common to all
new arms and occur within warm elapsed time. Server startup, cold
population, cold-store barriers, shutdown and exit-diagnostic collection
are outside that interval.

Restore counts/MiB/ops come from the existing `batched_get_blocking` log
lines, and warnings from `rolled back from`. Logged backend payload is
not a physical SSD-byte counter. Fixed-pool block recycling does not prove
physical HBM release to other processes or transparent disk-UVM paging.
The separate disk-UVM driver implementation has no runtime result yet.
