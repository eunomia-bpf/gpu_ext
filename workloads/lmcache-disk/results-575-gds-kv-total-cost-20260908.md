# LMCache total-recovery-cost policy comparison

Status: running, first four-arm block complete. This is not the completed
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
| 0 | BPF total cost | 69.6323735398 | 26499.3640565 | 15 | 1632 | 68 | 7 |

All four completed cells finish eight warm requests, with 8192 completed
output tokens each and zero warm failures. The remaining blocks are still
running. In this one block BPF total/native total throughput differs by
+0.9681%; BPF total/old native differs by +11.4861%, and BPF total/stock by
-1.7937%. One block does not establish a stable mechanism or policy gain.

In this first block, native total cost improves throughput by 10.4171%
against the original ratio rule but remains below stock. Its restoration
counts return to the stock counts. This is evidence consistent with the
denominator contributing to repeated restoration on this arrival order,
not proof that total cost wins on all orders or that every victim decision
matches stock. The final comparison retains all five blocks.

## Disk-full interruption and continuation

The first block finished before an actual ENOSPC failure during block 1's
native-ratio cold population. Its retained `server.log` records six cold
store messages followed by `No space left on device` and EngineCore failure.
The runner also failed to serialize `result.json` with `OSError: [Errno 28]`;
there is no complete numeric warm-phase result for this attempt. The raw
log and configuration remain in `block-01/position-0-native_ratio/`, not
overwritten or counted as a successful performance cell. The original
campaign process exited 1 and restored the old BPF loader, whose log says
`attached`.

Root moved only regenerable cache payload directories from the completed
15-cell grace campaign and these four completed block-0 cells to
`/var/tmp/lmcache-retired-cache-20260908.9ypd5z/`, retaining their relative
`raw/.../cache` paths. Nothing was deleted: all cache data is recoverable,
and every original result/log remains in place. Workspace available space
rose to about 25 GiB; archive filesystem available space is about 75 GiB.
The failed partial cache was left untouched.

The exact continuation is `python3 -u /tmp/lmcache-total-cost-resume-20260908.py`,
preserved as `resume-run.py` beside the original invocation. It reattaches
the same total-cost object, skips all four completed block-0 cells, and runs
blocks 1–4. Only the failed cold-population attempt is replaced by a new
directory, `block-01/position-0-native_ratio-after-enospc/`. Model, driver,
policy binaries, native library choices, arrival orders and measurement
code are unchanged. The cache remains on the same workspace NVMe device;
free space and the interruption differ across blocks and must be disclosed
when interpreting the final paired results.

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
