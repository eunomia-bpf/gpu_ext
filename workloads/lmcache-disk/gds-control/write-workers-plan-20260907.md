# LMCache shared I/O executor concurrency comparison

## Question

RQ2 (Mechanism Cost): What does the gpubpf mechanism cost, in hook and
observability overhead and in executing the same policy through gpubpf?

The completed buffer-reuse measurement reduces isolated BPF call time by
about 0.39 us, but does not measure storage-request improvement. This follow-up
tests a distinct, larger source of potential interference: concurrency of
the shared LMCache I/O executor under mixed read/write traffic. It is not
a repeat of the completed 10/200 ms write-budget parameter study.

The installed LMCache 0.5.4 `_async_save_bytes_to_disk` awaits
`asyncio.to_thread(self._save_gds, ...)`. The runner currently uses the default
loop executor. The opt-in change installs a standard four-worker default
executor before backend construction. This affects other `to_thread` work on
that loop too; it is not an exclusive write-only hardware queue or a new BPF
algorithm. Blocking demand reads keep their original arrival/worker path.

Hypothesis: limiting queued write I/O concurrency reduces read interference
and/or storage variability without losing the existing write-throughput gain.
An equally plausible outcome is lower write throughput with unchanged or worse
read latency. These are hypotheses, not causes established by comparing old
cross-campaign medians.

## New comparison

Five rotated blocks, each containing FIFO/default, native/default,
BPF/default, FIFO/four workers, native/four workers, BPF/four workers.
All arms use the same live-event-driven provider and 200 ms cumulative write
budget. All BPF arms use the already measured allocation-reuse implementation.
The actual policy, ABI, scalar defer clamp, arrivals and amount of I/O do not
change. Contemporary same-mode default controls are necessary to distinguish
executor changes from run-to-run storage variability. FIFO receives the same
executor change to avoid attributing its generic benefits to BPF.

Use the existing fresh-process single-cell runner with 64 demand reads,
96 background writes, 24 MiB objects, 2/4 ms scheduled arrivals and a
4096 MiB GPU staging pool. No new preflight, timeout, retry, clock study or
correctness campaign. Keep every attempted cell and all adverse results.

Primary metrics: scheduled-arrival read p99 and completed-write throughput,
with read p50 and total bandwidth reported alongside them. Report within-block
four/default changes per mode, BPF/native at each concurrency setting, and
policies/FIFO. With 64 reads the existing nearest-rank p99 is the maximum;
do not relabel this as serving TTFT or add confidence claims from five pairs.
The five rotations do not cover all six possible order positions per arm.

This supporting result can demonstrate an executor improvement or a
read/write tradeoff, but not that BPF intrinsically accelerates disk I/O.
If mixed or adverse, preserve it and leave the old default unchanged rather
than selecting another concurrency value after inspecting the outcomes.

## Execution and ownership

Root runs the completed launcher and publishes code, exact command/order,
per-request records, logs and analysis. Local Qwen 27B owns the minimal runner
parameter; a second local Qwen session owns the launcher after Qwen Next
terminated with an actual provider error after five retries. GLM remains live
on the earlier bottleneck analysis. At most three local sessions; none is
terminated for silence. No vendored code or driver is modified.

New raw directory: `raw/gds-write-workers-575-20260907-five-block/`.
Remove only each successful cell's own generated `cache/` after its process
exits, retaining its exact path/apparent size and all measurement files.
The workload's >100 GiB aggregate regenerable payload would otherwise exceed
free space. Apply this equally to all arms outside request timing; it can
still influence subsequent SSD state and is disclosed rather than ignored.
