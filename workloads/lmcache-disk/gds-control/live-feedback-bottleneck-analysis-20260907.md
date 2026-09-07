# Live-demand feedback: BPF-vs-native read-p99 bottleneck analysis (20260907)

Read-only diagnosis of `raw/gds-mixed-live-feedback-575-20260907-five-block`
(15 cells; `results-575-gds-mixed-live-feedback-20260907.md`), the adapter
(`lmcache_gds_backend_adapter.py`), deciders (`lmcache_gds_policy_adapter.py`),
and `run_gds_mixed_backend.py`. No new GPU runs, no source edits.

## Measurement semantics (fields, not claims)

- Reads: `submitted_s == offer_s` (read_worker stamps both before
  `get_blocking`). Read offer->completed therefore includes adapter admission,
  `_decision_lock`, decider, LMCache internal locking, and real I/O. No isolated
  read-admission or storage-transfer interval exists in the records.
- Writes: `submitted_s` is stamped at actual save-coroutine entry
  (`timed_save`), so write offer->submitted includes admission, deferral and
  loop scheduling; submit->completed includes subsequent saving and completion.
- `feedback_records` (writes only) carry `pending_demand_reads`,
  `remaining_budget_ns`, `decision`, `requested_wait_ns`: no timestamps, no
  per-decision durations, no lock-wait measurements.

## Interval evidence (per cell, from result.json)

- Read scheduled->offer jitter: med 0.065-0.178 ms, max 0.52-6.33 ms, similar in
  all arms (largest max is native, 6.33 ms, block 2). Not BPF-skewed.
- Read offer->completed contains the dominant native/BPF gap. Read p99
  (native->BPF, ms): 980.9->1179.5, 472.5->546.2, 415.9->868.8, 422.9->784.1,
  652.2->805.4; BPF is worse in 5/5. These are dispatch-based values, not
  the separately reported scheduled-arrival p99 values.
- Write admission offer->submitted: FIFO med 0.11-0.22 ms; native 10.34-10.49;
  BPF 10.38-10.55 ms; maxima 11.1-14.5 (native), 11.6-15.4 (BPF). Within-block
  BPF-native median deltas -0.11 to +0.21 ms.
- Write post-admission submit->completed (median ms): native 1101/504/675/370/693;
  BPF 1358/630/980/849/868 - BPF slower in all five blocks (+126 to +478 ms).
- Completion-time variability: same-arm medians swing across blocks:
  write-save 370-1443 ms; completed-write throughput 849-2882 MiB/s; total
  storage 1415-4803 MiB/s (FIFO alone spans 1415-3158). Per-block absolute
  swings rival paired deltas; these timings do not isolate disk-service state.
- Decision volume: decisions = 64 read decisions + write-record decisions.
  Native 932-993 decisions / 868-929 records / 772-833 defers; BPF 940-967 /
  876-903 / 780-807; FIFO fixed 160. Both feedback arms re-decide ~9-10x per
  write at <=1 ms steps; 95/96 final write submits occur at budget exhaustion.

## What can and cannot be separated with existing fields

- Repeating decisions: counts and requested waits are recorded, but actual
  callback cadence and latency cost are not measured separately.
- Synchronous ioctl: structurally present; BpfDecider decides via one libc ioctl
  under BpfDecider._lock inside the adapter `_decision_lock`
  (`lmcache_gds_policy_adapter.py:252-268`, `lmcache_gds_backend_adapter.py:421-443`),
  while NativeDecider is pure Python (`:209-212`). Cost is unmeasured.
  HYPOTHESIS (unproven): per-decision ioctl cost inflates the read tail.
- Locks/GIL: HYPOTHESIS (root-added, unproven): ctypes.CDLL releases the GIL
  during the ioctl while `_decision_lock` stays held; with 64 reader threads a
  blocked reader waits out the entire hold, so a fast ioctl can be amplified
  into convoy delay. No GIL or lock-wait data exists. Write-admission medians
  agree within 0.21 ms, but the ~10 ms deferral can mask smaller costs; this
  does not bound average decision overhead.
- Python scheduling: write completions and save handoffs share the loop thread
  that executes ~780-807 write re-decisions per feedback cell. HYPOTHESIS
  (unproven; LMCache-internal stamping path not examined here): loop congestion
  inflates measured write submit->completed and can shadow reads. Higher BPF
  write-save medians could reflect loop contention or different storage
  service; existing fields do not separate them.
- The runner spawns 64 read threads and one loop thread, not a read-worker
  pool. LMCache-internal pool occupancy was not measured, so saturation there
  cannot be ruled out. Thread-level GIL pressure is also hypothesis-only.
- Storage-service variability is a possible contributor, not an isolated
  measured cause. Rotating arm positions does not guarantee identical state.
- Write admission backlog: the combined wait is measured - deferred writes
  wait ~10 ms (the full budget) almost regardless of arm, because
  submit-while-pending is mostly budget-exhaustion, not early release; its
  feedback onto read latency via the shared lock and disk is NOT separable.

## Adverse data preserved

- BPF worse than native on scheduled read p99 in 5/5 pairs; worst +109.097
  percent (block 2).
- BPF cells are heterogeneous: block 0 dispatch-based read p50/p99 is
  794.878/1179.476 ms; the separate campaign report gives the scheduled metrics.
- BPF/FIFO completed-write throughput median -3.374 percent (range -37.832 to
  +26.358 percent); BPF/native median -15.138 percent (-51.780 to +7.053);
  ranges are not confidence intervals.
- FIFO is itself adverse in places: block 4 FIFO read p99 1258.279 ms is the
  campaign's single worst read p99; FIFO block 0 shows p50 168.969 ms with
  p99 1072.008 ms. No cell was discarded or retried; do not repeat cells.

## Tested follow-up already completed (not pending)

The earlier C++ cuFile executor also retained decision timing in
[`raw/gds-policy-campaign-20260906-summary.json`](raw/gds-policy-campaign-20260906-summary.json):
median per-run mean decision time is 0.039563 us for FIFO, 0.063484 us for
native and 1.004594 us for BPF (five runs each, 64 requests per run).
`gds_executor.cu` times the native function or BPF ioctl call with
`monotonic_ns()`, separately from transfer completion. This is a different
C++ executor with controlled policy inputs, not the current Python
live-feedback path; it omits Python request construction, GIL handoff and
adapter locking. It supplies an existing low-level reference without
rerunning cells, but cannot bound current per-decision cost or explain the
observed hundreds-of-milliseconds completion gap by itself.

[The completed GIL report](../results-575-gds-gil-handoff-20260907.md): default-off
`LMCACHE_GDS_IOCTL_KEEP_GIL=1` (PyDLL) construction (main `2d27777d`) in a
four-arm interleaved 20-cell FIFO/native/BPF-release/BPF-keep ablation ended
mixed, not a reliable improvement: the GIL convoy hypothesis is tested but
unconfirmed; keep default CDLL.

## Next experiments (at most two, non-redundant)

1. Decision localization: add per-decision timing to feedback_records (decider
   call duration, lock-acquire wait, monotonic timestamp) plus coarse
   read-decision timers; one interleaved 5-block native/BPF run in a NEW output
   directory only if event-driven results still need localization. This is
   optional performance diagnosis, not a precondition for measuring.
2. Critical-section shrink: build PolicyRequest and append feedback outside
   `_decision_lock`; hold it only for id assignment plus the decision, both
   arms. Unchanged policy semantics, same executors; tests whether lock
   hold-time amplification, not call count, drives the read-p99 tail.

Not recommended: rerunning completed cells, repeating the GIL ablation, or
implementing the event-driven executor here - that stays delegated to the Qwen
worktree and remains the main pending change.
