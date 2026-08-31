# RQ3 evidence reuse for revision R1/R4 — 2026-08-31

Scope: can the existing compute-scheduling and memory-priority artifacts
support the revision's distinction between scheduling and memory policy?
This is a read-only reanalysis of historical runs, not a new GPU experiment.
Source/data baseline: gpu_ext commit `e016aae`. The paper and raw CSVs are
unchanged. No claim, workload, baseline family, or RQ is being replaced.

Decision: the historical arithmetic is reproducible, but it does not close
the revision comparison. Preserve these artifacts; qualify the actual policy,
measurement semantics, and engagement before using them as final evidence.
New matched runs must retain the intended host-to-kernel latency and aggregate
BE throughput, rather than substitute the easier-to-measure historical proxy.

## Compute scheduling: what the 96% number measures

The paper figure's [plot source](../paper/img/results-raw/multi-tenant/plot_figures.py)
reads [simple_test_results](multi-tenant-scheduler/simple_test_results).
All 120 CSV files are present: 10 runs per mode, two LC and four BE processes,
200 records per process, hence 24,000 records across both modes.

Recompute without launching GPU work:

```bash
python3 docs/eval/multi-tenant-scheduler/analyze_results.py
```

| Historical statistic | Native | Policy |
| --- | ---: | ---: |
| Mean of ten per-run LC P99s (µs) | 1187.6 | 53.2 |
| Median of ten per-run LC P99s (µs) | 42.0 | 42.4 |
| P99 of all 4,000 LC records (µs) | 43.9 | 43.2 |
| Runs with LC P99 above 1 ms | 1/10 | 0/10 |
| Kernel count / summed BE duration (1/s) | 11.38 | 11.47 |

The 95.5% reduction is specifically the first row. Native run 0 has P99
11,401.5 µs; no sample or run was discarded. The difference between the first
and third rows is an aggregation distinction, not permission to select a
metric after seeing its result. Neither point estimate alone establishes
statistical significance or unchanged BE throughput.

The [launcher](multi-tenant-scheduler/simple_timeslice_test.py) runs
`gpu_sched_set_timeslices`, with LC=1 s and BE=200 µs. It does not launch the
explicit preemption tool. The separate March
[four-mode results](multi-tenant-scheduler/kfunc_preempt_results/mode_summary.csv)
include `kfunc_only` and `timeslice_kfunc`; those are different runs and must
not be silently substituted for the figure's data. The original launcher
also runs all native repetitions before all policy repetitions, not an
interleaved comparison.

The available [benchmark source](../../microbench/multi-stream/multi_stream_bench.cu)
records the so-called enqueue event on stream 0 and the start event on the
workload stream *before* launching the kernel. Thus its CSV difference is an
event-to-event gap, not a host-submission-to-kernel-entry observation. The
available Makefile uses the default stream semantics and the source creates
blocking streams. NVIDIA documents that legacy stream 0 synchronizes with
blocking streams; event recording itself is asynchronous. This makes the
instrumentation's effect on concurrency a concern, not proof of the exact
historical binary's behavior: that binary's build identity is not encoded in
these CSVs. See the official CUDA 12.9
[stream semantics](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/stream-sync-behavior.html)
and [event API](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/group__CUDART__EVENT.html).

The BE figure sums individual recorded durations. That is inverse mean
duration, not concurrent aggregate throughput. The CSVs do retain host clock
timestamps: using, per run, earliest BE `host_launch_us` through latest BE
`host_sync_us`, then dividing 8,000 kernels by the sum of the ten windows,
gives 46.2262 kernels/s native and 46.0143 kernels/s policy. This is a
retrospective host-window diagnostic, excludes pre-launch initialization,
and does not independently establish equivalence. Process-local CUDA-event
origins must not be combined to construct a cross-process wall-clock window.

The analyzer now reports the historical numerical summaries without the
former arbitrary >50% reduction / ±5% change tests that printed “ALL CLAIMS
VERIFIED.” Its six returned numerical values are unchanged. A successful
exit means the analysis ran, not that the paper claims were validated.

## Memory priority: arithmetic versus policy engagement

The [figure source](../paper/img/results-raw/multi-tenant/plot_all_kernels_stacked.py)
uses `max(high_latency_s, low_latency_s)` as completion time. Explicit complete
CSV sources, avoiding modification-time selection, are:

| Workload | Memory CSV timestamp | Scheduler CSV timestamp |
| --- | --- | --- |
| [HotSpot](multi-tenant-memory/results_hotspot) | `20251208_101609` | `20251208_113441` |
| [GEMM](multi-tenant-memory/results_gemm) | `20251208_102321` | `20251208_113846` |
| [K-Means](multi-tenant-memory/results_kmeans) | `20251208_103714` | `20251208_114516` |

Filenames are `policy_comparison_<timestamp>.csv` and
`sched_comparison_<timestamp>.csv`. HotSpot's earlier scheduler CSV
`20251208_113410` contains only a single-process row and cannot supply the
comparison. The plot currently picks files by modification time, which is
not a reproducible run selection rule.

| Workload | Memory-file no-policy (s) | Prefetch(20,80) (s; reduction) | Evict(20,80) (s; reduction) | Scheduler (s) |
| --- | ---: | ---: | ---: | ---: |
| HotSpot | 53.9473 | 23.9954; 55.52% | 23.8924; 55.71% | 53.8069 |
| GEMM | 135.7901 | 29.6159; 78.19% | 29.6901; 78.14% | 136.5450 |
| K-Means | 85.4899 | 6.7075; 92.15% | 6.5975; 92.28% | 85.7095 |

Every selected configuration has only `round=1`. The scheduler files also
contain their own no-policy rows: 53.8595, 135.5803, and 85.7207 s. Against
those rows, the scheduler changes are +0.098%, −0.712%, and +0.013% in reduction
notation; the figure instead uses the memory-file no-policy rows. These
single-round differences do not establish that a correctly engaged scheduling
policy is ineffective.

In [run_scheduler_comparison.py](multi-tenant-memory/run_scheduler_comparison.py),
both workloads start before the scheduler tool, which is delayed by 0.5 s.
The [timeslice policy](../../extension/gpu_sched_set_timeslices.bpf.c) changes
timeslices on task initialization; its bind callback only observes state.
The launcher neither requires a successful attach nor saves policy-hit
counters. Missing engagement evidence is not evidence that the scheduler ran
and lost. The CSVs cannot resolve whether their TSGs existed before attachment.

The scheduler launcher waits for the high process first, then timestamps the
low process after waiting for it. Its low-process duration is therefore not
an independent completion measurement if low finishes first. The pair's
maximum remains a launch-to-both-reaped interval, but these data cannot recover
independent per-tenant finish times in that case.

Finally, twice the single-process time is a sequential reference, not a proven
lower bound for concurrent execution. K-Means' selected memory policies finish
in 6.60–6.71 s, below that 7.80 s reference. This is not an algorithm failure;
the lower-bound interpretation lacks justification.

## Required continuation

Keep R1/R4 open. For fresh runs, measure the intended latency using correlated
host/kernel timestamps in one validated clock domain, measure BE work over a
shared wall-clock interval, attach initialization-time policies before starting
the workload, retain engagement/correctness evidence, and repeat the complete
matched comparison. The existing XSched proposal remains closed after three
review rounds; this audit does not reopen it or authorize another review of
the unchanged proposal. When the GPU becomes idle, the approved LMCache
preflight can proceed independently; it does not resolve these RQ3 defects.

Validation performed: existing scheduler analysis on all 120 CSVs; exact
before/after equality of all six numeric return values; rejection of
incomplete file counts; explicit inspection and recomputation of the six
complete memory/scheduler CSVs; no GPU process launched or signalled.

Independent read-only review confirmed the arithmetic, source interpretation,
unchanged six numerical outputs, and bounded scope. It found no remaining
material blocker for this audit, while explicitly leaving R1/R4 open. The
analyzer change removes 43 net lines; no raw data or paper text was changed.
