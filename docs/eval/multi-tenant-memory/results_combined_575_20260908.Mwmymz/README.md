# Original Fig.13 four-arm follow-up

Status: the three-workload measurement campaign is complete. All 60 cells
finished, all 120 tenant exit codes are zero, and all three runners exited
zero. K-Means finished at 2026-09-08 08:57 UTC. Runtime restoration is also
complete. The historical checkpoints below retain the collection timeline;
the final K-Means and unified summaries give the completed result.
Runner commit `a5d66a13` adds the
opt-in comparison to the original `run_policy_comparison.py`.

The planned campaign is five rotated blocks of baseline, memory-only,
scheduling-only and combined policy for each of the original three kernels
(60 two-tenant cells). HotSpot/GEMM use size factor 0.6; `kmeans_sparse`
uses 0.9. All use one iteration, managed UVM, memory parameters 20/80 and
scheduler timeslices 1000000/200 microseconds. No completed historical cell
is replaced or pooled with these fresh controls.

Commands, from the repository root, use this directory's absolute path as
`RESULT_ROOT` (notation only; the lifecycle log retains actual commands):

```sh
python3 docs/eval/multi-tenant-memory/run_policy_comparison.py --combined --blocks 5 --kernel hotspot --size-factor 0.6 --output RESULT_ROOT/hotspot
python3 docs/eval/multi-tenant-memory/run_policy_comparison.py --combined --blocks 5 --kernel gemm --size-factor 0.6 --output RESULT_ROOT/gemm
python3 docs/eval/multi-tenant-memory/run_policy_comparison.py --combined --blocks 5 --kernel kmeans_sparse --size-factor 0.9 --output RESULT_ROOT/kmeans
```

Both tenants stop before CUDA initialization, policies attach, then elapsed
completion uses one common release origin and independently observed exits.
Policy setup is outside the measured interval. This differs from the old
runner's before-Popen origin and the earlier fig13-fast per-SIGCONT times;
the fresh controls are required for that reason. Raw command lines, process
timestamps, return codes, logs and distinct per-tenant result CSVs are kept.

The headline completion metric includes process/CUDA initialization, the
benchmark's two built-in warmup runs, its one measured iteration, output
and teardown. It is not just the CUDA-event duration in `high_median_ms` or
`low_median_ms`. The two warmups are present in the original HotSpot,
GEMM and sparse K-Means implementations and were not changed for these
arms. For example, first-block K-Means memory-only high-priority completion
is 243.363910 s while its one measured kernel iteration is 78950.5 ms;
these describe different intervals, not competing estimates of one metric.

## Actual runtime and restoration

RTX 5090, NVIDIA 575.57.08, Linux 6.15.11-061511-generic. Root holds both
revision GPU/struct-ops locks. The current installed core lacks the scheduling
interface, so root temporarily loaded the already-built candidate core
(30216144 bytes) with the **current** `ff68a1d4` UVM (62413352 bytes).
The earlier 61945872-byte pre-KV-reclaim UVM was not substituted. All fresh
arms use the same current UVM, with `uvm_enable_builtin_tests=0`. The disk
backing implementation in progress was not built or loaded.

The immutable-for-this-run copies are in
`/var/tmp/fig13-original-modules-20260908.TkarSf/`; module binaries are not
publication artifacts and will not be committed. Root stopped only the two
known idle GDS/reclaim loaders (1744933 and 2106071), the GDM greeter and
persistence service after confirming no GPU compute process. The original
core, current UVM, services and both loaders were restored after the run.
`driver-lifecycle.log` records the actual commands and outcomes; restoration
completed with exit zero. The saved 62413352-byte current UVM was used, not
the older pre-KV-reclaim module. `uvm_enable_builtin_tests` is still zero.
GDM and nvidia-persistenced are active. The restored GDS loader (PID 2474310)
and KV-reclaim loader (PID 2474311) both reported `attached` and were alive
with their original binary/object arguments. Their startup logs are
`/tmp/fig13-restored-gds-policy-20260908.log` and
`/tmp/fig13-restored-kv-reclaim-20260908.log`. GPU/struct-ops locks were
released and the lifecycle shell closed; the background loaders do not
inherit those lock descriptors. No unfinished disk-backing module was loaded.

The system `bpftool struct_ops show` returned process exit 139 during
preparation. No global struct-ops cleanup was used. Experiment tools retain
their own loader output; this utility failure is not performance evidence.

## Completed HotSpot checkpoint

The full five-block CSV and per-cell records are under
`hotspot/combined_20260908_002905/`. Completion-time medians in seconds:

| Arm | High-priority tenant | Low-priority tenant |
| --- | ---: | ---: |
| Baseline | 56.286889 | 56.533284 |
| Memory only | 26.199263 | 27.876546 |
| Scheduling only | 3.963689 | 6.155824 |
| Combined | 3.560419 | 6.515025 |

These are arm medians, not paired effect estimates or intervals. Combined
improves high-priority completion versus scheduling alone but increases
low-priority completion time. This is a tradeoff, not an across-the-board
win. The first baseline block takes 80.798918/81.041481 seconds and remains
in the five-block results; it has not been dropped or rerun. No historical
HotSpot results are overwritten. GEMM is reported below; K-Means is still running.

Within each of the five blocks, combined versus scheduling-only changes
high-priority completion time by a mean **-10.5294%** (95% interval
[-11.4830%, -9.5759%]) and low-priority completion time by **+5.8852%**
([+5.1906%, +6.5798%]). Every pair has the same tradeoff direction.
Negative percentages mean faster completion. These pointwise intervals
enumerate all 3125 ordered whole-block bootstrap samples of the five paired
percent changes, with linearly interpolated 2.5/97.5 percentiles. The
[paired summary](hotspot/paired-summary.json) retains the source, all pairs,
method and comparisons with the memory-only and baseline arms. This is a
policy-composition result, not a BPF-versus-native mechanism-overhead result.

## Completed GEMM checkpoint

All five blocks and 20 cells are retained under `gemm/combined_20260908_003830/`.
Completion-time medians in seconds:

| Arm | High-priority tenant | Low-priority tenant |
| --- | ---: | ---: |
| Baseline | 138.788579 | 138.622167 |
| Memory only | 22.200545 | 28.115147 |
| Scheduling only | 13.370843 | 19.282938 |
| Combined | 13.041686 | 21.994206 |

Combined versus scheduling-only changes high-priority completion by a paired
mean **-2.5290%** (95% interval [-2.8204%, -2.2840%]) and low-priority
completion by **+14.0469%** ([+13.8065%, +14.2789%]). The corresponding
paired medians are -2.3130% and +14.0685%; do not confuse them with the mean
effects or ratios of arm medians. Every pair has the same tradeoff direction.
The [GEMM paired summary](gemm/paired-summary.json) uses the same method as
HotSpot and retains all source pairs and baseline/memory comparisons.
The small high-priority improvement comes with a larger low-priority cost;
it is not an overall latency win. K-Means remains outstanding.

## In-progress K-Means checkpoint: first block only

Block 0 finished all four arms with both tenant exit codes zero. Its closed
per-cell logs, metadata and tenant CSVs are preserved under
`kmeans/combined_20260908_005722/block00_*`. The aggregate CSV and lifecycle
log remain live while the other four blocks run; this is not a five-block
result and no completed cell is scheduled for rerun.

| Arm | High-priority completion (s) | Low-priority completion (s) |
| --- | ---: | ---: |
| Baseline | 335.357313 | 335.118057 |
| Memory only | 243.363910 | 258.582308 |
| Scheduling only | 31.487096 | 57.789015 |
| Combined | 31.084584 | 58.290425 |

These are individual observations, not medians. Combined is slightly faster
for the high-priority tenant and slower for the low-priority tenant in this
block. A progress message mistakenly quoted 32.40/58.40 seconds; the source
CSV values above correct that transcription, without altering any measurement.

At 08:21 UTC block 1 also finished all four arms, with both tenant exit codes
zero in every cell. Its closed `block01_*` files are collected alongside
block 0. K-Means is now 8/20 cells complete and running block 2; the overall
campaign is 48/60 cells complete. Aggregate five-block analysis and runtime
restoration remain pending; the first-block table is not promoted to a final
effect estimate.

At 08:33 UTC block 2 completed with both tenant exit codes zero in every
arm. Its closed `block02_*` records are retained. K-Means is 12/20 cells
complete (overall 52/60), with block 3 running. No completed block was
repeated or omitted; five-block analysis and restoration are still pending.

At 08:45 UTC block 3 completed with both tenant exit codes zero in all four
arms. The closed `block03_*` records are collected. K-Means is 16/20 cells
complete (overall 56/60), and the final block 4 is running. No five-block
K-Means effect estimate or completed runtime restoration is claimed yet.

## Memory-policy activity in the completed HotSpot/GEMM runs

The existing final `=== Summary ===` sections of each `mem_tool.log` give
the following medians across all five blocks; no additional run was used.

| Workload | Memory-only: `Total activated` | Combined: `Total activated` | Memory-only: `Total used calls` | Combined: `Total used calls` |
| --- | ---: | ---: | ---: | ---: |
| HotSpot | 31891697 | 10579172 | 87965 | 19260 |
| GEMM | 42581034 | 25355284 | 35024 | 19266 |

These are policy bookkeeping counters, not transferred pages/bytes or unique
GPU-memory accesses. In `extension/prefetch_eviction_pid.bpf.c`,
`update_prefetch_stats` increments `total_activate` for prefetch-tree
decisions; chunk activation also contributes to that field. `total_used`
is incremented by the block-activation policy path. Allow/deny totals mix
prefetch choices and LRU-reordering decisions, so the loader's generic
"moved" label must not be read as a count of physical transfers.

Adding scheduling is associated with substantially lower memory-policy
activity than memory-only in these runs. This supports describing the
composition as changing the memory-management workload, but does not
separately establish a reduction in page faults, DMA traffic or thrashing,
nor attribute all completion-time gains to one cause. The paired latency
tradeoffs above remain the performance results; these counters neither
replace them nor serve as a measurement-admission condition.

## Final K-Means result and three-workload comparison

All five blocks and 20 cells are retained under
`kmeans/combined_20260908_005722/`, including the final aggregate CSV, events,
metadata and per-tenant/tool logs. No cell was retried, removed or replaced.
Completion-time medians in seconds:

| Arm | High-priority tenant | Low-priority tenant |
| --- | ---: | ---: |
| Baseline | 332.056461 | 332.299329 |
| Memory only | 246.060516 | 260.657138 |
| Scheduling only | 31.597359 | 57.789015 |
| Combined | 31.125356 | 58.339609 |

Combined versus scheduling-only changes high-priority completion by a paired
mean **-1.3916%** (95% interval [-1.4890%, -1.2941%]) and low-priority
completion by **+0.9596%** ([+0.8663%, +1.0530%]). Paired medians are
-1.3573% and +0.9628%. Every pair has the same high-faster/low-slower direction.
The [K-Means paired summary](kmeans/paired-summary.json) uses the same
whole-block enumeration method and retains all individual pairs and
baseline/memory comparisons.

Across the three workloads, the marginal effect of adding memory policy to
scheduling is:

| Workload | High: paired mean change | High: 95% interval | Low: paired mean change | Low: 95% interval |
| --- | ---: | ---: | ---: | ---: |
| HotSpot | -10.5294% | [-11.4830%, -9.5759%] | +5.8852% | [+5.1906%, +6.5798%] |
| GEMM | -2.5290% | [-2.8204%, -2.2840%] | +14.0469% | [+13.8065%, +14.2789%] |
| K-Means | -1.3916% | [-1.4890%, -1.2941%] | +0.9596% | [+0.8663%, +1.0530%] |

This demonstrates a policy-composition tradeoff under these oversubscribed
two-tenant workloads: modest-to-larger high-priority gains, with low-priority
costs. It is not an all-tenant latency improvement, not a BPF-versus-native
mechanism-overhead comparison, and not evidence that old/new timing regimes
can share a paired baseline. Existing historical data remain unchanged.
