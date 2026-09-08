# Original Fig.13 four-arm follow-up

Status: running, not a completed three-workload result. HotSpot and GEMM each
finished all 20 cells with runner exit zero and both tenant exit codes zero
in every cell. K-Means started at 2026-09-08 07:57 UTC. Runner commit `a5d66a13` adds the
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
core, current UVM, services and both loaders must be restored after the run.
`driver-lifecycle.log` records the actual commands and outcomes; restoration
is not yet claimed complete.

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
