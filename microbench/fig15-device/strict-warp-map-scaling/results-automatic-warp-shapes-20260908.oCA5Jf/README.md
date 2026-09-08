# Automatic warp: original 32/1024-thread shape follow-up

Completed on RTX 5090 / driver 575.57.08 on 2026-09-08 at approximately
10:32 UTC. Thirty applications and twenty loaders exited zero. This follows
the existing device shape experiment, not a new paper reproduction or a
Table 1 throughput result. Previously completed 128-thread cells were not rerun.

## Results

Each cell runs one CUDA block, eight warmups and 128 measured launches.
The metric is total CUDA-event elapsed milliseconds over those 128 launches.
Both attached arms use the unchanged original shared_update BPF object.

| Threads per block | Native median ms | BPF off median ms | BPF on median ms | Paired mean on/off change | 95% bootstrap interval |
| --- | ---: | ---: | ---: | ---: | --- |
| 32 | 0.233472005 | 0.330303997 | 0.331808001 | +0.5214% | [-0.1772%, +1.2199%] |
| 1024 | 0.246976003 | 0.523904026 | 0.523679972 | -0.1292% | [-0.2620%, -0.0086%] |

There are five paired blocks per shape. Native times vary more than attached
times. Paired changes use each block's on/off ratio, not the ratio of medians.
Intervals enumerate all 3125 ordered whole-block bootstrap resamples and use
linear percentile quantiles. All observations are retained in cells.csv and
analysis.json; these intervals do not capture cross-session or hardware drift.

The 32-thread comparison does not distinguish a benefit or loss at this sample
size. The 1024-thread point estimate is a very small improvement, not a
substantial optimization. Neither shape reproduces the earlier 128-thread
+53.1303% slowdown. Shapes were measured sequentially, not interleaved with
the old campaign. These observations motivate investigation of shape-dependent
generated-code/resource behavior; they do not identify its cause. No generic
speedup, measured scalar execution-count reduction or Table 1 improvement is
claimed. The previous adverse campaigns remain intact.

## Execution and scope

Runtime libraries are the same build-auto-warp-575 build from bpftime
0c3b6d1; source HEAD d6db11e adds only the CPU PTX tests. No runtime rebuild
occurred. Existing warp-map-bench, warp-map-loader and warp-map-probe.bpf.o
under .output are unchanged. The only per-arm runtime switch is
BPFTIME_GPU_AUTO_WARP_EXECUTION=0/1. GPU thread slots match each shape.
No manual probe lane guard, altered data contract or event suppression was added.

Root directly reused the original commands; commands.sh records the exact
shell orchestration (a rerun creates a fresh directory). Orders rotate
native/off/on, off/on/native, on/native/off over five blocks per shape.
32 threads precedes 1024 threads. Every arm starts a fresh application;
attached arms use fresh private 256 MiB transport segments. Loader readback
happens after application exit and is outside the CUDA-event interval.
All ten on arms report automatic eligibility and all twenty attached loader
readbacks report key 0 = 6291613622346973184. The existing application reports
zero output mismatches. No separate clock/preflight/audit campaign was added.

Root held /tmp/gpubpf-revision-gpu0.lock and released it at completion.
All twenty task-owned private transport segments were removed after their
loaders exited; no large buffer or binary is archived. Deleted segments are
regenerable runtime storage, not retained measurements. Small raw logs remain.
Runtime-internal patch-cache key text alone is replaced with
<internal-cache-key-omitted>; timing, errors and output values are unchanged.
Native execution.log loader_rc=0 is only a shell placeholder; cells.csv leaves
it empty because native has no loader.

Outstanding implementation remains multiplicity-preserving Table 1 event
batching and the original device grid/work/actual-handler-count extensions.
These completed shape cells must not be repeated by the pending runner.
