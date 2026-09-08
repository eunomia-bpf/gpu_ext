# Five-policy oversubscription sweep: analysis only

This directory reads the running campaign in
`../results_oversubscription_20260908.iftn85ji` and writes independent previews.
It does not modify the paper, original figures, experiment sources or raw data.

Run from the repository root:

```sh
python3 docs/eval/multi-tenant-memory/oversubscription-analysis-20260908/analyze.py docs/eval/multi-tenant-memory/results_oversubscription_20260908.iftn85ji
```

The expected matrix is three workloads, four total ratios (0.8, 1.0, 1.2, 1.5), five policies and
five repetitions (300 two-tenant cells). `preview/status.json` reports the
current snapshot. Figures include only workload/ratio points for which all
25 cells are complete. Unmeasured ratios are left empty. Lines show medians
and error bars show the observed minimum and maximum of five repetitions.
They are not confidence intervals. The full requested x range remains visible
in previews to avoid presenting a partial sweep as a complete curve.

Policies are Default, Scheduler, Prefetch(20,80), Evict(20,80), and
Memory + Scheduler. Evict retains the original combined prefetch/eviction
memory implementation. The fifth policy adds the original scheduler to that
same memory implementation. It does not introduce a new memory algorithm.

## Metric and input review

All five policies use the original common-release runner. Both tenants stop
before CUDA initialization, policies attach, and the runner releases both
tenants with one time origin. Independent waiters record each completion.
Elapsed completion time includes CPU initialization, built-in warmup, measured
GPU work, and process cleanup. Policy attachment before release is excluded.
This is application completion time, not just CUDA-event kernel duration.
The separate kernel duration is retained in both output CSV files.

The script checks stopped states, configuration equality, tool exits, scheduler
policy hits, memory-policy calls for both tenant PIDs, independent timestamps,
and positive elapsed times for every completed cell. Successful process exits
and policy engagement do not constitute a numerical output correctness test.

The horizontal coordinate is combined managed allocation divided by the
recorded CUDA total GPU memory (33,669,316,608 bytes). It is not per-tenant
`--size_factor`, instantaneous resident memory, or the CSV's requested byte
budget. Allocation sizes follow the saved workload source:

- HotSpot rounds the side length down to a multiple of 16, with three float
  arrays. The script checks the geometry against each tenant's log.
- GEMM rounds down to whole 4096-by-11008 float weight layers and adds the two
  activation arrays. Its layer count is checked against the log.
- Sparse K-Means allocates three arrays at a 4096-byte stride, a dense distance
  matrix with 200 centroids, and six centroid arrays. All policies use the same
  point count, stride and byte budget.

The initial free memory was 33,140,637,696 bytes, less than total capacity.
Thus a requested ratio of 1.0 need not fit in available memory. The sparse
workload also need not touch every allocated page simultaneously. Neither
allocation ratio nor the free-memory snapshot alone establishes fault traffic.

`high_s.pdf` is the primary completion-time preview. `low_s.pdf` and
`both_finished_s.pdf` retain the other tenant and total completion behavior.
`cells.csv` keeps all completed cells, including partial points, and
`summary.csv` summarizes only complete points. No observation is removed
because it is slow or does not favor the combined policy. The original
single-tenant and old five-bar data remain untouched in their original files.

All figures are vector PDFs at 3.33 by 1.20 inches, with one row of three
panels. Red solid lines emphasize Memory + Scheduler; color, line pattern
and marker distinguish the five policies. The layout follows the original
`all_kernels_stacked.pdf`: a bottom legend, a y-axis label on each panel,
prominent titles, and thin gray frames and grid lines. It retains the original
red/blue/green/purple palette and 0.85 opacity. The five-policy legend occupies
one row, using Sched and Mem + Sched for the scheduling labels at the user's request.
Tick spacing avoids overlap at the single-column size.
This is a review draft, not a
paper replacement or a final performance conclusion.

## Coordinated CPU build window

The experiment owner supplied this incremental build for the automatic-warp
runtime candidate (reported source checkpoint `c4c83cd`):

```sh
cmake --build /home/yunwei37/workspace/gpu/bpftime-auto-warp/build-auto-warp-575 --target bpftime-agent bpftime-syscall-server -j2
```

Both targets built successfully, command exit 0. The output is retained in
`auto-warp-build.log`. This is build evidence only, not a GPU performance or
record-correctness result. No additional GPU experiment ran in this window.

The sweep runner PID 2664070 was temporarily stopped at epoch
1788868870.020934, before release of K-Means ratio 1.0, block 3, Scheduler.
Both tenant launchers (2750077 and 2750078) were confirmed stopped before
exec, and the GPU compute-process query was empty. The build command's EXIT
trap resumed the runner. The original runner subsequently recorded release
at epoch 1788868949.401446, after the CPU build ended. The wait therefore
does not enter that cell's elapsed time. Driver state and the two experiment
locks remained with the sweep. The pause was between measured executions,
not an interruption of a running GPU workload.
