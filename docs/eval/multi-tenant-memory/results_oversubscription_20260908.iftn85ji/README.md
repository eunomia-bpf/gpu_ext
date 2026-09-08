# Three-workload oversubscription sweep

User-authorized on 2026-09-08. Status: running since 04:00 PDT, 2026-09-08. Both experiment locks are held.
The default, scheduler, prefetch, and eviction tools have completed initial real cells.
No completed sweep or speedup is claimed yet.

Question: how do memory-only, scheduler-only, and joint policies affect
high-priority completion as the two-tenant working set exceeds GPU capacity?
Use HotSpot, GEMM, and sparse K-Means with requested total ratios
0.8, 1.0, 1.2, 1.5 and five rotated repetitions of five policies:
default, scheduler, Prefetch(20,80), Evict(20,80), memory plus scheduler.
All policies start before CUDA initialization. Completion uses one common
release timestamp and independent tenant-exit observation. The unchanged
benchmark includes initialization, two warmups, one measured iteration,
and process termination. Both tenant times and kernel-event times remain.

Reuse run_policy_comparison.py run_combined_mode/run_combined_arm. The local
sweep script configures those existing functions, selecting the original
prefetch_pid_tree or prefetch_eviction_pid tool. It does not reimplement
launch, attachment, timing, parsing, or cleanup. All source observations are
fresh, in per-workload/per-ratio directories; no old measurement is pooled.
The original runner source is copied under inputs for reference. Actual
commands and policy parameters are recorded by the original runner.

Primary plot: three panels, five policies, high-priority completion versus
combined workload size divided by GPU memory capacity. Median and range of
five observations per point. Allocation rounding in each original kernel
must be included when deriving actual x coordinates. Also retain low-priority
and total-completion curves for interpretation, regardless of direction.
A speedup is not assumed; a crossover or regression changes the interpretation.
The default policy supplies a control at each ratio, and scheduler/memory
variants isolate contributions within the same measurement convention.

Use the saved 575 scheduling core and current UVM from the earlier completed
campaign. Hold both GPU and struct-ops locks for all module changes and
measurements. Preserve and restore the installed core, saved current UVM,
gdm, nvidia-persistenced, and the two existing storage-policy loaders.
No paper edits, commits, pushes, or content digests are part of this run.

Invocation: sudo python3 run_sweep.py (after coordinated driver setup).

## User scope update

The user cancelled ratio 1.8 while ratio 1.5 was running. The final matrix is
300 cells (3 workloads, 4 ratios, 5 policies, 5 repetitions). The live runner
already loaded its original loop, so `hotspot/ratio-1.8/USER_CANCELLED.md`
documents an output-directory guard that prevents starting that ratio.
Its expected exit 1 at this guard is a scope stop, not a measured failure.
All ratio 1.5 work must finish and driver restoration must still succeed.
