# Original Fig.13 runner follow-up

Status: implementation dispatched, not measured. XSched's Qwen 27B session
ended with terminal HTTP 524 at 1788845412755 ms Unix time and its CLI exited;
it produced no source implementation. Its context is retained for continuation.
The freed slot now runs Qwen Next session `ses_f8079f034ffexM3YIHNS6mzN8I`
on this original-runner task. Disk UVM and automatic warp execution remain
live in the other two slots. No session was stopped for silence and no fourth
session was launched.

The user asks for the combined memory/scheduling result inside the original
HotSpot, GEMM and K-Means panels, using the original scripts. No paper file,
paper figure or paper number may be modified by this session. Preserve every
old CSV and plot. The existing `workloads/fig13-fast` HotSpot measurements are
complete and must not be repeated as that campaign or mixed with new controls.

## Question and comparison

Test whether combining the existing memory and scheduling policies improves
completion times over each policy alone across the original three workloads.
This is supporting evidence for policy composition, not a new policy or a
mechanism-overhead comparison. A high-priority improvement accompanied by a
low-priority slowdown is a tradeoff; a combined-policy loss must remain visible.
The completed fast HotSpot study does not settle GEMM/K-Means or provide
matched controls under the original runner's timing convention.

Use no-policy control, memory-only `prefetch_eviction_pid` (20/80), scheduling
only (1000000/200 us), and their combination. Preserve the original one
iteration, HotSpot/GEMM size factor 0.6 and K-Means sparse size factor 0.9;
confirm these against the original README/configuration before running.
Collect five interleaved blocks for each workload (60 paired-process cells),
with fixed settings across arms. Existing unrelated prefetch variants need
not be rerun to answer this comparison. Never combine historical single-round
controls with new combined-policy rows as a matched experiment.

## Bounded changes for local OpenCode

Own the existing `run_policy_comparison.py`, `run_scheduler_comparison.py`
and `plot_all_kernels_stacked.py` in this directory. Extend those files;
do not create a replacement experiment runner or framework.

- Replace the obsolete hard-coded `co-processor-demo/gpu_ext_policy/src`
  paths with the current repository's `extension` and `microbench/memory`
  paths, with optional overrides if needed.
- The scheduler runner currently starts CUDA tenants before attaching its
  initialization-time policy, then sleeps and waits for high before low.
  Install policies before CUDA initialization. Reuse the stopped-before-exec
  process launch approach already implemented in `workloads/fig13-fast`
  when PID-based memory policy configuration requires it.
- Record each tenant's completion independently. The old sequential waits
  overestimate low-priority completion when it exits first. The memory
  runner's polling is a useful starting point, but its post-attachment sleep
  and final-loop exit also delay observation of short completed processes.
  Apply the same timing implementation to all new comparison arms.
- Retain a common start origin for both tenants, not separate elapsed times
  from each SIGCONT. Record spawn, policy attachment, release, and independent
  exit observations so setup and release skew remain explicit. If attachment
  is moved outside the measured interval, state that change from the old
  before-Popen start convention and use fresh controls consistently.
- Preserve benchmark stdout, policy output, exit status, parameters, command
  lines and per-tenant timestamps instead of deleting temporary raw outputs.
  Keep available policy counters as evidence, not an additional admission gate.
- Cleanup must target this run's children/attachments only. Do not retain
  the old global `pkill`/struct-ops cleanup behavior that could remove the
  live LMCache policy. Coordinate GPU and struct-ops locks with root.
- Add opt-in combined-policy rows and plotting input/output directories;
  do not overwrite the old outputs. The plot currently uses `row.iloc[0]`:
  aggregate repeated rows explicitly instead of silently plotting block one.
  Keep three panels, show both tenants' completion, and compute overlap and
  remaining execution from the recorded common timeline. No fourth panel.

Root reviews the bounded patch, runs the original commands with fresh output
directories, analyzes block-paired changes for each tenant and total completion,
and commits/pushes code and lightweight raw results. Do not run GPU experiments
or switch modules from the implementation session. No clock-precision,
correctness campaign, or new review/preflight workflow is requested.

## Current source evidence

- `run_scheduler_comparison.py`: `start_time` precedes Popen, attachment follows
  launch, and `high_proc.wait()` precedes `low_proc.wait()`.
- `run_policy_comparison.py`: policy attachment requires live PIDs and is
  followed by a one-second sleep before completion polling.
- `extension/gpu_sched_set_timeslices.bpf.c`: timeslices are applied in
  `on_task_init`; current code also includes `on_timeslice_control`.
- `plot_all_kernels_stacked.py`: selected rows currently exclude combination
  and choose only the first matching CSV row.
- `workloads/fig13-fast/results-performance-575-20260907.md`: completed
  HotSpot-only evidence and module setup are reusable references, not a
  substitute for this requested original-script comparison.

The executable invocation and actual module/source revisions will be recorded
after the local implementation lands. There are no new performance numbers yet.

## Implementation dispatch prompt

```text
Implement the accepted ORIGINAL Fig.13 runner follow-up in /home/yunwei37/workspace/gpu/gpu_ext, current master. Read workspace AGENTS.md, docs/eval/multi-tenant-memory/README.md, and docs/eval/multi-tenant-memory/combined-followup-plan-20260908.md. This plan is the concrete task, not a request for another plan. Reuse the existing run_policy_comparison.py, run_scheduler_comparison.py, plot_all_kernels_stacked.py; do not create a replacement runner/framework. Scope ONLY those three scripts and a short usage addition to that directory's README if needed. No docs/paper, tex or tex-revision changes, new papers, hashes/checksums, subagents, Git commits/worktrees, module loads, GPU runs or BPF attachment. Root runs measurements and commits/pushes. Two other OpenCode sessions own disk-driver and bpftime-auto-warp; do not touch their sources.

Minimal implementation: add opt-in four-arm comparison (no policy, memory prefetch_eviction_pid 20/80, scheduling 1000000/200us, combined), five interleaved blocks, original HotSpot/GEMM size-factor0.6 and KMeans sparse0.9, iterations1. Preserve older modes/results and output columns. Fix obsolete base paths. Use stopped-before-exec children to know PIDs and attach initialization-time policies before CUDA starts; reuse mechanism from workloads/fig13-fast. Independently observe each child's completion, one common release origin; retain spawn/attach/release/exit timestamps and stdout, policy logs, statuses, settings and argv in fresh output directories. Do not sequentially wait high then low, do not use global pkill or remove others' structops. Only own processes/attachments cleanup. Root coordinates locks /tmp/gpubpf-revision-gpu0.lock and /tmp/gpubpf-revision-struct-ops.lock. Extend existing plotting to aggregate repeats, use explicit new input/output paths, keep three panels and show four arms with both tenants' overlap and remaining execution from common timestamps. Do not change any stored historical CSV or figure.

Implement small incremental apply_patch calls, one connected file portion at a time, not a huge monolithic tool argument; upstream model calls have sometimes ended in HTTP524. Do not spend the entire session on a broad source survey. This is no session timeout; preserve the full task and continue through real source edits. Ordinary Python syntax/--help checks permitted (must not launch GPU). No new gates or correctness campaigns. Leave source uncommitted with concise handoff: actual files, exact commands, checks and remaining limitations. Keep same settings across all four fresh arms. Existing completed fast HotSpot cells must not be rerun or substituted for the matched controls.
```
