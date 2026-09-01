# Experiment Plan: RQ4 native no-prefetch versus gpubpf no-prefetch

**Status: closed after preflight 2; no performance rows were admitted.** The
8M-node event-instrumented native preflight cell exceeded its declared 15-minute
limit before completing one measured epoch. See `results/preflight-2.md`. The
third preflight allowance was not used because the reviewed semantic gate was
infeasible under its declared bounds; this does not estimate uninstrumented GCN
runtime.

Revision 2 replaces the proposed LRU comparison after review found that
`gpu_block_access` does not fire on the target workload. No LRU timing was run.

## Research Question

- RQ exactly as written in the paper: **RQ4 (Overhead): What is the overhead
  of gpubpf's core mechanisms and observability capabilities?**
- Specific uncertainty tested here: when the driver's existing no-prefetch
  policy is expressed through the live `gpu_page_prefetch` struct_ops hook, does
  the general gpubpf mechanism materially change end-to-end performance relative
  to the driver's original built-in no-prefetch configuration?
- Why the answer matters: the shepherd explicitly asks for a matched comparison
  between an existing policy expressed through gpubpf and its original
  monolithic implementation. A policy sweep cannot answer this because it
  changes both mechanism and policy.

## Paper-Value Admission

- Planned role: decisive evidence for the policy-versus-mechanism clarification
  and supporting evidence for RQ4.
- Largest credible paper story this experiment could unlock: gpubpf can express
  an existing driver policy with a directly measured, disclosed mechanism cost;
  matching the native path is valuable and need not outperform it.
- Strongest reviewer reject argument or load-bearing uncertainty addressed: the
  paper reports improvements from policies without showing whether an unchanged
  policy pays a cost for using the general mechanism.
- Independent evidence added beyond existing runs and published results:
  historical no-policy hook overhead does not execute a BPF policy callback,
  while historical prefetch-policy sweeps change the selected policy and lack a
  revision-grade paired protocol. This experiment holds the observable policy
  outcome—no pages added by prefetch—constant while changing implementation.
- Why the result is not tautological, already settled, or dominated: gpubpf
  leaves the prefetch subsystem enabled, invokes the hook, executes verified BPF
  and a checked kfunc, and returns `BYPASS`; the native module parameter exits
  before prefetch analysis. Their difference is the cost of expressing the same
  no-prefetch decision through the general hook rather than the built-in switch.
- Paper decision if positive: report the matched cost with uncertainty and
  attribute larger gains elsewhere to the selected policy, not the mechanism.
- Paper decision if contradictory, mixed, or inconclusive: disclose the measured
  drawback and scope it to this memory hook and workload; do not claim that
  existing policies are reproduced for free.
- Best alternative experiment and why this one has higher decision value: a
  fresh Fig. 13 sweep asks which policy helps but cannot isolate same-policy
  cost. Scheduling equivalence currently requires the supported 575 maintenance
  window. The reviewed LRU alternative is invalid because its access hook is
  dead on this workload and the activation hook cannot bypass the native move.

## Expected And Alternative Outcomes

- Current expected answer: BPF no-prefetch preserves correctness and has a
  small positive cost relative to native no-prefetch; the estimate and interval,
  not a preset pass threshold, determine the reported result.
- Strongest competing explanation: prefetch bookkeeping plus callback and helper
  execution on the page-fault path adds a material slowdown.
- Result that would contradict the expectation: a paired estimate whose 95%
  interval excludes zero and shows more than 5% slowdown, or any repeatable
  correctness/stability failure unique to BPF no-prefetch.

## Published Precedent And Real Assets

- Closest published protocol: the paper's RQ4 host-runtime experiment measures
  synchronized application runtime under oversubscription; the existing GCN
  benchmark supplies synchronized epoch time and is already used by the paper's
  memory-policy evaluation.
- Official system/model/data/benchmark/tool and version: NVIDIA Open GPU Kernel
  Modules 610.43.02 with the in-tree gpubpf UVM hooks; PyTorch 2.9.0; the
  repository's `benchmark_gnn_uvm.py`; RTX 5090.
- What is reused: the native read-only module parameter
  `uvm_perf_prefetch_enable`, the benchmark's seeded random graph,
  managed-memory allocator, synchronized epoch timing, training accuracy, and
  atomic `configs/gnn.py` command.
- Necessary deviations or custom glue: `extension/prefetch_none_revision.bpf.c`
  is a minimal no-print BPF policy that sets an
  empty region with `bpf_gpu_set_prefetch_region(result_region, 0, 0)` and
  returns `BYPASS`. `extension/prefetch_none_revision.c` is its ownership-safe
  loader: it reports ordinary program/map/link identifiers, waits, and detaches.
  The timed policy contains no statistics map or per-fault logging.
  `uvm_migration_monitor.c` is an untimed semantic monitor over NVIDIA's UVM
  tools migration-event ABI. No experiment controller or content fingerprint
  is added.

## Comparison

- Proposed system or method: load the same custom 610 UVM module with native
  prefetch enabled, then attach the fixed gpubpf no-prefetch policy.
- Main baseline and the competing position it represents: load that exact UVM
  module with `uvm_perf_prefetch_enable=0` and attach no struct_ops policy. This
  is the driver's original built-in monolithic no-prefetch implementation.
- Why the main baseline needs a matched run instead of citation alone: no
  publication reports the cost of this callback/kfunc path on this RTX 5090,
  driver, kernel, and workload.
- Controls or ablations, labeled separately: one untimed two-cell semantic
  preflight monitors NVIDIA UVM migration events for the exact process. Both
  native and BPF cells must report zero migrations and zero bytes whose cause is
  `UvmEventMigrationCausePrefetch`, with zero dropped migration events. The BPF
  cell additionally traces `uvm_bpf_call_gpu_page_prefetch` and
  `bpf_gpu_set_prefetch_region` and requires nonzero counts. This covers the
  preferred-location first-touch branch, which can create prefetch migrations
  without calling the policy hook. Native parameter state and BPF attach
  absence are checked before every native run. These are not performance cells.
- Conclusion if the main baseline matches or wins: a match bounds mechanism
  cost; a native win quantifies the generality tax and must be disclosed.
- Information, tuning, and compute fairness: both cells use the same custom
  module file, benchmark, allocator, seed, graph scale, warmup, epochs, GPU, and
  exclusive host. Each atomic process starts after a fresh UVM unload/load, so
  the only module setting difference is native disable versus enabled plus the
  fixed BPF override. No configuration is selected from measured timing.
- Split or leakage rule when relevant: not applicable; the seeded graph and
  training procedure are regenerated identically for both cells.

## Workloads And Metrics

- Real workload: the existing two-layer PyTorch GCN training benchmark with
  8,000,000 nodes, ten edges per node, 128 features, chunked propagation, and
  managed allocation. Retained runs record about 36.1 GB peak allocation on the
  32 GB RTX 5090, so the workload exercises UVM oversubscription.
- Primary metric: per-run median synchronized epoch time in seconds, lower is
  better. The effect is the paired geometric mean of
  `BPF_no_prefetch / native_no_prefetch` across blocks, reported as percent
  overhead with a paired 95% bootstrap confidence interval.
- Correctness check or ground truth: every process must exit zero, report frozen
  parameters and 36 GB-class managed allocation, complete all epochs, and
  produce finite train accuracy. Within each pair, native and BPF accuracy must
  match exactly; a mismatch vetoes the block and routes it to result review.
- Repetitions, seeds, and uncertainty: ten paired blocks, seed 2025, one warmup
  and three measured epochs per process. Order alternates AB/BA, so each cell
  runs first five times. The analysis resamples ten paired log ratios using one
  declared pseudorandom seed; this is statistical reproducibility, not a content
  fingerprint.
- Cost estimate when material: approximately 20 benchmark processes and 20--40
  minutes based on retained 8M-node runs, plus one real preflight.

## Planned Runs

| Run group | Role | Workload | System/method | Repetitions | Decision consequence |
|---|---|---|---|---:|---|
| paired main | baseline | PyTorch GCN 8M UVM | native `uvm_perf_prefetch_enable=0`, no BPF | 10 | Original-policy runtime |
| paired main | proposed | PyTorch GCN 8M UVM | native prefetch enabled plus gpubpf empty-region `BYPASS` | 10 | Same-policy mechanism cost |
| preflight | semantic control | same GCN 8M, one measured epoch | native no-prefetch plus UVM migration monitoring | 1 | Prove the native cell emits no prefetch migrations; timing is excluded |
| preflight | semantic and engagement control | same GCN 8M, one measured epoch | gpubpf no-prefetch plus UVM migration monitoring and temporary external tracing | 1 | Prove the BPF cell emits no prefetch migrations while executing the callback/helper; timing is excluded |

## Execution

- Policy build: `prefetch_none_revision` is listed in `extension/Makefile`; run
  `make -C extension prefetch_none_revision`. Build success and source/object
  inspection must show exactly one prefetch callback, no BPF statistics map, no
  per-fault print, and the empty-region kfunc call. The loader build disables
  linker build IDs because the project forbids hash artifacts in experiment
  workflow paths.
- Semantic-monitor build: run
  `cc -O2 -Wall -Wextra -Werror -std=gnu11 -Wl,--build-id=none
  uvm_migration_monitor.c -o uvm_migration_monitor`. Its compile-time ABI
  assertions fix the 610 V2 queue-control and 72-byte migration-event layouts.
- Module workflow for each native cell: after proving zero compute clients and
  `nvidia_uvm` use count zero, run `sudo rmmod nvidia_uvm`, then
  `sudo insmod /home/yunwei37/workspace/gpu/gpu_ext-kernel-610/kernel-open/nvidia-uvm.ko uvm_perf_prefetch_enable=0`.
  Require parameter value 0, module version 610.43.02, matching running-kernel
  vermagic, and no attached memory struct_ops map.
- Module workflow for each BPF cell: perform the same safe unload, load the same
  module file with `uvm_perf_prefetch_enable=1`, require parameter value 1 and
  memory-hook BTF, then start `sudo extension/prefetch_none_revision` and wait
  for its ready record before the benchmark.
- Authoritative benchmark command: from `workloads/pytorch`, run
  `uv run python configs/gnn.py --nodes 8000000 --uvm --epochs 3 --warmup 1
  --no-cleanup -o <raw-path>` after the cell-specific admission above.
- Real preflight case: run one native and one BPF semantic cell using the final
  binaries and the same 8M command with one warmup and one measured epoch. Start
  `benchmark_gnn_uvm.py` directly with `--wait-for-monitor`; this initializes
  CUDA after selecting the UVM allocator, then pauses before the first benchmark
  allocation. Enumerate its owned `/dev/nvidia-uvm` fds in ascending numeric
  order. CUDA 610 may hold both a primary VA-space fd and an auxiliary MM fd;
  try candidates one at a time with
  `sudo -n ./revision-policy-mechanism/uvm_migration_monitor --pid PID
  --target-fd FD`. Select only the candidate that emits `ready`; an ioctl
  rejection marks that candidate ineligible, while zero or multiple ready
  candidates stop the preflight. Release the benchmark only after exactly one
  ready monitor exists. Root privilege is required because this host has Yama
  `ptrace_scope=1` and the monitor and benchmark are sibling processes; failure
  to duplicate every candidate or obtain exactly one `ready` monitor stops the
  preflight. The exact underlying command is
  `CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 uv run python benchmark_gnn_uvm.py
  --dataset random --nodes 8000000 --edges_per_node 10 --features 128 --hidden
  256 --epochs 1 --warmup 1 --prop chunked --use_uvm --wait-for-monitor
  --report_json <raw-path>`. Require both monitors to finish with
  `prefetch_migrations=0`, `prefetch_bytes=0`, and `dropped_migrations=0`. In the
  BPF cell, temporary `bpftrace` kprobes on
  `uvm_bpf_call_gpu_page_prefetch` and `bpf_gpu_set_prefetch_region` must also
  report nonzero calls. Both workloads must exit zero with identical finite
  accuracy and clean detach. All trace/monitor timing is excluded.
- Full completion rule: all 20 planned processes terminate successfully; every
  pair passes correctness/state checks; policy detach and UVM unload/load are
  clean; no foreign compute process overlaps a run. A row exceeding 15 minutes
  is retained as failed and stops execution for result review rather than being
  silently replaced.
- Raw-result path: `workloads/pytorch/revision-policy-mechanism/results/` with
  one JSON and console log per process, loader logs for BPF cells, a plain-text
  environment/admission log, exact command/order log, and Markdown analysis.
- Checkpoint or recovery approach: each process is one durable row. Resume only
  an unstarted suffix after checking the recorded order and live GPU/module
  state; do not rerun valid rows or create a resume controller.
- Recovery and final state: on any failure, detach only the owned BPF link and
  reload the custom 610 module with prefetch enabled. At experiment end leave
  custom 610 UVM loaded, prefetch enabled, no attached policy, and verify CUDA
  allocation. Never unload the display-owned core NVIDIA modules.

## Interpretation

- Positive result: correctness holds and the paired estimate/interval quantifies
  the concrete cost of the same no-prefetch outcome through gpubpf. A near-zero
  interval supports low-overhead equivalence; a modest cost remains valid.
- Negative or contradictory result: a material slowdown or BPF-only failure
  means the general mechanism has a drawback on this path and must be stated;
  it does not refute benefits from selecting a better gpubpf policy.
- Mixed or inconclusive result: an interval spanning meaningful speedup and more
  than 5% slowdown, nonstationary block effects, or invalid correctness leaves
  mechanism cost unresolved and cannot support a headline claim.
- Target paper figure or table: the expanded policy-versus-mechanism comparison
  associated with Fig. 13 or RQ4, only after paper edits are authorized.

## Reproducibility Notes

- Software and data versions: record Git revisions, package versions, driver and
  kernel versions, ordinary paths/sizes/timestamps, GPU name, and exact argv.
  Do not generate, refresh, compare, or record content hashes, checksums, or
  digests.
- Config and seed notes: seed 2025; 8M nodes; ten edges/node; 128 features; 256
  hidden units; chunk size 2M; one warmup plus three measured epochs.
- Known deviations: the live host uses validated 610 port revision `74a036fe`
  rather than the paper's earlier 575 stack. Both cells use this same module
  file, so the result supports only the mechanism-cost statement on that stack.
