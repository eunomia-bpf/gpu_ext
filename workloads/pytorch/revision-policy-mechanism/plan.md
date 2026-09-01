# Experiment Plan: RQ4 native LRU versus gpubpf LRU

## Research Question

- RQ exactly as written in the paper: **RQ4 (Overhead): What is the overhead
  of gpubpf's core mechanisms and observability capabilities?**
- Specific uncertainty tested here: when the driver's existing LRU refresh
  policy is implemented through the `gpu_block_access` struct_ops interface,
  does the general gpubpf mechanism materially change end-to-end performance
  relative to the original in-driver implementation?
- Why the answer matters: the shepherd explicitly asks for a matched comparison
  between an existing policy expressed through gpubpf and its original
  monolithic implementation. A policy sweep cannot answer this because it
  changes both mechanism and policy.

## Paper-Value Admission

- Planned role: decisive evidence for the policy-versus-mechanism clarification
  and supporting evidence for RQ4.
- Largest credible paper story this experiment could unlock: gpubpf can express
  an existing driver policy with a directly measured, disclosed mechanism cost;
  matching the native path is a valuable result and does not need to outperform
  it.
- Strongest reviewer reject argument or load-bearing uncertainty addressed: the
  paper attributes improvements to gpubpf without showing whether its general
  mechanism penalizes an unchanged policy.
- Independent evidence added beyond existing runs and published results:
  historical no-policy hook overhead does not execute a BPF policy callback,
  while historical LFU/FIFO/MRU sweeps change policy and used one trial. This
  experiment holds the LRU transition constant and executes the callback.
- Why the result is not tautological, already settled, or dominated: the gpubpf
  path adds RCU dispatch, BPF execution, a checked kfunc call, and a return to
  the driver while holding the eviction-list spinlock. These costs can be
  visible on an access-heavy oversubscribed workload.
- Paper decision if positive: report the matched overhead with uncertainty and
  explicitly attribute any larger memory-management gains elsewhere to policy,
  not to the mechanism alone.
- Paper decision if contradictory, mixed, or inconclusive: disclose the measured
  drawback and scope it to the tested memory hook/workload; do not claim that
  existing policies are reproduced for free.
- Best alternative experiment and why this one has higher decision value: a
  fresh Fig. 13 memory-policy sweep measures whether different policies help,
  but cannot isolate the cost of expressing the same policy. Scheduling-side
  equivalence currently requires the supported 575 driver maintenance window.

## Expected And Alternative Outcomes

- Current expected answer: BPF LRU preserves correctness and stays within a few
  percent of native LRU, but the estimate and interval, rather than a preset
  threshold, determine the reported cost.
- Strongest competing explanation: callback and helper execution inside the
  locked hot path adds a material slowdown even though the list transition is
  identical.
- Result that would contradict the expectation: a paired estimate whose 95%
  interval excludes zero and shows more than 5% slowdown, or any repeatable
  correctness or stability failure unique to BPF LRU.

## Published Precedent And Real Assets

- Closest published protocol: the paper's existing RQ4 host-runtime experiment
  measures synchronized application runtime under oversubscription; the
  repository's GCN benchmark measures synchronized training epoch time and is
  already used by the paper's memory-policy evaluation.
- Official system/model/data/benchmark/tool and version: NVIDIA Open GPU Kernel
  Modules 610.43.02 with the in-tree gpubpf UVM hooks; PyTorch 2.9.0; the
  repository's `benchmark_gnn_uvm.py`; RTX 5090.
- What is reused: the benchmark's seeded random graph, managed-memory allocator,
  synchronized epoch timing, training accuracy, and atomic `configs/gnn.py`
  command.
- Necessary deviations or custom glue: one minimal gpubpf LRU policy implements
  the driver's existing access transition by calling
  `bpf_gpu_block_move_tail(chunk, list)` and returning `BYPASS`. Its loader only
  attaches, reports ordinary program/map identifiers, waits for termination,
  and detaches. No experiment controller or content fingerprint is added.

## Comparison

- Proposed system or method: gpubpf LRU, which performs the same move-to-tail
  transition in a `gpu_block_access` callback and suppresses the duplicate
  in-driver transition.
- Main baseline and the competing position it represents: native UVM LRU on the
  same gpubpf-enabled module with no struct_ops policy attached. It represents
  the original monolithic implementation and is the only main baseline needed
  for this mechanism-cost question.
- Why the main baseline needs a matched run instead of citation alone: no
  publication reports the cost of this repository's callback/helper path on
  this RTX 5090, driver, kernel, and workload.
- Controls or ablations, labeled separately: one untimed engagement preflight
  traces calls to `bpf_gpu_block_move_tail` and
  `uvm_bpf_call_gpu_block_access`; it is not a performance cell. No-policy
  attach absence is checked before every native run.
- Conclusion if the main baseline matches or wins: a match bounds the mechanism
  cost; a native win quantifies the generality tax and must be disclosed.
- Information, tuning, and compute fairness: both cells use the identical
  module, benchmark, allocator, seed, graph scale, warmup, epoch count, GPU,
  clocks, and exclusive host. The only changed state is attachment of the fixed
  LRU struct_ops policy. No configuration is selected from measured timing.
- Split or leakage rule when relevant: not applicable; the seeded graph and
  training procedure are regenerated identically for both cells.

## Workloads And Metrics

- Real workload: the existing two-layer PyTorch GCN training benchmark with
  8,000,000 nodes, ten edges per node, 128 features, chunked propagation, and
  managed allocation. Historical runs record about 36.1 GB peak allocation on
  the 32 GB RTX 5090, so the workload exercises UVM oversubscription.
- Primary metric: per-run median synchronized epoch time in seconds, lower is
  better. The claim-matched effect is the paired geometric mean of
  `BPF_LRU / native_LRU` across blocks, reported as percent overhead with a
  paired 95% bootstrap confidence interval.
- Correctness check or ground truth: every process must exit zero, report the
  frozen parameters and 36 GB-class managed allocation, complete all epochs,
  and produce finite train accuracy. Within each paired block, native and BPF
  train accuracy must match exactly; any mismatch vetoes that block and triggers
  result review rather than silent replacement.
- Repetitions, seeds, and uncertainty: ten paired blocks, fixed seed 2025, one
  warmup epoch and three measured epochs per process. Order alternates AB/BA so
  each cell runs first five times. Bootstrap resamples the ten paired log ratios
  with a fixed, declared analysis seed only for interval reproducibility.
- Cost estimate when material: approximately 20 benchmark processes and 15--25
  minutes based on retained 8M-node runs, plus one real preflight.

## Planned Runs

| Run group | Role | Workload | System/method | Repetitions | Decision consequence |
|---|---|---|---|---:|---|
| paired main | baseline | PyTorch GCN 8M UVM | native in-driver LRU, no attached struct_ops | 10 | Establish original-policy runtime |
| paired main | proposed | PyTorch GCN 8M UVM | gpubpf move-to-tail LRU | 10 | Measure same-policy mechanism cost |
| preflight | engagement control | PyTorch GCN 8M UVM, one measured epoch | gpubpf LRU plus temporary external tracing | 1 | Prove attach and live callback/helper execution; never time this trace |

## Execution

- Authoritative command or workflow: from `workloads/pytorch`, run
  `uv run python configs/gnn.py --nodes 8000000 --uvm --epochs 3 --warmup 1
  --no-cleanup -o <raw-path>` after a read-only exclusive-GPU admission check.
  For the BPF cell, start the fixed loader before the same command and terminate
  it after the result is durable. For native, prove no memory struct_ops map is
  attached.
- Real preflight case: attach the final gpubpf LRU binary, start temporary
  external tracing of the wrapper and move-to-tail helper, run the same 8M
  command with one warmup and one measured epoch, then require nonzero counts,
  zero exit, finite accuracy, and clean detach. Trace overhead disqualifies its
  timing from analysis.
- Full completion rule: all 20 planned processes reach terminal success; every
  paired block passes correctness and state checks; policy attach/detach is
  clean; no foreign compute process overlaps a run. Failed or invalid rows are
  retained and sent to result review, not automatically replaced.
- Raw-result path: `workloads/pytorch/revision-policy-mechanism/results/` with
  one JSON and console log per process, BPF loader logs for BPF cells, a plain
  text environment/admission record, and a Markdown analysis after completion.
- Checkpoint or recovery approach: each process is an atomic durable row. Resume
  only the unstarted suffix after checking the recorded order and current GPU
  state; do not rerun completed valid rows or create a resume controller.

## Interpretation

- Positive result: correctness holds and the paired estimate/interval shows the
  concrete cost of the same LRU policy through gpubpf; a near-zero interval
  supports low-overhead equivalence, while a modest positive cost is still a
  valid mechanism result.
- Negative or contradictory result: a material slowdown or BPF-only failure
  means the general mechanism has a drawback on this path and must be stated;
  it does not by itself refute benefits from different gpubpf policies.
- Mixed or inconclusive result: an interval spanning both meaningful speedup
  and more than 5% slowdown, nonstationary block effects, or invalid correctness
  leaves the mechanism cost unresolved and cannot support a headline claim.
- Target paper figure or table: an expanded policy-versus-mechanism comparison
  associated with Fig. 13 or the RQ4 overhead table, after the user authorizes
  paper edits.

## Reproducibility Notes

- Software and data versions: record Git revisions, package versions, driver and
  kernel versions, ordinary paths/sizes/timestamps, GPU name, and exact argv.
  Do not generate, refresh, compare, or record content hashes, checksums, or
  digests.
- Config and seed notes: seed 2025; 8M nodes; 10 edges/node; 128 features; 256
  hidden units; chunk size 2M; one warmup plus three measured epochs.
- Known deviations: the live host uses the validated 610 port rather than the
  paper's earlier 575 stack. This experiment compares both cells within the
  same 610 stack and supports only the mechanism-cost statement on that stack.
