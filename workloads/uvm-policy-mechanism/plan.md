# Experiment Plan: same no-prefetch policy on a real UVM fault stream

## Research Question

- Paper RQ: **RQ4 (Overhead): What is the overhead of gpubpf's core mechanisms
  and observability capabilities?**
- Specific uncertainty: what end-to-end GPU page-fault service cost is added when
  the driver's existing no-prefetch policy is expressed through gpubpf's live
  page-prefetch callback and checked kfunc instead of its built-in module switch?
- Revision value: this directly answers the Shepherd's request to compare an
  existing policy implemented with the general mechanism against its original
  monolithic implementation. It does not claim a policy improvement.

## Admission And Assets

- Hardware/software: RTX 5090, NVIDIA 610.43.02, the same custom gpubpf UVM
  module file in both cells, and CUDA 12.9.
- Workload: `uvm_fault_stream.cu` allocates 8 GiB of managed memory, initializes
  one deterministic word in each 64 KiB region on the CPU, then times one GPU
  kernel that reads exactly those words. This generates real UVM GPU faults but
  avoids the event-monitor amplification that made the full GCN semantic
  preflight exceed its bound.
- Scope: the CPU initialization touches every expected GPU fault region before
  the pause and timed kernel. Every UVM VA block therefore has CPU residency;
  the claim is explicitly limited to the CPU-resident, non-first-touch fault
  path. It does not claim equivalence for the driver's preferred-location
  first-touch branch, which executes before the gpubpf callback.
- Correctness: the GPU writes one observed value per region. After timing, the
  host compares all 131,072 values with their exact expected values and requires
  zero mismatches. This is numerical validation, not a file/content fingerprint.
- Fixed policy artifact: reuse `extension/prefetch_none_revision.bpf.c` and its
  ownership-safe loader. The timed BPF program contains no map and no print.
- Untimed semantic artifact: reuse
  `workloads/pytorch/revision-policy-mechanism/uvm_migration_monitor`.

## Comparison

- Native cell: reload the custom 610 UVM module with
  `uvm_perf_prefetch_enable=0`; attach no struct_ops policy.
- gpubpf cell: reload the same module file with
  `uvm_perf_prefetch_enable=1`; attach the fixed empty-region `BYPASS` policy.
- Controlled dimensions: exact binary, allocation size, region stride, CPU
  initialization, GPU kernel, device, module file, and exclusive host.
- Mechanism difference: the native cell exits through the built-in switch; the
  gpubpf cell runs prefetch analysis, the struct_ops callback, verifier-approved
  BPF instructions, and `bpf_gpu_set_prefetch_region`, then returns the same
  no-prefetch decision on the scoped non-first-touch path.

## Metrics And Runs

- Primary metric: CUDA-event kernel time in milliseconds. Report the paired
  geometric mean of `gpubpf/native` and percent overhead with a paired 95%
  bootstrap confidence interval.
- Secondary descriptive metric: 8 GiB fault-span divided by kernel time. This is
  fault-span rate, not payload bandwidth, because only one word per 64 KiB region
  is read.
- Correctness veto: zero exit, finite positive kernel time, exactly 8 GiB,
  64 KiB regions, 131,072 regions, and zero mismatches in every retained row.
- Repetitions: 15 paired blocks, alternating AB/BA so each cell runs first in
  either seven or eight blocks. No timing-based tuning or row replacement.
- Timeout: five minutes per process. A timeout is retained as failure and stops
  the experiment for result review.

| Group | Role | Cell | Processes | Decision consequence |
|---|---|---|---:|---|
| paired main | original mechanism | native no-prefetch | 15 | Built-in policy cost |
| paired main | general mechanism | gpubpf no-prefetch | 15 | Same-policy mechanism cost |
| semantic preflight | untimed control | native and gpubpf | 2 | Prove zero actual prefetch migrations in both cells |

## Execution And Gates

- Build with `nvcc -O3 -std=c++17 -Xlinker --build-id=none
  uvm_fault_stream.cu -o uvm_fault_stream`. Do not inspect or record a build
  fingerprint.
- Before every cell, require no compute client and `nvidia_uvm` refcount zero.
  Never unload display-owned core NVIDIA modules.
- Native reload: unload only `nvidia_uvm`, load the fixed custom 610 module with
  prefetch parameter 0, require parameter 0 and no attached memory struct_ops.
- gpubpf reload: unload only `nvidia_uvm`, load that same file with parameter 1,
  start `sudo extension/prefetch_none_revision`, and require its ready record.
- Atomic timed command: `./uvm_fault_stream --gib 8 --region-kib 64 --output
  <row.json>`. No monitor or tracer runs during retained timing.
- One two-cell real semantic preflight uses `--wait-for-monitor`. Enumerate the
  paused process's `/dev/nvidia-uvm` fds and start the monitor with `sudo -n` on
  each candidate until exactly one emits `ready`; auxiliary candidates may
  reject tracker initialization. Release the workload only after unique ready.
- Both preflight cells must finish with nonzero total migrations and migrated
  bytes, zero prefetch migrations and bytes, zero dropped migration events,
  zero mismatches, and clean exit. The gpubpf cell additionally runs temporary
  external kprobes, started only after the workload is paused and its migration
  monitor is ready, so CPU initialization is excluded. Require
  `wrapper_calls == helper_calls` and both counts at least 131,072 for
  `uvm_bpf_call_gpu_page_prefetch` and `bpf_gpu_set_prefetch_region`, covering
  all unique demanded regions. Record the actual counts; extra calls from
  speculative hardware faults or VA-block retries do not invalidate the row.
- After every gpubpf process, send SIGINT only to the owned loader, require its
  detaching record and zero exit, confirm no memory struct_ops remains attached,
  and require `nvidia_uvm` refcount zero before the next AB/BA module reload.
- Recovery: stop only owned monitor/tracer/loader processes; reload the custom
  UVM module with prefetch 1; leave no BPF policy attached; verify a small CUDA
  allocation after the experiment.

## Interpretation

- A ratio near one with a narrow interval bounds the mechanism tax for this
  fault-intensive policy. A slowdown quantifies a real generality drawback and
  must be disclosed. A speedup is reported without attributing it to policy,
  because the observable no-prefetch decision is held constant.
- Any nonzero prefetch migration, dropped event, callback/helper non-engagement,
  correctness mismatch, timeout, or unstable paired trend makes the result
  inadmissible for a paper claim.
- Scope: this isolates the page-prefetch mechanism on a real UVM fault stream;
  it does not substitute for application-level policy-benefit experiments.

## Reproducibility

- Record Git revision, kernel/driver/CUDA versions, ordinary paths, sizes,
  timestamps, exact argv, module parameter, attach identifiers, cell order, and
  raw JSON rows.
- Do not generate, refresh, compare, or record file/content hashes, checksums,
  or digests anywhere in the workflow.
