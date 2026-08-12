# GPU UVM Basic Experiment Report

Date: 2026-08-12 UTC

## Scope

This experiment isolates NVIDIA Unified Virtual Memory behavior with one CUDA vector-add kernel:

```text
C[i] = A[i] + B[i]
```

It does not use PyTorch, an LLM workload, GPU scheduling changes, memory oversubscription, or a custom kernel module in the completed first stage. The measured CUDA/Nsight evidence class is `SYSTEM_NVIDIA_DRIVER_USERSPACE_UVM`; it is not a gpu_ext policy result.

## Status

| Area | Status | Evidence |
|---|---|---|
| CUDA managed/device implementation | PASS | Both allocation modes compiled and completed correctly |
| 256 MiB matrix | PASS | Five configurations, all correctness checks passed |
| 1 GiB matrix | PASS | Five configurations, all correctness checks passed |
| Nsight Unified Memory observation | PASS | CPU/GPU fault collection and phase-filtered migration reports available |
| gpu_ext userspace tools | PASS | Five trace/policy binaries and dependencies verified |
| gpu_ext custom module runtime | NOT EXECUTED | Distribution `nvidia_uvm` remains loaded; custom hook is not visible |
| Oversubscription | NOT EXECUTED | Deliberately excluded from the default safe experiment |

Overall first-stage status: `PASS_USERSPACE_UVM_BASIC`.

Second-stage status: `READY_FOR_MANUAL_MODULE_SWITCH`.

## Environment

- Host kernel: `6.15.11-gpuext-gpuext`.
- GPU: NVIDIA A30, 24 GiB.
- NVIDIA driver: 575.57.08 open kernel module.
- CUDA compiler: 12.8.93.
- Nsight Systems: 2024.6.2.
- CMake: 3.22.1.
- CPU page size: obtained at runtime with `sysconf(_SC_PAGESIZE)`.
- Final GPU memory use: 0 MiB.
- Kernel log Xid count during the experiment window: 0.

The custom `nvidia-uvm.ko` has matching driver version and kernel vermagic, but its `srcversion` differs from the loaded distribution module. It was not loaded automatically.

## Implementation

The CUDA program supports:

- `--bytes` with K, M, and G suffixes;
- `managed` and `device` allocation modes;
- no, page-stride, or full CPU retouch;
- optional CPU prefetch before retouch;
- optional explicit GPU prefetch;
- deterministic GPU-side sampled correctness validation;
- JSONL output and compact terminal tables.

Managed-memory execution order:

```text
cudaMallocManaged
  -> CPU first touch
  -> kernel_1_demand
  -> kernel_2_hot
  -> optional CPU prefetch
  -> CPU retouch
  -> kernel_3_after_cpu_touch
  -> optional GPU prefetch
  -> kernel_4_after_gpu_prefetch
```

The device-memory control uses `cudaMalloc`, explicit HtoD copies, two kernels, and an explicit DtoH copy. All CUDA calls, including cleanup paths, check return values.

The default runner rejects a per-array size above 1 GiB and requires the three arrays to fit within 20% of currently free GPU memory. Oversubscription is guarded by a separate opt-in environment variable and was not run.

## Runtime Matrix

The latest complete matrix contains ten runs: five configurations at 256 MiB per array and the same five at 1 GiB per array.

Representative final timings:

| Bytes per array | Case | First kernel | Second/hot kernel | After CPU page retouch |
|---:|---|---:|---:|---:|
| 256 MiB | Managed demand | 240.293 ms | 0.995 ms | N/A |
| 256 MiB | Managed page retouch | 244.428 ms | 0.995 ms | 177.140 ms |
| 256 MiB | Device control | 1.049 ms | 0.989 ms | N/A |
| 1 GiB | Managed demand | 910.858 ms | 4.029 ms | N/A |
| 1 GiB | Managed page retouch | 915.559 ms | 4.028 ms | 659.634 ms |
| 1 GiB | Device control | 4.071 ms | 4.020 ms | N/A |

These timings are observations, not page-residency proof. The migration and fault conclusions below come from Nsight Systems.

## Nsight UVM Evidence

The canonical profiled run used 256 MiB per array, page-stride CPU retouch, and explicit GPU prefetch after the third kernel.

Run totals:

- HtoD migration: 1319.707 MB.
- DtoH migration: 535.822 MB.
- CPU page faults: 3,833.
- GPU page faults: 9,600.

Phase-filtered evidence:

| NVTX phase | HtoD migration | DtoH migration | GPU page faults |
|---|---:|---:|---:|
| `cpu_first_touch` | 0 MB | 0 MB | 0 |
| `kernel_1_demand` | 782.836 MB | 0 MB | 5,472 |
| `kernel_2_hot` | 0 MB | 0 MB | 0 |
| `cpu_retouch` | 0 MB | 535.822 MB | 0 |
| `kernel_3_after_cpu_touch` | 536.871 MB | 0 MB | 4,128 |
| `explicit_gpu_prefetch` | 0 MB | 0 MB | 0 |
| `kernel_4_after_gpu_prefetch` | 0 MB | 0 MB | 0 |

The immediate second kernel had no observed GPU UVM faults. CPU retouch produced DtoH migration, and the following third kernel produced HtoD migration and GPU faults again. In the required sequence, the third kernel had already restored GPU access before explicit GPU prefetch, so the prefetch and fourth kernel produced no additional migration or faults.

Nsight repeats the run-wide CPU fault total in each NVTX-filtered `um_total_sum` report. CPU faults are therefore retained as a run total rather than incorrectly attributed to individual phases.

## Driver Call Path

The matching 575.57.08 source was inspected directly. The relevant chain is:

```text
service_fault_batch_dispatch
  -> service_fault_batch
  -> service_fault_batch_block_locked
  -> uvm_va_block_service_locked
  -> uvm_va_block_get_prefetch_hint
  -> uvm_perf_prefetch_get_hint_va_block
  -> uvm_perf_prefetch_prenotify_fault_migrations
  -> compute_prefetch_mask
  -> compute_prefetch_region
  -> uvm_bpf_call_gpu_page_prefetch
  -> gpu_mem_ops.gpu_page_prefetch
```

Exact source paths and line references are recorded in [CALL_PATH.md](CALL_PATH.md). The experiment does not depend on `gpu_block_access`, and the selected prefetch-only policies do not call `bpf_gpu_block_move_head()`.

## gpu_ext Stage 2 Preflight

The following existing extension programs were built from the repository source and passed ELF/dependency checks:

- `prefetch_trace`;
- `chunk_trace`;
- `prefetch_none`;
- `prefetch_always_max`;
- `prefetch_adaptive_sequential`.

The loaded module is `/lib/modules/6.15.11-gpuext-gpuext/updates/dkms/nvidia-uvm.ko`, with `srcversion=182AB87276B2337B4B1A4CD`. The custom module has `srcversion=2A011BD52759A63796A0B00`. The custom prefetch hook is not visible in `/proc/kallsyms`, so the trace runner correctly refused to attach.

No BPF program, struct_ops policy, or custom module was loaded during this work.

## Safety

- No `make modules_install` was run.
- No module was copied into `/lib/modules`.
- No `sudo`, `rmmod`, `insmod`, or `modprobe` command was executed.
- No system NVIDIA driver setting, MIG mode, clock, or power limit was changed.
- `SAFE_GPU_EXT_COMMANDS.sh` remains non-executable and defaults to inspection only.
- Trace policy cleanup uses exact child PIDs and a shell trap.
- Only one struct_ops policy may be attached by the trace runner.
- No eviction policy or `bpf_gpu_block_move_head()` is used in the initial policy matrix.

## Reproduction

First stage:

```bash
cd /home/peng/workspace/gpu_ext/experiments/uvm_basic
bash scripts/collect_environment.sh
bash scripts/run_matrix.sh
bash scripts/profile_nsys.sh
python3 analysis/summarize.py --experiment-dir .
```

Non-privileged Stage 2 preflight:

```bash
bash scripts/check_gpu_ext_stage2.sh
```

The custom module switch, trace attach, and system-driver restoration remain explicit manual actions in `scripts/SAFE_GPU_EXT_COMMANDS.sh`. They must only be used during a GPU maintenance window after reviewing active compute and display users.

## Artifacts

Committed lightweight evidence:

- `results/environment.txt`;
- `results/summary.csv`;
- `results/gpu_ext_stage2_preflight.json`;
- this report and the generated `RESULTS.md`.

Large or machine-specific raw artifacts remain local and are intentionally ignored by Git:

- JSONL and per-case logs;
- `.nsys-rep` and SQLite files;
- phase-level Nsight CSV exports;
- build products and extension binaries.

## Remaining Work

The only next step in this experiment is the manually authorized custom-module tracing matrix: baseline, `prefetch_none`, `prefetch_always_max`, and `prefetch_adaptive_sequential`. Oversubscription should remain separate and disabled until the non-oversubscribed trace matrix is stable and detached cleanly.
