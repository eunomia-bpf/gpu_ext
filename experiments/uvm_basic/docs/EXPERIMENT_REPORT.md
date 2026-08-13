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
| Stage 1B independent prefetch A/B | PASS | 16/16 correctness runs plus two separate Nsight profiles |
| gpu_ext userspace tools | PASS | Five trace/policy binaries and dependencies verified |
| gpu_ext custom module runtime | PASS | UVM-only switch, four-policy matrix, exact-PID detach, and distribution restore completed |
| Oversubscription | NOT EXECUTED | Deliberately excluded from the default safe experiment |

Overall first-stage status: `PASS_USERSPACE_UVM_BASIC`.

Stage 1B status: `PASS_UVM_PREFETCH_AB`.

Second-stage status: `PASS_GPU_EXT_STAGE2_POLICY_MATRIX`.

## Environment

- Host kernel: `6.15.11-gpuext-gpuext`.
- GPU: NVIDIA A30, 24 GiB.
- NVIDIA driver: 575.57.08 open kernel module.
- CUDA compiler: 12.8.93.
- Nsight Systems: 2024.6.2.
- CMake: 3.22.1.
- CPU page size: obtained at runtime with `sysconf(_SC_PAGESIZE)`.
- Final GPU memory use: 0 MiB.
- Original Stage 1 kernel log Xid count: 0.
- Stage 1B Xid count: unavailable because this unprivileged user cannot read `dmesg` and has restricted/empty journal visibility; no zero-Xid claim is made for that window.

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

Legacy managed-memory execution order, retained for compatibility:

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

Stage 1B adds `--after-retouch demand|prefetch`. Both variants run in separate processes and explicitly prefetch A/B/C to the CPU after the hot kernel. The demand process launches its post-retouch kernel directly; the prefetch process first explicitly migrates A/B/C to the GPU. `--stop-after-hot yes` supplies the exact two-kernel workload used by Stage 2.

## Stage 1B Prefetch A/B

All 16 independent processes passed correctness: five demand and five prefetch runs at 256 MiB per array, followed by three of each at 1 GiB per array.

| Bytes per array | Case | CPU prefetch-to-CPU mean | CPU retouch mean | GPU prefetch mean | Post-retouch kernel mean |
|---:|---|---:|---:|---:|---:|
| 256 MiB | demand | 174.672 ms | 6.813 ms | omitted | 243.066 ms |
| 256 MiB | prefetch | 173.514 ms | 6.694 ms | 34.841 ms | 1.008 ms |
| 1 GiB | demand | 690.965 ms | 27.551 ms | omitted | 911.878 ms |
| 1 GiB | prefetch | 696.315 ms | 27.928 ms | 137.094 ms | 4.052 ms |

Separate 256 MiB Nsight runs provide the migration evidence:

| Case | Run HtoD | Run DtoH | Run GPU faults | Post-retouch kernel HtoD | Post-retouch kernel GPU faults | Explicit-prefetch HtoD |
|---|---:|---:|---:|---:|---:|---:|
| demand | 1610.613 MB | 805.306 MB | 11,950 | 805.306 MB | 6,092 | N/A |
| prefetch | 1610.613 MB | 805.306 MB | 6,091 | 0 MB | 0 | 805.306 MB |

This validates the tested hypothesis: explicit prefetch moved the second HtoD transfer out of the post-retouch kernel interval, and that kernel had no recorded GPU UVM faults. CPU fault counts are run-wide totals (2,304 in each representative run), not phase-attributed values. Full statistics are in [PREFETCH_AB_RESULTS.md](PREFETCH_AB_RESULTS.md).

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
service_fault_batch
  -> service_fault_batch_dispatch
  -> service_fault_batch_block
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

## gpu_ext Stage 2

The following existing extension programs were built from the repository source and passed ELF/dependency checks:

- `prefetch_trace`;
- `chunk_trace`;
- `prefetch_none`;
- `prefetch_always_max`;
- `prefetch_adaptive_sequential`.

The distribution module has `srcversion=182AB87276B2337B4B1A4CD`; the custom module has `srcversion=2A011BD52759A63796A0B00`. A UVM-only switch loaded the custom module, exposed the gpu_ext hook, ran the matrix, and then restored the distribution module. The final loaded srcversion is again `182AB87276B2337B4B1A4CD`, the custom hook is absent, and `nvidia-smi` reports 0 MiB in use.

Stage 2 completed 80 runs: each of `custom_no_policy`, `prefetch_none`, `prefetch_always_max`, and `prefetch_adaptive_sequential` ran 10 timing, 3 trace, and 1 Nsight process at 256 MiB, plus 5 timing and 1 trace process at 1 GiB. All runs returned zero, passed correctness, and detached their policy. The custom no-policy result is not substituted with the distribution-driver baseline.

The UVM-only switch was first supported by static modversion evidence: 322 shared required symbols, 76 shared `nvUvmInterface*` symbols, and 93 shared `nvidia.ko` exports all had zero CRC mismatches. Runtime loading then confirmed that the UVM-only path works with the existing `nvidia.ko`.

At 256 MiB, custom no-policy kernel 1 averaged 240.731 ms versus the one-run system baseline of 240.293 ms (+0.182%). `prefetch_none` averaged 2111.166 ms with 29,820 representative GPU faults; `prefetch_always_max` averaged 78.379 ms with 1,369 faults; `prefetch_adaptive_sequential` averaged 186.242 ms with 7,088 faults. All hot kernels were approximately 1 ms. These are sequential vector-add results, not a general policy ranking.

All 20 trace runs attached successfully. The 256 MiB trace means were 17,742 callbacks for custom no-policy, 393,216 for no-prefetch, 768 for always-max, and 13,831 for adaptive sequential. Callback counts are not page-fault counts. Chunk activate was 384 per representative run and eviction prepare was zero, as expected without oversubscription.

## Safety

- No `make modules_install` was run.
- No module was copied into `/lib/modules`.
- The reviewed UVM-only switch used temporary `rmmod`/`insmod` and restored the distribution module; no permanent install occurred.
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

The custom module switch, trace attach, and system-driver restoration remain guarded actions in `scripts/SAFE_GPU_EXT_COMMANDS.sh`. They must only be used during a GPU maintenance window after reviewing active compute and display users.

## Artifacts

Committed lightweight evidence:

- `results/environment.txt`;
- `results/summary.csv`;
- `results/prefetch_ab_runs.csv` and `results/prefetch_ab_summary.csv`;
- `results/stage2_summary.csv` and `results/stage2_trace_summary.csv`;
- `results/gpu_ext_stage2_preflight.json`;
- this report and the generated `RESULTS.md`.

Large or machine-specific raw artifacts remain local and are intentionally ignored by Git:

- JSONL and per-case logs;
- `.nsys-rep` and SQLite files;
- phase-level Nsight CSV exports;
- build products and extension binaries.

## Stage 3 Runtime (2026-08-12 and 2026-08-13)

Status: `PARTIAL_GPU_EXT_STAGE3_STOPPED_AT_RUNTIME_LIMIT`.

Stage 3 adds two minimal probe points to the matching private custom-module tree:

- `uvm_bpf_trace_gpu_page_prefetch_decision()` after action processing and final region clamp;
- `uvm_bpf_trace_gpu_eviction_selected()` after UVM selects the victim and before eviction starts.

The rebuilt temporary `nvidia-uvm.ko` is version `575.57.08`, has kernel vermagic `6.15.11-gpuext-gpuext`, srcversion `5446825F901EFEAA48651FC`, and SHA256 `c785d8bcf2a953c238140c03ce7d19fc5953c94ac6d159c73e040839effe57a4`. It was loaded temporarily through the reviewed UVM-only switch; no `.ko` was installed. The distribution module was restored at the end with srcversion `182AB87276B2337B4B1A4CD`.

The enhanced `prefetch_trace` now distinguishes callback input, policy output, and final effective region with run-local `call_id`. `chunk_trace` distinguishes eviction preparation from the selected victim and hashes kernel identities before CSV output. The exact schema is in `PREFETCH_TRACE_SCHEMA.md`.

New CUDA modes first passed a distribution-driver smoke test at deliberately small sizes:

- full and page-stride CPU-first-touch-only at 16 MiB;
- `read-a`, `read-b`, `write-c`, and `vector-add` at 16 MiB;
- one 64 MiB A-B-A phase scan without oversubscription.

The enhanced runtime then established:

- callback/final-decision one-to-one correlation and complete candidate/policy/final region semantics;
- `prefetch_none` CPU first-touch slowdown from fine-grained empty-region decisions;
- exact A/B/C migration attribution, disproving the old 620.757 MB `always_max` interpretation;
- a complete, correct 0.95x four-policy matrix with no selected eviction;
- 13,355 selected evictions and 602 proven A-B-A refaulted VA blocks at 1.05x custom no-policy.

The first 1.05x run exposed an auxiliary verifier allocation bug after pressure. The verifier was changed to reserve its fixed 4096-index state before the scan, and the corrected no-policy matrix passed. `prefetch_none` then exceeded the unchanged 300 s limit on its first corrected 1.05x timing run. It was not retried and the timeout was not relaxed.

A later bounded continuation completed 23 additional runs: always-max and adaptive at 1.05x, plus no-policy, always-max, and adaptive timing/trace/Nsight characterization at 1.10x. All 23 returned zero, passed correctness, detached cleanly, and reported no new Xid. At 1.10x, selected-eviction counts were approximately 14.7k and same-block refault counts were approximately 1.2k for all three tested policies. `prefetch_none` remains missing at 1.10x, so the four-policy Stage 3C prerequisite is not complete and Stage 3D was not run. Full details are in `STAGE3_RESULTS.md`.

All policies detached, no Xid or correctness failure occurred in completed runs, GPU memory returned to 0 MiB, and the distribution UVM module was restored after both runtime windows. No GPU setting, full NVIDIA module stack, or permanent system module was changed.
