# UVM Basic Results

Final status: `PASS_USERSPACE_UVM_BASIC`.

Evidence class: `SYSTEM_NVIDIA_DRIVER_USERSPACE_UVM` for CUDA/Nsight runs. These are not gpu_ext hook results unless trace CSV files are listed below.

- Parsed runs: 11
- Managed runs: 9
- Device-memory runs: 2
- All recorded non-skipped phases correct: True
- Nsight Unified Memory fault evidence: AVAILABLE_IN_NSYS_CSV
- Nsight evidence files: uvm_basic_20260812T162845Z_phase_cpu_first_touch_um_total_sum_nvtx=cpu_first_touch.csv, uvm_basic_20260812T162845Z_phase_cpu_retouch_um_total_sum_nvtx=cpu_retouch.csv, uvm_basic_20260812T162845Z_phase_explicit_gpu_prefetch_um_total_sum_nvtx=explicit_gpu_prefetch.csv, uvm_basic_20260812T162845Z_phase_kernel_1_demand_um_total_sum_nvtx=kernel_1_demand.csv, uvm_basic_20260812T162845Z_phase_kernel_2_hot_um_total_sum_nvtx=kernel_2_hot.csv, uvm_basic_20260812T162845Z_phase_kernel_3_after_cpu_touch_um_total_sum_nvtx=kernel_3_after_cpu_touch.csv, uvm_basic_20260812T162845Z_phase_kernel_4_after_gpu_prefetch_um_total_sum_nvtx=kernel_4_after_gpu_prefetch.csv, uvm_basic_20260812T162845Z_stats_um_cpu_page_faults_sum.csv, uvm_basic_20260812T162845Z_stats_um_sum.csv, uvm_basic_20260812T162845Z_stats_um_total_sum.csv
- Nsight total HtoD migration: 1319.707 MB
- Nsight total DtoH migration: 535.822 MB
- Nsight total CPU page faults: 3833
- Nsight total GPU page faults: 9600

Timing alone does not prove page residency or migration. Fault and migration claims require the Nsight Unified Memory reports or a compatible gpu_ext trace.

## Run Summary

| Allocation | Bytes/array | CPU retouch | GPU prefetch | K1/K2 | K3/K4 | Correct |
|---|---:|---|---|---:|---:|---|
| managed | 268435456 | none | False | 241.420780 | UNAVAILABLE | True |
| managed | 268435456 | page | False | 245.575107 | UNAVAILABLE | True |
| managed | 268435456 | page | False | 244.353247 | UNAVAILABLE | True |
| managed | 268435456 | none | True | 246.924745 | 1.002066 | True |
| device | 268435456 | none | False | 1.060397 | UNAVAILABLE | True |
| managed | 1073741824 | none | False | 226.050820 | UNAVAILABLE | True |
| managed | 1073741824 | page | False | 227.275298 | UNAVAILABLE | True |
| managed | 1073741824 | page | False | 228.487673 | UNAVAILABLE | True |
| managed | 1073741824 | none | True | 225.192939 | 1.000509 | True |
| device | 1073741824 | none | False | 1.012560 | UNAVAILABLE | True |
| managed | 268435456 | page | True | 278.037839 | 207.490108 | True |

## Managed Demand vs Device Control

| Bytes/array | Managed kernel 1 / device kernel 1 |
|---:|---:|
| 268435456 | 229.084229 |
| 1073741824 | 223.758503 |

## Nsight Phase Evidence

| NVTX phase | HtoD MB | DtoH MB | GPU page faults |
|---|---:|---:|---:|
| cpu_first_touch | 0 | 0 | 0 |
| cpu_retouch | 0 | 535.822 | 0 |
| explicit_gpu_prefetch | 0 | 0 | 0 |
| kernel_1_demand | 782.836 | 0 | 5472 |
| kernel_2_hot | 0 | 0 | 0 |
| kernel_3_after_cpu_touch | 536.871 | 0 | 4128 |
| kernel_4_after_gpu_prefetch | 0 | 0 | 0 |

Nsight repeats the run-wide CPU page-fault total in NVTX-filtered `um_total_sum` reports, so CPU faults are reported only as a run total and are not attributed to individual phases.

## gpu_ext Trace Counts

- Stage 2 status: PASS_GPU_EXT_STAGE2_POLICY_MATRIX
- Stage 2 run manifests: 80
- All trace/policy binaries ready: True
- Custom gpu_ext module loaded during Stage 2: True
- All Stage 2 runs correct: True
- All policy instances detached: True
Detailed aggregated callback and chunk counts are in `stage2_trace_summary.csv` and `STAGE2_GPU_EXT_RESULTS.md`.

## Limitations

- CUDA Event durations include observed execution stalls but do not identify physical residency.
- `cudaMemGetInfo()` is auxiliary capacity information, not per-page residency evidence.
- The default matrix does not oversubscribe GPU memory and therefore is not an eviction experiment.
- No `gpu_block_access` conclusion is used because that hook is known to be unreliable in this branch.

## Conclusions

- CPU first touch followed by `kernel_1_demand` produced HtoD migration and GPU page faults in Nsight, while the immediate `kernel_2_hot` produced neither in this profiled run.
- Page-stride CPU retouch produced DtoH migration; `kernel_3_after_cpu_touch` then produced HtoD migration and GPU page faults again.
- After the third kernel had already restored GPU access, the explicit GPU prefetch and fourth kernel produced no additional UVM migration or GPU faults in this sequence.
- The device-memory control used explicit copies and did not exercise the same managed-memory demand-paging path.
