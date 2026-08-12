# Explicit GPU Prefetch A/B Results

Evidence classes: program timings are `PROGRAM_TIMING`; migration and fault counts are `NSIGHT_UVM`.

Demand and prefetch cases run in separate processes. Both start from a new managed allocation, run CPU first-touch and two kernels, prefetch A/B/C to the CPU, and retouch A/B. Only the explicit-prefetch case migrates A/B/C to the GPU before its measured post-retouch kernel.

## Timing

| Bytes/array | Case | Metric | Count | Mean ms | Median ms | p95 ms | Correct |
|---:|---|---|---:|---:|---:|---:|---|
| 268435456 | demand | cpu_prefetch_to_cpu_ms | 5 | 174.67181639999998 | 173.895299 | 177.321981 | True |
| 268435456 | demand | cpu_retouch_ms | 5 | 6.8126587999999995 | 6.79313 | 7.070807 | True |
| 268435456 | demand | gpu_prefetch_ms | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | True |
| 268435456 | demand | kernel_after_retouch_ms | 5 | 243.0662628 | 242.092026 | 245.178375 | True |
| 268435456 | prefetch | cpu_prefetch_to_cpu_ms | 5 | 173.51369 | 172.939894 | 176.435668 | True |
| 268435456 | prefetch | cpu_retouch_ms | 5 | 6.6942618 | 6.651941 | 6.846351 | True |
| 268435456 | prefetch | gpu_prefetch_ms | 5 | 34.841071799999995 | 34.778405 | 34.966673 | True |
| 268435456 | prefetch | kernel_after_retouch_ms | 5 | 1.0080256 | 1.007616 | 1.009664 | True |
| 1073741824 | demand | cpu_prefetch_to_cpu_ms | 3 | 690.9648903333333 | 691.085993 | 691.127529 | True |
| 1073741824 | demand | cpu_retouch_ms | 3 | 27.550762333333335 | 27.426623 | 28.000073 | True |
| 1073741824 | demand | gpu_prefetch_ms | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | True |
| 1073741824 | demand | kernel_after_retouch_ms | 3 | 911.8784793333334 | 912.270325 | 918.711304 | True |
| 1073741824 | prefetch | cpu_prefetch_to_cpu_ms | 3 | 696.3149943333334 | 696.694055 | 697.490036 | True |
| 1073741824 | prefetch | cpu_retouch_ms | 3 | 27.928430333333335 | 28.164977 | 28.18789 | True |
| 1073741824 | prefetch | gpu_prefetch_ms | 3 | 137.09427333333335 | 137.069186 | 137.210153 | True |
| 1073741824 | prefetch | kernel_after_retouch_ms | 3 | 4.051968 | 4.051968 | 4.052992 | True |

## Nsight

| Case | HtoD total MB | DtoH total MB | GPU faults total | CPU faults run total | Post-retouch kernel HtoD MB | Post-retouch kernel GPU faults | Explicit-prefetch HtoD MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| demand | 1610.613 | 805.306 | 11950 | 2304 | 805.306 | 6092 | UNAVAILABLE |
| prefetch | 1610.613 | 805.306 | 6091 | 2304 | 0 | 0 | 805.306 |

CPU fault counts are run-wide totals because this Nsight export does not safely attribute them to NVTX phases. Timing alone is not residency evidence. The hypothesis that explicit prefetch shifts HtoD migration out of the kernel and reduces GPU faults is accepted only if the Nsight phase data above supports it.
