# gpu_ext Stage 2 Results

Status: `PASS_GPU_EXT_STAGE2_POLICY_MATRIX`.

The system-driver baseline and custom-driver no-policy baseline are separate evidence classes and are not substituted for one another.

## Acceptance

- 80/80 runs returned zero and passed correctness.
- Four policies each completed 10 timing, 3 trace, and 1 Nsight run at 256 MiB, plus 5 timing and 1 trace run at 1 GiB.
- Every policy instance detached; no run added an NVIDIA Xid.
- The distribution `nvidia_uvm` was restored after the matrix.

## Key findings

- Custom no-policy 256 MiB kernel 1 mean: 240.731 ms; system-driver baseline: 240.293 ms; difference: +0.182%.
- `prefetch_none` kernel 1 mean: 2111.166 ms; representative GPU faults: 29820; trace callbacks: 393216.
- `prefetch_always_max` kernel 1 mean: 78.379 ms; representative GPU faults: 1369; trace callbacks: 768.
- `prefetch_adaptive_sequential` kernel 1 mean: 186.242 ms; representative GPU faults: 7088; trace callbacks: 13831.
- All four 256 MiB hot-kernel means are approximately 1 ms and all 1 GiB hot-kernel means are approximately 4 ms, close to the device-memory kernel controls.
- `always_max` is best for this sequential, non-oversubscribed vector-add case only; the result does not establish a generally best policy.

## Timing

| Policy | Size | Metric | Count | Mean | Median | p95 | Correct runs |
|---|---|---|---:|---:|---:|---:|---:|
| custom_no_policy | 1G | allocation_ms | 5 | 0.2348102 | 0.236079 | 0.246709 | 6/6 |
| custom_no_policy | 1G | cpu_first_touch_ms | 5 | 1363.5614902 | 1361.475446 | 1376.194195 | 6/6 |
| custom_no_policy | 1G | kernel_1_demand_ms | 5 | 910.8123535999999 | 910.266357 | 913.445862 | 6/6 |
| custom_no_policy | 1G | kernel_1_over_kernel_2 | 5 | 226.21191749160403 | 226.13380545861423 | 226.8082823822464 | 6/6 |
| custom_no_policy | 1G | kernel_2_hot_ms | 5 | 4.026368 | 4.026368 | 4.027392 | 6/6 |
| custom_no_policy | 256M | allocation_ms | 10 | 0.23466510000000002 | 0.235172 | 0.243513 | 14/14 |
| custom_no_policy | 256M | cpu_first_touch_ms | 10 | 337.4048354 | 336.998188 | 340.706438 | 14/14 |
| custom_no_policy | 256M | kernel_1_demand_ms | 10 | 240.7309311 | 240.515587 | 244.564987 | 14/14 |
| custom_no_policy | 256M | kernel_1_over_kernel_2 | 10 | 242.0631396011689 | 242.06498185887676 | 246.47367917150413 | 14/14 |
| custom_no_policy | 256M | kernel_2_hot_ms | 10 | 0.9945088 | 0.994304 | 0.997376 | 14/14 |
| prefetch_adaptive_sequential | 1G | allocation_ms | 5 | 0.23950299999999997 | 0.237391 | 0.248913 | 6/6 |
| prefetch_adaptive_sequential | 1G | cpu_first_touch_ms | 5 | 1332.0419564 | 1327.516299 | 1354.845317 | 6/6 |
| prefetch_adaptive_sequential | 1G | kernel_1_demand_ms | 5 | 634.3243774 | 647.125 | 654.848999 | 6/6 |
| prefetch_adaptive_sequential | 1G | kernel_1_over_kernel_2 | 5 | 157.5242231511989 | 160.5584369442327 | 162.72289455112914 | 6/6 |
| prefetch_adaptive_sequential | 1G | kernel_2_hot_ms | 5 | 4.0267776 | 4.025344 | 4.031488 | 6/6 |
| prefetch_adaptive_sequential | 256M | allocation_ms | 10 | 0.2375534 | 0.237311 | 0.256207 | 14/14 |
| prefetch_adaptive_sequential | 256M | cpu_first_touch_ms | 10 | 330.4737376 | 329.716683 | 336.646926 | 14/14 |
| prefetch_adaptive_sequential | 256M | kernel_1_demand_ms | 10 | 186.2423555 | 187.3336335 | 189.865982 | 14/14 |
| prefetch_adaptive_sequential | 256M | kernel_1_over_kernel_2 | 10 | 187.0794701749119 | 187.7504080212484 | 190.75719963670267 | 14/14 |
| prefetch_adaptive_sequential | 256M | kernel_2_hot_ms | 10 | 0.9955328 | 0.995328 | 0.999424 | 14/14 |
| prefetch_always_max | 1G | allocation_ms | 5 | 0.236724 | 0.233644 | 0.260495 | 6/6 |
| prefetch_always_max | 1G | cpu_first_touch_ms | 5 | 1272.2327194000002 | 1270.974527 | 1280.962563 | 6/6 |
| prefetch_always_max | 1G | kernel_1_demand_ms | 5 | 302.10539559999995 | 302.329865 | 304.243713 | 6/6 |
| prefetch_always_max | 1G | kernel_1_over_kernel_2 | 5 | 75.03182176681477 | 75.03024365668678 | 75.54360563858697 | 6/6 |
| prefetch_always_max | 1G | kernel_2_hot_ms | 5 | 4.026368 | 4.027392 | 4.02944 | 6/6 |
| prefetch_always_max | 256M | allocation_ms | 10 | 0.2334927 | 0.23232150000000001 | 0.250346 | 14/14 |
| prefetch_always_max | 256M | cpu_first_touch_ms | 10 | 317.4614788 | 317.0372085 | 320.858922 | 14/14 |
| prefetch_always_max | 256M | kernel_1_demand_ms | 10 | 78.3790078 | 78.1803515 | 79.203331 | 14/14 |
| prefetch_always_max | 256M | kernel_1_over_kernel_2 | 10 | 79.02417427098199 | 78.96856730016034 | 79.95134777756212 | 14/14 |
| prefetch_always_max | 256M | kernel_2_hot_ms | 10 | 0.9918464 | 0.991232 | 0.997376 | 14/14 |
| prefetch_none | 1G | allocation_ms | 5 | 0.24591539999999998 | 0.243633 | 0.268781 | 6/6 |
| prefetch_none | 1G | cpu_first_touch_ms | 5 | 8140.0981384 | 8142.052314 | 8176.373325 | 6/6 |
| prefetch_none | 1G | kernel_1_demand_ms | 5 | 9845.1257812 | 9704.108398 | 10497.893555 | 6/6 |
| prefetch_none | 1G | kernel_1_over_kernel_2 | 5 | 2440.35949632562 | 2404.0254585037733 | 2604.6364773385894 | 6/6 |
| prefetch_none | 1G | kernel_2_hot_ms | 5 | 4.0343552 | 4.035584 | 4.036608 | 6/6 |
| prefetch_none | 256M | allocation_ms | 10 | 0.2384531 | 0.238163 | 0.24776 | 14/14 |
| prefetch_none | 256M | cpu_first_touch_ms | 10 | 2025.3553304000002 | 2026.917589 | 2032.995364 | 14/14 |
| prefetch_none | 256M | kernel_1_demand_ms | 10 | 2111.1657469 | 2109.295654 | 2144.414795 | 14/14 |
| prefetch_none | 256M | kernel_1_over_kernel_2 | 10 | 2116.0630136404443 | 2112.6421623164824 | 2147.851357171475 | 14/14 |
| prefetch_none | 256M | kernel_2_hot_ms | 10 | 0.9976832 | 0.997376 | 0.999424 | 14/14 |
| custom_no_policy | 1G | chunk_activate_count | 1 | 1536.0 | 1536.0 | 1536.0 | 6/6 |
| custom_no_policy | 1G | eviction_prepare_count | 1 | 0.0 | 0.0 | 0.0 | 6/6 |
| custom_no_policy | 1G | max_region_pages_mean | 1 | 512.0 | 512.0 | 512.0 | 6/6 |
| custom_no_policy | 1G | prefetch_callback_count | 1 | 57562.0 | 57562.0 | 57562.0 | 6/6 |
| custom_no_policy | 256M | chunk_activate_count | 4 | 384.0 | 384.0 | 384.0 | 14/14 |
| custom_no_policy | 256M | eviction_prepare_count | 4 | 0.0 | 0.0 | 0.0 | 14/14 |
| custom_no_policy | 256M | gpu_fault_count | 1 | 5742.0 | 5742.0 | 5742.0 | 14/14 |
| custom_no_policy | 256M | h2d_migration_mb | 1 | 805.306 | 805.306 | 805.306 | 14/14 |
| custom_no_policy | 256M | kernel_1_demand_gpu_fault_count | 1 | 5742.0 | 5742.0 | 5742.0 | 14/14 |
| custom_no_policy | 256M | kernel_1_demand_h2d_migration_mb | 1 | 805.306 | 805.306 | 805.306 | 14/14 |
| custom_no_policy | 256M | kernel_2_hot_gpu_fault_count | 1 | 0.0 | 0.0 | 0.0 | 14/14 |
| custom_no_policy | 256M | kernel_2_hot_h2d_migration_mb | 1 | 0.0 | 0.0 | 0.0 | 14/14 |
| custom_no_policy | 256M | max_region_pages_mean | 4 | 512.0 | 512.0 | 512.0 | 14/14 |
| custom_no_policy | 256M | prefetch_callback_count | 4 | 17742.25 | 17304.0 | 19225.0 | 14/14 |
| prefetch_adaptive_sequential | 1G | chunk_activate_count | 1 | 1536.0 | 1536.0 | 1536.0 | 6/6 |
| prefetch_adaptive_sequential | 1G | eviction_prepare_count | 1 | 0.0 | 0.0 | 0.0 | 6/6 |
| prefetch_adaptive_sequential | 1G | max_region_pages_mean | 1 | 512.0 | 512.0 | 512.0 | 6/6 |
| prefetch_adaptive_sequential | 1G | prefetch_callback_count | 1 | 49145.0 | 49145.0 | 49145.0 | 6/6 |
| prefetch_adaptive_sequential | 256M | chunk_activate_count | 4 | 384.0 | 384.0 | 384.0 | 14/14 |
| prefetch_adaptive_sequential | 256M | eviction_prepare_count | 4 | 0.0 | 0.0 | 0.0 | 14/14 |
| prefetch_adaptive_sequential | 256M | gpu_fault_count | 1 | 7088.0 | 7088.0 | 7088.0 | 14/14 |
| prefetch_adaptive_sequential | 256M | h2d_migration_mb | 1 | 805.306 | 805.306 | 805.306 | 14/14 |
| prefetch_adaptive_sequential | 256M | kernel_1_demand_gpu_fault_count | 1 | 7088.0 | 7088.0 | 7088.0 | 14/14 |
| prefetch_adaptive_sequential | 256M | kernel_1_demand_h2d_migration_mb | 1 | 805.306 | 805.306 | 805.306 | 14/14 |
| prefetch_adaptive_sequential | 256M | kernel_2_hot_gpu_fault_count | 1 | 0.0 | 0.0 | 0.0 | 14/14 |
| prefetch_adaptive_sequential | 256M | kernel_2_hot_h2d_migration_mb | 1 | 0.0 | 0.0 | 0.0 | 14/14 |
| prefetch_adaptive_sequential | 256M | max_region_pages_mean | 4 | 512.0 | 512.0 | 512.0 | 14/14 |
| prefetch_adaptive_sequential | 256M | prefetch_callback_count | 4 | 13831.0 | 13531.5 | 15018.0 | 14/14 |
| prefetch_always_max | 1G | chunk_activate_count | 1 | 1536.0 | 1536.0 | 1536.0 | 6/6 |
| prefetch_always_max | 1G | eviction_prepare_count | 1 | 0.0 | 0.0 | 0.0 | 6/6 |
| prefetch_always_max | 1G | max_region_pages_mean | 1 | 512.0 | 512.0 | 512.0 | 6/6 |
| prefetch_always_max | 1G | prefetch_callback_count | 1 | 3072.0 | 3072.0 | 3072.0 | 6/6 |
| prefetch_always_max | 256M | chunk_activate_count | 4 | 384.0 | 384.0 | 384.0 | 14/14 |
| prefetch_always_max | 256M | eviction_prepare_count | 4 | 0.0 | 0.0 | 0.0 | 14/14 |
| prefetch_always_max | 256M | gpu_fault_count | 1 | 1369.0 | 1369.0 | 1369.0 | 14/14 |
| prefetch_always_max | 256M | h2d_migration_mb | 1 | 620.757 | 620.757 | 620.757 | 14/14 |
| prefetch_always_max | 256M | kernel_1_demand_gpu_fault_count | 1 | 1369.0 | 1369.0 | 1369.0 | 14/14 |
| prefetch_always_max | 256M | kernel_1_demand_h2d_migration_mb | 1 | 620.757 | 620.757 | 620.757 | 14/14 |
| prefetch_always_max | 256M | kernel_2_hot_gpu_fault_count | 1 | 0.0 | 0.0 | 0.0 | 14/14 |
| prefetch_always_max | 256M | kernel_2_hot_h2d_migration_mb | 1 | 0.0 | 0.0 | 0.0 | 14/14 |
| prefetch_always_max | 256M | max_region_pages_mean | 4 | 512.0 | 512.0 | 512.0 | 14/14 |
| prefetch_always_max | 256M | prefetch_callback_count | 4 | 768.0 | 768.0 | 768.0 | 14/14 |
| prefetch_none | 1G | chunk_activate_count | 1 | 1536.0 | 1536.0 | 1536.0 | 6/6 |
| prefetch_none | 1G | eviction_prepare_count | 1 | 0.0 | 0.0 | 0.0 | 6/6 |
| prefetch_none | 1G | max_region_pages_mean | 1 | 512.0 | 512.0 | 512.0 | 6/6 |
| prefetch_none | 1G | prefetch_callback_count | 1 | 1572864.0 | 1572864.0 | 1572864.0 | 6/6 |
| prefetch_none | 256M | chunk_activate_count | 4 | 384.0 | 384.0 | 384.0 | 14/14 |
| prefetch_none | 256M | eviction_prepare_count | 4 | 0.0 | 0.0 | 0.0 | 14/14 |
| prefetch_none | 256M | gpu_fault_count | 1 | 29820.0 | 29820.0 | 29820.0 | 14/14 |
| prefetch_none | 256M | h2d_migration_mb | 1 | 805.306 | 805.306 | 805.306 | 14/14 |
| prefetch_none | 256M | kernel_1_demand_gpu_fault_count | 1 | 29820.0 | 29820.0 | 29820.0 | 14/14 |
| prefetch_none | 256M | kernel_1_demand_h2d_migration_mb | 1 | 805.306 | 805.306 | 805.306 | 14/14 |
| prefetch_none | 256M | kernel_2_hot_gpu_fault_count | 1 | 0.0 | 0.0 | 0.0 | 14/14 |
| prefetch_none | 256M | kernel_2_hot_h2d_migration_mb | 1 | 0.0 | 0.0 | 0.0 | 14/14 |
| prefetch_none | 256M | max_region_pages_mean | 4 | 512.0 | 512.0 | 512.0 | 14/14 |
| prefetch_none | 256M | prefetch_callback_count | 4 | 393216.0 | 393216.0 | 393216.0 | 14/14 |
| system_driver_baseline | 268435456 | kernel_1_demand_ms | 1 | 240.292862 | 240.292862 | 240.292862 | 1/1 |
| system_driver_baseline | 268435456 | kernel_2_hot_ms | 1 | 0.995328 | 0.995328 | 0.995328 | 1/1 |
| system_driver_baseline | 268435456 | kernel_1_over_kernel_2 | 1 | 241.42078 | 241.42078 | 241.42078 | 1/1 |
| device_memory_baseline | 268435456 | kernel_1_demand_ms | 1 | 1.048928 | 1.048928 | 1.048928 | 1/1 |
| device_memory_baseline | 268435456 | kernel_2_hot_ms | 1 | 0.989184 | 0.989184 | 0.989184 | 1/1 |
| device_memory_baseline | 268435456 | kernel_1_over_kernel_2 | 1 | 1.060397 | 1.060397 | 1.060397 | 1/1 |
| system_driver_baseline | 1073741824 | kernel_1_demand_ms | 1 | 910.858215 | 910.858215 | 910.858215 | 1/1 |
| system_driver_baseline | 1073741824 | kernel_2_hot_ms | 1 | 4.02944 | 4.02944 | 4.02944 | 1/1 |
| system_driver_baseline | 1073741824 | kernel_1_over_kernel_2 | 1 | 226.05082 | 226.05082 | 226.05082 | 1/1 |
| device_memory_baseline | 1073741824 | kernel_1_demand_ms | 1 | 4.07072 | 4.07072 | 4.07072 | 1/1 |
| device_memory_baseline | 1073741824 | kernel_2_hot_ms | 1 | 4.020224 | 4.020224 | 4.020224 | 1/1 |
| device_memory_baseline | 1073741824 | kernel_1_over_kernel_2 | 1 | 1.01256 | 1.01256 | 1.01256 | 1/1 |

## Trace interpretation

The current prefetch CSV exposes callback context, page index, maximum candidate region, and PID fields. It does not expose the policy return action or the finally selected prefetch mask, so DEFAULT/BYPASS/ENTER_LOOP counts and selected prefetch bytes remain `UNAVAILABLE`.
A fault may invoke multiple callbacks and one callback may describe multiple pages; callback and fault counts are not one-to-one.
Trace rows are attributed using `fault_pid`/`owner_tgid` or chunk `pid`/`owner_pid`; each trace window permits no other UVM workload.

No oversubscription or eviction policy is part of this matrix. Zero eviction events is valid.
