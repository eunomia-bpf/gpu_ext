# Experiment Plan

## Question

For three equal arrays allocated with `cudaMallocManaged`, compare the first GPU access after CPU initialization, an immediate second GPU access, a GPU access after CPU retouch, and a GPU access after explicit `cudaMemPrefetchAsync` to the active device. Use ordinary `cudaMalloc` plus explicit copies as the control.

## Ordered Managed-Memory Phases

1. Allocate A, B, and C with `cudaMallocManaged`.
2. CPU initializes A=1, B=2, C=0 across every element.
3. Run `kernel_1_demand` without a GPU prefetch.
4. Run `kernel_2_hot` immediately, without CPU modification.
5. Optionally prefetch A and B to `cudaCpuDeviceId`, then retouch none, one float per host page, or every A/B element.
6. Run `kernel_3_after_cpu_touch`.
7. Optionally prefetch A, B, and C to the active GPU and run `kernel_4_after_gpu_prefetch`.

GPU-side sampled validation checks the first, middle, and last 1024 elements plus deterministic random positions. It compares C with A+B, including CPU-retouched values, without migrating the arrays to the CPU merely for validation.

## Safe Matrix

For 256 MiB and 1 GiB per array:

- managed, no retouch, no explicit prefetch;
- managed, page retouch;
- managed, CPU prefetch then page retouch;
- managed, explicit GPU prefetch;
- device-memory control.

Three arrays must fit under 20% of free GPU memory. The per-array default cap is 1 GiB. Oversubscription is a separate, opt-in script.

## Interpretation

Kernel time changes are observations, not residency proof. Nsight Unified Memory CPU/GPU fault reports or gpu_ext driver traces are the migration evidence. gpu_ext callback counts are not automatically page-fault counts.
