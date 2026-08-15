# Reduced-Capacity Calibration

Status: `IMPLEMENTED_NOT_EXECUTED`

The workload now allocates fixed verification state first, then allocates and touches a normal CUDA device buffer. It retains that buffer while running the managed A-B-A-B scan. The measured model is:

```text
effective_gpu_capacity = gpu_free_after_reserve - safety_headroom
working_set_ratio = managed_working_set / effective_gpu_capacity
```

The default Stage 4 target is 8 GiB with 1 GiB safety headroom. If the 8 GiB `prefetch_none` case times out, the matrix permits one restart at 6 GiB; it never goes below 4 GiB and never raises the 300 second timeout.

A non-pressure regression used 64 MiB managed memory plus a touched and byte-verified 64 MiB reserve. It passed all four phases and correctness, and GPU memory returned to zero. This regression validates implementation mechanics only; it is not the required 0.95x/1.05x/1.10x calibration.

Calibration acceptance remains pending manual custom-module execution:

- 0.95x should show no or little selected eviction.
- 1.05x should show selected eviction.
- 1.10x should increase eviction/refault relative to 1.05x.

All resulting measurements must retain the `REDUCED_EFFECTIVE_GPU_CAPACITY` evidence label.
