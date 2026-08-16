# Reduced-Capacity Calibration

Status: `FAILED_STAGE4_REDUCED_CAPACITY_CALIBRATION`

Runtime window: 2026-08-15 21:53-21:56 UTC.

The workload now allocates fixed verification state first, then allocates and touches a normal CUDA device buffer. It retains that buffer while running the managed A-B-A-B scan. The measured model is:

```text
effective_gpu_capacity = gpu_free_after_reserve - safety_headroom
working_set_ratio = managed_working_set / effective_gpu_capacity
```

The default Stage 4 target is 8 GiB with 1 GiB safety headroom. If the 8 GiB `prefetch_none` case times out, the matrix permits one restart at 6 GiB; it never goes below 4 GiB and never raises the 300 second timeout.

A non-pressure regression used 64 MiB managed memory plus a touched and byte-verified 64 MiB reserve. It passed all four phases and correctness, and GPU memory returned to zero. The full 8 GiB calibration was then executed with the temporary custom UVM module.

## Measured Capacity

All nine runs reported the same capacity boundary:

| Field | Bytes |
|---|---:|
| GPU free before reserve | 25,091,833,856 |
| Reserve requested | 15,428,157,440 |
| Reserve actual | 15,428,747,264 |
| GPU free after reserve | 9,663,086,592 |
| Safety headroom | 1,073,741,824 |
| Reported effective capacity | 8,589,344,768 |

The reserve was touched and verified in every run. Managed working sets were:

| Ratio | Managed bytes | Timing runs | Trace runs | Selected eviction | Same-block refault |
|---:|---:|---:|---:|---:|---|
| 0.95x | 8,159,875,072 | 2 | 1 | 0 | unavailable, no eviction |
| 1.05x | 9,018,810,368 | 2 | 1 | 0 | unavailable, no eviction |
| 1.10x | 9,448,275,968 | 2 | 1 | 0 | unavailable, no eviction |

All 9/9 runs returned zero, passed correctness, detached cleanly, and recorded Xid delta zero.

## Calibration Failure

The 0.95x condition passed its no-eviction expectation, but 1.05x and 1.10x did not produce selected eviction. The calibration gate therefore correctly stopped Stage 4B.

The observed cause is the capacity definition, not a policy result. The program computes:

```text
effective = free_after_reserve - 1 GiB headroom
```

but the 1 GiB headroom remains physically free and available to UVM. At 1.10x, the managed working set was 9,448,275,968 bytes, still below the actual 9,663,086,592 bytes free after reserve by 214,810,624 bytes. Consequently, even the largest calibration point did not exceed physical GPU availability. The chunk trace contained `POPULATE` and `ACTIVATE`, but no `EVICTION_SELECTED` event.

The next maintenance change must correct this pressure-model mismatch before rerunning calibration. Ratios must not be increased ad hoc, and no policy matrix should run against the failed calibration.
