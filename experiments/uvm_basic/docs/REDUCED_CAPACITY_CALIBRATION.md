# Reduced-Capacity Calibration

Status: `PASS_STAGE4A_PHYSICAL_GUARD_CALIBRATION`

Next gate: `READY_FOR_STAGE4B_PREFETCH_MATRIX`

## Legacy Failure

Evidence class: `LEGACY_MATHEMATICAL_HEADROOM_MODEL`

The 2026-08-15 runtime window completed nine `custom_no_policy` runs at nominal 0.95x, 1.05x, and 1.10x. All runs passed correctness, cleanup, and Xid checks, but selected eviction remained zero at every ratio.

The old model calculated:

```text
effective = gpu_free_after_reserve - 1 GiB safety_headroom
```

The 1 GiB headroom was never physically allocated. At nominal 1.10x, the 9,448,275,968-byte managed working set remained 214,810,624 bytes below the 9,663,086,592 bytes physically free after reserve. This failure remains in the original raw results, `calibration_status.json`, and `calibration_summary.csv`; the runner archives them with a `legacy_mathematical_headroom_` prefix before a new calibration writes canonical outputs.

## Physical Guard Repair

Evidence class: `PHYSICALLY_RESERVED_GUARD_MODEL`

The repaired allocation order is:

```text
fixed verification allocations
-> query gpu_free_initial
-> cudaMalloc/cudaMemset main_reserve_buffer
-> query gpu_free_after_main_reserve
-> cudaMalloc/cudaMemset guard_buffer
-> query gpu_free_after_guard
-> effective_gpu_capacity = gpu_free_after_guard
-> derive and allocate managed working set
-> CPU first touch
-> A first / B first / A reuse / B reuse
```

Both device buffers remain alive during every managed-memory phase. The 1 GiB guard is released first on exceptional cleanup; normal cleanup frees managed memory, guard, main reserve, and fixed verification allocations in that order.

The workload enforces:

- effective capacity within 2% of the requested target;
- actual working-set ratio within 0.01 of the requested ratio;
- a physical guard that is touched and byte-sampled;
- memory snapshots after managed allocation, CPU first touch, all four GPU phases, before cleanup, and after cleanup.

The deprecated `--safety-headroom-bytes` option is retained only as an alias that creates a real guard allocation. It is no longer subtracted mathematically from free memory.

## Non-Pressure Verification

Three distribution-driver regressions passed:

1. 64 MiB managed memory with 64 MiB main reserve and 64 MiB guard verified allocation order, all snapshots, A-B-A-B correctness, and process-exit GPU memory release.
2. An auto-sized 0.01x working set with a 22 GiB target and 64 MiB guard measured 23,621,730,304 bytes effective capacity, 0.00999996 actual ratio, and 0.0025% target error.
3. The production-shaped 8 GiB target with a physical 1 GiB guard and a non-pressure 0.01x working set allocated 15,428,747,264 bytes of main reserve and 1,073,741,824 bytes of guard. It measured 8,589,344,768 bytes effective capacity, 0.00999996 actual ratio, and 0.0069% target error; GPU memory returned to 0 MiB after process exit.

These runs validate implementation mechanics only. They are not Stage 4A pressure evidence and do not establish eviction behavior.

## Physical-Guard Calibration Result

Runtime window: 2026-08-16 09:29-09:32 UTC.

The reviewed UVM-only switch loaded the custom module for the nine calibration cases and restored the distribution module immediately afterward. No prefetch or eviction policy was attached.

All runs measured:

| Field | Value |
|---|---:|
| Main reserve allocated | 15,428,747,264 bytes |
| Physical guard allocated | 1,073,741,824 bytes |
| `gpu_free_after_guard` | 8,589,344,768 bytes |
| Effective capacity | 8,589,344,768 bytes |
| Target relative error | 0.0069% |

Pressure evidence from each enhanced-trace run:

| Requested ratio | Actual ratio | Selected eviction | Same-block refault | Refaulted bytes |
|---:|---:|---:|---:|---:|
| 0.95x | 0.95 | 0 | unavailable, no eviction | unavailable |
| 1.05x | 1.05 | 4,730 | 206 | 432,013,312 |
| 1.10x | 1.10 | 5,279 | 411 | 861,929,472 |

The 1.05x and 1.10x runs both crossed the physical pressure boundary. The 1.10x run produced more selected eviction, same-block refault, and refaulted bytes than 1.05x. This closes the failed legacy pressure gate without changing the requested ratios or reducing the target capacity.

All 9/9 processes returned zero and passed correctness. Capacity error remained below 2%, ratio error remained below 0.01, Xid delta was zero, every trace detached, every run released GPU memory, and no residual struct_ops remained. The canonical result files are `results/stage4/calibration_status.json` and `results/stage4/calibration_summary.csv`; the legacy failed result is preserved under the `legacy_mathematical_headroom_calibration_` prefix.
