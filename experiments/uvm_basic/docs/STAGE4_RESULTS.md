# GPU UVM Stage 4 Results

Overall status: `PASS_STAGE4A_PHYSICAL_GUARD_CALIBRATION`

Next gate: `READY_FOR_STAGE4B_PREFETCH_MATRIX`

## Scope Completed

- Replaced the mathematical-headroom model with touched, long-lived main-reserve and physical-guard device allocations.
- Preserved natural-capacity behavior when no reserve option is supplied.
- Added fixed 1 GiB safety headroom, a 4 GiB effective-capacity floor, and JSON capacity manifests.
- Added calibration, four-policy, eviction smoke, joint-policy, natural-confirmation, and trace-overhead runners.
- Added machine-readable static auditing and conservative runtime approval gates.
- Added aggregation that keeps reduced and natural capacity evidence separate.
- Passed a 64 MiB reserve/64 MiB managed implementation regression on the distribution driver.

## Runtime Status

| Stage | Status |
|---|---|
| 4A legacy mathematical-headroom calibration | EXECUTED, FAILED PRESSURE GATE |
| 4A physical-guard implementation | PASS (build and non-pressure regression) |
| 4A physical-guard 0.95x/1.05x/1.10x recalibration | PASS |
| 4B four-policy matrix | NOT EXECUTED |
| 4C static policy audit | PASS |
| 4C runtime smoke | NOT EXECUTED |
| 4D joint matrix | NOT EXECUTED |
| 4E natural confirmation | NOT READY |
| 4F fresh overhead measurement | NOT EXECUTED |

The 2026-08-15 custom-UVM window used only `custom_no_policy`; no prefetch or joint policy was attached. That legacy run remains evidence of the failed mathematical-headroom model.

## Runtime Preflight

The 2026-08-15 runtime preflight passed the driver version, kernel vermagic, binary, runner, GPU-idle, result-disk, and host-memory checks. The host-memory gate now includes the planned device reserve as well as the managed working set and 16 GiB margin.

Before the authorized temporary switch, runtime preflight observed the distribution UVM srcversion (`182AB87276B2337B4B1A4CD`) rather than the custom module srcversion (`5446825F901EFEAA48651FC`), and no gpu_ext hook symbol was visible. This was the expected pre-switch state recorded in [STAGE4_RUNTIME_PREFLIGHT.md](STAGE4_RUNTIME_PREFLIGHT.md), not the final maintenance-window outcome.

After explicit authorization, the custom UVM module was temporarily loaded and Stage 4A ran all nine legacy calibration cases. Every run passed correctness and cleanup, but 1.05x and 1.10x produced zero selected evictions. The calibration gate correctly prevented Stage 4B through Stage 4F.

The measured 1.10x working set remained about 205 MiB below actual GPU free memory after reserve because the 1 GiB safety headroom was subtracted from the reported effective capacity but remained physically usable. See [REDUCED_CAPACITY_CALIBRATION.md](REDUCED_CAPACITY_CALIBRATION.md).

The repaired model allocates and touches both `main_reserve_buffer` and a 1 GiB `guard_buffer`, retains both through the A-B-A-B scan, and defines effective capacity as the measured `gpu_free_after_guard`. The 2026-08-16 custom-UVM window completed all nine planned runs. The measured effective capacity was 8,589,344,768 bytes with actual ratios 0.95, 1.05, and 1.10. Selected eviction increased from 0 to 4,730 to 5,279; same-block refault was unavailable at 0.95x because there was no eviction, then increased from 206 to 411. All runs passed correctness, capacity, ratio, cleanup, and Xid gates.

The machine-readable maintenance-window result is in `results/stage4/runtime_status.json`.

## Static Audit

| Candidate | Static decision | Reason |
|---|---|---|
| `eviction_fifo` | rejected | Hot-path printk, unreliable access-hook dependency, and no implemented FIFO reorder |
| `prefetch_always_max_cycle_moe` | approved for smoke only | Bounded map and move-tail behavior; runtime validation still required |
| `prefetch_cooperative` | rejected | Workqueue and cross-VA-block migration require deeper proof |

The complete audit is in [EVICTION_POLICY_SAFETY_AUDIT_STAGE4.md](EVICTION_POLICY_SAFETY_AUDIT_STAGE4.md).

## Safety Boundary

All root/module/BPF runtime steps are isolated in the intentionally non-executable `scripts/SAFE_STAGE4_COMMANDS.sh`. The scripts retain the 300 second timeout, 32 GiB result-disk minimum, host working-set plus 16 GiB memory margin, exact PID cleanup, residual struct_ops checks, and Xid/correctness stop conditions.

The distribution `nvidia_uvm` was restored with srcversion `182AB87276B2337B4B1A4CD`; gpu_ext hook symbols are not visible, GPU memory is 0 MiB, and no compute process remains. Stage 4A is complete and Stage 4B may now be scheduled as a separate guarded maintenance window.
