# GPU UVM Stage 4 Results

Overall status: `FAILED_STAGE4_REDUCED_CAPACITY_CALIBRATION`

## Scope Completed

- Added a touched normal-device reserve to create measured 8 GiB or 6 GiB effective-capacity experiments.
- Preserved natural-capacity behavior when no reserve option is supplied.
- Added fixed 1 GiB safety headroom, a 4 GiB effective-capacity floor, and JSON capacity manifests.
- Added calibration, four-policy, eviction smoke, joint-policy, natural-confirmation, and trace-overhead runners.
- Added machine-readable static auditing and conservative runtime approval gates.
- Added aggregation that keeps reduced and natural capacity evidence separate.
- Passed a 64 MiB reserve/64 MiB managed implementation regression on the distribution driver.

## Runtime Status

| Stage | Status |
|---|---|
| 4A reduced-capacity implementation | PASS (code and small regression) |
| 4A 0.95x/1.05x/1.10x calibration | EXECUTED, FAILED PRESSURE GATE |
| 4B four-policy matrix | NOT EXECUTED |
| 4C static policy audit | PASS |
| 4C runtime smoke | NOT EXECUTED |
| 4D joint matrix | NOT EXECUTED |
| 4E natural confirmation | NOT READY |
| 4F fresh overhead measurement | NOT EXECUTED |

The custom UVM module was loaded temporarily for the calibration window and the distribution module was restored afterward. Only `custom_no_policy` calibration ran; no prefetch or joint policy was attached.

## Runtime Preflight

The 2026-08-15 runtime preflight passed the driver version, kernel vermagic, binary, runner, GPU-idle, result-disk, and host-memory checks. The host-memory gate now includes the planned device reserve as well as the managed working set and 16 GiB margin.

Before the authorized temporary switch, runtime preflight observed the distribution UVM srcversion (`182AB87276B2337B4B1A4CD`) rather than the custom module srcversion (`5446825F901EFEAA48651FC`), and no gpu_ext hook symbol was visible. This was the expected pre-switch state recorded in [STAGE4_RUNTIME_PREFLIGHT.md](STAGE4_RUNTIME_PREFLIGHT.md), not the final maintenance-window outcome.

After explicit authorization, the custom UVM module was temporarily loaded and Stage 4A ran all nine planned calibration cases. Every run passed correctness and cleanup, but 1.05x and 1.10x produced zero selected evictions. The calibration gate failed and prevented Stage 4B through Stage 4F.

The measured 1.10x working set remained about 205 MiB below actual GPU free memory after reserve because the 1 GiB safety headroom was subtracted from the reported effective capacity but remained physically usable. See [REDUCED_CAPACITY_CALIBRATION.md](REDUCED_CAPACITY_CALIBRATION.md).

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

The distribution `nvidia_uvm` was restored with srcversion `182AB87276B2337B4B1A4CD`; gpu_ext hook symbols are no longer visible, GPU memory is 0 MiB, and no compute process remains. Stage 4 is not sufficient to enter an LLM workload because calibration failed and the four-policy and joint-policy pressure matrices are absent.
