# GPU UVM Stage 4 Results

Overall status: `READY_FOR_MANUAL_GPU_EXT_STAGE4`

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
| 4A 0.95x/1.05x/1.10x calibration | NOT EXECUTED |
| 4B four-policy matrix | NOT EXECUTED |
| 4C static policy audit | PASS |
| 4C runtime smoke | NOT EXECUTED |
| 4D joint matrix | NOT EXECUTED |
| 4E natural confirmation | NOT READY |
| 4F fresh overhead measurement | NOT EXECUTED |

The loaded module was not switched during this work. No policy was attached, no pressure ratio was run, and no new Stage 4 fault, migration, eviction, refault, or timing conclusion is claimed.

## Static Audit

| Candidate | Static decision | Reason |
|---|---|---|
| `eviction_fifo` | rejected | Hot-path printk, unreliable access-hook dependency, and no implemented FIFO reorder |
| `prefetch_always_max_cycle_moe` | approved for smoke only | Bounded map and move-tail behavior; runtime validation still required |
| `prefetch_cooperative` | rejected | Workqueue and cross-VA-block migration require deeper proof |

The complete audit is in [EVICTION_POLICY_SAFETY_AUDIT_STAGE4.md](EVICTION_POLICY_SAFETY_AUDIT_STAGE4.md).

## Safety Boundary

All root/module/BPF runtime steps are isolated in the intentionally non-executable `scripts/SAFE_STAGE4_COMMANDS.sh`. The scripts retain the 300 second timeout, 32 GiB result-disk minimum, host working-set plus 16 GiB memory margin, exact PID cleanup, residual struct_ops checks, and Xid/correctness stop conditions.

The distribution `nvidia_uvm` remains loaded. Stage 4 is not sufficient to enter an LLM workload because the four-policy and joint-policy pressure matrices are still absent.
