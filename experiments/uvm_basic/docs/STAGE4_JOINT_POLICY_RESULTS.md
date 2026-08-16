# Stage 4 Joint Prefetch-Eviction Results

Status: `PASS_STAGE4_JOINT_POLICY_MATRIX`

Static audit rejected `eviction_fifo` and `prefetch_cooperative`. Only `prefetch_always_max_cycle_moe` entered runtime smoke and Stage 4D.

## Smoke Test

The approved policy passed all three smoke cases:

- 64 MiB timing;
- 64 MiB enhanced trace;
- 8 GiB effective-capacity 0.95x timing.

Every case returned zero, passed correctness, detached its struct_ops policy, produced no Xid, and released GPU memory. This promoted the policy to `APPROVED_FOR_STAGE4D`.

## Reduced-Capacity Matrix

All three policies completed three timings and one enhanced trace at 1.05x and 1.10x; each also completed one Nsight run at 1.10x.

Mean untraced phase times in milliseconds:

| Policy | Ratio | A first | B first | A reuse | B reuse | Total |
|---|---:|---:|---:|---:|---:|---:|
| `custom_no_policy` | 1.05x | 1,483.4 | 1,523.9 | 1,771.3 | 1,783.0 | 6,561.6 |
| `prefetch_always_max` | 1.05x | 512.9 | 541.4 | 805.6 | 813.5 | 2,673.4 |
| `prefetch_always_max_cycle_moe` | 1.05x | 511.3 | 541.6 | 806.6 | 815.1 | 2,674.6 |
| `custom_no_policy` | 1.10x | 1,546.7 | 1,611.3 | 1,856.4 | 1,826.5 | 6,840.9 |
| `prefetch_always_max` | 1.10x | 537.8 | 599.8 | 844.8 | 853.8 | 2,836.2 |
| `prefetch_always_max_cycle_moe` | 1.10x | 537.5 | 599.5 | 844.6 | 855.4 | 2,837.1 |

Enhanced traces likewise showed no meaningful separation between `always_max` and `cycle_moe`:

| Policy | Ratio | Selected eviction | Same-block refault | Refaulted bytes | Mean final pages |
|---|---:|---:|---:|---:|---:|
| `custom_no_policy` | 1.05x | 4,725 | 206 | 432,013,312 | 57.8 |
| `prefetch_always_max` | 1.05x | 4,708 | 206 | 432,013,312 | 256.0 |
| `prefetch_always_max_cycle_moe` | 1.05x | 4,710 | 206 | 432,013,312 | 256.0 |
| `custom_no_policy` | 1.10x | 5,284 | 411 | 861,929,472 | 58.1 |
| `prefetch_always_max` | 1.10x | 5,279 | 411 | 861,929,472 | 256.0 |
| `prefetch_always_max_cycle_moe` | 1.10x | 5,280 | 411 | 861,929,472 | 256.0 |

For this sequential A-B-A-B workload, `cycle_moe` behaved like `always_max`: it improved first access relative to no policy but did not improve reuse, repeated migration, eviction, or same-block refault. This does not show that the policy is generally ineffective; it shows that its MoE-cycle assumptions add no measured value to this non-MoE access pattern. No policy depending on the currently unreliable `gpu_block_access` hook was run.

Canonical status is `results/stage4/joint_matrix_status.json`. Raw evidence remains under ignored `results/stage4/joint_stage4/`.
