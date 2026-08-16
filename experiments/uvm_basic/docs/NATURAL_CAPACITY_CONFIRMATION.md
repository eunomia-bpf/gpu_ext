# Natural-Capacity Confirmation

Status: `PASS_STAGE4_NATURAL_CAPACITY_CONFIRMATION`

The 2026-08-16 window completed `custom_no_policy`, `prefetch_always_max`, and
`prefetch_always_max_cycle_moe` at natural A30 capacity and ratio 1.05x. Each
policy has two untraced timings and one enhanced trace. All nine runs passed
correctness, clean-detach, Xid, and GPU-memory-release checks.

Mean timings from the two untraced runs per policy were:

| Policy | A first ms | B first ms | A reuse ms | B reuse ms | Total ms |
|---|---:|---:|---:|---:|---:|
| `custom_no_policy` | 3,965.5 | 4,093.2 | 4,800.6 | 4,741.5 | 17,600.8 |
| `prefetch_always_max` | 1,431.5 | 1,526.8 | 2,293.0 | 2,316.2 | 7,567.5 |
| `prefetch_always_max_cycle_moe` | 1,434.4 | 1,529.8 | 2,300.6 | 2,318.1 | 7,582.9 |

The trace runs are retained as trace evidence and are not pooled with untraced
timing. They recorded 13,348/13,355/13,349 selected evictions and
600/599/599 same-block refaults in the policy order shown above.

The first attempt stopped safely before `prefetch_always_max`: the result
filesystem had 34,075,832,320 bytes available, slightly below the required
32 GiB threshold. No experiment evidence was deleted. Purging the unrelated
pip download cache recovered about 27 GB, after which the two remaining
policies were run in a new custom-UVM window and the distribution module was
restored.

The natural-capacity result confirms the reduced-capacity trend for this
sequential scan: `always_max` substantially improved first access, while
`cycle_moe` was effectively indistinguishable from `always_max` and showed no
reuse advantage. This is a limited trend confirmation, not a general workload
or native A30 performance claim.

Reduced-capacity and natural-capacity absolute timings remain separate data sets. Raw evidence is under ignored `results/stage4/natural_stage4/`.
