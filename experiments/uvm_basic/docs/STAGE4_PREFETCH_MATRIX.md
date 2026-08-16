# Stage 4 Reduced-Capacity Prefetch Matrix

Status: `PASS_STAGE4_REDUCED_CAPACITY_PREFETCH_MATRIX`

Runtime window: 2026-08-16 UTC. Evidence class: `PHYSICALLY_RESERVED_GUARD_MODEL`.

The matrix completed all four policies at 0.95x, 1.05x, and 1.10x measured effective capacity: three independent timing runs and one enhanced trace per cell, plus one Nsight representative per policy at 1.10x. All runs passed correctness, Xid, detach, and GPU-memory cleanup gates. The effective capacity was approximately 8 GiB with a physically allocated 1 GiB guard; these timings are not native A30 24 GiB timings.

## Timing Results

Mean milliseconds from the three untraced runs:

| Policy | Ratio | A first | B first | A reuse | B reuse | Total |
|---|---:|---:|---:|---:|---:|---:|
| `custom_no_policy` | 0.95x | 1,335.8 | 1,332.9 | 11.8 | 11.8 | 2,692.2 |
| `prefetch_none` | 0.95x | 17,435.2 | 17,330.3 | 11.8 | 11.8 | 34,789.0 |
| `prefetch_always_max` | 0.95x | 468.4 | 465.7 | 11.8 | 11.8 | 957.6 |
| `prefetch_adaptive_sequential` | 0.95x | 891.5 | 675.0 | 11.8 | 11.8 | 1,590.1 |
| `custom_no_policy` | 1.05x | 1,484.6 | 1,524.0 | 1,780.5 | 1,783.8 | 6,573.0 |
| `prefetch_none` | 1.05x | 19,548.2 | 19,415.2 | 16,144.9 | 15,961.9 | 71,070.2 |
| `prefetch_always_max` | 1.05x | 511.6 | 541.6 | 805.6 | 814.0 | 2,672.9 |
| `prefetch_adaptive_sequential` | 1.05x | 906.9 | 766.8 | 1,014.6 | 988.1 | 3,676.5 |
| `custom_no_policy` | 1.10x | 1,539.8 | 1,607.6 | 1,849.9 | 1,824.9 | 6,822.3 |
| `prefetch_none` | 1.10x | 20,384.6 | 20,419.6 | 16,998.1 | 17,212.0 | 75,014.3 |
| `prefetch_always_max` | 1.10x | 536.6 | 597.3 | 841.3 | 852.3 | 2,827.4 |
| `prefetch_adaptive_sequential` | 1.10x | 915.2 | 844.5 | 1,049.3 | 1,033.5 | 3,842.5 |

`prefetch_none` completed every run under the fixed 300 second limit, including 1.10x. Its trace recorded millions of BYPASS decisions with zero final pages, and its time increased sharply once reuse also faulted under pressure. The available points support page-granularity work as the dominant cost, but are insufficient to claim a precise linear model.

## Pressure Evidence

Enhanced-trace results:

| Policy | Ratio | Selected eviction | Same-block refault | Refaulted bytes | Mean final pages |
|---|---:|---:|---:|---:|---:|
| `custom_no_policy` | 0.95x | 0 | unavailable | unavailable | 65.2 |
| `prefetch_none` | 0.95x | 0 | unavailable | unavailable | 0.0 |
| `prefetch_always_max` | 0.95x | 0 | unavailable | unavailable | 256.0 |
| `prefetch_adaptive_sequential` | 0.95x | 0 | unavailable | unavailable | 96.9 |
| `custom_no_policy` | 1.05x | 4,726 | 206 | 432,013,312 | 57.8 |
| `prefetch_none` | 1.05x | 4,742 | 215 | 450,887,680 | 0.0 |
| `prefetch_always_max` | 1.05x | 4,711 | 206 | 432,013,312 | 256.0 |
| `prefetch_adaptive_sequential` | 1.05x | 4,743 | 214 | 448,790,528 | 126.9 |
| `custom_no_policy` | 1.10x | 5,283 | 411 | 861,929,472 | 58.2 |
| `prefetch_none` | 1.10x | 5,301 | 420 | 880,803,840 | 0.0 |
| `prefetch_always_max` | 1.10x | 5,284 | 411 | 861,929,472 | 256.0 |
| `prefetch_adaptive_sequential` | 1.10x | 5,310 | 420 | 880,803,840 | 129.5 |

The pressure boundary, not policy choice, was the primary determinant of eviction/refault count in this A-B-A-B scan. `always_max` greatly reduced first-access time without increasing the measured selected-eviction/refault counts relative to no policy. `adaptive_sequential` was an intermediate timing result, but did not reduce pressure counters at 1.05x or 1.10x. These conclusions are workload-specific and do not establish behavior for irregular or LLM access patterns.

Canonical machine-readable evidence is `results/stage4/prefetch_matrix_status.json` and `results/stage4/stage4_prefetch_summary.csv`. Raw traces remain under ignored `results/stage4/prefetch_matrix_stage4/`.
