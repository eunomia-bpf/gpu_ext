# always_max Migration Diagnosis

Status: `PASS_DIAGNOSIS_OLD_TOTAL_NOT_REPRODUCED`.

The workload records exact A/B/C virtual-address ranges. Nsight SQLite exposes `CUPTI_ACTIVITY_KIND_MEMCPY.virtualAddress`, so every UVM migration row was classified by allocation rather than inferred from totals.

| Policy | Kernel | A HtoD | B HtoD | C HtoD | Total HtoD | GPU faults |
|---|---|---:|---:|---:|---:|---:|
| custom no-policy | read A | 256 MiB | 0 | 0 | 256 MiB | 4,565 |
| custom no-policy | read B | 0 | 256 MiB | 0 | 256 MiB | 4,478 |
| custom no-policy | write C | 0 | 0 | 256 MiB | 256 MiB | 2,525 |
| custom no-policy | A+B->C | 256 MiB | 256 MiB | 256 MiB | 768 MiB | 5,864 |
| prefetch_none | A+B->C | 256 MiB | 256 MiB | 256 MiB | 768 MiB | 29,476 |
| prefetch_always_max | A+B->C | 256 MiB | 256 MiB | 256 MiB | 768 MiB | 1,679 |
| prefetch_adaptive_sequential | A+B->C | 256 MiB | 256 MiB | 256 MiB | 768 MiB | 7,309 |

`prefetch_always_max` also migrated exactly 256 MiB for each isolated A-only, B-only, and C-write-only run. The old representative total of about 620.757 MB was not reproduced; the controlled run reports 805.306 decimal MB (768 MiB), identical to the other policies. It must be treated as a prior-run or export/filtering anomaly, not evidence that `always_max` skipped 184 MB.

The policy changes fault aggregation, not the bytes required by this workload: `always_max` has far fewer GPU faults while all three initialized arrays still migrate completely. C's prior CPU-initialized content is migrated even for the write-only kernel in this implementation and profiler configuration.
