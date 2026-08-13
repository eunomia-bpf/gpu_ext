# CPU First-Touch Diagnosis

Status: `PASS_DIAGNOSIS_PREFETCH_NONE_FINE_GRAIN_FAULTING`.

Each cell below contains ten independent 256 MiB-per-array processes. Linux counters are run-wide `/usr/bin/time -v` evidence; decision counts are from representative enhanced traces.

| Policy | Touch mode | Mean (ms) | Minor faults mean | Representative decisions | Action / final pages |
|---|---|---:|---:|---:|---|
| custom no-policy | full | 353.105 | 7,717 | 2,304 | DEFAULT / 387,072 total selected pages |
| prefetch_none | full | 3,018.094 | 202,021 | 196,608 | BYPASS / 0 |
| prefetch_always_max | full | 320.644 | 5,797 | 384 | BYPASS / 196,608 total selected pages |
| prefetch_adaptive_sequential | full | 340.152 | 6,949 | 1,536 | BYPASS / 195,072 total selected pages |
| custom no-policy | page stride | 269.434 | 7,717 | 2,304 | DEFAULT |
| prefetch_none | page stride | 2,927.831 | 202,021 | 196,608 | BYPASS / 0 |

CPU first touch therefore does enter `compute_prefetch_region()` and the gpu_ext policy hook. Under `prefetch_none`, the 196,608 decisions equal the number of 4 KiB pages across the three 256 MiB arrays, and every decision returns an empty effective region. This supports the explanation that CPU initialization loses UVM's coarser prefetch grouping and degrades to fine-grained page handling. It does not mean every callback is a distinct hardware page fault.

Prefetching A/B/C to `cudaCpuDeviceId` before the write removes the policy difference: all four policies complete in approximately 148-154 ms, record about 5,418 minor faults, and produce zero prefetch decisions in the captured window. Major faults were zero in every condition. The anomaly is therefore assigned to UVM fault/prefetch granularity, not CPU arithmetic.
