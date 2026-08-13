# gpu_ext UVM Stage 3 Results

Status: `PARTIAL_GPU_EXT_STAGE3_STOPPED_AT_RUNTIME_LIMIT`.

Initial runtime window: 2026-08-12 20:51-21:56 UTC. Bounded continuation window: 2026-08-13 10:34-10:48 UTC. The custom UVM module was loaded temporarily for both windows and the distribution module was restored afterward.

## Completion

| Stage | Result |
|---|---|
| 3A enhanced decision trace | PASS |
| 3A trace-overhead target | BORDERLINE: +1.149% untraced versus old custom baseline |
| 3B CPU first-touch diagnosis | PASS |
| 3B array migration diagnosis | PASS |
| 3C 0.95x matrix | PASS, 16/16 correct, no eviction selected |
| 3C 1.05x | PARTIAL: no-policy, always-max, and adaptive passed; prefetch_none timed out at 300 s and was not retried |
| 3C 1.10x bounded characterization | PASS for no-policy, always-max, and adaptive; prefetch_none intentionally not run |
| 3D joint policy matrix | NOT RUN; 1.05x and 1.10x stability prerequisite not met |

No Stage 3 full-pass status is claimed.

## Trace Semantics

The enhanced event now distinguishes candidate, policy output, and final effective region. Callback and final-decision counts matched exactly in every Stage 3A trace.

- `custom_no_policy`: DEFAULT only; representative final-pages median 64, p95 256, max 512.
- `prefetch_none`: 393,216 BYPASS decisions; effective region always empty.
- `prefetch_always_max`: 768 BYPASS decisions; candidate, policy output, and final region were all 512 pages.
- `prefetch_adaptive_sequential`: BYPASS only in this workload; representative mean 187.85 pages, maximum 220.
- `ENTER_LOOP`: zero in the tested sequential vector-add.

One decision can cover multiple pages. A callback/decision is not equivalent to one GPU page fault.

## Diagnostic Conclusions

`prefetch_none` also affects CPU first touch. It increased the full initialization mean from 353.105 ms to 3,018.094 ms and run-wide minor faults from about 7,717 to 202,021. Its representative trace had 196,608 BYPASS decisions with zero final pages, exactly the number of 4 KiB pages in three 256 MiB arrays. Explicit CPU prefetch reduced all policies to approximately 148-154 ms and eliminated decisions in that window.

The old `always_max` 620.757 MB HtoD result was not reproduced. Address-classified Nsight runs show each isolated A, B, or C access migrated 256 MiB, and every A+B->C policy migrated all three arrays, 768 MiB / 805.306 decimal MB. `always_max` reduced GPU faults to 1,679 versus 5,864 no-policy and 29,476 `prefetch_none`, but did not reduce migration bytes.

## Oversubscription

At 0.95x, all four policies completed three timing runs and one trace run with correctness, Xid=0, and clean detach. Mean phase times in milliseconds:

| Policy | A first | B first | A reuse | B reuse |
|---|---:|---:|---:|---:|
| custom no-policy | 3,580 | 3,602 | 34.6 | 34.7 |
| prefetch_none | 57,676 | 57,206 | 34.6 | 34.7 |
| prefetch_always_max | 1,297 | 1,311 | 34.6 | 34.7 |
| prefetch_adaptive_sequential | 2,000 | 1,958 | 34.6 | 34.7 |

No selected eviction was observed at 0.95x.

At 1.05x, the initial verifier incorrectly allocated its small CUDA verification buffer only after filling GPU memory. That first run completed all scan phases but failed the auxiliary `cudaMalloc`. The verifier was corrected to reserve its fixed 4096-index state before pressure; a 64 MiB regression passed. The corrected no-policy 3 timing + 1 trace runs then passed.

The corrected 1.05x no-policy trace recorded 13,355 selected evictions and proved 602 same-VA-block A-first -> B eviction -> A-reuse refaults, totaling 1,262,485,504 bytes. Mean eviction-to-refault time was about 744,495 us. The next policy, `prefetch_none`, exceeded the fixed 300 s per-run limit during its first timing run and exited 124. It was not retried and the timeout was not relaxed.

After explicit continuation, the remaining bounded cases ran without `prefetch_none`. Timing-only means in milliseconds were:

| Ratio | Policy | A first | B first | A reuse | B reuse |
|---|---|---:|---:|---:|---:|
| 1.05x | custom no-policy | 3,971 | 4,107 | 4,797 | 4,815 |
| 1.05x | prefetch_always_max | 1,438 | 1,511 | 2,292 | 2,293 |
| 1.05x | prefetch_adaptive_sequential | 2,074 | 2,113 | 2,779 | 2,683 |
| 1.10x | custom no-policy | 4,147 | 4,371 | 5,034 | 5,008 |
| 1.10x | prefetch_always_max | 1,501 | 1,684 | 2,404 | 2,414 |
| 1.10x | prefetch_adaptive_sequential | 2,329 | 2,356 | 2,834 | 2,771 |

Each listed policy has three successful timing runs at the listed ratio. The 1.05x no-policy values use the corrected three-run series and exclude the failed auxiliary-verifier attempt.

Enhanced traces established increasing eviction/refault pressure:

| Ratio | Policy | Selected evictions | Same-block refaults | Refaulted bytes |
|---|---|---:|---:|---:|
| 1.05x | custom no-policy | 13,355 | 602 | 1,262,485,504 |
| 1.05x | prefetch_always_max | 13,359 | 601 | 1,260,388,352 |
| 1.05x | prefetch_adaptive_sequential | 13,388 | 626 | 1,312,817,152 |
| 1.10x | custom no-policy | 14,724 | 1,200 | 2,516,582,400 |
| 1.10x | prefetch_always_max | 14,724 | 1,199 | 2,514,485,248 |
| 1.10x | prefetch_adaptive_sequential | 14,755 | 1,225 | 2,569,011,200 |

The 1.10x Nsight representatives reported the following run-wide totals. MB follows the Nsight export's decimal unit and is not phase-attributed here.

| Policy | HtoD MB | DtoH MB | CPU faults | GPU faults |
|---|---:|---:|---:|---:|
| custom no-policy | 55,014.130 | 30,823.711 | 78,882 | 642,290 |
| prefetch_always_max | 55,704.101 | 26,556.006 | 13,147 | 210,266 |
| prefetch_adaptive_sequential | 55,351.542 | 30,735.630 | 52,588 | 475,873 |

These data show that always-max reduced page-fault counts for this sequential scan while moving a similar total HtoD volume. They do not establish a generally superior policy or justify Stage 3D, because the required four-policy 1.05x/1.10x matrix remains incomplete.

## Safety And Restoration

- The bounded continuation produced 23 additional runs; all returned zero, passed correctness, detached policy state, and reported Xid delta zero.
- Aggregation treats the timed-out `prefetch_none` case as incomplete and excludes it from correctness/performance statistics. Its immutable raw manifest described only the phase rows emitted before timeout; `results/stage3/timeout_corrections.json` records the effective whole-run interpretation.
- Every policy run reported clean struct_ops detach.
- Kernel-log comparisons reported zero new NVIDIA Xid, fatal UVM error, BUG, or Oops.
- GPU memory returned to 0 MiB and no compute process remained.
- No oversubscription policy calling `bpf_gpu_block_move_head()` was run.
- The distribution `nvidia_uvm` was restored with `srcversion=182AB87276B2337B4B1A4CD`; enhanced hook symbols were no longer visible.
- No GPU clock, power, MIG, ECC, persistence, full driver stack, or system module installation setting was modified.
- Raw Stage 3 evidence occupies approximately 8.7 GiB; the runner now requires at least 32 GiB result-disk headroom before each pressure case.

These results apply only to the synthetic sequential vector-add and two-region phase scan. They do not rank policies for LLM workloads and do not justify moving to an LLM experiment while `prefetch_none` remains outside the 1.05x runtime bound and Stage 3D remains untested.
