# GPU UVM Stage 4 Results

Overall status: `PASS_GPU_EXT_STAGE4_RUNTIME_AND_JOINT_POLICY`

Stages 4A through 4E completed. Stage 4F completed its required measurements,
but its <=1% disabled-path overhead target was not met.

## Runtime Status

| Stage | Status |
|---|---|
| 4A physical-guard calibration | `PASS_STAGE4A_PHYSICAL_GUARD_CALIBRATION` |
| 4B four-policy reduced-capacity matrix | `PASS_STAGE4_REDUCED_CAPACITY_PREFETCH_MATRIX` |
| 4C static audit | PASS; one policy eligible for smoke |
| 4C `cycle_moe` runtime smoke | PASS; `APPROVED_FOR_STAGE4D` |
| 4D joint matrix | `PASS_STAGE4_JOINT_POLICY_MATRIX` |
| 4E natural confirmation | `PASS_STAGE4_NATURAL_CAPACITY_CONFIRMATION` |
| 4F trace disabled-path overhead | `COMPLETE_TARGET_NOT_MET` |

## Stage 4A

The repaired model physically allocated and touched both the main reserve and a 1 GiB guard. It measured 8,589,344,768 bytes effective capacity. At 0.95x/1.05x/1.10x, selected eviction was 0/4,730/5,279 and same-block refault was unavailable/206/411. All nine runs passed correctness, cleanup, and Xid gates. See [REDUCED_CAPACITY_CALIBRATION.md](REDUCED_CAPACITY_CALIBRATION.md).

## Stage 4B

All 36 timing, 12 enhanced-trace, and four Nsight runs completed for `custom_no_policy`, `prefetch_none`, `prefetch_always_max`, and `prefetch_adaptive_sequential` at 0.95x, 1.05x, and 1.10x.

`prefetch_none` completed under the 300 second per-process limit, including 1.10x. Its mean total phase time rose from about 34.8 seconds at 0.95x to 71.1 and 75.0 seconds at 1.05x and 1.10x. `prefetch_always_max` was fastest in this sequential scan at about 0.96/2.67/2.83 seconds; adaptive was intermediate at about 1.59/3.68/3.84 seconds.

Eviction/refault changed primarily with ratio. At 1.10x, selected eviction was 5,283/5,301/5,284/5,310 and same-block refault was 411/420/411/420 in the same policy order. See [STAGE4_PREFETCH_MATRIX.md](STAGE4_PREFETCH_MATRIX.md).

## Stage 4C/D

Static audit continued to reject `eviction_fifo` and `prefetch_cooperative`. `prefetch_always_max_cycle_moe` passed 64 MiB timing/trace and 0.95x reduced-capacity smoke, then completed Stage 4D at 1.05x and 1.10x.

For this non-MoE sequential A-B-A-B scan, `cycle_moe` and `always_max` were effectively identical. At 1.10x their mean total phase times were 2,837.1 and 2,836.2 ms; both recorded 411 same-block refaults and about 862 MB refaulted data. No reuse benefit from the cycle policy was observed. See [STAGE4_JOINT_POLICY_RESULTS.md](STAGE4_JOINT_POLICY_RESULTS.md).

## Stage 4E/F

Natural-capacity 1.05x completed two timings and one trace for each of
`custom_no_policy`, `prefetch_always_max`, and
`prefetch_always_max_cycle_moe`. Their mean untraced total phase times were
17,600.8/7,567.5/7,582.9 ms. The reduced-capacity trend was confirmed for this
sequential scan, but `cycle_moe` again provided no measurable advantage over
`always_max`.

The first natural-capacity attempt stopped at the disk-headroom gate. Purging
only the unrelated pip cache recovered enough space, and the remaining cases
then completed. Stage 4F's 20 untraced and 20 traced runs measured kernel-1
means of 244.138 and 285.353 ms. The current disabled path is 1.415% above the
240.731 ms Stage 2 reference, so the <=1% target remains unmet. See
[NATURAL_CAPACITY_CONFIRMATION.md](NATURAL_CAPACITY_CONFIRMATION.md) and
[TRACE_DISABLED_OVERHEAD_STAGE4.md](TRACE_DISABLED_OVERHEAD_STAGE4.md).

## Safety And Restoration

- correctness failures: 0 in completed Stage 4B-F cases;
- Xid delta: 0;
- all completed policies detached cleanly;
- GPU memory returned to 0 MiB;
- no compute process remains;
- distribution `nvidia_uvm` srcversion `182AB87276B2337B4B1A4CD` was restored;
- custom gpu_ext hook symbols are no longer visible.

These are functional UVM control-policy experiments. Reduced-capacity timings are not native 24 GiB A30 performance, and none of these results should be generalized to LLM workloads without a separate workload phase.
