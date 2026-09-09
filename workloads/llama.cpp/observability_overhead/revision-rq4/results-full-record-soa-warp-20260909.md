# Warp-contiguous full records: five paired blocks completed

The opt-in physical-warp-contiguous storage index improves prefill throughput
against the existing SoA implementation in **all five blocks**. Median paired
improvement is **20.565%**, with observed range
19.508%–21.149%.
These are within-block ratios, not a ratio of campaign medians; the range
is not a confidence interval.

| Metric, median | Uninstrumented | Existing SoA | Warp-contiguous SoA |
| --- | ---: | ---: | ---: |
| Prefill token/s | 38336.733847 | 27726.515614 | 33381.157927 |
| Paired loss versus uninstrumented | — | 27.641% | 12.862% |
| Post-client whole-arena drain, ms | — | 1485.452 | 1501.739 |

| Block | Uninstrumented token/s | SoA token/s | Warp SoA token/s | Warp/SoA change |
| --- | ---: | ---: | ---: | ---: |
| 1 | 38336.734 | 27726.516 | 33135.292 | +19.508% |
| 2 | 38308.302 | 27791.847 | 33381.158 | +20.111% |
| 3 | 38359.697 | 27756.631 | 33464.875 | +20.565% |
| 4 | 38440.663 | 27646.792 | 33339.489 | +20.591% |
| 5 | 38093.725 | 27646.857 | 33493.785 | +21.149% |

All 15 clients and all ten collectors exit zero; the runner also exits zero.
Each collector reports 23068672 complete 80-byte records, 524288 active
slots, zero overflow/out-of-range records and 23068672 nonzero timestamps.
These are the existing collector outputs, not additional experiment gates.
The legacy generic parser's unrelated sentinel fields are not measurements:
analysis.json reads the full-record log lines directly.

## What changed and what did not

The local Qwen implementation changes only the storage-slot mapping, not
the event payload or hook placement. For the measured rope geometry,
grid 2048x1x1 and block 1x256x1, physical warp lanes differ in thread_y.
The old index puts successive lanes 2048 slots apart, spanning four banks;
the new block-major/thread-major index puts them in adjacent slots in the
same bank. Within each field, their stores are eight bytes apart.
The data support the layout change on this workload; they do not isolate
memory transactions from map-lookup or indexing-code effects.

Both arms preserve all ten u64 fields, individual timestamps, 32 banks,
16384 slots/bank, 256 records/slot and the final 10741613056-byte drain.
No events are sampled, deduplicated or replaced with leader-only records.
Automatic warp execution remains OFF and transport remains 3. The measured
improvement is not a new automatic compiler aggregation result.

This remains a finite GPU buffer for the fixed workload. The roughly
1.5-second post-client drain is outside the reported prefill throughput,
and is not an unbounded streaming or end-to-end logging cost.
Neither the old SoA/AoS/AoSoA results nor the original RTX 5090/P40 Table 1
numbers are replaced. This is a separate matched optimization comparison;
the earlier AoSoA regression remains recorded.

## Execution and records

RTX 5090, NVIDIA 575.57.08, CUDA 12.9; TinyLlama 1.1B Q4_K_M,
llama.cpp pp512/tg0, one llama-bench repetition per cell. Five blocks rotate
warp SoA / frozen SoA / uninstrumented. Both shared GPU/struct-ops leases
cover the campaign. No driver change occurs within it.

Implementation/build/start record: f144de04.
[Raw campaign](raw/full-record-soa-warp-20260909.cjoIog/README.md) includes
the four build inputs, successful isolated build log and exact command.
The new binary remains local; the old full-record-soa-20260908.HWfuRS
binary and shared build-auto-warp-575 runtime were reused, not rebuilt.
[paired_cells.json](raw/full-record-soa-warp-20260909.cjoIog/paired_cells.json)
retains every measurement and original client output.
[analysis.json](raw/full-record-soa-warp-20260909.cjoIog/analysis.json)
retains the per-block formulas and extracted full-record observations.

Recompute ratios as 100*(warp/SoA-1) and overhead as
100*(1-tool/baseline) within each block, then take the median across five
blocks. The complete rows are above and in analysis.json. No completed
old cell was rerun as recovery, and no manuscript was edited.

