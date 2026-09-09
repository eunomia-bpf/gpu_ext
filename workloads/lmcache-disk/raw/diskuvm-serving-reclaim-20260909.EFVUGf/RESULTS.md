# Per-range re-offload does not yet fix disk/UVM serving OOM

Source d6be28a3, RTX 5090, 2026-09-09 05:01:06–05:11:29 PDT.
Five observations were saved before the existing deferred stop, requested after
actual EngineCore OOM in both block-0 policy arms. The current cell finished;
runner exit was 3. The original UVM module and GDS/KV loaders were restored
(`RESTORATION_OK=1`). No local OpenCode session was stopped.

## All recorded observations

| Block | Reclaim arm | Recorded warm token/s | Completed tokens | Observed streamed tokens | HTTP failures | UVM restores / fallback |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | stock | 70.067316 | 8192 | 8192 | 0 | 31 / 37 |
| 0 | native | 64.882373 | 2048 | 2167 | 2 | 3 / 17 |
| 0 | BPF | 64.779982 | 2048 | 2167 | 2 | 3 / 17 |
| 1 | native | 71.095272 | 8192 | 8192 | 0 | 0 / 82 |
| 1 | BPF | 66.118279 | 8192 | 8192 | 0 | 7 / 33 |

The two block-0 policy rates describe interrupted output, not successful
serving throughput. Three cells complete the requested 8192 output tokens.
There is no complete three-arm block and no five-block performance estimate.
The block-1 native cell falls back for every restore; even its complete output
does not measure the new UVM transport. All observations remain recorded.

All five cells prepared 48 backings and reported the first restore exception:
`diskuvm_fault_read failed with code 2`. CUDA 12.9 defines error 2 as
`cudaErrorMemoryAllocation`; the helper currently returns errors from several
CUDA calls, so this does not identify the failing call. Block-0 native and BPF
server logs additionally record a real `torch.OutOfMemoryError` at line 862:
a 20 MiB allocation fails with 6.38 MiB free. The EngineCore then dies and
HTTP streams fail or end early. This is an execution failure, not rejection
on a clock, counter, or numerical-accuracy threshold.

## What the attempted repair established

The d6be28a3 per-backing lock and post-copy OFFLOAD/QUERY completion are present
in this run. They did not suffice to prevent OOM. The first-error logging
successfully exposed a previously swallowed CUDA allocation failure.
No redundant calibration, changed workload, or shortened request length was
used. All three arms use the same opt-in disk/UVM transport; stock means
the disabled KV-reclaim policy, not untouched GDS transport.

A source-level lead remains to be tested. In the current disk driver,
`uvm_disk_backing.c:669` migrates GPU-resident pages to CPU with
`uvm_va_block_make_resident(...EVICTION)`; the offload worker later unmaps
PTEs and removes CPU chunks. This is not evidence of releasing physical
GPU chunks. The ordinary PMM eviction path in `uvm_va_block.c:13375`
also unmaps the MMU chunk, marks the chunk evicted, and clears its slot.
That path has its own locking and PMM ownership contract and must not be
copied blindly. The next implementation task is to establish and repair
physical GPU-chunk reclamation for the disk-backed range using the existing
driver ownership rules. It is not yet proven to be the sole cause of OOM.

The helper also creates a CUDA stream and allocates 4 KiB of sample storage
for each traversal; samples are not read by the Python caller. That is another
bounded allocation-pressure lead, not an established explanation.

## Reproduction and retained data

`run-serving.sh` retains the exact invocation, fixed five rotated blocks,
62502 ns/token saved calibration, driver/module restoration and helper path.
`source/` preserves the frozen adapter and CUDA sources.
`cells/` retains every result, streamed response record, server log and
available per-process diagnostics. `cells/summary.json` is the runner's
unfiltered mechanical summary; it must not be quoted as a successful
comparison of these interrupted treatments.

Older LMCache disk I/O, serving, failed bring-up, Table 1 and trampoline
measurements remain unchanged. No manuscript was edited.

