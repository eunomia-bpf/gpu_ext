# Device/storage continuation, 2026-09-08

This session does not edit the manuscript. Completed measurements and adverse
results remain intact. Non-trivial development runs through local OpenCode;
the root reviews, builds, measures, records, commits and pushes. At most three
local sessions run concurrently, without stopping them for silence.

## Completed immediate requests

- [Disk-backed UVM restoration](../../workloads/lmcache-disk/results-disk-uvm-restore-20260908.md):
  five complete full-read runs, published in `cde80eeb`. At 256 MiB,
  median offload/CPU restore/GPU restore is 278.299/169.819/170.601 ms.
  This is a real file-backed same-address primitive, not a new end-to-end
  LMCache comparison or established GPU-direct P2P transfer. Restoration is
  CPU-first; subsequent GPU reads take 5.408 ms versus 0.337 ms initially.
- [Trampoline geometry/work timings](../../microbench/fig15-device/strict-warp-map-scaling/results-geometry-work-20260908.md):
  all 180 measurements published in `a0c5edf6`.
  Automatic execution is about 57–58% slower in the zero-work CTA sweep;
  larger compute reduces the relative difference. No old sample is removed.
- [Actual callback counts](../../microbench/fig15-device/strict-warp-map-scaling/results-observed-counts-20260908.md):
  all 12 separate diagnostics published in `8429a122`. Observed counts fall
  by 32x, which does not imply faster execution. Counts are not substituted
  for the counter-free timing measurements.

## Current GPU campaign

Completed follow-up: [all ten transport pairs](../../workloads/llama.cpp/observability_overhead/revision-rq4/results-original-ring-encoded-20260908.md)
now have numeric throughput and benchmark exit zero. Mean legacy/optimized
throughput is 34.514/194.059 token/s; the paired ratio median is 5.7702x.
Collector shutdown remains adverse (-9 in all twenty cells), and absolute
overhead is still about 99.5% for the optimized arm. The historical in-progress
checkpoint below is superseded for measurement completion, not for those
limitations. No completed cell was repeated.

The existing Table 1 runner compares original kernelretsnoop with legacy
transport versus aligned-word copy plus encoded-tail publication (mode 2).
The BPF object, record payload and 256-entry per-thread capacity are unchanged.
This is a transport optimization, not a new NVBit comparison or permission
to collapse distinct per-thread events into one observation.

The older 90.7051% kernelretsnoop result is a different probe configuration:
its retained cell records report 720,896 events, 16,384 coordinates and
44 entries per thread; its source patch selects one warp leader and emits
three coordinates plus a timestamp. The new original object emits all
thread coordinates/block dimensions and timestamp (80 bytes), with 256
entries per thread. The transport on/off pair preserves that original object;
its speedup must not be applied arithmetically to the older warp-level result.
Neither historical configuration or measurement is removed.

The first attempt completed ten baselines but failed all twenty loader starts
at the runtime's 10 GiB segment ceiling. Runtime `241872b` allows explicitly
requested segments up to 16 GiB, without changing the default. The resume
runs only those twenty failed attached cells, retaining the original ten
baselines. Baselines precede the repaired cells; it is not a fresh fully
interleaved thirty-cell campaign.

Records and resume command:
`workloads/llama.cpp/observability_overhead/revision-rq4/raw/table1-original-ring-encoded-resume-20260908.ceOL6M/`.
The initial attempt remains in sibling
`table1-original-ring-encoded-20260908.iVdbES/`.

Status at this checkpoint: in progress, not a final result. Numeric benchmark
exits and loader teardown errors are recorded separately. The runner's
`absent` setup-marker field must be read with its logging configuration:
`SPDLOG_LEVEL=warn` suppresses the runtime's informational transport marker.
An absent message does not by itself show whether optimization was disabled.
No added clock or logging gate blocks performance collection.

## Local implementation queue

1. Qwen 27B, session `ses_f7e9d6af4ffeyntfT0hZqhYC5c`: read-only source/log
   analysis of remaining transport overhead and loader teardown. No edits
   to the runtime currently used for measurement; no repeated GPU cells.
2. GLM, session `ses_f80c49c7cffebjDrdYLKy8X0ZC`: finish the XSched Level-2
   sm_120 native LDC adapter against actual NVCC-produced instructions.
   The equivalent branchless BPF guardian source separately awaits a real
   component build. Existing Level-1 results do not count as Level-2 results.
3. Qwen 27B, session `ses_f7e5a4134ffeup2HgHIRqLaiho`: prepare an isolated
   opt-in disk-restoration/GPU-promotion patch. CPU fault behavior and the
   measured CPU-first mode stay unchanged. No driver reload or performance
   claim before root integration and a new scoped run.

Qwen Next's earlier provider calls ended with actual HTTP 524 errors; the
current fallback therefore uses two Qwen 27B sessions and one GLM session,
not a fourth session. GLM's LDC generation was resumed after a terminal
length limit, not interrupted for lack of output. Heavy builds and GPU runs
wait for the active performance campaign. Hummingbird host/device work is
still queued, without new implementation or measurement claims.
