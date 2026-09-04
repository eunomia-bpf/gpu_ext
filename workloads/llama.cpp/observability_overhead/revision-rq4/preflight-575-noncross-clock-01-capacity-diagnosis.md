# Kernelretsnoop capacity diagnosis

`raw/preflight-575-noncross-clock-01` is a genuine failed preflight and remains
failed. Its pp32 gpubpf timing cell observed 44 launches with 32,768 logical
threads per launch, but the ring was allocated for 22,528 thread coordinates.
The resulting 450,560 OOB drops are exactly `(32,768 - 22,528) × 44`; the
collector correctly rejected the cell. The matched NVBit cell independently
reported 1,441,792 events (`32,768 × 44`).

The full width is derived from the selected llama.cpp rope launch, not copied
from pp32. The selected KV rope has four KV heads, one token dimension, and a
256-thread Y block. Therefore its logical coordinate count is
`4 × pp × 256`: 32,768 at pp32 and 524,288 at pp512. The fixed warm/timed
llama-bench path selects 44 launches, so full timing must observe multiplicity
44 at all 524,288 coordinates and 23,068,672 total events. The older pp512
result at `workloads/llama.cpp/results/exp_observability_overhead/20260706_180506`
used a capped/lossy 8,192-slot collector and is evidence that this workload was
run, not evidence for a safe current capacity.

The runtime's dense per-coordinate ring needs a 24-byte slot header and an
88-byte aligned allocation for each 80-byte record. A pp512 256-entry layout
would exceed 10 GiB, and 44 payload records alone exceed 1.7 GiB, so neither is
compatible with the frozen 1,000 MiB segment. The minimal repair is
phase-specific map sizing before skeleton load:

- correctness stays at 22,528 coordinates × 256 entries and retains the exact
  720,896-event/220-launch multiplicity oracle;
- timing uses `pp × 1,024` coordinates × 16 entries: about 44.75 MiB for pp32
  and 750,780,448 bytes (about 716 MiB) for pp512.

Sixteen is buffering capacity, not an allowed-loss threshold: the loader drains
concurrently, and any OOB/full/bad-size/other drop, dirty/pending record, second
drain event, invalid coordinate, wrong slot/entry allocation, or mismatch from
the exact 44-launch/event geometry rejects the cell. A new two-tool pp32
preflight must pass before any subset full run. If pp512 cannot drain this
layout without loss, the full experiment fails; it is not reclassified or
weakened.
