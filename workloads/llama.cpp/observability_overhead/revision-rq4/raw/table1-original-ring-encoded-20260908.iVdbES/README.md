# Original kernelretsnoop object: transport optimization comparison

First measurement of the mode-2 transport candidate, 2026-09-08. Reuses
the existing Table 1 runner and llama.cpp pp512/tg0 protocol: ten rotating
blocks, no-probe baseline, same kernelretsnoop object with optimization off,
and that object with aligned-word copy plus encoded-tail publication enabled.
This tests the combined transport candidate, not encoded-tail publication
in isolation. No source-policy replacement or event aggregation is requested.

Runtime bpftime `e61bdb3`; diagnostic invocation counting explicitly disabled.
Runner main `43c6d3e8` (later commits do not change its measurement code).
The original example source is prepared without the legacy runner's ring
capacity patch, with sufficient thread slots/shared memory for its own
declared map. Old Table 1, array and warp-map campaigns are retained.

Primary metric: prefill token/s, relative loss against the paired no-probe
baseline, and paired optimized/original transport change. No required
improvement threshold or clock-precision test. A slower result is retained.
This is a three-arm transport ablation, not a fresh NVBit head-to-head or a
replacement for the published seven-arm Table 1 comparison.

Command (from revision-rq4, under both shared experiment leases):

```sh
BPFTIME_GPU_WARP_HOOK_CALL_COUNT=0 python3 run_table1_perf.py \
  --auto-warp-three-arm --auto-warp-task kernelretsnoop \
  --auto-warp-transport 2 --blocks 10 \
  --bpftime-root /home/yunwei37/workspace/gpu/bpftime-auto-warp \
  --bpftime-build-dir /home/yunwei37/workspace/gpu/bpftime-auto-warp/build-auto-warp-575 \
  --output-dir raw/table1-original-ring-encoded-20260908.iVdbES
```

The first pass completed all 30 attempts: ten baseline throughputs were
measured, but all 20 attached loaders exited with signal 6 before timing.
The original 256-entry/thread map needs more than 10 GiB, while the runtime
clamped the requested 12,301 MiB segment to 10,240 MiB; allocation then threw
`boost::container::length_error`. This is not a transport performance result.

Runtime `241872b` raises only the explicit shared-memory size ceiling to
16,384 MiB (default remains 50 MiB). The resume uses the same compiled probe
object/source, preserving the original ring capacity, and reruns only the
20 failed attached cells in `../table1-original-ring-encoded-resume-20260908.ceOL6M/`.
The ten completed baselines are retained and reused, not repeated.
