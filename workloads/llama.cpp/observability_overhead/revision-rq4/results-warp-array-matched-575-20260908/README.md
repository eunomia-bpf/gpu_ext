# Matched warp-array kernelretsnoop comparison

Status: queued behind the active LMCache comparison; no results yet.

This tests the mechanism cost of the same warp-level observation and
GPU-local buffering in gpubpf and NVBit. The historical per-thread NVBit
channel collector is not the matched control. Its old results and the
completed ten-pair gpubpf array campaign remain unchanged.

Five new paired blocks rotate baseline, gpubpf kernelretsnoop GPU array,
and NVBit `OBS_MODE=kernelretsnoop_warp_array`. Each block runs all three
arms on the same RTX 5090 / NVIDIA 575.57.08, TinyLlama-1.1B Q4_K_M pp512
workload, CUDA graphs disabled, CPU affinity 8--15, with the existing
warmup and one measured repetition. These new controls are necessary for
an interleaved NVBit comparison; they do not repeat or overwrite completed
cells in another campaign.

The existing `run_table1_perf.run_arm_cell` and `write_records` entrypoints
run the full performance workload directly. No preflight, clock-accuracy
check, admission rule, new timeout, or additional correctness campaign is
introduced. Existing collector diagnostics are retained without filtering
performance measurements. The old parsers may leave their ring/channel
fields unset for the new arrays; their actual raw labels remain in logs.

Both arrays store full 32-byte coordinate/timestamp events at warp exits,
with 16384 per-warp counters and 44 events per warp in a 23199768-byte
device-local buffer. Each performs final whole-buffer host readback outside
prefill timing, reported separately. This bounded, serial-launch geometry
is not an unbounded concurrent-kernel streaming collector. The existing
gpubpf warning-mode runtime limitation remains unchanged.

Assets: gpubpf collector at
`/var/tmp/kernelretsnoop-onevalue-build.x1aICR/example/gpu/onevalue-array`,
bpftime `build-table1-575-warp` under `bpftime-table1-hostfix-plt`, and the
NVBit library built from `e895c6b1` at
`/tmp/nvbit-warp-array-build-GlqYJ8/observability/observability.so`.
See [the NVBit build handoff](../nvbit_adapters/warp-array-handoff.md) and
[the gpubpf build record](../onevalue-array-candidate/build-and-measurement.md).

The primary measurements are llama.cpp prefill token/s and same-block
percentage loss against baseline, with paired dispersion across blocks.
A win, tie, or loss is retained; no target ratio or desired winner filters
the measurements. Matching observation granularity does not guarantee the
same generated instructions or overhead. Full completion means all 15
planned cells have terminal outcomes recorded, not that one arm wins.
