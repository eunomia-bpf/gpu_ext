# Matched warp-array kernelretsnoop comparison

Status: complete, 2026-09-08 UTC. All 15 planned benchmarks returned zero;
no performance cells were excluded. Results below supersede the queued plan.

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

## Completed measurements

All values below come from `cells.json` and the per-cell `llama_bench.log`.
Loss is computed separately in each block as
`100 * (1 - tool_token_s / baseline_token_s)`; the aggregate is the mean
of those five losses, not a ratio using a baseline from another campaign.

| Block | Baseline token/s | gpubpf token/s | NVBit token/s | gpubpf loss % | NVBit loss % |
|---|---:|---:|---:|---:|---:|
| 1 | 38277.300200 | 36394.263894 | 34784.002891 | 4.919460 | 9.126290 |
| 2 | 38296.328010 | 36448.905575 | 34724.115546 | 4.824020 | 9.327820 |
| 3 | 38464.699548 | 36240.621589 | 34813.134044 | 5.782127 | 9.493290 |
| 4 | 38375.907605 | 36376.600695 | 34433.212178 | 5.209797 | 10.273882 |
| 5 | 38394.141054 | 36153.358877 | 34724.402859 | 5.836261 | 9.558068 |
| Mean | 38361.675283 | 36322.750126 | 34695.773504 | 5.314333 | 9.555870 |

Median paired loss is 5.209797% for gpubpf and 9.493290% for NVBit;
the observed ranges are 4.824020--5.836261% and 9.126290--10.273882%.
These are five-block observed dispersions, not confidence intervals.
The matched gpubpf arm has higher prefill throughput in all five blocks.

Both collectors retain 720896 full events per cell. Their final
23199768-byte buffer readback is outside the llama.cpp prefill interval:
gpubpf mean 10.955584 ms (10.225116--11.730516), NVBit mean 7.881000 ms
(7.446809--8.386095). These are the existing raw diagnostic timers, not
additional benchmark cells. The gpubpf readback is slower; the prefill
advantage must not be described as an end-to-end collector-lifecycle win.

## Relation to retained historical numbers

The original RTX 5090 kernelretsnoop losses, 90.705086% for gpubpf and
99.621030% for NVBit, remain unchanged. The ten-pair gpubpf-only array
follow-up remains unchanged at mean loss 5.400619%. This new campaign adds
the matched NVBit warp-array control; it does not replace any old result.
The P40 submission's 8% / 85% kernelretsnoop row is also retained.

GPU-local aggregation helps both implementations. Once NVBit receives
the same bounded warp-array strategy, its measured loss is 9.555870%,
not its historical per-thread channel loss of 99.621030%. Thus this
experiment supports a lower in-prefill mechanism cost for the measured
gpubpf implementation, but not preservation of the historical P40 ratio
or attribution of the entire aggregation benefit to BPF itself.
The bounded geometry and warning-mode limitations stated above still apply.
