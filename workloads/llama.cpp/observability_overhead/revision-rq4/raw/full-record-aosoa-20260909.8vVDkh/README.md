# Full-record grouped SoA: five paired blocks completed

The 32-slot grouped SoA (AoSoA) candidate is slower than the existing
field-major SoA implementation in every block. The median paired
throughput change is **-16.275%** (observed
range -16.490% to -15.972%).
Grouping the ten fields into smaller contiguous regions did not improve
this workload; the existing field-major SoA remains the faster measured
full-record implementation. No automatic replacement is made.

| Metric, median | Uninstrumented | Existing SoA | New AoSoA |
| --- | ---: | ---: | ---: |
| Prefill throughput, token/s | 38274.255674 | 27763.563111 | 23249.143131 |
| Paired loss versus uninstrumented | — | 27.449% | 39.250% |
| Post-client whole-arena drain, ms | — | 1495.369 | 1501.485 |

Ratios are calculated within each block before taking the median. The
observed ranges are not confidence intervals. All five blocks and all
15 client timings are included; each client exits zero. The runner
finishes with exit zero and records no collector-teardown errors.
The ten collectors each report 23068672 complete 80-byte records,
524288 active slots, zero overflow/out-of-range entries, and 23068672
nonzero timestamps. These are existing collector outputs, not an extra
performance admission test. The generic legacy parser leaves its
unrelated fields at -1; analysis.json reads the actual full-record log
lines instead of treating those placeholders as measurements.

## Measurement scope

RTX 5090, NVIDIA 575.57.08, CUDA 12.9; TinyLlama 1.1B Q4_K_M,
llama.cpp prefill pp512/tg0, one llama-bench repetition per cell.
Five blocks rotate AoSoA / SoA / uninstrumented. Both tool arms have
automatic warp execution disabled, transport 3 and per-call observation
disabled. The shared bpftime-auto-warp runtime is unchanged.
The throughput is the client's prefill token/s, excluding the later
10741613056-byte arena drain. The same ten u64 fields, per-thread
timestamps, coordinate mapping, 32 banks, 16384 slots/bank and
256 records/slot are retained. There is no event sampling or
leader-only record reduction.

AoSoA source is commit bc7ea6e6. Root copied its four committed build
inputs to:
`/home/yunwei37/workspace/gpu/bpftime-auto-warp/example/gpu/full-record-aosoa-20260909.wXFOqg`.
The successful build command was
`make -C <that-directory> -j2 LAYOUT=aosoa CUDA_HOME=/usr/local/cuda-12.9`.
The new binary was copied to this campaign as kernelretsnoop; the
existing SoA binary from full-record-soa-20260908.HWfuRS was reused.
Neither old build output was overwritten.

[The runner](paired-aosoa.py) was invoked as
`flock /tmp/gpubpf-revision-gpu0.lock flock /tmp/gpubpf-revision-struct-ops.lock python3 <this-directory>/paired-aosoa.py`.
Both leases covered the whole performance campaign. The final client
and collector completed before the queued, separate LMCache experiment
acquired those leases. No driver reload occurred during this comparison.

[analysis.json](analysis.json) contains per-block ratios, formulas,
all client throughputs and the existing collector outputs.
[paired_cells.json](paired_cells.json), per-cell execution/log files and
[run.log](run.log) preserve original results. All 15 cells measure this
new comparison; no completed cell from the previous campaigns was rerun
as recovery or overwritten. The prior AoS/SoA reports, original RTX 5090
Table 1, and P40 numbers remain unchanged. The new grouped layout is an
opt-in negative result, not a claimed optimization or a replacement
Table 1 number. No manuscript was edited.
