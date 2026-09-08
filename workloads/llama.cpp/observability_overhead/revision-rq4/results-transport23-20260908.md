# Original-record transport layout: five paired measurements

Transport 3 improves prefill throughput over transport 2 in all five pairs,
without changing the original per-thread BPF object or its 80-byte records.
This is a memory-layout optimization, not one-event-per-warp sampling.

| Metric | Transport 2 | Transport 3 |
| --- | ---: | ---: |
| Median prefill token/s | 208.756714 | 346.808416 |
| Mean prefill token/s | 209.208025 | 345.716344 |
| Completed measurements | 5 | 5 |

The median paired throughput ratio is **1.650416×** (+65.0416%), with a
paired-median bootstrap 95% interval of **1.632708–1.680261×** (10000
resamples, seed 1797). Prefill duration decreases by median 39.4092%.

## Measurement scope

- RTX 5090, NVIDIA 575.57.08; bpftime runtime `886b4ca`.
- Existing llama.cpp pp512/tg0 TinyLlama 1.1B Q4_K_M workload and Table 1
  `run_arm_cell` helper; invocation/helper commit `23e03e4a`.
- Five rotating pairs, both with AUTO_WARP_EXECUTION enabled. The outer
  transport labels select header/payload layout and host reassembly; the
  shared internal `auto_warp_on` label does not imply leader-only callbacks.
- Same original object, 256 entries per thread, 524288 allocated thread
  slots, 80-byte payloads and 11823743008-byte ring. Diagnostic counting off.
- No completed baseline, NVBit or earlier transport batch was repeated.

All ten benchmarks and collectors exit zero. Every collector reports
23068672 collected events, all timestamped, with zero OOB/full/bad-size/other
drops and zero pending events. Each private shared-memory segment is removed.
Collectors finish normally after SIGINT, with no artificial short deadline.
These are observations from the timed runs, not additional GPU test cells.
The logs retain the disabled oracle and segment-mismatch diagnostics;
this report does not claim that the disabled oracle passed.

## Interpretation and retained history

This establishes a layout optimization for the full per-thread stream, but
**does not establish low absolute overhead**. There is no fresh uninstrumented
baseline or NVBit arm in this batch, so it is not a replacement Table 1 H2H.
The earlier full-stream campaign remains in
[its report](results-original-ring-encoded-20260908.md).

The older Table 1 kernelretsnoop configuration used compact 32-byte,
warp-leader records (720896 events); this batch uses 80-byte per-thread
records (23068672 events). Do not multiply this speedup into the old
90.7051%/99.6210% loss figures, or replace any P40/5090 historical numbers.
GPU-local full-record buffering remains an optimization to investigate,
not a measured result of this batch. No manuscript file was changed.

Raw commands, all ten cells, collector logs and paired analysis:
[transport23-pairs-20260908.Bq3W8h](raw/transport23-pairs-20260908.Bq3W8h/README.md).
