# RTX 5090 device-observability result review

## Five-part verdict

- **Run status:** valid for the predeclared `kernelretsnoop` and `threadhist`
  subset. The independent analyzer accepted the five correctness cells and all
  10 randomized five-cell blocks, with no rejection, exclusion, or retry.
- **Tested hypothesis:** mixed and partly contradicted. gpubpf has lower
  overhead for the exit-count histogram, while matched NVBit is slightly
  faster for full exit records.
- **Research value:** supporting evidence for RQ4 and for a task-dependent
  mechanism boundary.
- **Paper impact:** replace the two corresponding device-observability rows
  with the RTX 5090 paired result and explicitly report both directions.
- **Next decision:** retain `launchlate` as invalid cross-clock evidence. Do
  not describe the original three-tool/seven-arm Table 1 campaign as complete.

## Accepted evidence

- Dependency gate: `raw/preflight-575-noncross-clock-04`
- Paper-value run: `raw/full-575-noncross-clock-02`
- Independent record: `raw/full-575-noncross-clock-02/independent-audit.json`
- Hardware/software: RTX 5090, driver 575.57.08, Linux 6.15.11, CUDA 12.9,
  NVBit 1.8, TinyLlama-1.1B Q4_K_M, llama.cpp build 7102 (`26836b27`)
- Workload: prefill, pp=512, tg=0, ten seed-1797 randomized paired blocks

| Task | gpubpf mean overhead | Matched NVBit mean overhead | Paired effect, NVBit - gpubpf |
|---|---:|---:|---:|
| 32-byte exit records | 99.6627% | 99.6208% | -0.04185 pp [-0.04355, -0.04029] |
| Final exit-count histogram | 4.0071% | 10.3006% | +6.29351 pp [6.12507, 6.47076] |

The no-probe geometric-mean prefill throughput was 38,056.928 token/s. The
corresponding instrumented geometric means were 128.374 and 144.302 token/s for
the gpubpf and NVBit exit-record arms, and 36,531.772 and 34,136.772 token/s
for the histogram arms.

## Validity and claim boundary

All processes exited zero and deterministic outputs matched. Every exit-record
arm produced exactly 23,068,672 nonzero-timestamp records from 44 selected
launches, with 524,288 coordinates at multiplicity 44 and no drops, malformed
records, pending data, or collector errors. Histogram aggregates matched at
23,068,672 samples, 524,288 nonzero logical threads, and 44 launches; gpubpf
also passed the complete 1,048,576-entry readback gate. NVBit retained only
aggregate histogram evidence, so the result does not establish elementwise
histogram-vector equality.

The metric is prefill token/s during the instrumented benchmark and excludes
collector/readback work after benchmark exit. The adapters match the selected
kernel and observable but retain each system's native transport. This is not a
stock-NVBit-tool comparison, an intrinsic callback/JIT microbenchmark, a
verifier-cost measurement, or evidence that one system wins for every device
task, prompt length, workload, or GPU. The pp=32 preflight is only a dependency
gate and is not a paper result.
