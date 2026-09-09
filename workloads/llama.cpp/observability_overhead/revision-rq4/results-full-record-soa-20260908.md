# Full-record SoA layout: five completed paired blocks

The field-major SoA implementation improves prefill throughput over the
record-major AoS implementation in all five blocks. Median paired gain is
**19.229%**, with observed range **18.538%–19.536%**. This is a layout
implementation comparison, not automatic warp execution or event sampling.

| Metric, median | Baseline | AoS | SoA |
| --- | ---: | ---: | ---: |
| Prefill throughput, token/s | 38455.024333 | 23273.037227 | 27713.335852 |
| Paired throughput loss versus baseline | — | 39.385% | 27.621% |
| Post-run bulk drain, ms | — | 1471.606 | 1474.772 |

The loss figures are medians of within-block ratios, not ratios of the
throughput medians. SoA's paired baseline loss ranges from 27.494% to
28.148%; substantial overhead remains. This does not reach the original
P40 Table 1 kernelretsnoop overhead, and the distinct original Table 1
record representation is not substituted with this full per-thread stream.

## Scope and records

All 15 clients and all 10 collectors exit zero. Each collector reports
23068672 committed records, 524288 active slots, zero overflow/out-of-range
events, and 23068672 nonzero timestamps. Each drains 10741613056 bytes
after client completion. This remains finite GPU-local capture followed
by whole-arena collection, not unbounded online streaming.

The same pp512/tg0 TinyLlama workload, RTX 5090, driver 575.57.08 and
bpftime `886b4ca` runtime are used. Both tool arms have automatic warp
execution disabled. SoA source is `e6b049bc`; the AoS arm uses the frozen
binary from the earlier completed campaign. Ten u64 fields, per-thread
timestamps, 32 banks, 16384 slots per bank and 256-record slot capacity
are unchanged. Physical placement changes from 80-byte record spacing
to 8-byte per-field spacing across adjacent threads. This experiment does
not isolate memory coalescing from every resulting compiler/codegen cost.

The [continuation](raw/full-record-soa-continuation-20260908.RBApri/paired.py)
rotates SoA/baseline/AoS over five blocks, retaining the successful
[initial SoA cell](raw/full-record-soa-20260908.HWfuRS/README.md) rather
than repeating it. Fourteen new cells complete the comparison. The first
SoA observation preceded its controls by about 24 minutes, including an
intervening separate GPU campaign. Excluding block 1 leaves a median
paired gain of 19.322% across the remaining four blocks. These are
single-machine observations; reported ranges are not confidence intervals.

The [per-block analysis](raw/full-record-soa-continuation-20260908.RBApri/analysis.json)
records every ratio, its formula, and each collector's existing outputs.
The runtime's map-pointer verifier warnings remain in the raw logs; this
is performance evidence, not strict-admission evidence. No clock-precision
gate or additional correctness campaign was introduced.

The [previous AoS/ring campaign](results-full-record-device-buffer-20260908.md),
original RTX 5090 Table 1, P40 numbers, and failed bring-up records remain
unchanged. GPU locks were released after this batch for the separate
Native/Off larger-CTA experiment; no shared runtime was rebuilt.
