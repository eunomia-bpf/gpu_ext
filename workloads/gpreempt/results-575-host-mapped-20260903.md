# GPreempt config-A comparison on the 575 / RTX 5090 port

The five-block native / original-C / BPF comparison completed on 2026-09-03
UTC: **15 of 15 cells passed**, without retries, missing cells or selected
subsets. This is the explicit `host_mapped` compatibility transport, **not a
reproduction of the original GDRCopy transport**. The original GDRCopy attempts
remain failed and separate.

The [independent raw audit](raw/575-host-mapped-three-way-01/audited-analysis-final.json)
recomputes the recorded request samples, numerical checks, policy engagement,
telemetry and cleanup rather than trusting the campaign's saved summary. The
[frozen plan](raw/575-host-mapped-three-way-01/plan.json) and all client/loader
logs, original six-stage request reports and per-cell telemetry are retained.

## Workload and implementation boundary

- One RTX 5090, Linux `6.15.11-061511-generic`, NVIDIA 575.57.08, explicitly
  loaded driver revision `849ea75d`, fixed 400 W.
- Original config A: 60 seconds per cell, VGG19 LC and ResNet152 BE, 100
  requests/s per role, batch one, CUDA graphs, 200 us preprocessing. Each cell
  keeps the original warmup/calibration sequence and checks every output.
- Both policy arms use the same portable host-mapped flag transport and original
  pair of blocking kernels. The original-C arm uses native decisions; the BPF
  arm uses actual host ubpf-JIT decisions plus the persistent kernel timeslice
  callback. Both request LC 1,000,000 us / BE 1 us.
- The baseline uses original native single-context stream priorities on the
  **same custom driver**, with no attached BPF policy. It is not a separate
  stock-driver measurement.
- Model exports are the documented seeded FP32 TVM testing models for `sm_120`,
  not recovered author-supplied binaries or pretrained-accuracy evidence.
- All clients run with the same root privilege and unpinned process affinity.
  GPU and struct-ops leases cover the entire campaign; no competing GPU job or
  compilation was admitted during these timed cells.

Post-reboot original and BPF real-GSP canaries passed immediately before this
campaign. They each validated 2,048 integer outputs and 17 negative cases, and
observed final LC/BE firmware requests of 1,000,000/1 us after CUDA's later
controls. This proves accepted persistent firmware requests for the canaries,
not the physical scheduling quantum. See the retained
[original](../xsched/raw/gpreempt-context-original-849ea75d-postboot-20260903-0147/result.json)
and [BPF](../xsched/raw/gpreempt-context-bpf-849ea75d-postboot-20260903-0147/result.json)
results.

## Results

Values below are medians of the five per-cell measurements. LC p99 is the
nearest-rank percentile of the **sum of the original six service stages**, not
arrival-to-completion latency.

| Arm | LC p99 (ms) | BE p99 (ms) | BE throughput (req/s) |
| --- | ---: | ---: | ---: |
| Native baseline | 1.414351 | 4.819063 | 100.0 |
| Original-C GPreempt | 1.415130 | 6.245411 | 100.0 |
| BPF GPreempt | 1.419397 | 6.268993 | 100.0 |

Paired geometric ratios and percentile bootstrap 95% intervals use complete
blocks as the resampling unit, with 10,000 draws:

| Comparison | Geometric ratio | Paired 95% interval |
| --- | ---: | --- |
| BPF / original-C LC p99, lower is better | 1.002575 | [1.001278, 1.003740] |
| BPF / native LC p99, lower is better | 1.003982 | [1.002388, 1.006216] |
| Original-C / native LC p99, lower is better | 1.001403 | [0.999074, 1.004112] |
| BPF / original-C BE throughput, higher is better | 0.999867 | [0.999600, 1.000000] |

BPF has a small, consistent LC latency cost relative to the original-C policy:
about **0.258%**, with a 95% interval of **0.128% to 0.374%**. These results do
not show BPF outperforming the original policy, and no post-hoc equivalence
claim is made. BE throughput is close to the fixed 100 requests/s offered-load
ceiling; this experiment does not establish saturated throughput or a general
preemption benefit. The native service-time means in the first block were
about 1.400 ms LC and 4.827 ms BE within a 10 ms period, consistent with a
relatively light workload on this newer GPU. No load was changed after seeing
the measurements.

## Correctness and engagement

Across all cells, 179,995 timed requests and 183,295 total requests were checked,
including warmup/calibration. Every 1,000-element result passed with maximum
absolute error zero: 183,295,000 output values checked. The five BPF cells
reported 117,589,908 actual JIT `due` decisions and 29,999 each of hint, block
and release actions, with zero bridge errors. Every BPF cell exercised one
persistent LC and one BE control decision. Every cell ended with zero UVM
references, empty struct-ops state, no compute clients, idle GPU and no new Xid.

The original periodic load generator skips stale slots; exact arrival/drop
counts are not instrumented. The five-request difference from a nominal
180,000 timed requests is **not** reported as five observed drops.

Recalculate without launching CUDA:

```sh
cd workloads/gpreempt
python3 -B analyze_three_way.py raw/575-host-mapped-three-way-01
```

The earlier native and original-C standalone canaries were correctness-only
runs that permitted concurrent CPU builds. They are not included in these
five paired blocks.
