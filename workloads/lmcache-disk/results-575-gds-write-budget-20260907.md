# LMCache live write protection: five completed paired blocks

All 25 fresh-process cells completed with child exit zero and all 4,000
requests completed (1,600 reads, 2,400 writes). The shared event-driven
executor's cumulative write budget changes from 10 ms to 200 ms. Native and
BPF both improve read tail and write throughput relative to their same-run
10 ms controls in all five blocks. This is a policy/executor-parameter gain,
not a claim that BPF instructions intrinsically accelerate storage.

Implementation: main `e380122e`, development `6008914a` (local Qwen 27B
OpenCode implementation, root review and experiments). The saved GDS-enabled
575.57.08 module and same GDS BPF program remained attached throughout.

## Performance

Medians across five cells per arm. Read latency starts at scheduled arrival;
write throughput includes deferred completion, not only physical I/O service.

| Metric | FIFO | Native 10 ms | BPF 10 ms | Native 200 ms | BPF 200 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Read p50, ms | 49.156 | 144.308 | 182.132 | 54.702 | 50.205 |
| Read p99, ms | 323.707 | 245.706 | 239.091 | 123.141 | 118.097 |
| Write throughput, MiB/s | 5043.566 | 5105.571 | 4988.735 | 5732.373 | 5786.561 |
| Total bandwidth, MiB/s | 8405.943 | 8509.285 | 8314.559 | 9553.955 | 9644.269 |

Paired changes below are computed within each block, not from the ratio of
the descriptive medians.

| Comparison | Read-p99 paired changes, blocks 0--4 (%) | Median |
| --- | --- | ---: |
| bpf200/bpf10 | -46.451, -51.011, -33.841, -62.689, -63.271 | -51.011% |
| native200/native10 | -42.327, -54.016, -53.999, -36.682, -59.426 | -53.999% |
| bpf200/native200 | -17.414, -4.096, +33.849, -39.718, +32.615 | -4.096% |
| bpf200/fifo | -50.562, -47.018, -61.904, -73.000, -67.965 | -61.904% |
| native200/fifo | -40.138, -44.755, -71.538, -55.211, -75.844 | -55.211% |

BPF 200/10 read p99 improves by paired median **51.011%** (range
33.841--63.271%); write throughput improves **15.315%** (11.013--23.871%).
Native 200/10 read p99 improves **53.999%**, and write throughput improves
**12.698%**. Every pair improves on both metrics in each implementation.
Relative to FIFO, BPF 200 improves p99 by median **61.904%** and write
throughput by **17.989%**, again in all five pairs.

BPF/native at 200 ms is mixed: read-p99 paired change has median -4.096%,
range -39.718% to +33.849%, with two adverse pairs. Write-throughput change
has median +0.527%, range -1.325% to +4.644%. BPF/native read-p50 changes
have median +28.805%, range -30.206% to +165.587%. Thus the data support
recovering the same qualitative policy benefit, not tight latency equivalence
or a universal BPF advantage. BPF 200/FIFO read-p50 also regresses in two
pairs; it is not an all-metrics/all-pairs win.

## What changed, and why the result is plausible

The [admission diagnostic](results-575-gds-admission-timing-20260907.md)
found tens of microseconds of median admission cost, while storage-request
tails were hundreds of milliseconds. Most 10 ms writes were therefore
released while demand reads remained outstanding.

In this campaign, native 10 ms exhausts the write budget for
74/82/78/77/74 writes per cell; BPF 10 ms exhausts it for 79/79/77/77/80.
Native 200 ms exhausts zero in every cell, and BPF 200 ms exhausts
0/1/0/0/0. Native/BPF total decisions also fall from 310--328/318--324 at
10 ms to 246--256/244--262 at 200 ms. The observations support the
interpretation that longer demand protection avoids premature write
submission and unnecessary re-decisions. They do not identify SSD-internal
service behavior or prove a general optimum at 200 ms.

The policy's actual native/BPF decision is still consulted at each admission.
Demand reads always submit; safe background writes defer when live pending
demand exists and budget remains. The trusted shared executor wakes at demand
drain or cumulative expiry, then obtains a fresh decision before submitting.
**200 ms is the executor's cumulative budget, not the scalar BPF defer
duration.** The existing scalar clamp and policy program are unchanged;
the existing event-driven executor's wakeup semantics are retained. This
does not establish faithful replay of the old 1 ms polling decision sequence.
The experiment tunes one disclosed parameter of this storage-policy adapter;
it does not claim a new LMCache paper algorithm or hardware GPUDirect P2P.

## All cells

| Block | Position | Arm | Read p99, ms | Write MiB/s | Budget-exhausted writes |
| --- | --- | --- | ---: | ---: | ---: |
| 0 | 0 | fifo | 258.975 | 4666.464 | 0 |
| 0 | 1 | native10 | 268.809 | 5144.632 | 74 |
| 0 | 2 | bpf10 | 239.091 | 4954.830 | 79 |
| 0 | 3 | native200 | 155.029 | 5732.373 | 0 |
| 0 | 4 | bpf200 | 128.032 | 5855.290 | 0 |
| 1 | 0 | native10 | 267.792 | 4827.007 | 82 |
| 1 | 1 | bpf10 | 241.067 | 5005.828 | 79 |
| 1 | 2 | native200 | 123.141 | 5527.966 | 0 |
| 1 | 3 | bpf200 | 118.097 | 5557.120 | 1 |
| 1 | 4 | fifo | 222.900 | 5087.603 | 0 |
| 2 | 0 | bpf10 | 228.673 | 4988.735 | 77 |
| 2 | 1 | native200 | 113.028 | 5753.868 | 0 |
| 2 | 2 | bpf200 | 151.287 | 5690.987 | 0 |
| 2 | 3 | fifo | 397.122 | 5238.211 | 0 |
| 2 | 4 | native10 | 245.706 | 5105.571 | 78 |
| 3 | 0 | native200 | 144.986 | 5864.246 | 0 |
| 3 | 1 | bpf200 | 87.401 | 5786.561 | 0 |
| 3 | 2 | fifo | 323.707 | 4782.271 | 0 |
| 3 | 3 | native10 | 228.980 | 4992.782 | 77 |
| 3 | 4 | bpf10 | 234.250 | 5018.065 | 77 |
| 4 | 0 | bpf200 | 105.266 | 5950.862 | 0 |
| 4 | 1 | fifo | 328.593 | 5043.566 | 0 |
| 4 | 2 | native10 | 195.636 | 5271.026 | 74 |
| 4 | 3 | bpf10 | 286.600 | 4804.066 | 80 |
| 4 | 4 | native200 | 79.377 | 5686.748 | 0 |

## Reproduction, retention and limitations

[Plan](gds-control/write-budget-experiment-20260907.md) and
[raw directory](raw/gds-write-budget-575-20260907-five-block-02/) contain
the exact 25 commands/order in `run-plan.json`, every `result.json`,
per-request and policy feedback records, child return codes, full runner
logs, shared cuFile log and `paired-analysis.json`. Read p99 uses the
existing nearest-rank metric; with 64 reads it is the maximum. Analysis
retains every block and computes 100*(numerator/denominator-1).

Each arm occupies every position once in the five rotations. Traffic remains
24 MiB objects, 64 reads/96 writes, 2/4 ms arrivals, 4096 MiB GPU pool, real
LMCache 0.5.4 cuFile compatibility-mode direct I/O on RTX 5090. GIL retention
and per-admission diagnostic timers are disabled. No other GPU workload ran.
This is storage-request latency, not vLLM TTFT.

Successful generated cache directories were removed after each process exited,
outside timing; each `cache-cleanup.txt` records its path and apparent size.
All result/log files remain. This common procedure keeps free disk space
roughly steady instead of retaining 94 GiB of regenerable payloads, but can
affect later SSD state. Therefore improvements here use contemporary matched
10 ms/FIFO controls, not historical cross-campaign timing as a causal control.
The original campaigns and their adverse numbers are untouched.

The first launcher invocation incorrectly precreated the runner-owned cell
directories. Its 25 failures occurred before I/O, are retained in the sibling
`gds-write-budget-575-20260907-five-block` directory and committed in
`90dd1ada`. The corrected launcher writes initial logs in parent directories.
No successfully measured cell was discarded, retried or overwritten.

All cuFile compatibility/mount-option diagnostic messages are retained. Five
blocks on one device/workload establish this measured improvement, not general
optimality, a confidence interval for all deployments, or tight BPF overhead.
The next useful work is integration of this scoped result and its policy versus
mechanism distinction, not repeating this completed matrix.
