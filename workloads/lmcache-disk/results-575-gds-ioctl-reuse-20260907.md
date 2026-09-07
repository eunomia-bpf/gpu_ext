# LMCache BPF allocation reuse: measured decision-path improvement

Implementation `b0eeb67c` reuses a 136-byte ioctl buffer and ctypes view,
without changing the policy or skipping BPF calls. It is default-off through
`LMCACHE_GDS_IOCTL_REUSE=1`. The historical storage results remain unchanged.

## Result

Six rotated blocks, three arms, 1,000 measured decisions per arm per block:
18,000 measured calls and 1,800 warmups. BPF uses the actual attached
command-82 driver policy, not a mocked timing path. The command exited zero;
all measured native/legacy/reuse action, delay, priority and batch tuples agree
for the same inputs. No storage transfers occur in this measurement.

| Per-block metric, median across six blocks | Native | Legacy BPF | Reuse BPF |
| --- | ---: | ---: | ---: |
| Median decision time, us | 0.4725 | 1.5755 | 1.1900 |
| Decision p99, us | 1.495 | 2.670 | 2.234 |

The within-block median-time changes are -24.937%, -25.756%, -23.766%,
-22.595%, -24.634%, and -21.651%: median **-24.200%**, improving in all six
blocks. Paired p99 changes have median **-17.114%** (range -18.198% to
-10.637%), also improving in all six blocks. These ranges are not confidence
intervals. Loop throughput improves by paired median 27.182%, but includes
timer calls, Python loop and record construction; it is not storage bandwidth.

## What this does and does not establish

Allocation reuse reduces measured isolated decision cost by about 0.39 us.
The optimized BPF path still costs more than native; this is not native parity.
It does not establish lower LMCache request p99 or higher disk throughput.
The earlier concurrent admission diagnostic measured 1.827/17.962 us for
native/BPF calls under storage traffic, with scheduling and contention included.
It must not be compared numerically as an old/new control for this isolated
single-thread measurement.

The completed [write-budget comparison](results-575-gds-write-budget-20260907.md)
already improved BPF read p99 by 51.011% and write throughput by 15.315%,
with the same qualitative benefit in native. Those are shared policy/executor
parameter gains. This new result isolates a separate implementation cost.
Saving fractions of a microsecond does not by itself explain or eliminate the
remaining hundred-millisecond storage tails. No completed storage cells were
repeated to search for a favorable result, and no end-to-end claim is added.

## Execution and retention

[Raw decisions](raw/gds-ioctl-reuse-575-20260907/decisions.json),
[exact command](raw/gds-ioctl-reuse-575-20260907/command.txt), and
[per-block/paired analysis](raw/gds-ioctl-reuse-575-20260907/analysis.json)
retain every measured call. The raw fields define request ID, nanoseconds,
action, defer time, priority and batch target. Median is the ordinary sample
median; p99 uses nearest rank (sorted sample 990 of 1,000). Each pair compares
the two arms in the same rotated block, as 100*(reuse/legacy-1).

Each arm occupies each position twice. Four repeated inputs cover demand
reads, safe writes with and without live pending reads, and speculative reads.
Every arm receives the same sequence and IDs; each instance gets 100 warmups.
The run is one process on the RTX 5090 host, Linux 6.15.11, NVIDIA 575.57.08,
Python 3.12, GIL-release path. CPU affinity and frequency were not pinned.
The GPU is not doing a storage workload during this call microbenchmark.

Local Qwen 27B implemented the change; root reviewed it, restored the exact
legacy hot path apart from the option branch, ran the existing and focused
development tests (12 passing), and ran the short recorded measurement command
while Qwen Next encountered provider retries. GLM analysis and Qwen Next's
offline review remain separate local tasks; no session was stopped for silence.
