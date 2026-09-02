# BPF HPF on the original XSched frontend: measured outcome

A real BPF implementation of HPF, executed by bpftime's ubpf JIT, completed
five randomized paired blocks against native CUDA and original XSched HPF.
All **15 cells completed**, with no failed or excluded cells. Its LC queueing
latency returned to the original XSched scale. The paired interval does not
establish that it is faster than XSched, or prove strict performance equivalence.

| Configuration | Median LC P99/max (ms) | Median BE kernels/s |
|---|---:|---:|
| Native CUDA | 7251.551 | 10.2658 |
| Original XSched Level-1 HPF | 3936.093 | 9.6557 |
| BPF HPF / bpftime JIT / original XSched frontend | 3695.408 | 9.5602 |

The paired BPF-minus-XSched LC difference has a 95% bootstrap interval of
**−518.009 to +258.531 ms**. Relative BE throughput has interval
**−1.759% to +0.295%**, satisfying the predeclared 5% BE noninferiority margin
in this short campaign. Since the LC interval crosses zero, the runner's
strict-improvement classification is **inconclusive**, not a performance win.
Against native CUDA, BPF HPF's LC interval is −3840.300 to −2818.720 ms and
its relative BE throughput interval is −7.786% to −6.135%: the expected
latency/throughput tradeoff is visible. These are exploratory paired
intervals, not a multiple-comparison-adjusted search result.

## What this implementation establishes

Only the HPF policy factory is replaced through link-time wrapping. The
upstream XSched server, CUDA interception frontend, 24 XQueues, scheduling
events, and Level-1 suspend/resume actuator are unchanged. The server loads
compiled BPF bytecode and executes its priority decisions through bpftime's
ubpf JIT; it does not merely call a C++ rewrite with a BPF label.

CPU tests compare this actual BPF execution against actual upstream HPF on
**5,000 snapshots and 158,769 queue decisions**, with zero mismatches. Cases
include five devices, priority ties, idle queues, default and clamped
priorities, and the 64-queue bound; larger snapshots are rejected. The GPU
campaign then exercises those decisions on the real 24-queue workload.
This supports expressibility of the tested, bounded HPF semantics in BPF.
It is **not a driver-only gpubpf scheduler**, a full XSched reimplementation,
or an XSched Level-3 reproduction on sm_120. The BPF policy uses an O(n²)
bounded scan where upstream uses a map-based procedure, so this comparison
also does not isolate pure VM overhead.

The [original driver-policy negative result](performance-575-20260902.md)
and [two driver-candidate negative results](driver-candidates-575-20260902.md)
remain intact. This new route changes the execution frontend/actuator
relative to those driver-only routes; its improvement must not be attributed
to merely tuning their preemption cooldown or timeslice settings.

## Protocol and validation

The workload and measurement protocol remain 2 LC + 4 BE processes, four
streams each, five kernels per stream, 340 blocks × 256 threads, 9,496,464
FMA repetitions (previously calibrated to approximately 80 ms isolated), and
seed 1797 for within-block configuration order. All cells ran Linux 6.15.11,
custom NVIDIA 575.57.08, RTX 5090, and a 400 W power limit. Each cell held
the shared GPU/struct-ops leases. LC was released at least 5 ms after every
BE process confirmed that its first kernel was active.

The pilot deliberately retains only five complete blocks and five kernels
per stream instead of the original ten blocks and 50 kernels. Each cell has
40 LC samples, making nearest-rank P99 the sample maximum. LC latency means
host submission to device entry; BE throughput divides 80 completed kernels
by the duration from common BE release to the latest BE completion. These
are workload metrics, not a 50 ms XSched scheduling-period measurement.

- All 90 worker records exited successfully and validated **156,672,000
  output values and 1,800 completed kernels**. Each configuration contributed
  200 LC and 400 BE kernels.
- All ten XSched/frontend cells established 24 queues and successful
  suspend/resume for all 16 BE queues per cell. Across the five BPF cells,
  there were **80 successful BE suspend and 80 resume transitions**.
- Separately, the JIT policy recorded **537 invocations and 7,175 queue
  decisions**: 752 suspend decisions and 6,423 resume decisions. These are
  policy decisions, not counts of hardware preemptions or state transitions.
- All 30 before/after safety observations had zero UVM references, zero GPU
  utilization, no compute clients, empty struct-ops state, and no Xid or
  kernel abnormality. The maximum conservative device/host clock error
  bound was **0.160849 ms**, below the fixed 1 ms limit.
- The preceding two-kernel BPF preflight also passed correctness, all
  16 BE queue transitions, JIT engagement, and post-run cleanup.

The runner revision was `b5def86`; upstream XSched is
`f49289f0220931df78de948ed841ecbaf960a919`, bpftime is
`d6316fa73edaac4fdfe21b89d4470da6cd9b8ae8`, and the loaded custom driver
source revision was `28b1d30c`. No driver reload or GSP propagation repair
occurred during this campaign.

## Evidence and replay

- [Frozen protocol](raw/bpftime-hpf-pilot-575-20260902/protocol.json)
- [Aggregate results and paired intervals](raw/bpftime-hpf-pilot-575-20260902/summary.json)
- [BPF preflight](raw/bpftime-hpf-preflight-575-20260902/bpftime_hpf/result.json)
- All worker stdout, policy stdout, engagement audits, clock observations,
  and before/after safety records are retained beside the results.

CPU-only replay:

```sh
make -C workloads/xsched test-bpftime-hpf
python3 -B workloads/xsched/test_run_xsched_rq3.py
python3 -B workloads/xsched/run_xsched_rq3.py analyze --output workloads/xsched/raw/bpftime-hpf-pilot-575-20260902
```

A new real run uses new output directories and requires the shared GPU slot:

```sh
python3 -B workloads/xsched/run_xsched_rq3.py preflight --configs bpftime_hpf --output NEW_PREFLIGHT_DIR --reps 9496464 --timeout 120
python3 -B workloads/xsched/run_xsched_rq3.py pilot --configs native,xsched,bpftime_hpf --output NEW_PILOT_DIR --reps 9496464 --timeout 120
```
