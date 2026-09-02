# XSched revision baseline

This directory stages the R1 scheduling baseline named in the ASPLOS'27
revision plan. The evaluated configuration must be described as **XSched
Level-1 on sm_120**: current upstream falls back to `CudaQueueLv1` for unknown
CUDA architectures, while its Level-2 guardian and Level-3 trap handlers do
not implement sm_120.

The ignored upstream checkout is `deps/xsched`. It is pinned in the build-smoke
record rather than vendored into gpu_ext.

## Current status

- **A real repeated three-way performance campaign is complete:** five
  randomized blocks, native/original XSched/our gpubpf, all 15 cells valid.
  [Measured results](performance-575-20260902.md) show lower LC tail latency
  for XSched and modestly higher BE throughput for gpubpf versus XSched.
  This is the explicitly scoped short-budget campaign, not the original
  50-kernel/10-block full protocol.
- The old review in [plan-review.md](plan-review.md) is a historical record,
  not an additional execution-approval requirement. The user has authorized
  automatic performance experiments. The runner now brackets the CUPTI/device
  clock offset before and after each cell and uses mutually exclusive result
  categories, repairing the two old defects.
- The official CUDA platform builds successfully on this host.
- A finite native smoke and a finite XSched Level-1 shim/HPF smoke now complete
  with exact recurrence checks; see `build-smoke.md`.
- A six-process smoke additionally establishes 24 unique XQueues and successful
  suspend/resume engagement for all 16 BE queues. It is not a repeated or
  paper-scale performance comparison.
- GPU runs hold the shared GPU/struct-ops leases. Each cell saves idle/power,
  UVM, struct-ops, and kernel/Xid observations before and after its owned
  processes. It never stops an unrelated process.

## Three-way measured comparison

`pilot` is a frozen short-budget **performance** campaign: five complete
seeded randomized blocks of native CUDA, original upstream XSched Level-1
HPF, and our gpubpf priority-preemption policy. It retains two LC/four BE
processes, four streams per process, and the calibrated approximately 80 ms
kernel. Only kernels per stream (5 instead of 50) and complete blocks (5
instead of 10) are reduced. This gives 40 LC and 80 BE samples per cell;
nearest-rank P99 is consequently the LC sample maximum. P50, P95, mean,
BE throughput, raw per-kernel times, correctness, engagement, and paired
bootstrap intervals are also saved. A completed pilot is **not** completion
of the original 50-kernel/10-block protocol or reproduction of XSched Level-3.

```sh
python3 -B workloads/xsched/test_run_xsched_rq3.py
python3 -B workloads/xsched/run_xsched_rq3.py admission
python3 -B workloads/xsched/run_xsched_rq3.py calibrate --output workloads/xsched/raw/calibration-575 --reps 1000000 --timeout 120
# Use the emitted frozen_reps in both following commands.
python3 -B workloads/xsched/run_xsched_rq3.py preflight --output workloads/xsched/raw/preflight-575 --reps FROZEN_REPS --timeout 120
python3 -B workloads/xsched/run_xsched_rq3.py pilot --output workloads/xsched/raw/pilot-575 --reps FROZEN_REPS --timeout 120
```

All output paths must be new. `full` retains ten blocks, 50 kernels per
stream, and three isolated controls per role. `analyze --output DIR` is
CPU-only and refuses missing configurations or incomplete blocks. See
[plan.md](plan.md) for the matched settings and interpretation limits.
