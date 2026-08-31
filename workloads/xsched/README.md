# XSched revision baseline

This directory stages the R1 scheduling baseline named in the ASPLOS'27
revision plan. The evaluated configuration must be described as **XSched
Level-1 on sm_120**: current upstream falls back to `CudaQueueLv1` for unknown
CUDA architectures, while its Level-2 guardian and Level-3 trap handlers do
not implement sm_120.

The ignored upstream checkout is `deps/xsched`. It is pinned in the build-smoke
record rather than vendored into gpu_ext.

## Current status

- The first proposal is closed after three review rounds; see
  [plan-review.md](plan-review.md). Its clock-domain and interpretation defects
  must be resolved in a new proposal before GPU execution.
- The official CUDA platform builds successfully on this host.
- Runtime engagement and preemption behavior have not yet been established.
- GPU runs must not overlap external services; the shared RTX 5090 is currently
  occupied by SGLang.
- A paper-facing comparison requires a separately reviewed workload, metric,
  correctness check, and interleaved repetition plan.
