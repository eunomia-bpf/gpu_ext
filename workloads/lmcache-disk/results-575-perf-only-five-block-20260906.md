# LMCache local-storage performance — five blocks on RTX 5090

The performance-only runner completed five rotated blocks and 15/15 cells on
RTX 5090 with driver 575.57.08. Each cell served eight warm requests with a
1,536-token cached prefix and generated 16 tokens. The disk arm used LMCache's
real `file://` local backend with `use_odirect=true`. The runner applied no
correctness, admission, retry, or result-filtering gates.

Source records: [`raw/storage-575-perf-only-20260906-01`](raw/storage-575-perf-only-20260906-01).

| Arm | warm TTFT median (ms) | warm requests/s median | output token/s median |
|---|---:|---:|---:|
| recompute | 67.1691 | 1.9151 | 30.6422 |
| LMCache CPU | 72.6468 | 1.8823 | 30.1168 |
| LMCache local disk | 96.3280 | 1.7858 | 28.5723 |

Relative to recompute, CPU caching increases median TTFT by 8.1551% and lowers
output throughput by 1.7146%; local-disk caching increases median TTFT by
43.4112% and lowers output throughput by 6.7551%. Disk is 32.5977% slower in
TTFT and 5.1284% lower in output throughput than CPU caching.

For this short-prefix, single-request workload, recomputing the prefix is
cheaper than recovering its KV state from CPU or local SSD. This is a valid
negative storage-tier result rather than evidence that storage offload is
generally harmful: longer prefixes, shared prefixes, or memory pressure can
change the trade-off.

This completes the recompute/CPU/disk baseline portion only. The planned
native and gpubpf recoverability-aware arms remain required to compare the
same disk-aware placement policy through the original and BPF decision paths.
The current host has CUDA GDS userspace libraries, but no loaded or installed
`nvidia-fs` module; `gdscheck` reports NVMe unsupported and compatibility mode.
Consequently, these data are local-SSD/O_DIRECT results and are not presented
as GPUDirect Storage measurements.
