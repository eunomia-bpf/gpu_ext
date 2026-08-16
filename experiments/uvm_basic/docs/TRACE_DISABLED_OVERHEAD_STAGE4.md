# Stage 4 Trace Disabled-Path Overhead

Status: `COMPLETE_TARGET_NOT_MET`

Stage 4F ran 20 fresh untraced and 20 trace-attached independent 256 MiB
vector-add processes. All 40 runs passed correctness, clean-detach, Xid, and
GPU-memory-release checks.

| Mode | Runs | Mean ms | Median ms | Stddev ms | P95 ms | 95% CI ms |
|---|---:|---:|---:|---:|---:|---:|
| current custom, trace not attached | 20 | 244.138 | 244.343 | 3.012 | 247.611 | 242.818-245.459 |
| current custom, trace attached | 20 | 285.353 | 285.014 | 3.193 | 289.948 | 283.954-286.753 |

The Stage 2 custom no-policy reference was 240.731 ms. The current untraced
mean is therefore 1.415% higher, so the <=1% disabled-path target was not met.
Attaching the enhanced trace increased kernel-1 time by 16.882% relative to
the current untraced runs. No outliers were removed.

This window measured the current implementation; it did not alter the kernel
hot path. The summarizer's streaming CSV change only reduces offline analysis
memory use and is not a kernel-overhead optimization.
