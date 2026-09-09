# First actual KV disk/UVM serving run — prepared, not executed

2026-09-09 PDT. This directory currently contains the root-owned driver and
serving lifecycle only. There are no request measurements here yet. The
published backing module and CUDA helper are not a completed serving result.

The next run uses one block of the existing stock/native/BPF KV-reclaim
workload, with disk/UVM transport enabled identically in all three arms.
It reuses the measured 62502 ns/token calibration from
`../kv-reclaim-recompute-calibration-575-20260907-01/calibration.json`.
No completed earlier cell or calibration is repeated. This first block
checks the new application path in the ordinary performance workload;
it is not yet a five-block comparison or a transport-speedup claim against
the original GDS backend. The ordinary throughput, request latency, failures,
and restore/fallback observations must all remain visible.

`run-serving.sh` adapts the completed disk-UVM primitive lifecycle to this
runner. Root has syntax-checked the shell script, but has not loaded the new
UVM or launched this serving campaign. The local model is still finishing
the `--disk-uvm` and `--disk-uvm-fault-lib` runner wiring and per-cell result
collection; do not execute a half-wired CLI. Existing source work continues,
not cancelled or timed out. Both shared leases must cover the actual run.

Prepared components:

- Main backing module and default-off bootstrap: published `8e79a2bc`.
- Python exact-allocation helper: published `20abefec`.
- CUDA fault traversal: published `5d730f49`, built shared object at
  `/tmp/lmcache-diskuvm-serving-build-20260909.7dZdC8/libdiskuvm_fault.so`
  (26648 bytes).
- Built disk/UVM module: driver `dea1fefc`,
  `/home/yunwei37/workspace/gpu/gpu_ext-kernel-575-gds/kernel-open/nvidia-uvm.ko`.
- Saved original module:
  `/var/tmp/fig13-original-modules-20260908.TkarSf/nvidia-uvm.ko`.

The script's old-loader PIDs 4135269/4135270 were checked live during
preparation. Recheck them before invocation; these values are not a reusable
service-discovery mechanism. Root owns module replacement/restoration and
all GPU use. The transport remains driver fault hydration through CPU
staging, not demonstrated NVMe-to-GPU P2P or automatic memory-pressure
offload. Previous application results and disk/UVM primitive results remain
unchanged. This session does not edit the manuscript.
