# First actual KV disk/UVM serving run — running

2026-09-09 PDT. The real campaign started at 01:51:30 PDT with runner
`6993574f`. At this update it is loading the model for stock, position 0.
There is no completed serving comparison yet. The published backing module
and CUDA helper alone are not a completed serving result.

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
runner. Root syntax-checked the shell script and runner, then loaded the new
UVM and started the real campaign. The local model supplied the per-cell
environment forwarding; root supplied the two CLI flags and campaign-to-cell
keyword forwarding. Later diagnostic-collection helper work remains separate
from the already-imported runner version. The per-cell raw diagnostic output
path is enabled in the actual server environment. Main backing, bootstrap,
and CUDA sources are frozen for the run. Both shared leases cover it.

The first invocation (`sudo flock ...`) exited 66 before touching the
driver because root could not create/open the user-owned lock in `/tmp`;
that error is preserved in `lifecycle.log`. Root acquired both locks as the
workspace user and invoked only the lifecycle via sudo instead:

```sh
flock /tmp/gpubpf-revision-gpu0.lock \
  flock /tmp/gpubpf-revision-struct-ops.lock \
  sudo -n bash workloads/lmcache-disk/raw/diskuvm-serving-20260909.9uBMtq/run-serving.sh
```

`lifecycle-run.log` records the actual module change and active policy
loaders 330946/330947; both report `attached`. The failed lock invocation
was not a serving attempt and did not rerun any cell.

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
