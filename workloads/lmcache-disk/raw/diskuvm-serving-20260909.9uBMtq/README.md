# First disk/UVM-enabled LMCache serving block — three cells complete

2026-09-09 PDT, RTX 5090 / CUDA 12.9 / 575.57.08. The real campaign ran
01:51:30–01:59:54 PDT with runner `6993574f`. All three server processes
return zero; each completes eight warm requests and 8192 generated tokens,
with no recorded request failures. The runner returns zero. Original-driver
restoration completed on a separate retry at 02:01:23 PDT.

| Configuration (all with disk/UVM opt-in) | Warm output token/s | Median warm TTFT, ms |
| --- | ---: | ---: |
| Stock KV-reclaim behavior | 73.199047 | 25076.623476 |
| Native disk-aware KV-reclaim policy | 66.625019 | 30249.547229 |
| Same KV-reclaim policy through BPF | 64.477420 | 31266.933853 |

These are **one sequential block**, not medians across repeated blocks.
BPF throughput is 3.2234% below native in this single pair; no confidence
interval or stable throughput advantage/disadvantage follows. Warm throughput
is 8192 generated tokens divided by the complete warm-burst duration,
excluding startup, cold population, barriers, and shutdown. TTFT is measured
from actual HTTP send to first token and includes server queuing; it is not
the earlier disk-read p99 metric. Source policy, workload, and older results
are not replaced by this first block.

All three arms enable the same experimental transport, so this does **not**
isolate transport improvement over original GDS. The raw per-cell environment
records both the opt-in and built helper path. However, the diagnostic files
contain only `retained_total: 0`, without engine restore/fallback counters.
One such file existed while the engine was still live, consistent with an
auxiliary process's atexit writing the shared path. At 01:59 PDT root counted
48 open cache-file descriptors in BPF engine PID 338730, consistent with
retained backings; that observation is not an exact restore count or proof
that every read used the new path. Thus these are real **opt-in serving
measurements**, not yet conclusive evidence of successful disk/UVM restoration
for all requests. The performance is retained without a counter gate; a small
engine-close dump repair is the next implementation task.

The run used one block of the existing stock/native/BPF KV-reclaim
workload, with disk/UVM transport enabled identically in all three arms.
It reuses the measured 62502 ns/token calibration from
`../kv-reclaim-recompute-calibration-575-20260907-01/calibration.json`.
No completed earlier cell or calibration was repeated. This first block
uses the new application connection in the ordinary performance workload;
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
and CUDA sources were frozen for the run. Both shared leases covered it;
the freeze was released after completion. The later diagnostics collector
was not loaded by this runner, and original `result.json` files keep that
field null. Separately preserved raw diagnostic JSON files are not silently
substituted into the original records.

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

The normal exit handler initially could not unload UVM (`Module is in use`)
after the loaders exited; its failure remains in `lifecycle-run.log` as
`DISK_RUN_EXIT=0 RESTORATION_OK=0`. A subsequent read observed refcount zero.
Under both leases, `restore-after-exit.sh` then unloaded the experimental
module and restored the exact saved original. `restoration-retry.log` records
`RESTORATION_OK=1` and replacement loader PIDs 344198/344199; their logs both
say `attached`. Post-restoration GPU is idle at 0%, 1 MiB. No reboot or
forced process termination was needed.

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

## Payload cleanup

After all raw records and this report were pushed in `7fad6676`, root
removed only the three completed cells' `cache` directories. Each had an
apparent directory size of 1208352768 bytes (3625058304 bytes in total,
about 3.38 GiB). No serving process remained. These were generated KV
payloads, excluded from Git; the original files cannot be restored from
Git, but the workload can regenerate its cache. All original request
responses, server logs, timing records, diagnostic files, and adverse
outcomes remain committed. No model weights or source worktrees were removed.
