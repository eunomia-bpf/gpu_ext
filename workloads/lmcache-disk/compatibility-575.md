# LMCache 575 startup diagnostic

2026-09-03. This is dependency diagnosis for the existing storage-tier study,
not a new paper experiment or a completed LMCache performance result. Root
completed the two tiny diagnostics below and has launched the repaired
full-model disk preflight; no outcome is claimed yet. Root owns GPU leases,
safety checks and launches.

## Known failure, not yet a cause

The closed traced preflight is
`raw/storage-575-preflight-01/disk/`. Its `execution.json` records failure
before readiness, server exit 1, unchanged boot, no reported kernel/Xid
abnormality, and no cleanup errors. `server.log:39–45` reports 29.04 GiB used
for model loading, followed by `!!!!!!! Segfault encountered !!!!!!!` with
`cuModuleLoadData` and `loadBinary` on the native stack. The earlier log selects
the Triton FP8 MoE backend. It does not identify the failing kernel or prove
that LMCache disk I/O, memory exhaustion, tracing, or a particular driver
version caused the failure. No requests were served or storage-tier result
produced. The adjacent missing MoE tuning configuration is a performance
warning, not an established cause of the crash.

## V3 readiness attempt

The first V3 attempt, `raw/storage-575-v3-preflight-01/disk`, stopped in
artifact admission because the local frozen model snapshot was incomplete.
The server and GPU workload were never started. This admission-only failure is
preserved, must not be overwritten or reused, and provides no V3 correctness,
storage-engagement, or performance evidence. After restoring the complete
model, the batch requires the fresh traced path
`raw/storage-575-v3-preflight-02/disk`.

On 2026-09-04, that second V3 attempt passed artifact admission and launched
the server, but EngineCore stopped during PyTorch CUDA initialization with
`NV_ERR_NO_MEMORY`. GPU memory use was 15 MiB before launch; monitoring found
no Xid and cleanup succeeded. An immediate root invocation in the same venv
successfully allocated a one-element CUDA tensor. This is retained narrowly as
a transient startup failure, not V3 correctness evidence or an established
V3 failure. The model workload and requests never ran. Before execution, the
only authorized traced retry is the fresh path
`raw/storage-575-v3-preflight-03/disk`; the correctness and formal paths remain
unchanged.

That third V3 attempt reached the real model and the first cold request. The
log proves that V3 discovered the physical fused NHD cache as
`(788, 16, 4, 256)` per layer and initialized its pointers, while the engine
metadata created earlier still described the legacy split layout
`(48, 2, 256, 4, 128)`. Consequently, `CacheEngine.store` allocated a
last-dimension-512 memory object before V3's lazy discovery; the subsequent
copy from V3's last-dimension-1024 temporary buffer failed with a tensor-size
mismatch. The request returned an empty HTTP 200 response, no cold store
completed, and there is no correctness, disk-engagement, or performance
result. The retained execution record reports unchanged boot, no kernel/Xid
abnormality, successful cleanup, and GPU memory returning to 15 MiB.

The source repair in
`patches/lmcache-v0.5.4-v3-register-before-storage.patch` adds a generic
registration hook and makes the V3 connector discover the actual cache layout
during vLLM's `register_kv_caches`, before `post_init` creates storage. It does
not select V3 by private type. The rebuilt fixed wheel passed 39 CPU tests plus
the existing exact 32-slot fused-NHD GPU D2H-to-pinned-CPU-to-H2D roundtrip;
the latter checked every selected value and allocator cleanup. These are
implementation gates only. A fresh full-model run is still required before
claiming that the compatibility failure is closed, and the failed preflight-03
directory will not be reused.

For reproduction, start from the source revision recorded in
`artifacts-current.json`, apply the listed patch with `git apply`, and rebuild
with the unchanged environment in `build-smoke.md`. The resulting wheel keeps
the existing `dist-current-py312/lmcache-0.5.4-cp312-cp312-linux_x86_64.whl`
path and is installed into `current-venv` with `pip install --force-reinstall
--no-deps`; admission therefore continues to exercise a source-built wheel,
not an ad-hoc edit under `site-packages`.

This repair is scoped to vLLM's in-process V3 connector. The default hook is a
no-op, so the legacy, layerwise, multiprocess/NIXL, and non-vLLM connectors are
unchanged; they are also outside this storage-tier validation and receive no
new compatibility claim here.

The second attempt's saved telemetry summary peaks at 30,724 MiB of 32,607 MiB.
That observation
does not establish either an allocation failure or adequate headroom at every
instant. `--enforce-eager` disables torch compilation/CUDA graphs, not the
separately selected Triton MoE kernels.

## V3 preflight-04 and preflight-05

The fourth V3 attempt, `raw/storage-575-v3-preflight-04/disk`, stopped in
artifact admission because the dependency source tree was dirty. The server
and the GPU workload were never started; the preserved record contains only
`execution.json`, telemetry, and an empty kernel log. This admission-only
failure is preserved, must not be overwritten or reused, and provides no V3
correctness, storage-engagement, or performance evidence.

The fifth attempt, `raw/storage-575-v3-preflight-05/disk`, completed the
traced preflight, and `validate-cell --require-trace` exits 0. All eight cold
and all eight warm requests returned HTTP 200; each warm request reports
exactly 1536 LMCache-hit and 1536 retrieved tokens. The cache footprint is
48 files totaling 1,207,959,552 bytes, and the warm aggregate for this cell
is 8 requests, 128 output tokens, 4.308207764 seconds, and
29.710730543 output token/s. This is one traced preflight cell on eight
prefixes: it is not a correctness comparison against the untraced
`recompute`/`lmcache_cpu` arms and not a performance reproduction, and
neither is claimed here. The failed preflight-04 directory will not be
reused.

## Correctness-01 shutdown-gate failure

The disk arm under `raw/storage-575-v3-correctness-01/lmcache_disk` completed
all 16 HTTP requests. Its eight warm requests each reported exactly 1536
LMCache-hit and 1536 retrieved tokens. After the runner sent SIGINT and vLLM
logged `[shutdown] MPClient: complete`, the AsyncLLM output handler emitted its
shutdown-race traceback ending in `EngineDeadError`; FastAPI then completed
application shutdown. The generic `Traceback` fatal scan rejected the cell
before `result.json` was written. This directory is preserved as a
shutdown-gate failure and will not be reused or retroactively promoted. The
three-arm correctness comparison is incomplete, and this attempt supplies no
correctness or performance claim.

The validator now excludes only that exact vLLM 0.27.1 traceback signature
when it is uniquely enclosed by the observed SIGINT, MPClient, API-server, and
FastAPI shutdown boundaries. It still scans the original surrounding and
traceback text for CUDA, O_DIRECT, fallback, allocation, and eviction failures;
all near-miss or additional tracebacks remain fatal.

## Installed toolchain: read-only source and metadata evidence

All paths below are relative to
`current-venv/lib/python3.12/site-packages/`, except the system assembler.
No torch/Triton/CUDA import, compilation, or GPU operation was used in this
inspection; assembler commands were limited to `--version`/`--help`.

| Component | Observed evidence |
| --- | --- |
| torch | `torch/version.py`: 2.13.0+cu129, CUDA build 12.9; its installed metadata requires Triton 3.7.1 and CUDA toolkit 12.9.1. |
| vLLM | Installed metadata: 0.27.1+cu129, requiring torch 2.13.0. |
| Triton | Installed metadata: 3.7.1. |
| Ordinary bundled assembler | `triton/backends/nvidia/bin/ptxas --version`: CUDA 12.8, V12.8.93. |
| Blackwell bundled assembler | `triton/backends/nvidia/bin/ptxas-blackwell --version`: CUDA 13.1, V13.1.80. |
| System assembler | `/usr/local/cuda-12.9/bin/ptxas --version`: CUDA 12.9, V12.9.86; its help lists `sm_120a`. |

The installed `triton/backends/nvidia/compiler.py:34–35` selects
`knobs.nvidia.ptxas_blackwell` for architecture >= 100. Therefore sm_120 uses
the Blackwell tool; the ordinary `TRITON_PTXAS_PATH` does not switch that
branch. `triton/knobs.py:195–215,488–492` defines the exact override
`TRITON_PTXAS_BLACKWELL_PATH`. An inaccessible override silently falls back
to the bundled binary, so record the **selected** path, not just the requested
environment variable. `CUDA_HOME` and `PATH` alone do not select this tool.

`compiler.py:71–76,440–467` derives the default PTX version from the selected
assembler and feeds the generated PTX to that assembler. Thus changing the
override changes the compilation pipeline, not just a path label. The source
also unconditionally supplies `--regAllocOptLevel=2` at lines 494–499;
the subsequent 12.9 diagnostic compiled and ran the tiny kernel without
changing that source. `triton/backends/nvidia/driver.c:156–184`
implements `loadBinary` and calls `cuModuleLoadData` at line 182. This matches
the observed stack names, but does not establish the origin of the fault.

`lmcache_primitives.py` constructs a fixed server environment without
inheriting caller variables. Setting the override only outside
`run_lmcache_disk.py` would therefore **not** change the formal server.

## Actual diagnostic and adopted 575-only repair

`raw/triton-575-diagnostic-01/{default,cuda129}` each retain `client.log`,
`execution.json`, `gpu-telemetry.csv`, and `kernel-follow.log`. Both used the
same script/venv, one RTX 5090 on 575.57.08, no model or strace, separate fresh
caches, and 33,137,426,432 free device bytes before the Triton call.

| Arm | Selected assembler in client log | Actual outcome |
| --- | --- | --- |
| `default` | Bundled `ptxas-blackwell`, release 13.1 | Exit -11 after entering the tiny compile/load/launch phase; no PASS. |
| `cuda129` | `/usr/local/cuda-12.9/bin/ptxas`, release 12.9 | Exit 0; all 4,096 FP32 outputs exactly matched the CPU oracle. |

Both execution records have unchanged boot, live monitors until cleanup,
empty kernel/Xid/cleanup error lists and GPU memory restored to 2 MiB. The
tiny default log does not contain a native fault backtrace, so it alone does
not narrow the fault to an individual compile/load/launch step. Model-scale
memory pressure and strace are not necessary to reproduce the tiny failure;
the contrast supports compiler-pipeline sensitivity, not a universal
575/13.1 incompatibility or proof of the full model's original root cause.

Based on this diagnostic, all `recompute`, `lmcache_cpu`, and `lmcache_disk`
cells explicitly selecting 575.57.08 now get
`TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda-12.9/bin/ptxas` from the common
server-environment builder. Admission requires an accessible binary reporting
12.9.86 and records its normal file inventory and full version output. The
full target environment is saved in the existing `environment.json` before
launch, including failed starts, and in `result.json` upon success; offline
575 validation rejects missing/wrong pins or inconsistent saved environments.
The legacy 610 default stays unpinned and its old records remain valid under
their original rules. No dependencies, model, thresholds, workload, memory
budget, or schedule change. Root has launched the fresh traced full-model test
`raw/storage-575-preflight-02/disk`, to be followed by all three correctness arms and
the same 30 formal cells if the original gates pass. Failed raw is preserved.

No extra cache directory is required by the inspected source: Triton's
`runtime/cache.py:315–317`, `compiler/compiler.py:245–251`, and
`backends/nvidia/compiler.py:34–43,558–561` partition disk compilation entries
using, among other inputs, the target architecture and the **selected**
assembler's full version output. The 13.1 and 12.9 pipelines therefore use
different entries. Each server is a fresh process and its workers use spawn,
so an earlier process's in-memory kernel/version cache is not reused. This is
source inspection, not an audit of old cache contents; no cache values were
generated, printed, or compared, and no existing cache was cleared.

The repair passed all 27 CPU tests, including exact frozen-prompt rederivation
and tests of all three launch environments, failed-start environment capture,
missing/wrong compiler evidence rejection, and legacy 610 compatibility:
`taskset -c 17 ./current-venv/bin/python -B -m unittest -v test_runner`
from this workload directory (5.731 seconds). This does not replace the next
real full-model preflight.

## Minimal root-run discriminator

`triton_575_smoke.py` uses the exact venv, no vLLM/LMCache/model import, 4,096
deterministic FP32 elementwise sums, one Triton kernel, synchronization, and
an exact check of every output against CPU arithmetic. It prints requested
and selected compiler paths, versions, GPU architecture, memory availability,
and the phase reached. It is not a benchmark, FP8 MoE test, or storage test.

After acquiring the existing GPU leases and performing safety admission,
root can use a fresh temporary directory, retain both logs/caches, and run
these two commands separately from `workloads/lmcache-disk`. These commands
are suggestions, not a substitute for the coordinator's safety wrapper.
They leave the script, venv, workload size, and driver unchanged. Set no
ambient `LD_PRELOAD`, `LD_AUDIT`, or CUDA injection variables.

```bash
task_diagnostic_dir=$(mktemp -d /tmp/lmcache-triton-575-XXXXXX)
env -u TRITON_PTXAS_PATH -u TRITON_PTXAS_BLACKWELL_PATH \
  PATH=/usr/local/cuda-12.9/bin:/usr/bin:/bin CUDA_HOME=/usr/local/cuda-12.9 \
  CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
  TRITON_CACHE_DIR="$task_diagnostic_dir/default-cache" \
  /usr/bin/timeout --signal=TERM --kill-after=5s 180s \
  /usr/bin/taskset -c 8-15 ./current-venv/bin/python -u -B triton_575_smoke.py \
  > "$task_diagnostic_dir/default.log" 2>&1

env -u TRITON_PTXAS_PATH \
  TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda-12.9/bin/ptxas \
  PATH=/usr/local/cuda-12.9/bin:/usr/bin:/bin CUDA_HOME=/usr/local/cuda-12.9 \
  CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
  TRITON_CACHE_DIR="$task_diagnostic_dir/cuda129-cache" \
  /usr/bin/timeout --signal=TERM --kill-after=5s 180s \
  /usr/bin/taskset -c 8-15 ./current-venv/bin/python -u -B triton_575_smoke.py \
  > "$task_diagnostic_dir/cuda129.log" 2>&1
```

Start both **without strace**, in fresh processes with separate caches; do not
clear or alter the existing cache. If a process faults, preserve its output
and repeat GPU/kernel safety admission before any next launch. If the default
faults here but 12.9 passes, model-scale memory pressure and strace are not
necessary to reproduce that small-kernel failure, and compiler-pipeline
sensitivity is supported. It still does not prove a general 575/13.1
incompatibility or that the full model will work with 12.9. If both pass, the
full model's failure remains unresolved; a same-default tiny smoke under the
original `strace -ff -qq -s 4096 -e trace=open,openat -o NEW_TRACE_PATH` can
test whether tracing alone reproduces a fault. A negative tiny test cannot
exclude a kernel-specific FP8 MoE failure or model-scale memory pressure.
If both fail, first use the printed phase and full logs rather than attributing
the failure to either assembler from version metadata alone.
