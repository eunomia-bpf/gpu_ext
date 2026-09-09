# LMCache disk: source, build and measured campaigns

This is a reproduction map, not a new measurement or manuscript edit.
Start at [ARTIFACT.md](../../ARTIFACT.md) for all systems.

## Keep these two results separate

| Campaign | Metric and controls | Evidence |
| --- | --- | --- |
| Write-budget storage, 25 cells | Scheduled-arrival read p99 and write throughput; FIFO, native/BPF at 10/200 ms budgets. | [Report](../../workloads/lmcache-disk/results-575-gds-write-budget-20260907.md), [recorded commands](../../workloads/lmcache-disk/raw/gds-write-budget-575-20260907-five-block-02/run-plan.json). |
| Physical-reclaim serving, 15 cells | Warm generation token/s; stock/native/BPF. The BPF throughput result is adverse. | [Analysis](../../workloads/lmcache-disk/gds-control/physical-reclaim-performance-analysis-20260909.md), [raw results](../../workloads/lmcache-disk/raw/diskuvm-physical-reclaim-20260909.pXYN4F/RESULTS.md). |

Storage read latency is not vLLM TTFT. CPU-staged restoration is not evidence
of GPU-direct P2P. The 200 ms write budget is a shared executor parameter,
not a claim that BPF instructions make physical storage faster.

## Build the storage components

From a normal checkout, with a BPF-capable clang, a C compiler, make,
libelf/zlib development libraries and CUDA 12.9 including cuFile installed:

```sh
git submodule update --init --depth 1 -- libbpf
make -C workloads/lmcache-disk/gds-control -j2 \
  CUDA_HOME=/usr/local/cuda-12.9 \
  gds_policy.bpf.o gds_policy ioctl_probe gds_executor
```

The [Makefile](../../workloads/lmcache-disk/gds-control/Makefile) builds libbpf
from the pinned submodule when needed. It takes `CLANG`, `CC` and `CUDA_HOME`
overrides. The command writes regenerable outputs beside their sources;
it neither loads a policy nor changes the driver. On this shared host,
root wraps heavy builds with the two leases documented in `ARTIFACT.md`.

This exact target set [built successfully from an independent published
checkout](../../workloads/lmcache-disk/raw/published-component-build-20260909.VqA85f/README.md)
on 2026-09-09: gpu_ext `fcb77b7d`, libbpf `02bdeb7`, existing host CUDA 12.9.
This does not establish a fresh OS installation or complete serving setup.

## Run records and remaining runtime setup

The 25-cell plan names `run_gds_mixed_backend.py`, its five arm schedules,
64 reads / 96 writes per cell, a 4096 MiB buffer, the event-driven variant
and the per-arm write budget. It also records the old Python environment
and absolute output paths. Read it as an execution record, not a relocatable
installer; choose new output locations rather than reusing completed cells.

The Python LMCache/CUDA environment and the saved modified 575.57.08 module
providing the storage-decision interface remain runtime dependencies.
Stock driver installation alone does not recreate that interface. The build
above does not load that module or establish its fresh-host deployment.
Physical-reclaim serving has additional adapter/model dependencies described
in its linked report; these are not supplied by the four storage targets.

To reproduce existing statistics now, without those runtime dependencies:

```sh
python3 -B scripts/artifact/reanalyze.py --campaign storage
python3 -B scripts/artifact/reanalyze.py --campaign lm
```

Both commands retain every published arm and adverse pair. They recompute
statistics from cell summaries, not percentiles from every request timestamp.
