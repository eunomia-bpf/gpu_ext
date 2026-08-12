# Minimal CUDA UVM Experiment

This directory isolates a reproducible vector-add experiment from the existing workloads. The default path uses the currently loaded NVIDIA driver and never reloads kernel modules.

The consolidated design, execution record, and measured evidence are documented in [docs/EXPERIMENT_REPORT.md](docs/EXPERIMENT_REPORT.md). Detailed generated measurements are in [docs/RESULTS.md](docs/RESULTS.md).

## Quick Start

```bash
bash scripts/collect_environment.sh
bash scripts/run_matrix.sh
bash scripts/profile_nsys.sh
bash scripts/run_prefetch_ab.sh
bash scripts/profile_prefetch_ab.sh
bash scripts/check_gpu_ext_stage2.sh
```

The matrix uses 256 MiB and 1 GiB per array only when three arrays fit within 20% of currently free GPU memory. Each run records JSONL under `results/`; `analysis/summarize.py` writes `results/summary.csv` and `docs/RESULTS.md`.

The independent-process prefetch A/B results are in [docs/PREFETCH_AB_RESULTS.md](docs/PREFETCH_AB_RESULTS.md). Stage 2 status is in [docs/STAGE2_PREFLIGHT.md](docs/STAGE2_PREFLIGHT.md) and [docs/STAGE2_GPU_EXT_RESULTS.md](docs/STAGE2_GPU_EXT_RESULTS.md).

Static module-version evidence for the UVM-only switch is documented in [docs/UVM_ONLY_COMPATIBILITY.md](docs/UVM_ONLY_COMPATIBILITY.md). Stage 2 completed as `PASS_GPU_EXT_STAGE2_POLICY_MATRIX`; the distribution UVM module was restored afterward.

## Evidence Boundary

- CUDA Event timing compares demand access, hot access, CPU retouch, and explicit prefetch.
- Nsight Systems page-fault reports are required before claiming a UVM fault or migration count.
- `cudaMemGetInfo()` does not establish page residency.
- `device` allocation is a `cudaMalloc`/`cudaMemcpy` control, not a UVM demand-paging path.
- gpu_ext tracing is optional and requires the user to manually load a compatible custom module.

`scripts/SAFE_GPU_EXT_COMMANDS.sh` is intentionally non-executable and is never sourced by another script. Review it manually before any driver reload. Oversubscription is separately guarded and is not part of the default run.
