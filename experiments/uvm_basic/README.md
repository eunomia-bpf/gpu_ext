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

## Stage 3

Stage 3 added a closed prefetch-decision schema, CPU-first-touch and per-array migration diagnostics, and a guarded A-B-A managed-memory phase scan. The complete 0.95x matrix passed. At 1.05x, `prefetch_none` exceeded the fixed 300 s limit and remains a recorded resource limit; it was not retried. A bounded continuation completed no-policy, always-max, and adaptive characterization at 1.10x, including trace and Nsight evidence, without unlocking Stage 3D. The exact partial status and restoration evidence are in [docs/STAGE3_RESULTS.md](docs/STAGE3_RESULTS.md).

Non-privileged preparation:

```bash
bash scripts/check_stage3.sh
python3 analysis/analyze_prefetch_decisions.py --experiment-dir .
python3 analysis/analyze_eviction_refault.py --experiment-dir .
python3 analysis/summarize_stage3.py --experiment-dir .
```

All module switching, BPF attachment, and oversubscription commands remain isolated in the intentionally non-executable `scripts/SAFE_STAGE3_COMMANDS.sh`. The runtime used that reviewed path, detached every policy, and restored the distribution `nvidia_uvm`.

## Evidence Boundary

- CUDA Event timing compares demand access, hot access, CPU retouch, and explicit prefetch.
- Nsight Systems page-fault reports are required before claiming a UVM fault or migration count.
- `cudaMemGetInfo()` does not establish page residency.
- `device` allocation is a `cudaMalloc`/`cudaMemcpy` control, not a UVM demand-paging path.
- gpu_ext tracing is optional and requires the user to manually load a compatible custom module.

`scripts/SAFE_GPU_EXT_COMMANDS.sh` is intentionally non-executable and is never sourced by another script. Review it manually before any driver reload. Oversubscription is separately guarded and is not part of the default run.
