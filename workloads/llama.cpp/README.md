# llama.cpp Expert Offloading Benchmark (Experiment 1)

Benchmarks llama.cpp with GPT-OSS-120B MoE model (~59 GiB) on a 32GB GPU, comparing
framework CPU offloading vs UVM with gpu_ext eBPF policies.

- **Paper**: Figure 6 (RQ1), MoE expert offloading
- **Model**: GPT-OSS-120B MXFP4 MoE (116.83B params, 59.02 GiB)
- **GPU**: RTX 5090 (32 GB VRAM) — model is ~1.8x GPU memory

## Reproduction status

The current stack has a fail-closed workflow, but the historical five-cell
Figure 6 result is **not yet an exact reproduction**. The surviving historical
record is a console transcript, and the plain-UVM, user-hint, and gpubpf source
states were not preserved as three independently buildable revisions. The
current `uvm_current` and `gpubpf_current` cells are therefore current-stack
replays and must not be relabelled as either historical UVM variant.

The replay admission now targets Linux 6.15.11 / NVIDIA 575.57.08 and the same
400 W limit as the current revision experiments. It checks the selected model
against the admitted GPT-OSS snapshot and checks every benchmark row's model,
source build, and RTX 5090 identity. Intended software power-cap activity is
recorded; thermal/hardware slowdown still invalidates a result.

Two archived CPU-offload control runs are within 5% of the historical pp512 and
tg128 values. That supports those controls only; it does not validate the
historical UVM or gpubpf speedups. `legacy_exp1_reference.json` records the
reported values and all known evidence gaps.

## Quick Start (Tested on RTX 5090 / CUDA 12.9 / Ubuntu 24.04)

```bash
cd workloads/llama.cpp

# 1. Build llama.cpp from source (submodule)
git submodule update --init llama.cpp
make build-cuda-no-vmm                       # CUDA + NO_VMM for UVM compatibility

# 2. Create venv and install dependencies
uv venv
uv sync                                      # installs requests, numpy, matplotlib, etc.

# 3. Download models
bash download_models.sh 20b                  # gpt-oss-20b (~12 GiB, quick test)
bash download_models.sh 120b                 # gpt-oss-120b (~59 GiB, paper experiment)
# Models cached to ~/.cache/llama.cpp/

# 4. Verify normal mode (20B, fits in VRAM)
MODEL_20B=~/.cache/llama.cpp/ggml-org_gpt-oss-20b-GGUF_gpt-oss-20b-mxfp4.gguf
./build/bin/llama-bench -m $MODEL_20B -r 1
# Expected: pp512 ~9600 tok/s, tg128 ~354 tok/s

# 5. Smoke-test the current adaptive UVM implementation
MODEL_120B=~/.cache/llama.cpp/ggml-org_gpt-oss-120b-GGUF_gpt-oss-120b-mxfp4-00001-of-00003.gguf
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 ./build/bin/llama-bench -m $MODEL_120B -r 1
# Archived current-stack observation: pp512 ~138 tok/s, tg128 ~48 tok/s.
# This is not the historical plain-UVM Figure 6 cell.
```

## Build Details

**`build-cuda-no-vmm`** builds with `-DGGML_CUDA_NO_VMM=ON`. This is required because:
- VMM (Virtual Memory Management) uses `cuMemCreate`/`cuMemMap` which conflicts with
  `cudaMallocManaged` (UVM)
- Without NO_VMM, the memory pool tries VMM first, which fails on RTX 5090 (VMM: no)
- Uses GCC-12 for CUDA 12.9 compatibility

**UVM changes** in this fork (`eunomia-bpf/llama.cpp`):
- `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` switches allocations to `cudaMallocManaged`
- First allocation sets `cudaMemAdviseSetPreferredLocation = CPU` (data stays in RAM,
  avoids OOM during model loading since 59 GiB > 32 GB VRAM)
- On first compute forward, switches preferred location to GPU (enables GPU-initiated
  page migration for hot data)
- Forces legacy memory pool (bypasses VMM) when UVM is enabled

## Safe audit and replay

```bash
# Read-only readiness and surviving-evidence audit. No CUDA work is launched.
./run_exp1_reproduction.py audit

# Low-risk current-stack control. The historical command did not pin CPU cores.
./run_exp1_reproduction.py run --configs ncmoe64 --repetitions 5

# Full current-stack replay, strictly serial with a cooldown between cells.
# gpubpf_current must report nonzero fault, prefetch, LFU-access, and eviction
# engagement before its result is accepted.
./run_exp1_reproduction.py run \
  --configs ncmoe64,ncmoe32,uvm_current,gpubpf_current \
  --repetitions 5 --timeout-seconds 300 --cooldown-seconds 60
```

Each cell has an exclusive GPU/struct_ops lease, owned-process-only cleanup, a
hard timeout, power/thermal telemetry, and pre/post checks for Xid, kernel
errors, residual UVM references, foreign GPU processes, and residual struct_ops
state. Results include raw machine-readable `llama-bench` output. The runner
stops the matrix on the first failed gate.

## Archived observations (2026-02-16, RTX 5090)

### GPT-OSS-20B (12 GiB, fits in VRAM)

| Mode | pp512 (tok/s) | tg128 (tok/s) |
|------|--------------|--------------|
| Normal (cudaMalloc) | 9609.57 | 354.57 |
| UVM (cudaMallocManaged) | 79.49 | 2.32 |

### GPT-OSS-120B (59 GiB, UVM oversubscription on 32 GB GPU)

| Mode | pp512 (tok/s) | tg128 (tok/s) |
|------|--------------|--------------|
| Current adaptive UVM | 137.81 | 48.28 |

### Historical paper reference (not yet exactly reproduced)

| Config | pp512 (tok/s) | tg128 (tok/s) |
|--------|--------------|--------------|
| ncmoe=64 (framework offload) | 245.63 | 16.34 |
| ncmoe=32 (framework offload) | 260.14 | 18.18 |
| UVM baseline | 238.48 | 7.72 |
| UVM + user hint | 144.00 | 49.31 |
| **UVM + gpu_ext eBPF** | **229.67** | **86.89** |

## Directory Structure

```
llama.cpp/
├── README.md                 # This file
├── Makefile                  # Build targets + benchmark shortcuts
├── pyproject.toml            # uv dependencies
├── uv.lock                   # Locked dependency versions
├── llama.cpp/                # [submodule] eunomia-bpf/llama.cpp (with UVM patches)
├── download_models.sh        # One-click model downloader (20b/120b/all)
├── download_sharegpt.py      # ShareGPT dataset for server benchmarks
├── run_exp1_reproduction.py  # Fail-closed audit/current-stack replay
├── legacy_exp1_reference.json # Historical values and evidence limits
├── test_exp1_reproduction.py # Offline parser/protocol tests
├── run_exp5_two_tenant.sh    # Co-location experiment (Figure 12)
├── uvm/                      # UVM test scripts & visualization
│   ├── visbasic.py           # Figure generation
│   └── plot_colocated_results.py
├── docs/                     # Analysis & investigation notes
└── results/                  # Benchmark output logs
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `VMM: no` in CUDA init | Normal for RTX 5090; use `build-cuda-no-vmm` |
| OOM with 120B model | Ensure `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` is set |
| `llama-server: command not found` | Build first: `make build-cuda-no-vmm` |
| Slow UVM performance (~1-2 tok/s tg128) | The preferred-location CPU→GPU switch happens on first inference; subsequent runs should be faster |
