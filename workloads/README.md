# Workloads: Reproducing gpu_ext Paper Experiments

This directory contains all benchmark workloads used in the gpu_ext paper evaluation.

For the current revision's measured cohorts, native/BPF comparisons and
supplementary results, start with the [artifact guide](../ARTIFACT.md).
The original experiment numbering and setup descriptions below are retained;
they are not a current revision-completion checklist. Use each linked dated
report for its hardware, configuration, metric and source version.

Artifact maintenance updates repository instructions and evidence links;
manuscript edits require separate authorization. If a change touches
device-side behavior, check the corresponding bpftime implementation first.

## Hardware Requirements

| Server | CPU | RAM | GPU | Used For |
|--------|-----|-----|-----|----------|
| Server A | Intel Core Ultra 9 285K (24 cores) | 128 GB DDR5 | NVIDIA RTX 5090 (32GB) | Most experiments |
| Server B | Dual Intel Gold 6138 (80 cores) | 256 GB | NVIDIA P40 | Observability overhead |

## Software Requirements

- CUDA 12.8+ with nvcc
- Python 3.12+ with `uv` package manager
- PyTorch 2.9.0+
- vLLM 0.11.0+
- Modified NVIDIA kernel module (from `kernel-module/`)
- `bpftool` for loading eBPF policies (from `bpftool/`)

## Directory Structure

```
workloads/
├── README.md                # This file
├── REPRODUCTION_LOG.md      # Experiment session logs
├── cleanup_gpu.py           # Kill stale GPU processes
├── scripts/                 # Shared tools (Layer 2)
│   ├── common.py            # Shared utilities (cleanup_gpu, wait_for_server, etc.)
│   ├── run_trials.py        # Generic N-trial runner
│   └── collect_results.py   # Geomean/stddev aggregation
├── llama.cpp/               # LLM inference (Expert Offloading, Multi-Tenant)
│   ├── Makefile             # Build & benchmark targets
│   ├── llama.cpp/           # [submodule] eunomia-bpf/llama.cpp
│   ├── configs/             # Atomic benchmark scripts (Layer 1)
│   │   ├── bench.py         # llama-bench single run
│   │   └── server_bench.py  # llama-server + vllm bench serve
│   ├── run_exp5_two_tenant.sh  # Co-location experiment (not yet atomized)
│   ├── datasets/            # ShareGPT (gitignored)
│   └── results/
├── vllm/                    # LLM serving (KV-cache Offloading)
│   ├── Makefile             # Quick bench + dataset download
│   ├── configs/             # Atomic benchmark scripts (Layer 1)
│   │   └── serve_bench.py   # vLLM server + vllm bench serve
│   ├── datasets/            # ShareGPT (gitignored)
│   └── results/
├── pytorch/                 # GNN Training (UVM oversubscription)
│   ├── Makefile             # Build allocator .so
│   ├── benchmark_gnn_uvm.py # Core GNN benchmark
│   ├── configs/             # Atomic benchmark scripts (Layer 1)
│   │   └── gnn.py           # GNN training single run
│   └── result/
├── faiss/                   # Vector Search (SIFT dataset)
│   ├── bench_gpu_1bn.py     # Core FAISS benchmark
│   ├── faiss/               # [submodule] eunomia-bpf/faiss
│   ├── configs/             # Atomic benchmark scripts (Layer 1)
│   │   └── search.py        # FAISS search single run
│   └── results/
└── _deprecated/             # Old monolithic scripts (superseded by configs/*.py)
```

### Script Architecture

**Layer 1 (Atomic scripts)**: Each `configs/*.py` runs one config, one trial, outputs JSON.

```bash
# Example: single llama-bench run
uv run python configs/bench.py --uvm --output results/bench_uvm.json
```

**Layer 2 (Multi-trial)**: `scripts/run_trials.py` runs N trials, `scripts/collect_results.py` computes geomean/stddev.

```bash
# Example: 10 trials of llama-bench with UVM
python scripts/run_trials.py --trials 10 \
  --command "uv run python llama.cpp/configs/bench.py --uvm" \
  --results-dir results/exp1/uvm_baseline/
```

---

## Quick Start: Loading gpu_ext eBPF Policies

Before running any experiment with gpu_ext, you need the modified kernel module and eBPF policies loaded:

```bash
# 1. Build and load the modified nvidia-uvm kernel module
cd /path/to/gpu_ext/kernel-module
make && sudo make install

# 2. Build the eBPF extension policies
cd /path/to/gpu_ext/extension
make

# 3. Load a policy (example: stride prefetch + LFU eviction)
sudo bpftool struct_ops register prefetch_stride.bpf.o
sudo bpftool struct_ops register eviction_lfu.bpf.o

# 4. Verify
sudo bpftool struct_ops show
```

To unload policies:
```bash
sudo bpftool struct_ops show          # find the ID
sudo bpftool struct_ops unregister id <ID>
```

---

## Experiment 1: llama.cpp — Expert Offloading (RQ1, Figure 5)

**Paper claim**: gpu_ext achieves 4.8x decode speedup over framework offloading on GPT-OSS-120B MoE.

### Setup

```bash
cd workloads/llama.cpp

# Build llama.cpp with CUDA (no VMM, needed for UVM)
git submodule update --init llama.cpp
make build-cuda

# Download ShareGPT dataset
python download_sharegpt.py

# Download model (GPT-OSS-120B, ~59 GiB, auto-cached to ~/.cache/llama.cpp/)
make download-models
```

### Run Benchmarks

Five configurations, using atomic scripts:

```bash
cd workloads/llama.cpp

# Config 1: Framework CPU offload (ncmoe=64)
uv run python configs/bench.py --ncmoe 64 -o results/ncmoe64.json

# Config 2: Framework CPU offload (ncmoe=32)
uv run python configs/bench.py --ncmoe 32 -o results/ncmoe32.json

# Config 3: Default UVM (no policy)
uv run python configs/bench.py --uvm -o results/uvm_baseline.json

# Config 4: UVM + user hints (cudaMemAdvise)
# (requires llama.cpp built with cudaMemAdvise hints enabled)
uv run python configs/bench.py --uvm -o results/uvm_user_hint.json

# Config 5: UVM + gpu_ext eBPF (load policies first, then same --uvm flag)
sudo bpftool struct_ops register /path/to/gpu_ext/extension/prefetch_stride.bpf.o
sudo bpftool struct_ops register /path/to/gpu_ext/extension/eviction_lfu.bpf.o
uv run python configs/bench.py --uvm -o results/uvm_ebpf.json
```

For 10-trial runs with geomean aggregation:
```bash
cd workloads
python scripts/run_trials.py --trials 10 \
  --command "uv run --directory llama.cpp python configs/bench.py --uvm" \
  --results-dir llama.cpp/results/exp1/uvm_baseline/
```

### Expected Output

```
| model                   | size    | params   | backend | ngl | test  | t/s            |
| gpt-oss 120B MXFP4 MoE | 59.02G  | 116.83B  | CUDA    | 99  | pp512 | 238.48 ± 1.43  |
| gpt-oss 120B MXFP4 MoE | 59.02G  | 116.83B  | CUDA    | 99  | tg128 | 86.89 ± 0.50   |
```

Key metrics: `pp512` = prefill throughput (tok/s), `tg128` = decode throughput (tok/s).

### Generate Figure

```bash
cd workloads/llama.cpp/uvm && python visbasic.py   # produces llama_uvm_combined_color.pdf
```

### Reference Results (RTX 5090)

| Config | pp512 (tok/s) | tg128 (tok/s) |
|--------|--------------|--------------|
| ncmoe=64 (framework offload) | 245.63 | 16.34 |
| ncmoe=32 (framework offload) | 260.14 | 18.18 |
| UVM baseline | 238.48 | 7.72 |
| UVM + user hint | 144.00 | 49.31 |
| **UVM + gpu_ext eBPF** | **229.67** | **86.89** |

---

## Experiment 2: vLLM — KV-cache Offloading (RQ1, Figure 6)

**Paper claim**: gpu_ext improves TTFT by 1.7-2x and decoding throughput by 1.3x over vLLM CPU-offload.

### Setup

```bash
cd workloads/vllm

# Initialize submodule and install vLLM from local source (with UVM support)
git submodule update --init workloads/vllm/vllm
uv sync
uv pip install -e vllm/

# Download dataset (if not present)
make download-datasets
# Model (Qwen3-30B-A3B-FP8) is auto-downloaded by vLLM on first run
```

### Run Benchmarks

Using the atomic script `configs/serve_bench.py`:

```bash
cd workloads/vllm

# Config 1: CPU offload (8GB)
uv run python configs/serve_bench.py --mode cpu_offload -o results/cpu_offload.json

# Config 2: UVM baseline (no policy)
uv run python configs/serve_bench.py --mode uvm -o results/uvm_baseline.json

# Config 3: UVM + gpu_ext eBPF (load policies first, then same --mode uvm)
sudo bpftool struct_ops register /path/to/gpu_ext/extension/prefetch_adaptive_sequential.bpf.o
sudo bpftool struct_ops register /path/to/gpu_ext/extension/eviction_lfu.bpf.o
uv run python configs/serve_bench.py --mode uvm -o results/uvm_ebpf.json
```

For 10-trial runs:
```bash
cd workloads
python scripts/run_trials.py --trials 10 \
  --command "uv run --directory vllm python configs/serve_bench.py --mode cpu_offload" \
  --results-dir vllm/results/exp2/cpu_offload/
```

**Note**: `serve_bench.py` defaults to `workloads/vllm/vllm/` (submodule) for vLLM server. Override via `VLLM_SERVER_DIR` environment variable.

### Expected Output

Results saved to `results/uvm_baseline_results_YYYYMMDD_HHMMSS.json` with metrics:
- Mean/Median/P99 TTFT (ms), TPOT (ms), ITL (ms)
- Request throughput (req/s), Output token throughput (tok/s)

### Generate Figure

```bash
cd uvm/first-iter && python generate_figures.py
# produces ttft_tpot_combined.pdf
```

### Reference Results (RTX 5090, 100 concurrent requests)

| Config | Mean TTFT (ms) | Mean TPOT (ms) | Throughput (tok/s) |
|--------|---------------|----------------|-------------------|
| CPU Offload (8GB) | 8387.80 | 324.13 | 391.14 |
| UVM Baseline | 9642.27 | 374.23 | 307.26 |
| **UVM + gpu_ext eBPF** | **5042.22** | **235.68** | **376.53** |
| LMCache | 5401.71 | 222.24 | 571.54 |

---

## Experiment 3: PyTorch GNN — Graph Neural Network Training (RQ1, Figure 7)

**Paper claim**: gpu_ext achieves 2.65x speedup without user prefetch, 1.44x additional with user prefetch at 15M nodes.

### Setup

```bash
cd workloads/pytorch

# Build custom CUDA allocators
make all    # produces uvm_allocator.so and gpu_allocator.so

# Create venv
uv venv && source .venv/bin/activate
uv pip install torch psutil torch-geometric
```

No external dataset needed — graphs are randomly generated.

### Run Benchmarks

Using the atomic script `configs/gnn.py`:

```bash
cd workloads/pytorch

# No UVM baseline (pure GPU, OOMs beyond ~7M nodes)
uv run python configs/gnn.py --nodes 5000000 -o results/gnn_5M_normal.json

# UVM baseline (default driver, no eBPF)
uv run python configs/gnn.py --nodes 10000000 --uvm -o results/gnn_10M_uvm.json

# UVM + gpu_ext eBPF (load policies first, then same --uvm flag)
sudo bpftool struct_ops register /path/to/gpu_ext/extension/prefetch_adaptive_sequential.bpf.o
uv run python configs/gnn.py --nodes 10000000 --uvm -o results/gnn_10M_uvm_ebpf.json
```

For 10-trial runs across node counts:
```bash
cd workloads
for NODES in 5000000 7000000 8000000 10000000 12000000 15000000; do
  python scripts/run_trials.py --trials 10 \
    --command "uv run --directory pytorch python configs/gnn.py --nodes $NODES --uvm" \
    --results-dir pytorch/result/exp3/uvm_${NODES}/
done
```

### Expected Output

Each run produces a JSON file:
```json
{
  "config": {"nodes": 10000000, "use_uvm": true, ...},
  "epoch_times": [69.87, 70.12, ...],
  "avg_epoch_time": 69.87,
  "uvm_stats": {"peak_allocated_gb": 45.11, "allocations": 23934}
}
```

### Generate Figure

```bash
python visualize_all.py
# produces uvm_benchmark_comparison.pdf
```

### Reference Results (RTX 5090, without user prefetch)

| Nodes | No UVM | UVM Baseline | UVM + gpu_ext | Speedup |
|-------|--------|-------------|--------------|---------|
| 5M | 1.14s | 34.23s | 12.76s | 2.68x |
| 7M | 1.79s | 48.28s | 17.81s | 2.71x |
| 8M | OOM | 55.36s | 20.51s | 2.70x |
| 10M | OOM | 70.06s | 26.47s | 2.65x |
| 12M | OOM | 93.71s | 39.74s | 2.36x |
| 15M | OOM | 292.77s | 168.73s | 1.74x |

---

## Experiment 4: Faiss — Vector Search (RQ1, Figure 8)

**Paper claim**: gpu_ext reduces build time by 21-29% and query latency by 10-16%.

### Setup

```bash
cd workloads/faiss

# Create venv and install Python deps
uv sync

# Build FAISS from submodule
git submodule update --init faiss
cd faiss
cmake -B build \
  -DFAISS_ENABLE_GPU=ON \
  -DFAISS_ENABLE_PYTHON=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_OPT_LEVEL=avx2 \
  -DBUILD_TESTING=OFF \
  -DBLA_VENDOR=OpenBLAS
make -C build -j$(nproc) swigfaiss
cd ..

# Install FAISS from local build into the venv
uv pip install -e faiss/build/faiss/python/

# Download SIFT dataset
bash download_sift.sh
```

### Run Benchmarks

Using the atomic script `configs/search.py`:

```bash
cd workloads/faiss

# UVM baseline (no eBPF)
uv run python configs/search.py --dataset SIFT50M --uvm -o results/sift50m_uvm.json
uv run python configs/search.py --dataset SIFT100M --uvm -o results/sift100m_uvm.json

# UVM + gpu_ext eBPF (load policies first, then same --uvm flag)
sudo bpftool struct_ops register /path/to/gpu_ext/extension/prefetch_adaptive_sequential.bpf.o
uv run python configs/search.py --dataset SIFT100M --uvm -o results/sift100m_uvm_ebpf.json
```

For 10-trial runs:
```bash
cd workloads
python scripts/run_trials.py --trials 10 \
  --command "uv run --directory faiss python configs/search.py --dataset SIFT100M --uvm" \
  --results-dir faiss/results/exp4/sift100m_uvm/
```

### Expected Output

Console output includes:
```
Add time: 68.407 s
probe=1  : 5.135 s 1-R@1: 0.4486
probe=4  : 14.393 s 1-R@1: 0.7655
probe=16 : 56.511 s 1-R@1: 0.9476
```

Results also saved as JSON in `results/` directory.

### Generate Figure

```bash
cd results && python vis_faiss.py
# produces faiss_benchmark_results.pdf
```

---

## Experiment 5: Multi-Tenant — Two-Tenant Co-location (RQ2, Figure 11)

**Paper claim**: gpu_ext achieves mutual improvement: LC TPOT reduced by 40-45%, BE training improved by 28%.

### Setup

Requires both llama.cpp and PyTorch workloads set up (see Experiments 1 & 3).

### Run Benchmarks

**Step 1: Single-tenant baselines**
```bash
# Baseline: llama.cpp single-tenant (gpt-oss-20b, UVM)
cd workloads/llama.cpp
GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
  ./build/bin/llama-server --gpt-oss-20b-default -c 65536 &
LLAMA_PID=$!
sleep 30

# Run llama.cpp benchmark
uv run vllm bench serve \
  --model Qwen/Qwen3-30B-A3B-FP8 \
  --dataset-name sharegpt --num-prompts 100 \
  --dataset-path datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --base-url http://127.0.0.1:8013 \
  --max-concurrency=1 --request-rate 0.2 \
  2>&1 | tee uvm/results_single_llama.log
kill $LLAMA_PID

# Baseline: GNN single-tenant
cd workloads/pytorch
CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 uv run python benchmark_gnn_uvm.py \
  --dataset random --nodes 8000000 \
  --edges_per_node 10 --features 128 --hidden 256 \
  --epochs 25 --warmup 1 --prop chunked --use_uvm \
  --report_json result/single_gnn.json
```

**Step 2: Co-located (default UVM, no policy)**
```bash
# Terminal 1: Start llama.cpp server
GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
  ./build/bin/llama-server --gpt-oss-20b-default -c 65536 &

# Terminal 2: Start GNN training (wait for server to load)
sleep 30
CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 uv run python benchmark_gnn_uvm.py \
  --dataset random --nodes 8000000 --use_uvm --epochs 25 \
  --report_json result/colocated_gnn_baseline.json &

# Terminal 3: Run llama.cpp benchmark (wait for GNN to start)
sleep 10
uv run vllm bench serve \
  --dataset-name sharegpt --num-prompts 100 \
  --dataset-path datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --base-url http://127.0.0.1:8013 \
  --max-concurrency=1 --request-rate 0.2
```

**Step 3: Co-located with gpu_ext per-tenant policies**
```bash
# Load per-tenant memory policies
sudo bpftool struct_ops register /path/to/gpu_ext/extension/prefetch_eviction_pid.bpf.o

# Then repeat Step 2 with policies active
```

### Generate Figure

```bash
cd workloads/llama.cpp/uvm
python plot_colocated_results.py
# produces fig_colocated_results.pdf
```

### Reference Results (RTX 5090)

| Metric | Single llama.cpp | Co-located (UVM) | Co-located (gpu_ext) |
|--------|-----------------|-------------------|---------------------|
| TPOT (ms) | 3.67 | 19.73 (5.38x worse) | 10.86 (2.96x, **45% better**) |
| GNN epoch (s) | - | 23.23 | 16.72 (**28% better**) |

---

## Experiment 6: Multi-Tenant Microbenchmarks (RQ2, Figures 9-10)

These experiments use CUDA microbenchmarks from `microbench/` directory.

### Compute-bound Timeslice Scheduler (Figure 9)
Uses gpu_ext's BPF struct_ops scheduling policies with LC+BE processes.

### Memory-bound Priority Differentiation (Figure 10)
Uses HotSpot, GEMM (PolyBench), and K-Means (UVMBench) kernels.
Scripts and data are in `microbench/memory/`.

---

## Experiment 7: Mechanism Overhead (RQ3, Figures 12, Tables 2-3)

### Host Runtime Overhead
Uses GEMM and HotSpot with hooks enabled but no policy attached.
Expected overhead: <0.2%.

### Device-side Observability
Uses llama.cpp prefill with Llama 1B on P40 GPU.
Tools: `kernelretsnoop`, `threadhist`, `launchlate` from `extension/` directory.

### Device-side Microbenchmarks (Figure 12)
Vector-add from cuda-samples, comparing eGPU-style vs gpu_ext SIMT-aware execution.
Scripts in `microbench/memory/`.

---

## Path Configuration

Scripts use relative paths by default. Override via environment variables when needed:

| Variable | Used By | Default | Description |
|----------|---------|---------|-------------|
| `VLLM_SERVER_DIR` | `vllm/configs/serve_bench.py` | `vllm/` (submodule) | vLLM installation directory |
| `DATASET_PATH` | `vllm/configs/serve_bench.py` | `datasets/ShareGPT_V3_...json` | ShareGPT dataset path |
| `MODEL_120B_CACHE` | `llama.cpp/Makefile` | `$HOME/.cache/llama.cpp/...` | GPT-OSS-120B model path |

---

## Paper Reproduction Checklist (Strict)

All experiments must strictly follow the paper configurations. Each experiment has a **one-click run script** that saves full logs and outputs results.

### Software Versions (Paper §6.1)

| Software | Version | Notes |
|----------|---------|-------|
| PyTorch | 2.10.0+cu128 | with custom UVM/GPU allocator |
| vLLM | 0.11.0rc2 | UVM fork submodule (`workloads/vllm/vllm/`) |
| llama.cpp | build 7101 (commit 65b578b1) | eunomia-bpf/llama.cpp fork |
| Faiss | 1.13.0 | eunomia-bpf/faiss fork |
| CUDA | 12.8 | Driver 570.133.20 |

All tests: **10 trials, geometric mean** unless otherwise specified.

### Experiment → Figure/Table Mapping

| # | Experiment | Paper | Model/Dataset | Atomic Script | Needs Modified KM? |
|---|-----------|-------|---------------|---------------|-------------------|
| 1 | llama.cpp Expert Offloading | RQ1, Fig 6 | GPT-OSS-120B (59 GiB) | `llama.cpp/configs/bench.py` | Configs 4-5 only |
| 2 | vLLM KV-cache Offloading | RQ1, Fig 7 | Qwen3-30B-A3B-FP8 (~30GB) | `vllm/configs/serve_bench.py` | Config 3 only |
| 3 | PyTorch GNN Training | RQ1, Fig 8 | Random graphs 1M-15M nodes | `pytorch/configs/gnn.py` | Configs 4-5 only |
| 4 | FAISS Vector Search | RQ1, Fig 9 | SIFT 20M/50M/100M | `faiss/configs/search.py` | Config 2 only |
| 5 | Two-Tenant Co-location | RQ2, Fig 12 | llama.cpp 20B + GNN 8M | `llama.cpp/run_exp5_two_tenant.sh` | gpu_ext config only |
| 6 | Compute-bound Scheduler | RQ2, Fig 10 | 2LC+4BE processes | (microbench) | Yes |
| 7 | Memory-bound Priority | RQ2, Fig 11 | HotSpot/GEMM/K-Means | (microbench) | Yes |
| 8 | Mechanism Overhead | RQ3, Fig 13 | vector-add, GEMM, HotSpot | (microbench) | Yes |

### Configurations Per Experiment

**Exp 1: llama.cpp (Fig 6) — 5 configs:**
1. `ncmoe=64` — Framework CPU offload (64 experts on CPU)
2. `ncmoe=32` — Framework CPU offload (32 experts on CPU)
3. `UVM baseline` — Default CUDA Unified Memory, no policy
4. `UVM + user hints` — cudaMemAdvise hints (needs modified KM)
5. `UVM + gpu_ext eBPF` — Stride prefetch + LFU eviction (needs modified KM)

**Exp 2: vLLM (Fig 7) — 4 configs:**
1. `cpu_offload` — `vllm serve --cpu-offload-gb 8`
2. `uvm_baseline` — `VLLM_USE_UVM=1 vllm serve`
3. `uvm_ebpf` — UVM + gpu_ext sequential prefetch (needs modified KM)
4. `lmcache` — LMCache KV-cache offloading

**Exp 3: PyTorch GNN (Fig 8) — 5 configs × 8 node counts:**
- Node counts: 1M, 3M, 5M, 7M, 8M, 10M, 12M, 15M
1. `native GPU` — Default allocator, no UVM (1M-7M only, OOMs beyond)
2. `UVM baseline` — cudaMallocManaged, default driver (5M-15M)
3. `UVM + user prefetch` — cudaMemPrefetchAsync (5M-15M)
4. `UVM + gpu_ext eBPF` — Sequential eBPF prefetch (needs modified KM)
5. `UVM + user prefetch + gpu_ext` — Combined (needs modified KM)

**Exp 4: FAISS (Fig 9) — 2 configs × 3 dataset sizes:**
- Datasets: SIFT20M (9.5GB), SIFT50M (24GB), SIFT100M (48GB)
- Index: IVF4096,Flat, nprobe=1,4,16
1. `UVM baseline` — Default UVM
2. `UVM + gpu_ext eBPF` — Adaptive prefetch (needs modified KM)

**Exp 5: Two-Tenant (Fig 12) — 2 configs:**
- T1 (LC): llama.cpp server (gpt-oss-20b, UVM, ctx=65536), 100 ShareGPT @ 0.2 RPS
- T2 (BE): GNN training (8M nodes, UVM, 25 epochs)
1. `default UVM` — Both compete under driver FIFO/LRU
2. `gpu_ext per-tenant` — LC gets prefetch priority (needs modified KM)

### Baseline Configs (Runnable Without Modified KM)

| Experiment | Runnable Configs | Atomic Script |
|-----------|-----------------|---------------|
| llama.cpp | 1 (ncmoe=64), 2 (ncmoe=32), 3 (UVM baseline) | `llama.cpp/configs/bench.py` |
| vLLM | 1 (cpu_offload), 2 (uvm_baseline), 4 (lmcache) | `vllm/configs/serve_bench.py` |
| PyTorch GNN | 1 (native GPU), 2 (UVM baseline), 3 (UVM+prefetch) | `pytorch/configs/gnn.py` |
| FAISS | 1 (UVM baseline) | `faiss/configs/search.py` |
| Two-Tenant | 1 (default UVM) | `llama.cpp/run_exp5_two_tenant.sh` |

### Reference Results (Paper, RTX 5090)

**llama.cpp GPT-OSS-120B (tok/s):**
| Config | pp512 | tg128 |
|--------|-------|-------|
| ncmoe=64 | 245.63 | 16.34 |
| ncmoe=32 | 260.14 | 18.18 |
| UVM baseline | 238.48 | 7.72 |
| UVM+hints | 144.00 | 49.31 |
| UVM+gpu_ext | 229.67 | 86.89 |

**vLLM Qwen-30B (100 concurrent requests):**
| Config | Mean TTFT (ms) | Mean TPOT (ms) | Throughput (tok/s) |
|--------|---------------|---------------|-------------------|
| CPU offload | 8387.80 | 324.13 | 391.14 |
| UVM baseline | 9642.27 | 374.23 | 307.26 |
| UVM+gpu_ext | 5042.22 | 235.68 | 376.53 |
| LMCache | 5401.71 | 222.24 | 571.54 |

**PyTorch GNN (epoch time, without user prefetch):**
| Nodes | Native GPU | UVM Baseline | UVM+gpu_ext |
|-------|-----------|-------------|-------------|
| 5M | 1.14s | 34.23s | 12.76s |
| 7M | 1.79s | 48.28s | 17.81s |
| 8M | OOM | 55.36s | 20.51s |
| 10M | OOM | 70.06s | 26.47s |
| 12M | OOM | 93.71s | 39.74s |
| 15M | OOM | 292.77s | 168.73s |

**FAISS SIFT100M (IVF4096,Flat):**
| Metric | UVM Baseline | UVM+gpu_ext |
|--------|-------------|-------------|
| Add time | 68.41s | 49.31s |
| Search nprobe=1 | 5.14s | 4.53s |
| Search nprobe=4 | 14.39s | 13.11s |
| Search nprobe=16 | 56.51s | 51.44s |

**Two-Tenant (llama.cpp 20B + GNN 8M):**
| Metric | Single llama.cpp | Co-located (UVM) | Co-located (gpu_ext) |
|--------|-----------------|------------------|---------------------|
| TPOT (ms) | 3.67 | 19.73 | 10.86 |
| GNN epoch (s) | — | 23.23 | 16.72 |

---

## What's Still Missing From This Repo

The following items are needed to fully reproduce all experiments but are not yet in the gpu_ext repository:

| Item | Needed For | Size | How to Get |
|------|-----------|------|-----------|
| GPT-OSS-120B model | llama.cpp (Exp 1) | ~59 GiB | `huggingface-cli download ggml-org/gpt-oss-120b-GGUF` |
| Qwen3-30B-A3B-FP8 model | vLLM (Exp 2) | ~30 GB | Auto-downloaded by vLLM on first run |
| SIFT1B dataset | Faiss (Exp 4) | ~60 GB | `bash faiss/download_sift.sh` |
| vLLM (modified for UVM) | vLLM (Exp 2) | — | submodule at `workloads/vllm/vllm/`, install with `uv pip install -e vllm/` |
| LMCache | vLLM baseline (Exp 2) | — | Clone from github.com/LMCache/LMCache |
| HotSpot/GEMM/K-Means | Multi-tenant microbench (Exp 7) | small | From Rodinia/PolyBench/UVMBench |
| cuda-samples vector-add | Device microbench (Exp 8) | small | From NVIDIA cuda-samples |
