# gpu_ext

Extending Linux GPU drivers with eBPF for programmable memory offloading and scheduling.

## Overview

Modern GPU workloads (LLM inference, vector databases, DNN training) exhibit diverse memory access patterns and scheduling requirements. However, GPU drivers use fixed, one-size-fits-all policies that cannot adapt to workload-specific needs.

**gpu_ext** enables customizable GPU resource management through eBPF struct_ops:

- **Memory Management**: Pluggable eviction and prefetch policies at the driver level
- **Scheduling**: Per-process timeslice and priority control for multi-tenant GPU sharing
- **Observability**: Tracing tools for memory and scheduling events

Inspired by Linux kernel's `sched_ext`, gpu_ext brings the same extensibility to GPU drivers.

## Structure

```
├── extension/          # eBPF policies, userspace loaders, trace tools
├── kernel-module/      # Modified NVIDIA kernel modules with eBPF hooks
│   └── nvidia-module/  #   NVIDIA Open GPU Kernel Modules v575.57.08
├── workloads/          # Benchmark workloads (llama.cpp, vLLM, PyTorch, FAISS)
├── libbpf/             # libbpf submodule
├── bpftool/            # bpftool submodule
├── vmlinux/            # vmlinux BTF headers
├── microbench/         # Microbenchmarks (compute/memory)
├── scripts/            # Shared utilities
├── tools/              # Helper tools
└── docs/               # Documentation
```

**Policies in `extension/`**:

| Category | Policies |
|----------|----------|
| Eviction | FIFO, LFU, MRU, PID-quota, freq-decay, FIFO-chance |
| Prefetch | none, always-max, adaptive-sequential, adaptive-tree-iter, stride, PID-tree, PID-eviction |
| Scheduling | timeslice control, preemption control |
| Tracing | chunk_trace, prefetch_trace, gpu_sched_trace |

## Prerequisites

```bash
# Ubuntu 22.04+
sudo apt-get install -y --no-install-recommends \
    build-essential gcc g++ make \
    clang llvm \
    libelf1 libelf-dev zlib1g-dev \
    pkg-config

# Or use the Makefile shortcut:
make install
```

Additional requirements:
- **Kernel module build**: Linux kernel headers (`linux-headers-$(uname -r)`), CUDA 12.8+
- **Workloads**: Python 3.12+ with [`uv`](https://docs.astral.sh/uv/) package manager
- **Nix users**: `nix develop` provides a ready-to-use shell environment

## Build

### 1. Build eBPF Policies

```bash
make build    # Compiles all BPF policies + userspace loaders
```

This builds libbpf and bpftool from submodules, then compiles each `.bpf.c` policy into BPF bytecode (`.bpf.o`) and a userspace loader binary. BPF objects and skeleton headers go to `extension/.output/`; loader binaries are placed directly in `extension/`.

Some optional extension binaries require extra host dependencies:
- `sched_gpu_*` needs `SCX_INCLUDE_DIR=/path/to/linux/tools/sched_ext/include`
- `prefetch_adaptive_*` needs CUDA/NVML headers and stubs
- `test_preempt_demo` / `test_preempt_multi` need CUDA driver headers and stubs

### 2. Build Kernel Module

The modified NVIDIA kernel module (based on Open GPU Kernel Modules v575.57.08) adds BPF struct_ops hook points to `nvidia-uvm` for memory management and to `nvidia` for GPU scheduling.

```bash
cd kernel-module/nvidia-module
make modules -j$(nproc)
```

This runs two stages automatically: first builds OS-agnostic driver objects (`src/nvidia/`, `src/nvidia-modeset/`), then builds kernel modules via Kbuild.

Output:
```
kernel-open/nvidia.ko
kernel-open/nvidia-modeset.ko
kernel-open/nvidia-drm.ko
kernel-open/nvidia-uvm.ko      # Contains eBPF hooks
```

### 3. Load Custom Kernel Module (insmod only)

> **IMPORTANT**: Only use `insmod` for temporary loading. **NEVER run `make modules_install`** or copy `.ko` files to `/lib/modules/`. The custom modules are loaded into the running kernel only and automatically revert to the system NVIDIA driver on reboot. This ensures system stability — if anything goes wrong, a simple reboot restores the original driver.

```bash
# Unload system modules
sudo systemctl stop nvidia-persistenced 2>/dev/null || true
sudo systemctl stop gdm3 2>/dev/null || true
sleep 2
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null || true

# Load custom modules via insmod (in dependency order)
sudo insmod kernel-module/nvidia-module/kernel-open/nvidia.ko
sudo insmod kernel-module/nvidia-module/kernel-open/nvidia-modeset.ko
sudo insmod kernel-module/nvidia-module/kernel-open/nvidia-drm.ko
sudo insmod kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko

# Restart display manager
sudo systemctl start gdm3 2>/dev/null || true

# Verify
lsmod | grep nvidia
```

To revert to system modules at any time (without reboot):
```bash
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia && sudo modprobe nvidia_uvm
```

For detailed troubleshooting, see [docs/driver_docs/MODULE_LOAD_UNLOAD_GUIDE.md](docs/driver_docs/MODULE_LOAD_UNLOAD_GUIDE.md).

### 4. Load an eBPF Policy

With the custom kernel module loaded, attach a policy:

```bash
# Run a policy loader (stays in foreground, Ctrl-C to detach)
sudo ./extension/prefetch_adaptive_sequential

# Or run in background
sudo ./extension/eviction_lfu &

# Verify eBPF programs are attached
sudo bpftool prog list | grep struct_ops
```

## Workloads

Benchmark workloads for reproducing the paper experiments. See [`workloads/README.md`](workloads/README.md) for full setup and instructions.

| Workload | Paper | Description |
|----------|-------|-------------|
| llama.cpp | RQ1, Fig 6 | MoE expert offloading (GPT-OSS-120B, 59 GiB) |
| vLLM | RQ1, Fig 7 | KV-cache offloading (Qwen3-30B-A3B-FP8) |
| PyTorch | RQ1, Fig 8 | GNN training with UVM oversubscription (1M-15M nodes) |
| FAISS | RQ1, Fig 9 | Vector search on SIFT 20M/50M/100M |

Quick start:
```bash
cd workloads/llama.cpp
uv sync
uv run python configs/bench.py --uvm -o results/uvm_baseline.json
```

## Paper

> **gpu_ext: Extensible OS Policies for GPUs via eBPF**
> Yusheng Zheng, Tong Yu, Yiwei Yang, Minghui Jiang, Xiangyu Gao, Jianchang Su, Yanpeng Hu, Wenan Mao, Wei Zhang, Dan Williams, Andi Quinn
> [arXiv:2512.12615](https://arxiv.org/abs/2512.12615)

## Roadmap

### Kernel Driver Extensible Framework

- [x] **Cross-VA-block proactive prefetch**: eBPF workqueue-based prefetch that breaks the 2MB per-fault-page limit. ~20% improvement on microbenchmarks. Pending end-to-end testing on real workloads.
- [x] **GPU kernel submission-level scheduling**: `bpf_nv_gpu_preempt_tsg` kfunc for cross-process GPU TSG preemption. Two trigger paths verified: bpf_wq from struct_ops hooks, and sleepable uprobe on `cuLaunchKernel` (avg 312us, no bpf_wq needed). (see `docs/gpu_preempt_kfunc_plan.md`)
- [x] **CPU-GPU coordinated scheduling**: Combined sched_ext + GPU memory/scheduling policies (FPRS). ~5% improvement on multi-tenant serving. (see `docs/xcoord_plan.md`)
- [ ] **Better coordinated scheduling policy**: Exploring AI-driven policy search for improved CPU-GPU coordination.

### Device-Side Extensions

- [ ] **Safety verification**: Evaluate how the BPF verifier can reject unsafe GPU extension programs. Design exists in paper; no implementation yet.
- [x] **NVBit + eBPF integration (POC)**: Proof-of-concept completed via [bpftime](https://github.com/eunomia-bpf/bpftime).
- [ ] **NVBit + eBPF integration (production)**: Full integration with bpftime runtime. In progress.

### Policy and Evaluation

- [x] **Combined host-side policies**: Multiple compositions implemented and benchmarked (always_max + cycle_moe, always_max + xCoord, FPRS coord v2).
- [ ] **More complex combined policies**: Explore richer compositions (prefetch + eviction + scheduling + CPU coordination) once framework capabilities are expanded.
- [ ] **Dynamism**: Fast policy injection and fast runtime via compiler techniques, enabling both rapid development iteration and low-overhead execution.
- [ ] **Paper improvements**: Strengthen evaluation methodology, add new workloads, and refine the writing.

## Related

- [bpftime](https://github.com/eunomia-bpf/bpftime) - GPU device-side eBPF support

## License

MIT
