# Makefile for Cluster Launch Control Benchmark
# Tested on RTX 5090 (CC 12.0) with CUDA 12.9

# CUDA paths
CUDA_12_9 = /usr/local/cuda-12.9

# Compiler settings
NVCC = $(CUDA_12_9)/bin/nvcc

# Architecture
ARCH = sm_120

# Compiler flags
NVCC_FLAGS = -arch=$(ARCH) -O3

# ============================================
# BUILD TARGETS
# ============================================

.PHONY: all clean test help

# Default target
all: clc_policy_benchmark

# Unified policy benchmark (comprehensive + specialized modes)
clc_policy_benchmark: clc_policy_benchmark.cu clc_policy_framework.cuh clc_policies.cuh benchmark_kernels.cuh
	$(NVCC) $(NVCC_FLAGS) -o clc_policy_benchmark clc_policy_benchmark.cu
	@echo "Built: clc_policy_benchmark"

# ============================================
# TEST TARGETS
# ============================================

test: clc_policy_benchmark
	@echo "=== Testing Comprehensive CLC Benchmark ==="
	@echo ""
	./clc_policy_benchmark
	@echo ""

# Run policy benchmark (default: 1M elements)
run: clc_policy_benchmark
	@echo "=== Running Comprehensive CLC Benchmark (1M elements) ==="
	./clc_policy_benchmark

# Run with different problem sizes
run-small: clc_policy_benchmark
	@echo "=== Running Comprehensive CLC Benchmark (256K elements) ==="
	./clc_policy_benchmark 262144 256

run-large: clc_policy_benchmark
	@echo "=== Running Comprehensive CLC Benchmark (4M elements) ==="
	./clc_policy_benchmark 4194304 256


# ============================================
# INFORMATION
# ============================================

info:
	@echo "=== Environment Information ==="
	@echo "GPU:"
	@nvidia-smi --query-gpu=name,compute_cap --format=csv
	@echo ""
	@echo "Driver:"
	@cat /proc/driver/nvidia/version | head -1
	@echo ""
	@echo "CUDA Version:"
	@$(NVCC) --version | grep "release"

# ============================================
# CLEANUP
# ============================================

clean:
	rm -f clc_policy_benchmark
	rm -f /tmp/clc*.ptx /tmp/clc*.cubin
	rm -f *.csv *.png
	@echo "Cleaned all binaries, temp files, CSV and PNG files"

# ============================================
# HELP
# ============================================

help:
	@echo "CLC AI Workload Benchmark - Makefile"
	@echo ""
	@echo "BUILD TARGETS:"
	@echo "  make                         - Build all benchmarks"
	@echo "  make all                     - Build all benchmarks"
	@echo "  make clc_benchmark_workloads - Build AI workload benchmark"
	@echo "  make clc_policy_benchmark    - Build the policy scheduler benchmark"
	@echo ""
	@echo "TEST TARGETS:"
	@echo "  make test          - Build and run the original benchmark"
	@echo "  make run           - Run the original benchmark"
	@echo "  make run-small     - Run the original benchmark (256K elements)"
	@echo "  make run-large     - Run the original benchmark (4M elements)"
	@echo "  make run-policy    - Run the new policy benchmark"
	@echo ""
	@echo "BENCHMARK USAGE:"
	@echo "  ./clc_benchmark_workloads [n] [threads]"
	@echo "    n       - Number of elements (default: 1048576)"
	@echo "    threads - Threads per block (default: 256)"
	@echo ""
	@echo "EXAMPLES:"
	@echo "  ./clc_benchmark_workloads           # 1M elements (default)"
	@echo "  ./clc_benchmark_workloads 262144    # 256K elements (quick test)"
	@echo "  ./clc_benchmark_workloads 4194304   # 4M elements (production scale)"
	@echo ""
	@echo "AI SCENARIOS TESTED:"
	@echo "  1. NLP: Variable Sequence Lengths (BERT/GPT)"
	@echo "  2. AI Inference: Dynamic Batching"
	@echo "  3. Transformer: Sparse Attention"
	@echo "  4. GNN: Variable Graph Node Degrees"
	@echo "  5. MoE: Mixture of Experts Routing"
	@echo "  6. CV: Video Frame Processing"
	@echo ""
	@echo "INFORMATION:"
	@echo "  make info          - Show GPU, driver, and CUDA info"
	@echo ""
	@echo "CLEANUP:"
	@echo "  make clean         - Remove all binaries and temp files"
	@echo ""
	@echo "REQUIREMENTS:"
	@echo "  - CUDA 12.9+"
	@echo "  - Compute Capability 10.0+ (Blackwell or newer)"
	@echo "  - Driver 575+"

.DEFAULT_GOAL := all
