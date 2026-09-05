BPFTIME_ROOT ?= ../../../../bpftime-table1-575
BPFTIME_BUILD ?= $(BPFTIME_ROOT)/build-table1-575-strict
CUDA_ROOT ?= /usr/local/cuda-12.9
CLANG ?= clang
NVCC ?= $(CUDA_ROOT)/bin/nvcc
LIBBPF_INCLUDE := $(BPFTIME_BUILD)/libbpf
LIBBPF_STATIC := $(BPFTIME_BUILD)/libbpf/libbpf/libbpf.a
VMLINUX_INCLUDE := $(BPFTIME_ROOT)/third_party/vmlinux/x86
BPF_INCLUDES := -I$(LIBBPF_INCLUDE) -I$(VMLINUX_INCLUDE) \
	-I/usr/include/x86_64-linux-gnu

.PHONY: all test
all: .output/warp-map-probe.bpf.o .output/warp-map-loader \
	.output/warp-map-bench .output/warp-map-bench.ptx

test:
	python3 -m unittest -v test_strict_warp_map_scaling.py

.output:
	mkdir -p $@

.output/warp-map-probe.bpf.o: warp_map_probe.bpf.c | .output
	$(CLANG) -g -O2 -Wall -Werror -target bpf -D__TARGET_ARCH_x86 \
		$(BPF_INCLUDES) -c $< -o $@

.output/warp-map-loader: warp_map_loader.c | .output
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -I$(LIBBPF_INCLUDE) \
		$< $(LIBBPF_STATIC) -lelf -lz -o $@

.output/warp-map-bench: warp_map_bench.cu | .output
	$(NVCC) -std=c++17 -O2 -lineinfo -cudart shared \
		-gencode arch=compute_120,code=sm_120 \
		-gencode arch=compute_120,code=compute_120 \
		-Xcompiler=-Wall,-Wextra -Xlinker --build-id=none $< -o $@

.output/warp-map-bench.ptx: warp_map_bench.cu | .output
	$(NVCC) -std=c++17 -O2 -lineinfo --ptx \
		-gencode arch=compute_120,code=compute_120 $< -o $@
