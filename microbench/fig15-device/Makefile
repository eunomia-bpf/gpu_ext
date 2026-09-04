BPFTIME_ROOT ?= ../../../bpftime-table1-575
BPFTIME_BUILD ?= $(BPFTIME_ROOT)/build-table1-575
CUDA_ROOT ?= /usr/local/cuda-12.9
CLANG ?= clang
NVCC ?= $(CUDA_ROOT)/bin/nvcc
LIBBPF_INCLUDE := $(BPFTIME_BUILD)/libbpf
LIBBPF_STATIC := $(BPFTIME_BUILD)/libbpf/libbpf/libbpf.a
VMLINUX_INCLUDE := $(BPFTIME_ROOT)/third_party/vmlinux/x86
BPF_INCLUDES := -I$(LIBBPF_INCLUDE) -I$(VMLINUX_INCLUDE) \
	-I/usr/include/x86_64-linux-gnu

.PHONY: all clean
all: .output/map-probe.bpf.o .output/map-probe .output/map-bench \
	.output/map-bench.ptx

.output:
	mkdir -p $@

.output/map-probe.bpf.o: map_probe.bpf.c | .output
	$(CLANG) -g -O2 -Wall -Werror -target bpf -D__TARGET_ARCH_x86 \
		$(BPF_INCLUDES) -c $< -o $@

.output/map-probe: map_probe.c | .output
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -I$(LIBBPF_INCLUDE) \
		$< $(LIBBPF_STATIC) -lelf -lz -o $@

.output/map-bench: map_bench.cu | .output
	$(NVCC) -std=c++17 -O2 -lineinfo -cudart shared \
		-gencode arch=compute_120,code=sm_120 \
		-gencode arch=compute_120,code=compute_120 \
		-Xcompiler=-Wall,-Wextra -Xlinker --build-id=none $< -o $@

.output/map-bench.ptx: map_bench.cu | .output
	$(NVCC) -std=c++17 -O2 -lineinfo --ptx \
		-gencode arch=compute_120,code=compute_120 $< -o $@

clean:
	rm -rf .output
