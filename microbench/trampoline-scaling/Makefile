BPFTIME_ROOT ?= ../../../bpftime-table1-575
BPFTIME_BUILD ?= $(BPFTIME_ROOT)/build-table1-575
CUDA_ROOT ?= /usr/local/cuda-12.9
CLANG ?= clang
NVCC ?= $(CUDA_ROOT)/bin/nvcc
LIBBPF_INCLUDE := $(BPFTIME_BUILD)/libbpf
LIBBPF_STATIC := $(BPFTIME_BUILD)/libbpf/libbpf/libbpf.a
VMLINUX_INCLUDE := $(BPFTIME_ROOT)/third_party/vmlinux/x86
BPF_INCLUDES := -I$(LIBBPF_INCLUDE) -I$(VMLINUX_INCLUDE) -I/usr/include/x86_64-linux-gnu -I.

.PHONY: all clean
all: .output/probe .output/probe-noop.bpf.o .output/probe-counter.bpf.o \
     .output/scaling .output/scaling.ptx

.output:
	mkdir -p $@

.output/probe-noop.bpf.o: probe.bpf.c matrix.h | .output
	$(CLANG) -g -O2 -Wall -Werror -target bpf -D__TARGET_ARCH_x86 \
		-DTRAMPOLINE_SCALING_NOOP $(BPF_INCLUDES) -c $< -o $@

.output/probe-counter.bpf.o: probe.bpf.c matrix.h | .output
	$(CLANG) -g -O2 -Wall -Werror -target bpf -D__TARGET_ARCH_x86 \
		$(BPF_INCLUDES) -c $< -o $@

.output/probe: probe.c matrix.h | .output
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -I$(LIBBPF_INCLUDE) -I. \
		$< $(LIBBPF_STATIC) -lelf -lz -o $@

.output/scaling: scaling.cu matrix.h | .output
	$(NVCC) -std=c++17 -O2 -lineinfo -cudart shared \
		-gencode arch=compute_120,code=sm_120 \
		-gencode arch=compute_120,code=compute_120 \
		-Xcompiler=-Wall,-Wextra -Xlinker --build-id=none $< -o $@

.output/scaling.ptx: scaling.cu matrix.h | .output
	$(NVCC) -std=c++17 -O2 -lineinfo --ptx \
		-gencode arch=compute_120,code=compute_120 $< -o $@

clean:
	rm -rf .output

