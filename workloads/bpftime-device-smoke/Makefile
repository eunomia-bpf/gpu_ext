BPFTIME_R5 ?= ../../../bpftime-r5
CLANG ?= clang
NVCC ?= /usr/local/cuda-12.9/bin/nvcc
LIBBPF := $(BPFTIME_R5)/build-r5-v2/libbpf/libbpf
INCLUDES := -I$(LIBBPF)/include -I$(BPFTIME_R5)/third_party/vmlinux/x86 -I/usr/include/x86_64-linux-gnu

.PHONY: all strict
all: .output/probe .output/probe.bpf.o .output/vector
strict: all .output/probe-negative.bpf.o

.output:
	mkdir -p $@

.output/probe.bpf.o: probe.bpf.c | .output
	$(CLANG) -g -O2 -target bpf -D__TARGET_ARCH_x86 $(INCLUDES) -c $< -o $@

.output/probe-negative.bpf.o: probe.bpf.c | .output
	$(CLANG) -g -O2 -target bpf -D__TARGET_ARCH_x86 -DDEVICE_SMOKE_NEGATIVE_LANE_BRANCH $(INCLUDES) -c $< -o $@

.output/probe: probe.c | .output
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -I$(LIBBPF)/include $< $(LIBBPF)/libbpf.a -lelf -lz -o $@

.output/vector: vector.cu | .output
	$(NVCC) -O2 -cudart shared -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120 $< -o $@ -Xlinker --build-id=none
