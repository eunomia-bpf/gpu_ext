# Builds the GDS storage-policy BPF object, the libbpf struct_ops loader,
# and the nvidia_uvm gpu-storage-decide ioctl probe.

CLANG ?= clang
CC    ?= gcc
CUDA_HOME ?= /usr/local/cuda-12.9
NVCC  ?= $(CUDA_HOME)/bin/nvcc
RM    ?= rm -f

CFLAGS     ?= -O2 -g -Wall -Wextra
NVCCFLAGS  ?= -O2 -std=c++17 -Wno-deprecated-gpu-targets \
	      -Xcompiler=-Wall,-Wextra
BPF_CFLAGS := -g -O2 -target bpf -D__TARGET_ARCH_x86 \
	      -I../../../vmlinux/x86 -I../../../libbpf/src

VMLINUX_DIR := ../../../vmlinux/x86
LIBBPF_DIR  := ../../../libbpf/src
LIBBPF_A    := $(LIBBPF_DIR)/libbpf.a
CUDA_TARGET := $(CUDA_HOME)/targets/x86_64-linux

all: gds_policy.bpf.o gds_policy ioctl_probe gds_executor \
     kv_reclaim_policy.bpf.o kv_reclaim_native.so kv_reclaim_loader \
     kv_reclaim_total_cost.bpf.o kv_reclaim_total_cost_native.so

# Tree libbpf (built in place; no libbpf/bootstrap bpftool needed).
$(LIBBPF_A): $(wildcard $(LIBBPF_DIR)/*.[ch]) $(wildcard $(LIBBPF_DIR)/include/*.h)
	$(MAKE) -C $(LIBBPF_DIR)

gds_policy.bpf.o: gds_policy.bpf.c $(VMLINUX_DIR)/vmlinux.h
	$(CLANG) $(BPF_CFLAGS) -c $< -o $@

gds_policy: gds_policy.c $(LIBBPF_A)
	$(CC) $(CFLAGS) -I$(LIBBPF_DIR) -o $@ $< $(LIBBPF_A) -lelf -lz

ioctl_probe: ioctl_probe.c
	$(CC) $(CFLAGS) -o $@ $<

gds_executor: gds_executor.cu
	$(NVCC) $(NVCCFLAGS) -I$(CUDA_TARGET)/include -o $@ $< \
		-L$(CUDA_TARGET)/lib -Xlinker=-rpath \
		-Xlinker=$(CUDA_TARGET)/lib -lcufile

kv_reclaim_policy.bpf.o: kv_reclaim_policy.bpf.c kv_reclaim_abi.h $(VMLINUX_DIR)/vmlinux.h
	$(CLANG) $(BPF_CFLAGS) -c $< -o $@

kv_reclaim_native.so: kv_reclaim_native.c kv_reclaim_abi.h
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $<

kv_reclaim_loader: kv_reclaim_loader.c $(LIBBPF_A)
	$(CC) $(CFLAGS) -I$(LIBBPF_DIR) -o $@ $< $(LIBBPF_A) -lelf -lz

# Total-cost policy ablation: the same existing sources, with the shared
# ranking selector defined. Default per-byte targets above are untouched.
kv_reclaim_total_cost.bpf.o: kv_reclaim_policy.bpf.c kv_reclaim_abi.h \
	$(VMLINUX_DIR)/vmlinux.h
	$(CLANG) $(BPF_CFLAGS) -DUVM_KV_RECLAIM_POLICY_TOTAL_COST \
		-c $< -o $@

kv_reclaim_total_cost_native.so: kv_reclaim_native.c kv_reclaim_abi.h
	$(CC) $(CFLAGS) -fPIC -shared -DUVM_KV_RECLAIM_POLICY_TOTAL_COST \
		-o $@ $<

clean:
	$(RM) gds_policy.bpf.o gds_policy ioctl_probe gds_executor \
	     kv_reclaim_policy.bpf.o kv_reclaim_native.so kv_reclaim_loader \
	     kv_reclaim_total_cost.bpf.o kv_reclaim_total_cost_native.so

.PHONY: all clean
