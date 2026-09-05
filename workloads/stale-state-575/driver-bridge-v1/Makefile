CC ?= cc
CLANG ?= /usr/bin/clang-18
ROOT := ../../..
VMLINUX_DIR := $(ROOT)/vmlinux/x86
BPF_INCLUDE := $(ROOT)/extension/.output
UAPI_INCLUDE := $(ROOT)/libbpf/include/uapi
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Werror
BPF_CFLAGS := -g -O2 -target bpf -D__TARGET_ARCH_x86 -Wall -Wextra -Werror \
	-I$(VMLINUX_DIR) -I$(BPF_INCLUDE) -I$(UAPI_INCLUDE)

.PHONY: all test clean

all: abi_test stale_state_v1.bpf.o live_loader

abi_test: abi_test.c abi.h ../stale_state_policy_model.h
	$(CC) $(CFLAGS) $< -o $@

stale_state_v1.bpf.o: stale_state_v1.bpf.c abi.h ../stale_state_policy_model.h
	$(CLANG) $(BPF_CFLAGS) -c $< -o $@

live.tmp.bpf.o: live.bpf.c abi.h observer.h ../stale_state_policy_model.h
	$(CLANG) $(BPF_CFLAGS) -c $< -o $@

live.bpf.o: live.tmp.bpf.o $(ROOT)/extension/.output/bpftool/bootstrap/bpftool
	$(ROOT)/extension/.output/bpftool/bootstrap/bpftool gen object $@ $<

live.skel.h: live.bpf.o
	$(ROOT)/extension/.output/bpftool/bootstrap/bpftool gen skeleton $< > $@

live_loader: live_loader.c observer.h live.skel.h $(ROOT)/extension/.output/libbpf.a
	$(CC) -O2 -g -std=c11 -Wall -Wextra -Werror \
		-I$(BPF_INCLUDE) -I$(UAPI_INCLUDE) -I. $< \
		$(ROOT)/extension/.output/libbpf.a -lelf -lz \
		-Wl,--build-id=none -o $@

test: all
	./abi_test
	/usr/bin/llvm-objdump-18 -h stale_state_v1.bpf.o | grep -q 'struct_ops/gpu_stale_state_prefetch_v1'
	/usr/bin/llvm-objdump-18 -h stale_state_v1.bpf.o | grep -q '[.]struct_ops'
	/usr/bin/llvm-objdump-18 -h live.bpf.o | grep -q 'fentry/uvm_stale_state_v1_diagnostic'
	/usr/bin/llvm-objdump-18 -h live.bpf.o | grep -q 'struct_ops/gpu_stale_state_prefetch_v1'
	/usr/bin/llvm-objdump-18 -h live.bpf.o | grep -q '[.]struct_ops[.]link'

clean:
	rm -f abi_test stale_state_v1.bpf.o live.tmp.bpf.o live.bpf.o live.skel.h live_loader
