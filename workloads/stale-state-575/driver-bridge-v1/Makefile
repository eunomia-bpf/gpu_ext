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

all: abi_test stale_state_v1.bpf.o

abi_test: abi_test.c abi.h ../stale_state_policy_model.h
	$(CC) $(CFLAGS) $< -o $@

stale_state_v1.bpf.o: stale_state_v1.bpf.c abi.h ../stale_state_policy_model.h
	$(CLANG) $(BPF_CFLAGS) -c $< -o $@

test: all
	./abi_test
	/usr/bin/llvm-objdump-18 -h stale_state_v1.bpf.o | grep -q 'struct_ops/gpu_stale_state_prefetch_v1'
	/usr/bin/llvm-objdump-18 -h stale_state_v1.bpf.o | grep -q '[.]struct_ops'

clean:
	rm -f abi_test stale_state_v1.bpf.o
