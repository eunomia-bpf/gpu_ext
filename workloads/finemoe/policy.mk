# Standalone CPU-only selector build; no CUDA or shared environment changes.
BPFTIME ?= $(abspath ../../../bpftime)
UBPF_LIB ?= $(BPFTIME)/build/vm/compat/ubpf-vm/ubpf/lib/libubpf.a
UBPF_INCLUDE ?= $(BPFTIME)/third_party/ubpf/vm/inc
UBPF_CONFIG ?= $(BPFTIME)/build/vm/compat/ubpf-vm/ubpf/vm
CC := /usr/bin/gcc-13
CXX := /usr/bin/g++-13
BPF_CLANG ?= clang
OBJCOPY ?= llvm-objcopy
CPUSET ?= 12

.PHONY: policy test-policy
policy: build/libfinemoe_policy.so build/finemoe_policy.bin

build:
	mkdir -p $@

build/finemoe_policy.o: finemoe_policy.c finemoe_policy.h | build
	taskset -c $(CPUSET) $(CC) -std=c11 -O2 -fPIC -fno-fast-math -ffp-contract=off -Wall -Wextra -Werror -c $< -o $@

build/libfinemoe_policy.so: finemoe_policy_bridge.cpp finemoe_policy.h build/finemoe_policy.o
	taskset -c $(CPUSET) $(CXX) -std=c++17 -O2 -fPIC -shared -Wl,--build-id=none -Wall -Wextra -Werror -I$(UBPF_INCLUDE) -I$(UBPF_CONFIG) $< build/finemoe_policy.o $(UBPF_LIB) -o $@

build/finemoe_policy.bpf.o: finemoe_policy.bpf.c finemoe_policy.h | build
	taskset -c $(CPUSET) $(BPF_CLANG) -O2 -target bpf -mcpu=v3 -fno-builtin -Wall -Wextra -Werror -c $< -o $@

build/finemoe_policy.bin: build/finemoe_policy.bpf.o
	$(OBJCOPY) --dump-section .text=$@ $<

test-policy: policy
	taskset -c $(CPUSET) /usr/bin/python3 -B test_finemoe_policy.py
