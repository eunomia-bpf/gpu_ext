# Small, CPU-only targets; never configure or rebuild bpftime/the offloader.
BPFTIME ?= $(abspath ../../../../bpftime)
UBPF_LIB ?= $(BPFTIME)/build/vm/compat/ubpf-vm/ubpf/lib/libubpf.a
UBPF_INCLUDE ?= $(BPFTIME)/third_party/ubpf/vm/inc
UBPF_CONFIG ?= $(BPFTIME)/build/vm/compat/ubpf-vm/ubpf/vm
CC := /usr/bin/gcc-13
CXX := /usr/bin/g++-13
BPF_CLANG ?= clang
OBJCOPY ?= llvm-objcopy
CPUSET ?= 17

.PHONY: policy test
policy: build/libeb_policy.so build/eb_policy.bin

build:
	mkdir -p $@

build/policy.o: policy.c policy.h | build
	taskset -c $(CPUSET) $(CC) -std=c11 -O2 -fPIC -Wall -Wextra -Werror -c $< -o $@

build/libeb_policy.so: jit_bridge.cpp policy.h build/policy.o
	taskset -c $(CPUSET) $(CXX) -std=c++17 -O2 -fPIC -shared -Wl,--build-id=none -Wall -Wextra -Werror -I$(UBPF_INCLUDE) -I$(UBPF_CONFIG) $< build/policy.o $(UBPF_LIB) -o $@

build/policy.bpf.o: policy.c policy.h | build
	taskset -c $(CPUSET) $(BPF_CLANG) -O2 -target bpf -mcpu=v3 -fno-builtin -Wall -Wextra -Werror -c $< -o $@

build/eb_policy.bin: build/policy.bpf.o
	$(OBJCOPY) --dump-section .text=$@ $<

test: policy
	taskset -c $(CPUSET) /usr/bin/python3 -B test_policy.py
