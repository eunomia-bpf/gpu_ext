CC ?= cc
CXX := /usr/bin/g++-13
NVCC ?= nvcc
CFLAGS ?= -O2 -g -Wall -Wextra -Werror
NVCCFLAGS ?= -O3 -std=c++17 -Xlinker --build-id=none
BPFTIME_ROOT ?= $(abspath ../../../bpftime)
BPFTIME_BUILD ?= $(BPFTIME_ROOT)/build
UBPF_INCLUDE ?= $(BPFTIME_ROOT)/third_party/ubpf/vm/inc
UBPF_CONFIG ?= $(BPFTIME_BUILD)/vm/compat/ubpf-vm/ubpf/vm
UBPF_LIB ?= $(BPFTIME_BUILD)/vm/compat/ubpf-vm/ubpf/lib/libubpf.a
BPF_CLANG ?= /usr/bin/clang-18
OBJCOPY ?= /usr/bin/llvm-objcopy-18

.PHONY: test-offline test-driver-bridge build-sources clean

test_policy_model: test_policy_model.c stale_state_policy_model.h
	$(CC) $(CFLAGS) -std=c11 test_policy_model.c -o $@

test_json_precision: test_json_precision.cpp
	$(CXX) -O2 -std=c++17 -Wall -Wextra -Werror -Wl,--build-id=none $< -o $@

build:
	mkdir -p $@

build/stale_state_policy.bpf.o: stale_state_policy.bpf.c stale_state_policy_jit.h stale_state_policy_model.h | build
	$(BPF_CLANG) -O2 -target bpf -mcpu=v3 -fno-builtin -Wall -Wextra -Werror -c $< -o $@

build/stale_state_policy.bin: build/stale_state_policy.bpf.o
	$(OBJCOPY) --dump-section .text=$@ $<

build/test_policy_jit: test_policy_jit.cpp stale_state_policy_jit.cpp stale_state_policy_jit.h stale_state_policy_model.h build/stale_state_policy.bin
	$(CXX) -O2 -std=c++17 -Wall -Wextra -Werror -Wl,--build-id=none \
		-I$(UBPF_INCLUDE) -I$(UBPF_CONFIG) test_policy_jit.cpp stale_state_policy_jit.cpp \
		$(UBPF_LIB) -o $@

uvm_event_monitor: uvm_event_monitor.c
	$(CC) $(CFLAGS) -std=c11 -Wl,--build-id=none uvm_event_monitor.c -o $@

stale_state_workload: stale_state_workload.cu
	$(NVCC) $(NVCCFLAGS) stale_state_workload.cu -o $@

test-driver-bridge:
	$(MAKE) -C driver-bridge-v1 test

test-offline: test_policy_model test_json_precision uvm_event_monitor build/test_policy_jit test-driver-bridge
	./test_policy_model
	./test_json_precision
	./build/test_policy_jit ./build/stale_state_policy.bin
	PYTHONDONTWRITEBYTECODE=1 python3 -B test_offline.py

build-sources: stale_state_workload uvm_event_monitor test_policy_model

clean:
	rm -f stale_state_workload uvm_event_monitor test_policy_model test_json_precision
	rm -rf build
	$(MAKE) -C driver-bridge-v1 clean
