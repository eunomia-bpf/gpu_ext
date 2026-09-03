# CPU control-flow tests; no CUDA/torch and no offloader build.
include Makefile
FINEMOE ?= $(abspath ../../finemoe)
.PHONY: test-adapter-control

build/test_adapter_live: test_adapter_live.cpp adapter_live.inc adapter_state.cpp adapter_state.h policy.h
	taskset -c $(CPUSET) $(CXX) -std=c++17 -O2 -Wall -Wextra -Werror -Wl,--build-id=none -I$(FINEMOE) adapter_state.cpp test_adapter_live.cpp -ldl -pthread -o $@

test-adapter-control: policy build/test_adapter_live
	taskset -c $(CPUSET) ./build/test_adapter_live ./build/libeb_policy.so ./build/eb_policy.bin
