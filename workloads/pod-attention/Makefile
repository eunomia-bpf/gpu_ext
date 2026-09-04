SHELL := /bin/bash
BPFTIME ?= $(abspath ../../../bpftime)
BPFTIME_BUILD ?= $(BPFTIME)/build-cuda-pr503
PTX_LIBDIR := $(BPFTIME_BUILD)/attach/nv_attach_impl/pass/ptxpass_kprobe_entry
PTX_INCLUDES := -I$(BPFTIME)/attach/nv_attach_impl/pass/ptxpass_core/include -I$(BPFTIME)/third_party
CXX ?= g++
CLANG ?= clang-15
OBJCOPY ?= llvm-objcopy-15
CPUSET ?= 12-15
PYTHON ?= python3
TORCH_PYTHON ?= $(abspath ../moe-infinity/.venv/bin/python)
CUDA_HOME ?= /usr/local/cuda-12.9
POD_ROOT := $(CURDIR)
PTX_FLAGS := -std=c++20 -O2 -fPIC $(PTX_INCLUDES) -L$(PTX_LIBDIR) -lptxpass_kprobe_entry -Wl,-rpath,$(PTX_LIBDIR)

.PHONY: cpu test prepare build test-cuda-abi
cpu: build/selector.bin build/pod-ptx-compile build/libpod_ptx_adapter.so build/test-selector build/pod-loader build/libpod_launch_bridge.so build/test-launch-bridge

prepare:
	bash prepare.sh

build: cpu prepare
	cd deps/vattention/pod_attn && env CUDA_VISIBLE_DEVICES= CCACHE_DISABLE=1 CUDA_HOME=$(CUDA_HOME) \
	  TORCH_CUDA_ARCH_LIST='12.0+PTX' MAX_JOBS=2 NVCC_THREADS=1 \
	  POD_EXPERIMENT_DIR=$(POD_ROOT) $(TORCH_PYTHON) $(POD_ROOT)/host_link.py build_ext \
	  --build-lib $(POD_ROOT)/build/python --build-temp $(POD_ROOT)/build/torch

build/.dir:
	mkdir -p build
	touch $@

build/selector.bpf.o: selector.bpf.c selector_policy.h selector_abi.h | build/.dir
	$(CLANG) -O2 -g -target bpf -mcpu=v3 -c $< -o $@

build/selector.bin: build/selector.bpf.o
	$(OBJCOPY) --dump-section pod_selector=$@ $<

build/pod-ptx-compile: ptx_adapter.cpp | build/.dir
	$(CXX) $< -DPOD_PTX_ADAPTER_CLI -o $@ $(PTX_FLAGS)

build/libpod_ptx_adapter.so: ptx_adapter.cpp | build/.dir
	$(CXX) $< -shared -o $@ $(PTX_FLAGS)

build/test-selector: test_selector.cpp selector_abi.h selector_policy.h | build/.dir
	$(CXX) -std=c++17 -O2 -Wall -Wextra $< -o $@

build/pod-loader: loader.cpp | build/.dir
	$(CXX) -std=c++17 -O2 -Wall -Wextra $< -o $@ -lbpf

build/libpod_launch_bridge.so: launch_bridge.cpp launch_bridge.h | build/.dir
	$(CXX) -std=c++17 -O2 -Wall -Wextra -fPIC -shared -I$(CUDA_HOME)/include $< -o $@ \
	  -Wl,--no-as-needed -L$(CUDA_HOME)/lib64 -lcudart -lcuda -Wl,--as-needed \
	  -Wl,-rpath,$(CUDA_HOME)/lib64 -ldl -pthread

build/test-launch-bridge: test_launch_bridge.cpp launch_bridge.cpp launch_bridge.h | build/.dir
	$(CXX) -std=c++17 -O2 -Wall -Wextra -I$(CUDA_HOME)/include $< -o $@ -ldl -pthread

build/selector-abi.ptx: test_selector_abi.cu selector_cuda.cuh selector_policy.h selector_abi.h | build/.dir
	$(CUDA_HOME)/bin/nvcc -O3 -std=c++17 --keep-device-functions -arch=sm_120 --ptx $< -o $@

test-cuda-abi: build/selector-abi.ptx
	$(PYTHON) -c 'import pathlib,re; p=pathlib.Path("build/selector-abi.ptx").read_text(); assert re.search(r"\.func pod_device_selector\(\s*\.param \.b64 pod_device_selector_param_0,\s*\.param \.b64 pod_device_selector_param_1",p); assert re.search(r"call(?:\.uni)?\s+pod_device_selector(?:\x24[0-9]+)?,\s*\(\s*\w+,\s*\w+\s*\)",p); print("PASS: NVCC preserves real two-argument device ABI (compile only)")'

test: cpu
	taskset -c $(CPUSET) build/test-selector
	taskset -c $(CPUSET) build/test-launch-bridge
	taskset -c $(CPUSET) $(PYTHON) test_ptx_adapter.py
	taskset -c $(CPUSET) $(PYTHON) test_bench.py
	taskset -c $(CPUSET) $(PYTHON) -m unittest test_ptx_prune test_ptx_partition test_prepare_ptx test_host_link test_run_study test_phase_study
