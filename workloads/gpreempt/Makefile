CUDA_HOME ?= /usr/local/cuda-12.9
CUDA_ARCH ?= 120
JOBS ?= 4
CPUSET ?= 8-15
HOST_CC ?= /usr/bin/gcc-13
HOST_CXX ?= /usr/bin/g++-13
GDRCOPY_DIR ?= $(CURDIR)/deps/gdrcopy-2.5.2
FRESH ?=
BRIDGE ?= 0
LOAD_STUDY ?= 0
UPSTREAM := $(abspath deps/upstream)
BUILD := $(abspath build/ninja)
ifeq ($(LOAD_STUDY),1)
BUILD := $(abspath build/load-study)
endif

.PHONY: all prepare configure build build-bridge build-load-study test-load-study test-cpu gdrcopy-driver prepare-tvm configure-tvm build-tvm build-host-flag
all: build

prepare:
	bash ./prepare.sh $(if $(filter 1,$(LOAD_STUDY)),--load-study,$(if $(filter 1,$(BRIDGE)),--bridge,))

configure: prepare
	taskset -c $(CPUSET) env -u MAKEFLAGS -u MFLAGS cmake $(FRESH) -G Ninja -S $(UPSTREAM) -B $(BUILD) \
		-DCMAKE_C_COMPILER=$(HOST_CC) -DCMAKE_CXX_COMPILER=$(HOST_CXX) \
		-DCMAKE_CUDA_HOST_COMPILER=$(HOST_CXX) \
		-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=$(CUDA_HOME)/bin/nvcc \
		-DCUDA_TOOLKIT_ROOT_DIR=$(CUDA_HOME) -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH) \
		-DGPREEMPT_CUDA_ARCH=$(CUDA_ARCH) -DWITH_GFLAGS=OFF -DBUILD_TESTING=OFF \
		-DGPREEMPT_ARTIFACT_DIR=$(CURDIR) -DGPREEMPT_BRIDGE_DIR=$(abspath ../../extension) \
		-DGPREEMPT_LOAD_STUDY=$(if $(filter 1,$(LOAD_STUDY)),ON,OFF)

build: configure
	taskset -c $(CPUSET) env -u MAKEFLAGS -u MFLAGS cmake --build $(BUILD) --target \
		gpreempt baseclient gpreemptclient gpreemptclient_wo block_cubin test-basic -j$(JOBS)

build-bridge:
	$(MAKE) build BRIDGE=1

# Separate output tree: never overwrite the frozen original config-A binaries.
build-load-study:
	$(MAKE) build BRIDGE=1 LOAD_STUDY=1

test-load-study:
	mkdir -p build/load-study
	taskset -c $(CPUSET) $(HOST_CXX) -std=c++17 -O2 -Wall -Wextra -Werror -pthread \
		test_load_study_measurement.cpp -o build/load-study/test-measurement
	taskset -c $(CPUSET) build/load-study/test-measurement
	bash -n prepare.sh
	taskset -c $(CPUSET) python3 -B test_load_study.py
	taskset -c $(CPUSET) python3 -B test_analyze_load_study.py

test-cpu: prepare
	mkdir -p $(BUILD)
	taskset -c $(CPUSET) $(HOST_CXX) -std=c++17 -O2 -DCUDA -I$(UPSTREAM)/include \
		test_transport.cpp $(UPSTREAM)/src/gpreempt.cpp \
		-Wl,--wrap=open,--wrap=ioctl -o $(BUILD)/test-transport
	taskset -c $(CPUSET) $(BUILD)/test-transport
	bash -n prepare.sh
	python3 -B run_smoke.py --help
	python3 -B test_smoke.py
	taskset -c $(CPUSET) $(HOST_CXX) -std=c++17 -O1 test_client_checks.cpp -o $(BUILD)/test-client-checks
	taskset -c $(CPUSET) $(BUILD)/test-client-checks
	taskset -c $(CPUSET) $(HOST_CXX) -std=c++17 -O2 -Wall -Wextra -Werror -Wl,--build-id=none \
		-I$(CUDA_HOME)/include -I$(GDRCOPY_DIR)/include test_flag_transport.cpp -o $(BUILD)/test-flag-transport
	taskset -c $(CPUSET) $(BUILD)/test-flag-transport
	python3 -B test_three_way.py

# Official v2.5.2 (c91ad9f); build only, never install or load.
gdrcopy-driver:
	taskset -c $(CPUSET) $(MAKE) -C $(GDRCOPY_DIR) driver -j$(JOBS) \
		CUDA=$(CUDA_HOME) \
		NVIDIA_SRC_DIR=/home/yunwei37/workspace/gpu/gpu_ext-kernel-575/kernel-open/nvidia \
		NVIDIA_IS_OPENSOURCE=y KVER=6.15.11-061511-generic \
		CC=/usr/bin/gcc-14

prepare-tvm: prepare
	bash ./prepare_tvm.sh

configure-tvm: prepare-tvm
	taskset -c $(CPUSET) env -u MAKEFLAGS -u MFLAGS cmake -G Ninja \
		-S $(abspath deps/tvm) -B $(abspath build/tvm) \
		-DCMAKE_C_COMPILER=$(HOST_CC) -DCMAKE_CXX_COMPILER=$(HOST_CXX) \
		-DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=$(CUDA_HOME) \
		-DUSE_LLVM=/usr/bin/llvm-config-14 -DUSE_LIBBACKTRACE=OFF \
		-DUSE_GTEST=OFF -DUSE_RPC=OFF -DUSE_CCACHE=OFF

# Heavy compiler jobs: run only in a separately coordinated cooldown window.
build-tvm:
	taskset -c $(CPUSET) env -u MAKEFLAGS -u MFLAGS cmake --build $(abspath build/tvm) \
		--target tvm tvm_runtime -j$(JOBS)

# CPU-only build; run --case host_flag later under both experiment leases.
build-host-flag:
	mkdir -p build
	taskset -c $(CPUSET) $(CUDA_HOME)/bin/nvcc -arch=sm_$(CUDA_ARCH) -ccbin $(HOST_CXX) \
		-std=c++17 -O2 test_host_flag.cu -o build/test-host-flag
