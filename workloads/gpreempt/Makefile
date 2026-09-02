CUDA_HOME ?= /usr/local/cuda-12.9
CUDA_ARCH ?= 120
JOBS ?= 4
CPUSET ?= 8-15
HOST_CC ?= /usr/bin/gcc-13
HOST_CXX ?= /usr/bin/g++-13
FRESH ?=
UPSTREAM := $(abspath deps/upstream)
BUILD := $(abspath build/ninja)

.PHONY: all prepare configure build test-cpu gdrcopy-driver
all: build

prepare:
	bash ./prepare.sh

configure: prepare
	taskset -c $(CPUSET) env -u MAKEFLAGS -u MFLAGS cmake $(FRESH) -G Ninja -S $(UPSTREAM) -B $(BUILD) \
		-DCMAKE_C_COMPILER=$(HOST_CC) -DCMAKE_CXX_COMPILER=$(HOST_CXX) \
		-DCMAKE_CUDA_HOST_COMPILER=$(HOST_CXX) \
		-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=$(CUDA_HOME)/bin/nvcc \
		-DCUDA_TOOLKIT_ROOT_DIR=$(CUDA_HOME) -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH) \
		-DGPREEMPT_CUDA_ARCH=$(CUDA_ARCH) -DWITH_GFLAGS=OFF -DBUILD_TESTING=OFF

build: configure
	taskset -c $(CPUSET) env -u MAKEFLAGS -u MFLAGS cmake --build $(BUILD) --target \
		gpreempt baseclient gpreemptclient gpreemptclient_wo block_cubin test-basic -j$(JOBS)

test-cpu: prepare
	mkdir -p $(BUILD)
	taskset -c $(CPUSET) $(HOST_CXX) -std=c++17 -O2 -DCUDA -I$(UPSTREAM)/include \
		test_transport.cpp $(UPSTREAM)/src/gpreempt.cpp \
		-Wl,--wrap=open,--wrap=ioctl -o $(BUILD)/test-transport
	taskset -c $(CPUSET) $(BUILD)/test-transport
	bash -n prepare.sh
	python3 -B run_smoke.py --help

# Root-owned dependency checkout, official v2.5 (bda1f60); build only, never load.
gdrcopy-driver:
	taskset -c $(CPUSET) $(MAKE) -C deps/gdrcopy driver -j$(JOBS) \
		CUDA=$(CUDA_HOME) \
		NVIDIA_SRC_DIR=/home/yunwei37/workspace/gpu/gpu_ext-kernel-575/kernel-open/nvidia \
		NVIDIA_IS_OPENSOURCE=y KVER=6.15.11-061511-generic \
		CC=/usr/bin/gcc-14 HAVE_VM_FLAGS_SET=y
