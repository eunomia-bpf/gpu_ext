CUDA_HOME ?= /usr/local/cuda-12.9
NVCC := $(CUDA_HOME)/bin/nvcc
CXXFLAGS := -O3 -std=c++17 -lineinfo -gencode arch=compute_120,code=sm_120
CUPTI_LIB := $(CUDA_HOME)/targets/x86_64-linux/lib

.PHONY: all clean

all: build/priority_workload build/bench_lc build/bench_be

build:
	mkdir -p $@

build/priority_workload: priority_workload.cu | build
	$(NVCC) $(CXXFLAGS) -I$(CUDA_HOME)/targets/x86_64-linux/include $< \
		-L$(CUPTI_LIB) -Xlinker=-rpath -Xlinker=$(CUPTI_LIB) -lcupti -o $@

build/bench_lc: build/priority_workload
	ln -f $< $@

build/bench_be: build/priority_workload
	ln -f $< $@

clean:
	rm -rf build
