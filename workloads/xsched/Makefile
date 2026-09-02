CUDA_HOME ?= /usr/local/cuda-12.9
NVCC := $(CUDA_HOME)/bin/nvcc
CXXFLAGS := -O3 -std=c++17 -lineinfo -gencode arch=compute_120,code=sm_120
CUPTI_LIB := $(CUDA_HOME)/targets/x86_64-linux/lib

# Optional same-frontend HPF policy port. Reuse existing bpftime and XSched
# build archives; neither third-party source tree is modified by these rules.
BPFTIME_ROOT ?= ../../../bpftime
BPFTIME_BUILD ?= $(BPFTIME_ROOT)/build
XSCHED_ROOT := deps/xsched
XSCHED_BUILD := $(XSCHED_ROOT)/build
HPF_INCLUDES := -I$(XSCHED_ROOT)/include -I$(XSCHED_ROOT)/sched/include \
	-I$(XSCHED_ROOT)/utils/include -I$(BPFTIME_ROOT)/vm/vm-core/include \
	-I$(BPFTIME_ROOT)/vm/compat/include
HPF_VM_LIBS := $(BPFTIME_BUILD)/vm/vm-core/libbpftime_vm.a \
	-Wl,--whole-archive $(BPFTIME_BUILD)/vm/compat/ubpf-vm/libbpftime_ubpf_vm.a \
	-Wl,--no-whole-archive $(BPFTIME_BUILD)/vm/compat/ubpf-vm/ubpf/lib/libubpf.a \
	$(BPFTIME_BUILD)/third_party/spdlog/libspdlogd.a
HPF_XSCHED_LIBS := -Wl,--start-group $(XSCHED_BUILD)/3rdparty/ipc/src/libipc.a \
	$(XSCHED_BUILD)/3rdparty/jsoncpp/src/lib_json/libjsoncpp.a \
	$(XSCHED_BUILD)/utils/libutils.a $(XSCHED_BUILD)/sched/libsched.a \
	$(XSCHED_BUILD)/protocol/libprotocol.a $(XSCHED_BUILD)/service/libservice_common.a \
	$(XSCHED_BUILD)/3rdparty/ipc/3rdparty/boost-ipc/container/libboost_container.a \
	-Wl,--end-group -lrt -lpthread -ldl
HPF_WRAP := -Wl,--wrap=_ZN6xsched5sched12CreatePolicyE11XPolicyType

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

build/bpftime_hpf.bpf.o: bpftime_hpf.bpf.c bpftime_hpf.h | build
	clang -O2 -g -target bpf -c $< -o $@

build/bpftime_hpf.bin: build/bpftime_hpf.bpf.o
	llvm-objcopy --only-section=.text -O binary $< $@

build/bpftime_hpf.o: bpftime_hpf.cpp bpftime_hpf.h | build
	$(CXX) -O2 -std=c++17 -Wall -Wextra $(HPF_INCLUDES) -c $< -o $@

build/xserver-bpftime: build/bpftime_hpf.o build/bpftime_hpf.bin
	$(CXX) build/bpftime_hpf.o \
		$(XSCHED_BUILD)/service/CMakeFiles/xserver.dir/server/src/main.cpp.o \
		$(XSCHED_BUILD)/service/CMakeFiles/xserver.dir/server/src/server.cpp.o \
		$(HPF_WRAP) $(HPF_XSCHED_LIBS) $(HPF_VM_LIBS) -lpthread -ldl -o $@

build/bpftime_hpf_test: bpftime_hpf_test.cpp build/bpftime_hpf.o build/bpftime_hpf.bin
	$(CXX) -O2 -std=c++17 -Wall -Wextra $(HPF_INCLUDES) $< build/bpftime_hpf.o \
		$(HPF_WRAP) $(HPF_XSCHED_LIBS) $(HPF_VM_LIBS) -lpthread -ldl -o $@

.PHONY: test-bpftime-hpf
test-bpftime-hpf: build/bpftime_hpf_test
	GPUBPF_HPF_CODE=$(abspath build/bpftime_hpf.bin) ./build/bpftime_hpf_test

clean:
	rm -rf build
