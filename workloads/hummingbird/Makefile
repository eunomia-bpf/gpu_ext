HB_CXX := /usr/bin/g++-13
HB_CC := /usr/bin/gcc-13
HB_CLANG ?= /usr/bin/clang
HB_OBJCOPY ?= /usr/bin/llvm-objcopy
HB_CPUSET ?= 4-7
HB_OUTPUT := $(CURDIR)/build
HB_BPFTIME := $(abspath ../../../bpftime)
HB_VM_BUILD := $(HB_BPFTIME)/build
HB_INCLUDES := -I$(HB_BPFTIME)/vm/vm-core/include -I$(HB_BPFTIME)/vm/compat/include
HB_VM_LIBS := $(HB_VM_BUILD)/vm/vm-core/libbpftime_vm.a \
	-Wl,--whole-archive $(HB_VM_BUILD)/vm/compat/ubpf-vm/libbpftime_ubpf_vm.a \
	-Wl,--no-whole-archive $(HB_VM_BUILD)/vm/compat/ubpf-vm/ubpf/lib/libubpf.a \
	$(HB_VM_BUILD)/third_party/spdlog/libspdlogd.a -lpthread -ldl
HB_UPSTREAM := $(abspath ../gpreempt/deps/upstream)
HB_GP_BUILD := $(abspath ../gpreempt/build/load-study)
HB_GP_ARTIFACT := $(abspath ../gpreempt)
HB_EXTENSION := $(abspath ../../extension)
HB_CUDA := /usr/local/cuda-12.9
HB_DISB_ARCHIVE ?= $(HB_OUTPUT)/disb-trace/libdisb.a
HB_CUDA_INCLUDES := -I$(HB_CUDA)/include -include cuda_runtime.h \
	-I$(HB_OUTPUT)/disb-src/include -I$(CURDIR) -I$(HB_UPSTREAM)/include \
	-I$(HB_UPSTREAM)/third_party/jsoncpp/include -I$(HB_GP_BUILD)/third_party/glog \
	-I$(HB_UPSTREAM)/third_party/glog/src -I$(HB_GP_ARTIFACT) -I$(HB_EXTENSION)
HB_DEFINES := -DCUDA -DGPREEMPT_LOAD_STUDY -DGLOG_USE_GLOG_EXPORT \
	-DMODEL_PATH='"$(HB_UPSTREAM)/model"' -DBUILD_PATH='"$(HB_GP_BUILD)"'
HB_RPATH := $(HB_GP_BUILD):$(HB_GP_BUILD)/third_party/glog:$(HB_GP_BUILD)/third_party/jsoncpp/src/lib_json:$(HB_EXTENSION)/.output:$(HB_CUDA)/lib64
HB_GPU_LIBS := -L$(HB_GP_BUILD) -lexecutor -lworkloads -lgpreempt \
	-L$(HB_GP_BUILD)/third_party/glog -lglog \
	-L$(HB_GP_BUILD)/third_party/jsoncpp/src/lib_json -ljsoncpp \
	-L$(HB_CUDA)/lib64 -lcuda -lcudart -lpthread -ldl \
	-Wl,--build-id=none -Wl,-rpath,$(HB_RPATH)
HB_CPPFLAGS := -std=c++17 -O2 -Wall -Wextra $(HB_INCLUDES) $(HB_CUDA_INCLUDES) $(HB_DEFINES)

.PHONY: test-cpu check-model clients profile-client split-cubin
test-cpu: $(HB_OUTPUT)/test_idle_policy $(HB_OUTPUT)/test_split_grid $(HB_OUTPUT)/test_profile $(HB_OUTPUT)/idle_policy.bin
	taskset -c $(HB_CPUSET) /usr/bin/python3 -B -m unittest test_split_model -v
	taskset -c $(HB_CPUSET) $(HB_OUTPUT)/test_idle_policy $(HB_OUTPUT)/idle_policy.bin
	taskset -c $(HB_CPUSET) $(HB_OUTPUT)/test_split_grid
	taskset -c $(HB_CPUSET) $(HB_OUTPUT)/test_profile $(HB_OUTPUT)/idle_policy.bin

clients: $(HB_OUTPUT)/hummingbird_client $(HB_OUTPUT)/fixed_client $(HB_OUTPUT)/idle_policy.bin
profile-client: $(HB_OUTPUT)/hummingbird_profile
split-cubin: $(HB_OUTPUT)/resnet152-split/mod.cubin

check-model:
	taskset -c $(HB_CPUSET) /usr/bin/python3 -B split_model.py \
		--source ../gpreempt/deps/upstream/model/resnet152/mod.cu \
		--host ../gpreempt/deps/upstream/model/resnet152/host.json --check

$(HB_OUTPUT):
	mkdir -p $@

$(HB_OUTPUT)/idle_policy.o: idle_policy.c idle_policy.h | $(HB_OUTPUT)
	taskset -c $(HB_CPUSET) $(HB_CC) -std=c11 -O2 -Wall -Wextra -Werror -c $< -o $@

$(HB_OUTPUT)/idle_policy.bpf.o: idle_policy.bpf.c idle_policy.c idle_policy.h | $(HB_OUTPUT)
	taskset -c $(HB_CPUSET) $(HB_CLANG) -O2 -target bpf -c $< -o $@

$(HB_OUTPUT)/idle_policy.bin: $(HB_OUTPUT)/idle_policy.bpf.o
	$(HB_OBJCOPY) --only-section=.text -O binary $< $@

$(HB_OUTPUT)/test_idle_policy: test_idle_policy.cpp idle_policy.h $(HB_OUTPUT)/idle_policy.o
	taskset -c $(HB_CPUSET) $(HB_CXX) -std=c++17 -O2 -Wall -Wextra -Werror \
		-Wl,--build-id=none $(HB_INCLUDES) $< $(HB_OUTPUT)/idle_policy.o $(HB_VM_LIBS) -o $@

$(HB_OUTPUT)/test_split_grid: test_split_grid.cpp split_grid.h | $(HB_OUTPUT)
	taskset -c $(HB_CPUSET) $(HB_CXX) -std=c++17 -O2 -Wall -Wextra -Werror \
		-Wl,--build-id=none $< -o $@

$(HB_OUTPUT)/test_profile: test_profile.cpp idle_executor.h $(HB_OUTPUT)/idle_executor.o $(HB_OUTPUT)/idle_policy.o
	taskset -c $(HB_CPUSET) $(HB_CXX) $(HB_CPPFLAGS) $< $(HB_OUTPUT)/idle_executor.o \
		$(HB_OUTPUT)/idle_policy.o $(HB_VM_LIBS) $(HB_GPU_LIBS) -o $@

$(HB_OUTPUT)/idle_executor.o: idle_executor.cpp idle_executor.h idle_policy.h split_grid.h Makefile | $(HB_OUTPUT)
	taskset -c $(HB_CPUSET) $(HB_CXX) $(HB_CPPFLAGS) -c $< -o $@

$(HB_OUTPUT)/hummingbird_client.o: hummingbird_client.cpp idle_executor.h $(HB_OUTPUT)/disb-src/include/benchmark.h Makefile | $(HB_OUTPUT)
	taskset -c $(HB_CPUSET) $(HB_CXX) $(HB_CPPFLAGS) -c $< -o $@

$(HB_OUTPUT)/hummingbird_profile.o: hummingbird_profile.cpp idle_executor.h Makefile | $(HB_OUTPUT)
	taskset -c $(HB_CPUSET) $(HB_CXX) $(HB_CPPFLAGS) -c $< -o $@

$(HB_OUTPUT)/hummingbird_client: $(HB_OUTPUT)/hummingbird_client.o $(HB_OUTPUT)/idle_executor.o $(HB_OUTPUT)/idle_policy.o $(HB_DISB_ARCHIVE)
	taskset -c $(HB_CPUSET) $(HB_CXX) $^ $(HB_VM_LIBS) $(HB_GPU_LIBS) -Wl,-Map,$@.map -o $@

$(HB_OUTPUT)/hummingbird_profile: $(HB_OUTPUT)/hummingbird_profile.o $(HB_OUTPUT)/idle_executor.o $(HB_OUTPUT)/idle_policy.o
	taskset -c $(HB_CPUSET) $(HB_CXX) $^ $(HB_VM_LIBS) $(HB_GPU_LIBS) -o $@

$(HB_OUTPUT)/fixed_client: $(HB_UPSTREAM)/src/cuda-clients/gpreemptclient.cpp $(HB_DISB_ARCHIVE) Makefile $(HB_OUTPUT)/disb-src/include/benchmark.h
	taskset -c $(HB_CPUSET) $(HB_CXX) $(HB_CPPFLAGS) $< $(HB_DISB_ARCHIVE) \
		-L$(HB_EXTENSION)/.output -lgpreempt_bridge -lgdrapi $(HB_GPU_LIBS) -Wl,-Map,$@.map -o $@

$(HB_OUTPUT)/resnet152-split/mod.cu: split_model.py $(HB_UPSTREAM)/model/resnet152/mod.cu $(HB_UPSTREAM)/model/resnet152/host.json | $(HB_OUTPUT)
	taskset -c $(HB_CPUSET) /usr/bin/python3 -B split_model.py \
		--source $(HB_UPSTREAM)/model/resnet152/mod.cu --host $(HB_UPSTREAM)/model/resnet152/host.json \
		--output $(HB_OUTPUT)/resnet152-split

$(HB_OUTPUT)/resnet152-split/mod.cubin: $(HB_OUTPUT)/resnet152-split/mod.cu
	taskset -c $(HB_CPUSET) $(HB_CUDA)/bin/nvcc -ccbin $(HB_CXX) -O3 -arch=sm_120 -cubin $< -o $@
