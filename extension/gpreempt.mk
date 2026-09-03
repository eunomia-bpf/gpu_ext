# Standalone GPReempt policy build. Reuses already-built bpftime/libbpf archives;
# deliberately does not rebuild third-party projects or touch the main Makefile.
GP_OUTPUT ?= .output
BPFTIME_ROOT ?= ../../bpftime
BPFTIME_BUILD ?= $(BPFTIME_ROOT)/build
GP_BPFTOOL ?= .output/bpftool/bootstrap/bpftool
GP_DRIVER_ROOT ?= ../../gpu_ext-kernel-575
GP_CUDA_ROOT ?= /usr/local/cuda-12.9
GP_INCLUDES := -I$(BPFTIME_ROOT)/vm/vm-core/include -I$(BPFTIME_ROOT)/vm/compat/include
GP_VM_LIBS := $(BPFTIME_BUILD)/vm/vm-core/libbpftime_vm.a \
	-Wl,--whole-archive $(BPFTIME_BUILD)/vm/compat/ubpf-vm/libbpftime_ubpf_vm.a \
	-Wl,--no-whole-archive $(BPFTIME_BUILD)/vm/compat/ubpf-vm/ubpf/lib/libubpf.a \
	$(BPFTIME_BUILD)/third_party/spdlog/libspdlogd.a -lpthread -ldl

.PHONY: all bridge policy test
all: bridge policy
bridge: $(GP_OUTPUT)/libgpreempt_bridge.so $(GP_OUTPUT)/gpreempt_hint.bin
policy: $(GP_OUTPUT)/gpreempt_policy

$(GP_OUTPUT)/gpreempt_context_smoke: gpreempt_context_smoke.cpp gpreempt_bridge.h $(GP_OUTPUT)/libgpreempt_bridge.so
	$(CXX) -O2 -g -std=c++17 -Wall -Wextra -Wl,--build-id=none \
		-I$(GP_CUDA_ROOT)/include -I$(GP_DRIVER_ROOT)/src/common/sdk/nvidia/inc \
		-I$(GP_DRIVER_ROOT)/kernel-open/common/inc $< \
		-L$(GP_OUTPUT) -Wl,-rpath,'$$ORIGIN' -lgpreempt_bridge \
		-L$(GP_CUDA_ROOT)/lib64/stubs -lcuda -lpthread -o $@

$(GP_OUTPUT)/gpreempt_context_smoke_rpc.bpf.o: gpreempt_context_smoke_rpc.bpf.c gpreempt_context_smoke_rpc.h
	clang -O2 -g -target bpf -D__TARGET_ARCH_x86 -I.output -I../libbpf/include/uapi \
		-I../vmlinux/x86 -c $< -o $@

$(GP_OUTPUT)/gpreempt_context_smoke_rpc.skel.h: $(GP_OUTPUT)/gpreempt_context_smoke_rpc.bpf.o
	$(GP_BPFTOOL) gen skeleton $< > $@

$(GP_OUTPUT)/gpreempt_context_smoke_rpc: gpreempt_context_smoke_rpc.c gpreempt_context_smoke_rpc.h $(GP_OUTPUT)/gpreempt_context_smoke_rpc.skel.h
	$(CC) -O2 -g -Wall -Wextra -I$(GP_OUTPUT) -Wl,--build-id=none $< .output/libbpf.a -lelf -lz -o $@

.PHONY: context-smoke
context-smoke: $(GP_OUTPUT)/gpreempt_context_smoke $(GP_OUTPUT)/gpreempt_context_smoke_rpc

$(GP_OUTPUT):
	mkdir -p $@

$(GP_OUTPUT)/gpreempt_hint.bpf.o: gpreempt_hint.bpf.c gpreempt_bridge.h | $(GP_OUTPUT)
	clang -O2 -g -target bpf -c $< -o $@

$(GP_OUTPUT)/gpreempt_hint.bin: $(GP_OUTPUT)/gpreempt_hint.bpf.o
	llvm-objcopy --only-section=.text -O binary $< $@

$(GP_OUTPUT)/libgpreempt_bridge.so: gpreempt_bridge.cpp gpreempt_bridge.h | $(GP_OUTPUT)
	$(CXX) -O2 -g -fPIC -shared -std=c++17 -Wall -Wextra -Wl,--build-id=none \
		$(GP_INCLUDES) $< $(GP_VM_LIBS) -o $@

$(GP_OUTPUT)/gpreempt_policy.bpf.o: gpreempt_policy.bpf.c gpreempt_bridge.h gpu_sched_set_timeslices.h | $(GP_OUTPUT)
	clang -O2 -g -target bpf -D__TARGET_ARCH_x86 -I.output -I../libbpf/include/uapi \
		-I../vmlinux/x86 -c $< -o $@

$(GP_OUTPUT)/gpreempt_policy.skel.h: $(GP_OUTPUT)/gpreempt_policy.bpf.o
	$(GP_BPFTOOL) gen skeleton $< > $@

$(GP_OUTPUT)/gpreempt_policy: gpreempt_policy.c gpreempt_bridge.h $(GP_OUTPUT)/gpreempt_policy.skel.h
	$(CC) -O2 -g -Wall -Wextra -I$(GP_OUTPUT) -Wl,--build-id=none $< \
		.output/libbpf.a -lelf -lz -o $@

$(GP_OUTPUT)/gpreempt_policy_test: gpreempt_policy_test.cpp gpreempt_bridge.h $(GP_OUTPUT)/libgpreempt_bridge.so
	$(CXX) -O2 -std=c++17 -Wall -Wextra -Wl,--build-id=none $< \
		-L$(GP_OUTPUT) -Wl,-rpath,'$$ORIGIN' -lgpreempt_bridge -o $@

$(GP_OUTPUT)/gpreempt_policy_cpu_test: gpreempt_policy_cpu_test.c gpreempt_policy.bpf.c gpreempt_bridge.h gpu_sched_set_timeslices.h
	$(CC) -O2 -g -Wall -Wextra -Wno-unused-parameter -Wl,--build-id=none $< -o $@

test: $(GP_OUTPUT)/gpreempt_policy_test $(GP_OUTPUT)/gpreempt_hint.bin $(GP_OUTPUT)/gpreempt_policy_cpu_test
	./$(GP_OUTPUT)/gpreempt_policy_cpu_test
	GPREEMPT_POLICY=original ./$(GP_OUTPUT)/gpreempt_policy_test
	GPREEMPT_POLICY=bpf GPREEMPT_HINT_CODE=$(abspath $(GP_OUTPUT)/gpreempt_hint.bin) ./$(GP_OUTPUT)/gpreempt_policy_test
