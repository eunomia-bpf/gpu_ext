# CPU-only bpftime/ubpf JIT build; reuse existing VM archives, never rebuild deps.
MEP_OUTPUT ?= .output
MEP_CXX ?= /usr/bin/g++-13
MEP_CLANG ?= clang
BPFTIME_ROOT ?= ../../bpftime
BPFTIME_BUILD ?= $(BPFTIME_ROOT)/build
MEP_INCLUDES := -I$(BPFTIME_ROOT)/vm/vm-core/include -I$(BPFTIME_ROOT)/vm/compat/include
MEP_VM_LIBS := $(BPFTIME_BUILD)/vm/vm-core/libbpftime_vm.a \
	-Wl,--whole-archive $(BPFTIME_BUILD)/vm/compat/ubpf-vm/libbpftime_ubpf_vm.a \
	-Wl,--no-whole-archive $(BPFTIME_BUILD)/vm/compat/ubpf-vm/ubpf/lib/libubpf.a \
	$(BPFTIME_BUILD)/third_party/spdlog/libspdlogd.a -lpthread -ldl

.PHONY: all test
all: $(MEP_OUTPUT)/libmoe_expert_policy.so $(MEP_OUTPUT)/moe_expert_policy.bin

$(MEP_OUTPUT):
	mkdir -p $@

$(MEP_OUTPUT)/moe_expert_policy.bpf.o: moe_expert_policy.bpf.c moe_expert_policy.h | $(MEP_OUTPUT)
	$(MEP_CLANG) -O2 -g -target bpf -c $< -o $@

$(MEP_OUTPUT)/moe_expert_policy.bin: $(MEP_OUTPUT)/moe_expert_policy.bpf.o
	llvm-objcopy --only-section=.text -O binary $< $@

$(MEP_OUTPUT)/libmoe_expert_policy.so: moe_expert_policy.cpp moe_expert_policy.h | $(MEP_OUTPUT)
	$(MEP_CXX) -O2 -g -fPIC -shared -std=c++17 -Wall -Wextra -Werror -Wl,--build-id=none \
		$(MEP_INCLUDES) $< $(MEP_VM_LIBS) -o $@

$(MEP_OUTPUT)/moe_expert_policy_test: moe_expert_policy_test.cpp moe_expert_policy.h $(MEP_OUTPUT)/libmoe_expert_policy.so
	$(MEP_CXX) -O2 -g -std=c++17 -Wall -Wextra -Werror -Wl,--build-id=none $< \
		-L$(MEP_OUTPUT) -Wl,-rpath,'$$ORIGIN' -lmoe_expert_policy -lpthread -o $@

test: all $(MEP_OUTPUT)/moe_expert_policy_test
	./$(MEP_OUTPUT)/moe_expert_policy_test $(abspath $(MEP_OUTPUT)/moe_expert_policy.bin)
