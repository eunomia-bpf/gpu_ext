# CPU-only WARC policy-port build and differential test. No GPU, no driver.
# Reuses the existing gpubpf selector artifacts from extension/.output
# (built by extension/moe_expert_policy.mk); never rebuilds them here.
CC ?= cc
CFLAGS ?= -O2 -g -Wall -Wextra -Werror -std=c11

HERE := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
OUTPUT ?= $(HERE)/.output
REPO_ROOT := $(abspath $(HERE)/../..)
INC := -I$(REPO_ROOT)/extension
EXTENSION_OUTPUT := $(REPO_ROOT)/extension/.output
BPF_LIB ?= $(EXTENSION_OUTPUT)/libmoe_expert_policy.so
BPF_CODE ?= $(EXTENSION_OUTPUT)/moe_expert_policy_rank.bin
TRACE ?= $(HERE)/../expert-buffering-policy/raw/timing/block-01/llama_ncmoe32/trace.jsonl

.PHONY: all test clean
.SECONDARY:

all: $(OUTPUT)/libdali_policy.so $(OUTPUT)/test_dali_policy

$(OUTPUT):
	mkdir -p $@

$(OUTPUT)/libdali_policy.so: dali_policy.c dali_policy_abi.h dali_policy_model.h \
		$(REPO_ROOT)/extension/moe_expert_policy.h | $(OUTPUT)
	$(CC) $(CFLAGS) $(INC) -fPIC -shared dali_policy.c -o $@.tmp -ldl
	mv $@.tmp $@

$(OUTPUT)/test_dali_policy: test_dali_policy.c dali_policy_abi.h \
		dali_policy_model.h $(OUTPUT)/libdali_policy.so
	$(CC) $(CFLAGS) $(INC) test_dali_policy.c -L$(OUTPUT) -ldali_policy \
		-Wl,-rpath,'$$ORIGIN' -o $@

test: all
	$(OUTPUT)/test_dali_policy "$(TRACE)" "$(BPF_LIB)" "$(BPF_CODE)"

clean:
	rm -rf $(OUTPUT)
