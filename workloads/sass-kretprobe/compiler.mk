# compiler.mk -- builds ONLY the bpf_to_ptx exporter (GLM-owned files).
#
# It does not reconfigure or rebuild the bpftime tree: the already-built
# archives in build-table1-575-warp are reused as-is, and exactly one
# verifier translation unit (gpu_verifier.cpp of revision fd976ea) is
# compiled fresh so the strict GPU verifier exposes
# verify_gpu_program_with_context (8-byte PREVAIL context; uniformity +
# SIMT checks run inside the same call chain).
#
# Targets:
#   make -f compiler.mk                  build build/bpf_to_ptx (relative)
#   make -f compiler.mk clean-compiler   remove only exporter artifacts
#
# BIN is the relative build/bpf_to_ptx so the main Makefile (Qwen-owned)
# can depend on it without a second absolute-path target.
#
# Run (from this directory):
#   ./build/bpf_to_ptx INPUT_BPF_OBJECT SECTION OUTPUT_PTX [SYMBOL=bpf_exit] [SM=sm_120]
#   e.g. ./build/bpf_to_ptx build/kretprobe_sass.bpf.o cuda__/kretprobe_sass build/bpf_exit.ptx
#
# This file owns nothing else in workloads/sass-kretprobe; BPF C, NVBit
# tool and target embedding belong to the other agent.

SASS_TREE ?= /home/yunwei37/workspace/gpu/bpftime-sass-existing-application
MAIN_TREE ?= /home/yunwei37/workspace/gpu/bpftime-table1-hostfix-plt
BPFTIME_BUILD ?= $(MAIN_TREE)/build-table1-575-warp

HERE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
BUILD ?= build
OUTDIR := $(BUILD)
BIN := $(OUTDIR)/bpf_to_ptx
PTX_EXPORTER_SRC := $(HERE)bpf_to_ptx.cpp
FD_VERIFIER_SRC := $(SASS_TREE)/bpftime-verifier/src/gpu/gpu_verifier.cpp

CXX ?= /usr/bin/c++

# Flags copied from $(BPFTIME_BUILD)/bpftime-verifier/CMakeFiles/bpftime-verifier.dir/flags.make
# so the fresh TU is ABI-compatible with the archived verifier objects
# (simt_safety_check.cpp.o / uniformity_analysis.cpp.o / gpu_platform.cpp.o
# are byte-identical sources between fd976ea and the built tree).
VERIFIER_DEFINES := -DBPFTIME_BUILD_WITH_LIBBPF=1 -DLLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1
VERIFIER_INCLUDES := -I$(SASS_TREE)/bpftime-verifier/include \
	-I$(MAIN_TREE)/bpftime-verifier/ebpf-verifier/src \
	-I$(BPFTIME_BUILD)/libbpf
VERIFIER_CXXFLAGS := -fno-omit-frame-pointer -g -std=gnu++20 -fPIC -D_DEBUG \
	-O0 -g3 -fno-omit-frame-pointer

# Exporter TU: needs the fd976ea verifier header (with_context declaration)
# and the existing ptxpass core header.
EXPORTER_DEFINES := $(VERIFIER_DEFINES)
EXPORTER_INCLUDES := -I$(SASS_TREE)/bpftime-verifier/include \
	-I$(SASS_TREE)/attach/nv_attach_impl/pass/ptxpass_core/include \
	-I$(MAIN_TREE)/third_party
EXPORTER_CXXFLAGS := -fno-omit-frame-pointer -g -std=gnu++20 -fPIC -O2

# LLVM static link set copied from the already-proven consumer link line
# $(BPFTIME_BUILD)/vm/compat/llvm-vm/llvm-jit/example/CMakeFiles/maps-example.dir/link.txt
LLVM_LIBS := $(sort $(wildcard /usr/lib/llvm-15/lib/libLLVM*.a))

VERIFIER_ARCHIVE := $(BPFTIME_BUILD)/bpftime-verifier/libbpftime-verifier.a
PREVAIL_ARCHIVE := $(BPFTIME_BUILD)/bpftime-verifier/ebpf-verifier/libebpfverifier.a
LIBBTF_ARCHIVE := $(BPFTIME_BUILD)/bpftime-verifier/ebpf-verifier/external/libbtf/libbtf/liblibbtf.a
PTXPASS_ARCHIVE := $(BPFTIME_BUILD)/attach/nv_attach_impl/pass/ptxpass_core/libptxpass_core.a
LLVMBPF_ARCHIVE := $(BPFTIME_BUILD)/vm/compat/llvm-vm/libllvmbpf_vm.a
SPDLOG_ARCHIVE := $(BPFTIME_BUILD)/third_party/spdlog/libspdlogd.a

.PHONY: all clean-compiler
all: $(BIN)

$(OUTDIR):
	mkdir -p $@

$(OUTDIR)/gpu_verifier.o: $(FD_VERIFIER_SRC) | $(OUTDIR)
	$(CXX) $(VERIFIER_DEFINES) $(VERIFIER_INCLUDES) $(VERIFIER_CXXFLAGS) \
		-c $< -o $@

$(OUTDIR)/bpf_to_ptx.o: $(PTX_EXPORTER_SRC) | $(OUTDIR)
	$(CXX) $(EXPORTER_DEFINES) $(EXPORTER_INCLUDES) $(EXPORTER_CXXFLAGS) \
		-c $< -o $@

$(BIN): $(OUTDIR)/bpf_to_ptx.o $(OUTDIR)/gpu_verifier.o $(LIBBTF_ARCHIVE)
	$(CXX) -fno-omit-frame-pointer -g -o $@ \
		$(OUTDIR)/bpf_to_ptx.o $(OUTDIR)/gpu_verifier.o \
		-Wl,--start-group \
		$(VERIFIER_ARCHIVE) \
		$(PREVAIL_ARCHIVE) \
		$(LIBBTF_ARCHIVE) \
		$(PTXPASS_ARCHIVE) \
		$(LLVMBPF_ARCHIVE) \
		$(LLVM_LIBS) \
		$(SPDLOG_ARCHIVE) \
		-Wl,--end-group \
		-lelf -lrt -ldl -lm -lz -lzstd -ltinfo -lpthread

# Named clean-compiler so this file can be included by the main Makefile
# (Qwen-owned) without clashing with its clean recipe.
clean-compiler:
	rm -f $(OUTDIR)/bpf_to_ptx.o $(OUTDIR)/gpu_verifier.o $(BIN)
