# compiler.mk -- builds ONLY the hb_map_exporter (Hummingbird host-device BPF
# -> PTX exporter).
#
# It is a local adapter over the exact machinery proven by the SASS
# kretprobe exporter: it does NOT reconfigure or rebuild the bpftime tree.
# The already-built archives in build-table1-575-warp are reused as-is, and
# exactly one verifier translation unit (gpu_verifier.cpp of revision
# fd976ea) is compiled fresh so the strict GPU verifier exposes
# verify_gpu_program_with_context. The context bound is the 48-byte
# HbMapContext pinned by hostdev/hb_map_abi.h (the ctx48 shape).
#
# Targets:
#   make -f compiler.mk                  build build/hb_map_exporter
#   make -f compiler.mk clean-compiler   remove only exporter artifacts
#
# Run (from this directory):
#   ./build/hb_map_exporter INPUT_BPF_OBJECT SECTION OUTPUT_PTX [SYMBOL] [SM]
#   e.g. ./build/hb_map_exporter build/hb_map.bpf.o cuda__/hb_device_map \
#            build/hb_map.ptx hb_device_bpf_map sm_120
#
# This file owns nothing else; BPF C, source transform, PTX patcher and
# cubin build belong to the rest of hostdev.

SASS_TREE ?= /home/yunwei37/workspace/gpu/bpftime-sass-existing-application
MAIN_TREE ?= /home/yunwei37/workspace/gpu/bpftime-table1-hostfix-plt
BPFTIME_BUILD ?= $(MAIN_TREE)/build-table1-575-warp

HERE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
BUILD ?= build
OUTDIR := $(BUILD)
BIN := $(OUTDIR)/hb_map_exporter
PTX_EXPORTER_SRC := $(HERE)hb_map_exporter.cpp
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

# Exporter TU: needs the fd976ea verifier header (with_context declaration),
# the existing ptxpass core header, and hostdev/hb_map_abi.h for the context
# bound.
EXPORTER_DEFINES := $(VERIFIER_DEFINES)
EXPORTER_INCLUDES := -I$(HERE) \
	-I$(SASS_TREE)/bpftime-verifier/include \
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

$(OUTDIR)/hb_map_exporter.o: $(PTX_EXPORTER_SRC) hb_map_abi.h | $(OUTDIR)
	$(CXX) $(EXPORTER_DEFINES) $(EXPORTER_INCLUDES) $(EXPORTER_CXXFLAGS) \
		-c $< -o $@

$(BIN): $(OUTDIR)/hb_map_exporter.o $(OUTDIR)/gpu_verifier.o $(LIBBTF_ARCHIVE)
	$(CXX) -fno-omit-frame-pointer -g -o $@ \
		$(OUTDIR)/hb_map_exporter.o $(OUTDIR)/gpu_verifier.o \
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

# Named clean-compiler so this file can be included by the main hostdev
# Makefile without clashing with its clean recipe.
clean-compiler:
	rm -f $(OUTDIR)/hb_map_exporter.o $(OUTDIR)/gpu_verifier.o $(BIN)
