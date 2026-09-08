# XSched Level-2 bounded component build.
#
# Builds ONLY the Level-2 guardian components from the read-only sibling
# source tree ../level2 (workloads/xsched/level2/, GLM-owned). All outputs go
# under .output/ in this directory. No GPU work, no driver reload, no
# experiment execution, no writes outside this directory.
#
# Explicit component targets:
#   bpf-ptx        real BPF ELF -> device-callable PTX of xsched_guardian
#   guardian-cubin merged trampoline+BPF PTX -> sm_120 astoolspatch SASS
#   guard-tool     NVBit xg guard shared library (tool + embedded fatbin)
#   native         sm_120 probe/stub cubins + host xg_ldc_patcher binary
#   all            the four components; nothing is run
#
# Non-default extras:
#   exporter       ensure the GLM bpf_to_ptx exporter exists
#   ldc-patch      host-only: xg_ldc_patcher -> xg_sm120_guardian_arrays.h
#   hal-arrays     print XG_SM120_GENERATED_HEADER wiring for the deps/xsched build
#   clean          remove .output/
#
# Pipeline (mirrors the verified workloads/sass-kretprobe build):
#   bpf/xsched_guardian.bpf.c --clang -target bpf--> BPF ELF
#     --bpf_to_ptx (GLM exporter; CLI: OBJ SECTION OUT [SYMBOL] [SM])-->
#     device PTX (.visible .func, 2-arg context ABI: ptr, length)
#   tramp/xsched_guard_tramp.cu --nvcc -ptx -rdc--> trampoline PTX
#   both --level2/tool/merge_ptx.sh--> one TU, one header, no .extern
#     --ptxas -astoolspatch--> sm_120 SASS cubin
#     --fatbinary --embedded-fatbin--> fatbin C include
#     --g++ -x c++--> carrier object owning .nv_fatbin
#   tool/xsched_guard_tool.cu + carrier --nvcc -shared -lnvbit--> .so
#
# The native-C vs BPF decision is made ON DEVICE by xg_tramp's
# decision_mode argument (1 = device eBPF, 0 = native C); both arms share
# the same actuator glue in one trampoline TU. The host only selects the
# insertion-time constant; there is no host decision fallback.

CUDA_HOME ?= /usr/local/cuda-12.9
.DEFAULT_GOAL := all
NVCC ?= $(CUDA_HOME)/bin/nvcc
CXX ?= g++
CLANG ?= clang
PTXAS ?= $(CUDA_HOME)/bin/ptxas
FATBINARY ?= $(CUDA_HOME)/bin/fatbinary
CUOBJDUMP ?= $(CUDA_HOME)/bin/cuobjdump
NVDISASM ?= $(CUDA_HOME)/bin/nvdisasm

CUDA_INC := $(CUDA_HOME)/targets/x86_64-linux/include
CUDA_LIBDIR := $(CUDA_HOME)/targets/x86_64-linux/lib

SM ?= sm_120
SM_NUM := $(patsubst sm_%,%,$(SM))

# BPF symbol/section from level2/xsched_guardian_abi.h
# (XG_BPF_SYMBOL / XG_BPF_SECTION).
BPF_SYMBOL ?= xsched_guardian
BPF_SECTION ?= cuda__/xsched_guardian

# NVBit dependency path from the existing workloads/sass-kretprobe Makefile
# (same workloads/ tree, re-based to this directory).
NVBIT_ROOT ?= ../../llama.cpp/observability_overhead/revision-rq4/deps/nvbit_release_x86_64
NVBIT_CORE := $(NVBIT_ROOT)/core

SRC := ../level2
SASS_DIR := ../../sass-kretprobe

BUILD ?= .output
BPF_OBJECT := $(BUILD)/xsched_guardian.bpf.o
BPF_PTX := $(BUILD)/ptx/$(BPF_SYMBOL).ptx
TRAMP_PTX := $(BUILD)/tramp/xsched_guard_tramp.ptx
MERGED_PTX := $(BUILD)/merged/xg_guardian.ptx
GUARDIAN_CUBIN := $(BUILD)/xg_guardian.cubin
FATBIN_C := $(BUILD)/xg_guardian.fatbin.c
CARRIER_O := $(BUILD)/xg_guardian_carrier.o
TOOL_O := $(BUILD)/xsched_guard_tool.o
TOOL_SO := $(BUILD)/xsched_guard_tool.so
PROBE_CUBIN := $(BUILD)/native/xg_ldc_probe.cubin
CHECK_CUBIN := $(BUILD)/native/check_preempt_port.cubin
RESTORE_CUBIN := $(BUILD)/native/restore_exec_port.cubin
PATCHER := $(BUILD)/native/xg_ldc_patcher
PATCHED_HEADER := $(BUILD)/xg_sm120_guardian_arrays.h

# GLM-owned exporter (workloads/sass-kretprobe/compiler.mk contract),
# rebuilt here through the Level-2 context adapter. The reused un-adapted
# 8-byte exporter cannot justify the Level-2 snapshot decision context
# (6 x u64 = 48 bytes, level2/xsched_guardian_abi.h); the adapter patch
# level2/bpf/bpf_to_ptx_ctx48.patch widens the strict PREVAIL verify-time
# context descriptor to 48 bytes on a COPY of the exporter source. The
# frozen workloads/sass-kretprobe source/build is never edited.
ADAPTER_DIR := $(BUILD)/adapter
ADAPTER_SRC := $(ADAPTER_DIR)/bpf_to_ptx.cpp
EXPORTER := $(BUILD)/compiler/bpf_to_ptx

$(ADAPTER_SRC): $(SASS_DIR)/bpf_to_ptx.cpp $(SRC)/bpf/bpf_to_ptx_ctx48.patch
	mkdir -p $(ADAPTER_DIR)
	cp $< $(ADAPTER_DIR)/bpf_to_ptx.cpp
	patch -d $(ADAPTER_DIR) -p1 --forward < $(SRC)/bpf/bpf_to_ptx_ctx48.patch

$(EXPORTER): $(ADAPTER_SRC)
	$(MAKE) -f $(SASS_DIR)/compiler.mk BUILD=$(abspath $(BUILD)/compiler) \
		PTX_EXPORTER_SRC=$(abspath $(ADAPTER_SRC)) BIN=$(abspath $(EXPORTER))

NVCC_DEVFLAGS := -arch=$(SM) -O3 -std=c++14 -Xcompiler -fPIC
NVCC_CUBINFLAGS := -cubin -O3 -gencode arch=compute_$(SM_NUM),code=$(SM)

.PHONY: all bpf-ptx guardian-cubin guard-tool native ldc-patch hal-arrays exporter clean

all: bpf-ptx guardian-cubin guard-tool native

# ------------------------------------------------------------- BPF object
$(BPF_OBJECT): $(SRC)/bpf/xsched_guardian.bpf.c $(SRC)/tramp/xg_bpf_abi.h
	mkdir -p $(dir $@)
	$(CLANG) -g -O2 -target bpf -I$(SRC)/tramp -c $< -o $@

# -------------------------------------- BPF ELF -> device-callable PTX .func
$(BPF_PTX): $(BPF_OBJECT) $(EXPORTER)
	mkdir -p $(dir $@)
	$(EXPORTER) $(BPF_OBJECT) $(BPF_SECTION) $@ $(BPF_SYMBOL) $(SM)
bpf-ptx: $(BPF_PTX)

# ------------------------------------------ trampoline CUDA -> PTX (nvcc stage)
$(TRAMP_PTX): $(SRC)/tramp/xsched_guard_tramp.cu $(SRC)/xsched_guardian_abi.h
	mkdir -p $(dir $@)
	$(NVCC) -ptx -rdc=true -arch=$(SM) -O3 -std=c++14 --keep-device-functions \
		$(SRC)/tramp/xsched_guard_tramp.cu -o $@

# ------------------------------------------- single merged astoolspatch TU
$(MERGED_PTX): $(BPF_PTX) $(TRAMP_PTX) $(SRC)/tool/merge_ptx.sh
	mkdir -p $(dir $@)
	bash $(SRC)/tool/merge_ptx.sh $(BPF_PTX) $(TRAMP_PTX) $@

# ------------------------------------ ptxas astoolspatch SASS (no -c mode)
$(GUARDIAN_CUBIN): $(MERGED_PTX)
	$(PTXAS) -arch=$(SM) -astoolspatch $< -o $@
guardian-cubin: $(GUARDIAN_CUBIN)

# --------------------------------------------- fatbin + host carrier object
$(FATBIN_C): $(GUARDIAN_CUBIN)
	$(FATBINARY) -64 --embedded-fatbin $@ \
		--image3=kind=elf,sm=$(SM_NUM),file=$<

$(CARRIER_O): $(FATBIN_C)
	$(CXX) -x c++ -fPIC -O2 -I$(CUDA_INC) -c $< -o $@

# ------------------------------------------------------------------ NVBit tool
$(TOOL_O): $(SRC)/tool/xsched_guard_tool.cu $(NVBIT_CORE)/nvbit.h
	mkdir -p $(dir $@)
	$(NVCC) -c $(NVCC_DEVFLAGS) -I$(NVBIT_CORE) \
		$(SRC)/tool/xsched_guard_tool.cu -o $@

$(TOOL_SO): $(TOOL_O) $(CARRIER_O)
	$(NVCC) -arch=$(SM) -O3 \
		$(TOOL_O) $(CARRIER_O) \
		-L$(NVBIT_CORE) -lnvbit -L$(CUDA_LIBDIR) -lcuda -lcudart_static \
		-shared -o $@
	@echo "XG tool built: $@"
guard-tool: $(TOOL_SO)

# --------------------------------------------------- native sm_120 components
$(PROBE_CUBIN): $(SRC)/native/probe.cu
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_CUBINFLAGS) $< -o $@

$(CHECK_CUBIN): $(SRC)/native/check_preempt_port.cu
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_CUBINFLAGS) $< -o $@

$(RESTORE_CUBIN): $(SRC)/native/restore_exec_port.cu
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_CUBINFLAGS) $< -o $@

$(PATCHER): $(SRC)/native/ldc_patcher.cpp
	mkdir -p $(dir $@)
	$(CXX) -O2 -std=c++17 $< -o $@

native: $(PROBE_CUBIN) $(CHECK_CUBIN) $(RESTORE_CUBIN) $(PATCHER)

# ------------------------------------------- host-only guardian extraction
# Feeds GLM's HAL arch/sm120.cpp integration; never part of the default.
# The generated arrays header is consumed by a deps/xsched build that
# applies ../level2/xsched-level2-sm120.patch, configured with
#   -DXG_SM120_GENERATED_HEADER=$(abspath .output/xg_sm120_guardian_arrays.h)
# which enables GuardianSM120 and the arch-120 Guardian factory case.
$(PATCHED_HEADER): $(PATCHER) $(CHECK_CUBIN) $(RESTORE_CUBIN)
	mkdir -p $(dir $@)
	$(PATCHER) $(CHECK_CUBIN) $(RESTORE_CUBIN) $(NVDISASM) $@

ldc-patch: $(PATCHED_HEADER)

# Print the exact CMake variable that wires the generated arrays into a
# patched deps/xsched HAL build (see ../level2/xsched-level2-sm120.patch).
hal-arrays: $(PATCHED_HEADER)
	@echo "XG_SM120_GENERATED_HEADER=$(abspath $(PATCHED_HEADER))"

exporter: $(EXPORTER)

clean:
	rm -rf $(BUILD)
