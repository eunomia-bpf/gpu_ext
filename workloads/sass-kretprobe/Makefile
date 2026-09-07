# sass-kretprobe: verified BPF -> NVBit EXIT injection into a SASS-only sm_120
# target. Owns: BPF source, NVBit tool, embedding glue, target build.
# GLM-owned (do not edit here): bpf_to_ptx.cpp, compiler.mk,
# compiler-notes.md. Their contract: `include compiler.mk` provides
# $(BUILD)/bpf_to_ptx (host exporter). This Makefile invokes it:
#   bpf_to_ptx INPUT_BPF_OBJECT SECTION OUTPUT_PTX [SYMBOL] [SM]
#
# Embedding pipeline (verified against the pinned NVBit 1.8 tool build):
#   BPF ELF --(GLM exporter)--> bpf_exit.ptx (.visible .func, 2-arg)
#   tool/sk_device.cu --(nvcc -ptx -rdc --keep-device-functions)--> wrapper PTX
#   both --(tool/merge_ptx.sh)--> one TU, one header, no .extern
#   --(ptxas -astoolspatch, no compile-only)--> astoolspatch SASS cubin
#   --(fatbinary --embedded-fatbin)--> fatbin C include
#   --(g++ -x c++)--> carrier object owning .nv_fatbin
#   host tool object + carrier --(nvcc -shared -lnvbit)--> sass_kretprobe.so

NVCC ?= nvcc
CXX ?= g++
CLANG ?= clang
PTXAS ?= ptxas
FATBINARY ?= fatbinary
CUOBJDUMP ?= cuobjdump

SM ?= sm_120
SM_NUM := $(patsubst sm_%,%,$(SM))
BPF_SYMBOL ?= bpf_exit
BPF_SECTION ?= cuda__/kretprobe_sass

NVBIT_ROOT ?= ../llama.cpp/observability_overhead/revision-rq4/deps/nvbit_release_x86_64
NVBIT_CORE := $(NVBIT_ROOT)/core

BUILD ?= build
BPF_OBJECT := $(BUILD)/kretprobe_sass.bpf.o
BPF_PTX := $(BUILD)/ptx/bpf_exit.ptx
EXPORTER := $(BUILD)/bpf_to_ptx
TARGET_APP := $(BUILD)/vectoradd
TOOL_SO := $(BUILD)/sass_kretprobe.so

CUDA_BIN := $(shell dirname $(shell which $(NVCC)))
CUDA_LIBDIR := $(CUDA_BIN)/../lib64
CUDA_INC := $(CUDA_BIN)/../targets/x86_64-linux/include

NVCC_DEVFLAGS := -arch=$(SM) -O3 -std=c++14 -Xcompiler -fPIC

.PHONY: all bpf target tool verify clean
all: bpf target tool

# ---------------------------------------------------------------- BPF object
$(BPF_OBJECT): bpf/kretprobe_sass.bpf.c
	mkdir -p $(dir $@)
	$(CLANG) -g -O2 -target bpf -c $< -o $@
bpf: $(BPF_OBJECT)

# --------------------------------------------------- GLM-owned exporter bin
include compiler.mk

# ------------------------------------------------- BPF object -> PTX .func
$(BPF_PTX): $(BPF_OBJECT) $(EXPORTER)
	mkdir -p $(dir $@)
	$(EXPORTER) $(BPF_OBJECT) $(BPF_SECTION) $@ $(BPF_SYMBOL) $(SM)

# ------------------------------------------ wrapper CUDA -> PTX (nvcc stage)
$(BUILD)/tool/sk_device.ptx: tool/sk_device.cu tool/sk_bpf.h
	mkdir -p $(BUILD)/tool
	$(NVCC) -ptx -rdc=true -arch=$(SM) -O3 -std=c++14 --keep-device-functions \
		tool/sk_device.cu -o $@

# ------------------------------------------- single merged astoolspatch TU
$(BUILD)/tool/sk_combined.ptx: $(BPF_PTX) $(BUILD)/tool/sk_device.ptx tool/merge_ptx.sh
	bash tool/merge_ptx.sh $(BPF_PTX) $(BUILD)/tool/sk_device.ptx $@

# ------------------------------------ ptxas astoolspatch SASS (no -c mode)
$(BUILD)/tool/sk_bpf.cubin: $(BUILD)/tool/sk_combined.ptx
	$(PTXAS) -arch=$(SM) -astoolspatch $< -o $@

# --------------------------------------------- fatbin + host carrier object
$(BUILD)/tool/sk_bpf.fatbin.c: $(BUILD)/tool/sk_bpf.cubin
	$(FATBINARY) -64 --embedded-fatbin $@ \
		--image3=kind=elf,sm=$(SM_NUM),file=$<

$(BUILD)/tool/carrier.o: $(BUILD)/tool/sk_bpf.fatbin.c
	$(CXX) -x c++ -fPIC -O2 -I$(CUDA_INC) -c $< -o $@

# ------------------------------------------------------------------ NVBit tool
$(BUILD)/tool/sass_kretprobe.o: tool/sass_kretprobe.cu $(NVBIT_CORE)/nvbit.h
	mkdir -p $(BUILD)/tool
	$(NVCC) -c $(NVCC_DEVFLAGS) -I$(NVBIT_CORE) tool/sass_kretprobe.cu -o $@

$(TOOL_SO): $(BUILD)/tool/sass_kretprobe.o $(BUILD)/tool/carrier.o
	$(NVCC) -arch=$(SM) -O3 \
		$(BUILD)/tool/sass_kretprobe.o $(BUILD)/tool/carrier.o \
		-L$(NVBIT_CORE) -lnvbit -L$(CUDA_LIBDIR) -lcuda -lcudart_static \
		-shared -o $@
	@echo "SKRET tool built: $@"
tool: $(TOOL_SO)

# ------------------------------------------------------------------- target
$(TARGET_APP): $(NVBIT_ROOT)/test-apps/vectoradd/vectoradd.cu
	mkdir -p $(BUILD)
	$(NVCC) -ccbin=$(CXX) -O3 -gencode arch=compute_120,code=sm_120 \
		$< -o $@
	@echo "SKRET target built (SASS-only sm_120): $@"
target: $(TARGET_APP)

# ------------------------------------- optional read-only artifact inspection
verify: $(TOOL_SO) $(TARGET_APP) $(BPF_PTX)
	@echo "== BPF PTX entry (device .func, no .entry) =="
	grep -F ".func $(BPF_SYMBOL)" $(BPF_PTX)
	@! grep -F ".entry" $(BPF_PTX)
	@echo "== embedded SASS symbols in tool fatbin =="
	$(CUOBJDUMP) --dump-elf-symbols $(TOOL_SO) | \
		grep -E "bpf_exit|sk_bpf_trampoline"
	@echo "== target fatbin: PTX absent, sm_120 SASS present =="
	$(CUOBJDUMP) --list-ptx $(TARGET_APP)
	$(CUOBJDUMP) --list-elf $(TARGET_APP)

clean:
	rm -rf $(BUILD)
