# hostdev Makefile: builds the Hummingbird original host+device BPF arms.
#
# Produces, under build/, the device arms consumed by the unchanged
# host/pipeline client:
#   build/resnet152-bpf/mod-native.cubin  -- native ADAPTER control
#        (mod-bpf.cu calling the native callable wrapper; callable device
#         function + 48-byte context, native computation)
#   build/resnet152-bpf/mod-bpf.cubin     -- the BPF device arm
#        (same mod-bpf.cu, clone call targets rewritten to the compiled eBPF
#         engine hb_device_bpf_map)
#
# The ORIGINAL inline control is the frozen, reused
# ../build/resnet152-split/mod.cubin (built by the parent Makefile); it is not
# rebuilt here. All arms share the same tile budget / host policy / arrival
# trace / DNN; only the device mapping mechanism differs.
#
# Pipeline:
#   clang -target bpf            -> build/hb_map.bpf.o
#   make -f compiler.mk          -> build/hb_map_exporter (48-byte ctx verifier)
#   hb_map_exporter              -> build/hb_map.ptx (.func hb_device_bpf_map)
#   hb_split_model.py            -> build/resnet152-bpf/mod-bpf.cu
#   nvcc -ptx -rdc --keep-dev    -> build/resnet152-bpf/mod-native.ptx
#   ptxas                        -> build/resnet152-bpf/mod-native.cubin
#   patch_device_map.py          -> build/resnet152-bpf/mod-bpf-patched.ptx
#   ptxas                        -> build/resnet152-bpf/mod-bpf.cubin
#
# No GPU is invoked by any build step; ptxas only assembles PTX to SASS.
# All paths are anchored to $(HERE) so `make -C .../hostdev` is CWD-safe.

HB_CUDA ?= /usr/local/cuda-12.9
NVCC  := $(HB_CUDA)/bin/nvcc
PTXAS := $(HB_CUDA)/bin/ptxas
CLANG ?= /usr/bin/clang
CXX   ?= /usr/bin/g++
PYTHON ?= /usr/bin/python3

HERE     := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
BUILD    ?= build
OUT      := $(BUILD)/resnet152-bpf
UPSTREAM ?= $(HERE)../../gpreempt/deps/upstream/model/resnet152
SM       ?= sm_120
HB_CPUSET ?=

BPF_OBJ     := $(BUILD)/hb_map.bpf.o
EXPORTER    := $(BUILD)/hb_map_exporter
BPF_PTX     := $(BUILD)/hb_map.ptx
TRANS_CU    := $(OUT)/mod-bpf.cu
NATIVE_PTX  := $(OUT)/mod-native.ptx
NATIVE_BIN  := $(OUT)/mod-native.cubin
PATCHED_PTX := $(OUT)/mod-bpf-patched.ptx
BPF_BIN     := $(OUT)/mod-bpf.cubin

BPF_SECTION ?= cuda__/hb_device_map
BPF_SYMBOL  ?= hb_device_bpf_map

.PHONY: all check test clean
all: $(BPF_PTX) $(NATIVE_BIN) $(BPF_BIN)

$(OUT):
	mkdir -p $@

# ------------------------------------------------------------- BPF object
$(BPF_OBJ): $(HERE)hb_map.bpf.c $(HERE)hb_map_policy.h $(HERE)hb_map_abi.h | $(BUILD)
	$(CLANG) -g -O2 -target bpf -c $< -o $@

# ------------------------------------------------- bounded 48-byte exporter
# compiler.mk owns the `build/` directory rule and build/hb_map_exporter.
include compiler.mk

# ------------------------------------------------------ BPF object -> PTX
$(BPF_PTX): $(BPF_OBJ) $(EXPORTER)
	$(EXPORTER) $(BPF_OBJ) $(BPF_SECTION) $@ $(BPF_SYMBOL) $(SM)

# --------------------------------------------------- BPF-variant transform
$(TRANS_CU): $(HERE)hb_split_model.py $(UPSTREAM)/mod.cu $(UPSTREAM)/host.json | $(OUT)
	$(PYTHON) -B $(HERE)hb_split_model.py \
		--source $(UPSTREAM)/mod.cu --host $(UPSTREAM)/host.json \
		--output $(OUT)

# --------------------------------------------------- callable-native PTX
$(NATIVE_PTX): $(TRANS_CU) $(HERE)hb_map_cuda.cuh $(HERE)hb_map_policy.h $(HERE)hb_map_abi.h
	$(NVCC) -ccbin $(CXX) -O3 -arch=$(SM) -std=c++14 -ptx \
		-rdc=true --keep-device-functions -I$(HERE) $(TRANS_CU) -o $@

# ------------------------------------------------- native ADAPTER cubin
$(NATIVE_BIN): $(NATIVE_PTX)
	$(PTXAS) -arch=$(SM) -O3 $< -o $@

# ------------------------------------------------------- BPF-clone patch
$(PATCHED_PTX): $(NATIVE_PTX) $(BPF_PTX) $(HERE)patch_device_map.py
	$(PYTHON) -B $(HERE)patch_device_map.py --kernel $(NATIVE_PTX) --bpf $(BPF_PTX) \
		--output $@

# ----------------------------------------------------------- BPF cubin
$(BPF_BIN): $(PATCHED_PTX)
	$(PTXAS) -arch=$(SM) -O3 $< -o $@

# ------------------------------------------------------------- checks
check:
	$(PYTHON) -B $(HERE)hb_split_model.py \
		--source $(UPSTREAM)/mod.cu --host $(UPSTREAM)/host.json --check

test:
	$(PYTHON) -B $(HERE)test_hb_map.py --build $(BUILD)

clean:
	rm -rf $(BUILD)
