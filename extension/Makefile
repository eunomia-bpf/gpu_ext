# SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
OUTPUT := .output
CLANG ?= clang
LLVM_STRIP ?= llvm-strip
LIBBPF_SRC := $(abspath ../libbpf/src)
BPFTOOL_SRC := $(abspath ../bpftool/src)
LIBBPF_OBJ := $(abspath $(OUTPUT)/libbpf.a)
BPFTOOL_OUTPUT ?= $(abspath $(OUTPUT)/bpftool)
BPFTOOL ?= $(BPFTOOL_OUTPUT)/bootstrap/bpftool
ARCH ?= $(shell uname -m | sed 's/x86_64/x86/' \
			 | sed 's/aarch64/arm64/' \
			 | sed 's/ppc64le/powerpc/' \
			 | sed 's/mips.*/mips/' \
			 | sed 's/arm.*/arm/' \
			 | sed 's/riscv64/riscv/')
VMLINUX := ../vmlinux/$(ARCH)/vmlinux.h
CUDA_INCLUDE_DIR ?= /usr/local/cuda/include
CUDA_STUB_DIR ?= /usr/local/cuda/lib64/stubs
# Use our own libbpf API headers and Linux UAPI headers distributed with
# libbpf to avoid dependency on system-wide headers, which could be missing or
# outdated
INCLUDES := -I$(OUTPUT) -I../libbpf/include/uapi -I$(dir $(VMLINUX)) -I$(CUDA_INCLUDE_DIR)
CFLAGS := -g -Wall
ALL_LDFLAGS := $(LDFLAGS) $(EXTRA_LDFLAGS)

# sched_ext (scx) headers for GPU-aware scheduler
SCX_INCLUDE_DIR ?=
SCX_HEADER := $(if $(SCX_INCLUDE_DIR),$(SCX_INCLUDE_DIR)/scx/common.bpf.h,)
SCX_VMLINUX := ../vmlinux/$(ARCH)/scx/vmlinux.h
SCX_INCLUDES := $(if $(SCX_INCLUDE_DIR),-I$(SCX_INCLUDE_DIR),) -I$(dir $(SCX_VMLINUX))
HAVE_SCX := $(if $(SCX_INCLUDE_DIR),$(if $(wildcard $(SCX_HEADER)),$(if $(wildcard $(SCX_VMLINUX)),1,0),0),0)
HAVE_NVML := $(if $(wildcard $(CUDA_INCLUDE_DIR)/nvml.h),$(if $(firstword $(wildcard $(CUDA_STUB_DIR)/libnvidia-ml.so $(CUDA_STUB_DIR)/libnvidia-ml.so.*)),1,0),0)
HAVE_CUDA := $(if $(wildcard $(CUDA_INCLUDE_DIR)/cuda.h),$(if $(firstword $(wildcard $(CUDA_STUB_DIR)/libcuda.so $(CUDA_STUB_DIR)/libcuda.so.*)),1,0),0)

BPF_APPS = struct_ops prefetch_always_max prefetch_none prefetch_adaptive_tree_iter prefetch_adaptive_sequential prefetch_pid_tree prefetch_eviction_pid prefetch_stride eviction_fifo eviction_lfu eviction_mru eviction_pid_quota eviction_freq_pid_decay eviction_fifo_chance eviction_cycle_moe chunk_trace prefetch_trace gpu_sched_trace gpu_preempt_ctrl gpu_sched_set_timeslices eviction_lfu_xcoord prefetch_always_max_cycle_moe prefetch_max_mru_expert prefetch_max_passive_mru prefetch_cross_block_v2 prefetch_template_belady prefetch_proactive_layer prefetch_always_max_xcoord prefetch_always_max_qos test_preempt_kfunc test_preempt_demo test_preempt_multi test_uprobe_preempt prefetch_faiss_phase prefetch_serving_adaptive # BPF apps that need skeletons
SCX_APPS = sched_gpu_baseline sched_gpu_minimal sched_gpu_serving sched_gpu_xcoord sched_gpu_xcoord_noad sched_gpu_coord # sched_ext apps (need scx headers)
TOOL_APPS = cleanup_struct_ops_tool # Pure userspace tools

# Apps that need NVML
NVML_APPS = prefetch_adaptive_tree_iter prefetch_adaptive_sequential

# Apps that need pthread
PTHREAD_APPS = gpu_preempt_ctrl

# Apps that need CUDA driver API + pthread
CUDA_APPS = test_preempt_demo test_preempt_multi

CORE_BPF_APPS = $(filter-out $(NVML_APPS) $(CUDA_APPS),$(BPF_APPS))
ENABLED_NVML_APPS = $(if $(filter 1,$(HAVE_NVML)),$(NVML_APPS),)
ENABLED_CUDA_APPS = $(if $(filter 1,$(HAVE_CUDA)),$(CUDA_APPS),)
ENABLED_SCX_APPS = $(if $(filter 1,$(HAVE_SCX)),$(SCX_APPS),)
APPS = $(CORE_BPF_APPS) $(ENABLED_NVML_APPS) $(PTHREAD_APPS) $(ENABLED_CUDA_APPS) $(ENABLED_SCX_APPS) $(TOOL_APPS) # minimal minimal_legacy uprobe kprobe fentry usdt sockfilter tc ksyscall
ALL_APPS = $(BPF_APPS) $(SCX_APPS) $(TOOL_APPS)

CARGO ?= $(shell which cargo)
ifeq ($(strip $(CARGO)),)
BZS_APPS :=
else
BZS_APPS := # profile
APPS += $(BZS_APPS)
ALL_APPS += $(BZS_APPS)
# Required by libblazesym
ALL_LDFLAGS += -lrt -ldl -lpthread -lm
endif

ifeq ($(HAVE_NVML),0)
$(info Skipping NVML apps; set CUDA_INCLUDE_DIR/CUDA_STUB_DIR to enable: $(NVML_APPS))
endif

ifeq ($(HAVE_CUDA),0)
$(info Skipping CUDA apps; set CUDA_INCLUDE_DIR/CUDA_STUB_DIR to enable: $(CUDA_APPS))
endif

ifeq ($(HAVE_SCX),0)
$(info Skipping sched_ext apps; set SCX_INCLUDE_DIR to a kernel tools/sched_ext include tree and ensure $(SCX_VMLINUX) exists: $(SCX_APPS))
endif

# Get Clang's default includes on this system. We'll explicitly add these dirs
# to the includes list when compiling with `-target bpf` because otherwise some
# architecture-specific dirs will be "missing" on some architectures/distros -
# headers such as asm/types.h, asm/byteorder.h, asm/socket.h, asm/sockios.h,
# sys/cdefs.h etc. might be missing.
#
# Use '-idirafter': Don't interfere with include mechanics except where the
# build would have failed anyways.
CLANG_BPF_SYS_INCLUDES ?= $(shell $(CLANG) -v -E - </dev/null 2>&1 \
	| sed -n '/<...> search starts here:/,/End of search list./{ s| \(/.*\)|-idirafter \1|p }')

ifeq ($(V),1)
	Q =
	msg =
else
	Q = @
	msg = @printf '  %-8s %s%s\n'					\
		      "$(1)"						\
		      "$(patsubst $(abspath $(OUTPUT))/%,%,$(2))"	\
		      "$(if $(3), $(3))";
	MAKEFLAGS += --no-print-directory
endif

define allow-override
  $(if $(or $(findstring environment,$(origin $(1))),\
            $(findstring command line,$(origin $(1)))),,\
    $(eval $(1) = $(2)))
endef

$(call allow-override,CC,$(CROSS_COMPILE)cc)
$(call allow-override,LD,$(CROSS_COMPILE)ld)

.PHONY: all
all: $(APPS)

.PHONY: clean
clean:
	$(call msg,CLEAN)
	$(Q)rm -rf $(OUTPUT) $(ALL_APPS)

$(OUTPUT) $(OUTPUT)/libbpf $(BPFTOOL_OUTPUT):
	$(call msg,MKDIR,$@)
	$(Q)mkdir -p $@

# Build libbpf
$(LIBBPF_OBJ): $(wildcard $(LIBBPF_SRC)/*.[ch] $(LIBBPF_SRC)/Makefile) | $(OUTPUT)/libbpf
	$(call msg,LIB,$@)
	$(Q)$(MAKE) -C $(LIBBPF_SRC) BUILD_STATIC_ONLY=1		      \
		    OBJDIR=$(dir $@)/libbpf DESTDIR=$(dir $@)		      \
		    INCLUDEDIR= LIBDIR= UAPIDIR=			      \
		    install

# Build bpftool
$(BPFTOOL): | $(BPFTOOL_OUTPUT)
	$(call msg,BPFTOOL,$@)
	$(Q)$(MAKE) ARCH= CROSS_COMPILE= OUTPUT=$(BPFTOOL_OUTPUT)/ -C $(BPFTOOL_SRC) bootstrap


$(LIBBLAZESYM_SRC)/target/release/libblazesym.a::
	$(Q)cd $(LIBBLAZESYM_SRC) && $(CARGO) build --features=cheader,dont-generate-test-files --release

$(LIBBLAZESYM_OBJ): $(LIBBLAZESYM_SRC)/target/release/libblazesym.a | $(OUTPUT)
	$(call msg,LIB, $@)
	$(Q)cp $(LIBBLAZESYM_SRC)/target/release/libblazesym.a $@

$(LIBBLAZESYM_HEADER): $(LIBBLAZESYM_SRC)/target/release/libblazesym.a | $(OUTPUT)
	$(call msg,LIB,$@)
	$(Q)cp $(LIBBLAZESYM_SRC)/target/release/blazesym.h $@

# Build BPF code (standard)
$(OUTPUT)/%.bpf.o: %.bpf.c $(LIBBPF_OBJ) $(wildcard %.h) $(VMLINUX) | $(OUTPUT) $(BPFTOOL)
	$(call msg,BPF,$@)
	$(Q)$(CLANG) -g -O2 -target bpf -D__TARGET_ARCH_$(ARCH)		      \
		     $(INCLUDES) $(CLANG_BPF_SYS_INCLUDES)		      \
		     -c $(filter %.c,$^) -o $(patsubst %.bpf.o,%.tmp.bpf.o,$@)
	$(Q)$(BPFTOOL) gen object $@ $(patsubst %.bpf.o,%.tmp.bpf.o,$@)

# Build BPF code (sched_ext apps - use scx vmlinux.h + scx headers)
$(patsubst %,$(OUTPUT)/%.bpf.o,$(SCX_APPS)): $(OUTPUT)/%.bpf.o: %.bpf.c $(LIBBPF_OBJ) $(wildcard %.h) $(SCX_VMLINUX) | $(OUTPUT) $(BPFTOOL)
	$(call msg,SCX-BPF,$@)
	$(Q)$(CLANG) -g -O2 -target bpf -D__TARGET_ARCH_$(ARCH)		      \
		     -I$(OUTPUT) -I../libbpf/include/uapi			      \
		     $(SCX_INCLUDES)						      \
		     $(CLANG_BPF_SYS_INCLUDES)				      \
		     -c $(filter %.c,$^) -o $(patsubst %.bpf.o,%.tmp.bpf.o,$@)
	$(Q)$(BPFTOOL) gen object $@ $(patsubst %.bpf.o,%.tmp.bpf.o,$@)

# Generate BPF skeletons
$(OUTPUT)/%.skel.h: $(OUTPUT)/%.bpf.o | $(OUTPUT) $(BPFTOOL)
	$(call msg,GEN-SKEL,$@)
	$(Q)$(BPFTOOL) gen skeleton $< > $@

# Build user-space code - BPF apps need skeletons
$(patsubst %,$(OUTPUT)/%.o,$(BPF_APPS)): %.o: %.skel.h
$(patsubst %,$(OUTPUT)/%.o,$(SCX_APPS)): %.o: %.skel.h

# Tool apps don't need skeletons
$(patsubst %,$(OUTPUT)/%.o,$(TOOL_APPS)): $(OUTPUT)/%.o: %.c cleanup_struct_ops.h | $(OUTPUT)
	$(call msg,CC,$@)
	$(Q)$(CC) $(CFLAGS) $(INCLUDES) -c $(filter %.c,$^) -o $@

# SCX apps need scx headers for userspace compilation
$(patsubst %,$(OUTPUT)/%.o,$(SCX_APPS)): $(OUTPUT)/%.o: %.c $(wildcard %.h) | $(OUTPUT)
	$(call msg,SCX-CC,$@)
	$(Q)$(CC) $(CFLAGS) $(INCLUDES) $(SCX_INCLUDES) -c $(filter %.c,$^) -o $@

$(OUTPUT)/%.o: %.c $(wildcard %.h) | $(OUTPUT)
	$(call msg,CC,$@)
	$(Q)$(CC) $(CFLAGS) $(INCLUDES) -c $(filter %.c,$^) -o $@

$(patsubst %,$(OUTPUT)/%.o,$(BZS_APPS)): $(LIBBLAZESYM_HEADER)

$(BZS_APPS): $(LIBBLAZESYM_OBJ)

# Build application binary (exclude special apps from generic rule)
$(filter-out $(NVML_APPS) $(PTHREAD_APPS) $(CUDA_APPS),$(APPS)): %: $(OUTPUT)/%.o $(LIBBPF_OBJ) | $(OUTPUT)
	$(call msg,BINARY,$@)
	$(Q)$(CC) $(CFLAGS) $^ $(ALL_LDFLAGS) -lelf -lz -o $@

# Special rule for apps with NVML
$(NVML_APPS): %: $(OUTPUT)/%.o $(LIBBPF_OBJ) | $(OUTPUT)
	$(call msg,BINARY,$@)
	$(Q)$(CC) $(CFLAGS) $^ $(ALL_LDFLAGS) -lelf -lz -L$(CUDA_STUB_DIR) -lnvidia-ml -o $@

# Special rule for apps with pthread
$(PTHREAD_APPS): %: $(OUTPUT)/%.o $(LIBBPF_OBJ) | $(OUTPUT)
	$(call msg,BINARY,$@)
	$(Q)$(CC) $(CFLAGS) $^ $(ALL_LDFLAGS) -lelf -lz -lpthread -o $@

# Special rule for apps with CUDA + pthread
$(CUDA_APPS): %: $(OUTPUT)/%.o $(LIBBPF_OBJ) | $(OUTPUT)
	$(call msg,BINARY,$@)
	$(Q)$(CC) $(CFLAGS) $^ $(ALL_LDFLAGS) -lelf -lz -L$(CUDA_STUB_DIR) -lcuda -lpthread -o $@

# delete failed targets
.DELETE_ON_ERROR:

# keep intermediate (.skel.h, .bpf.o, etc) targets
.SECONDARY:
