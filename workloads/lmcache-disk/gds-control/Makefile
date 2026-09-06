# Builds the GDS storage-policy BPF object, the libbpf struct_ops loader,
# and the nvidia_uvm gpu-storage-decide ioctl probe.

CLANG ?= clang
CC    ?= gcc
RM    ?= rm -f

CFLAGS     ?= -O2 -g -Wall -Wextra
BPF_CFLAGS := -g -O2 -target bpf -D__TARGET_ARCH_x86 \
	      -I../../../vmlinux/x86 -I../../../libbpf/src

VMLINUX_DIR := ../../../vmlinux/x86
LIBBPF_DIR  := ../../../libbpf/src
LIBBPF_A    := $(LIBBPF_DIR)/libbpf.a

all: gds_policy.bpf.o gds_policy ioctl_probe

# Tree libbpf (built in place; no libbpf/bootstrap bpftool needed).
$(LIBBPF_A): $(wildcard $(LIBBPF_DIR)/*.[ch]) $(wildcard $(LIBBPF_DIR)/include/*.h)
	$(MAKE) -C $(LIBBPF_DIR)

gds_policy.bpf.o: gds_policy.bpf.c $(VMLINUX_DIR)/vmlinux.h
	$(CLANG) $(BPF_CFLAGS) -c $< -o $@

gds_policy: gds_policy.c $(LIBBPF_A)
	$(CC) $(CFLAGS) -I$(LIBBPF_DIR) -o $@ $< $(LIBBPF_A) -lelf -lz

ioctl_probe: ioctl_probe.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	$(RM) gds_policy.bpf.o gds_policy ioctl_probe

.PHONY: all clean
