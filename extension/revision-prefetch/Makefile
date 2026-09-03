# Independent outputs; use existing libbpf/bpftool, never rebuild the driver.
CLANG ?= clang
CC ?= cc
OUT := build
LIBBPF := ../.output/libbpf.a
BPFTOOL := ../.output/bpftool/bootstrap/bpftool
INCLUDES := -I../.output -I../../libbpf/include/uapi -I../../vmlinux/x86 -I..

.PHONY: all
all: $(OUT)/prefetch_safety

$(OUT):
	mkdir -p $@

$(OUT)/fixture.tmp.bpf.o: fixture.bpf.c fixture.h ../uvm_types.h ../bpf_testmod.h | $(OUT)
	$(CLANG) -g -O2 -target bpf -D__TARGET_ARCH_x86 $(INCLUDES) \
	  -I/usr/include/x86_64-linux-gnu -c $< -o $@

$(OUT)/fixture.bpf.o: $(OUT)/fixture.tmp.bpf.o $(BPFTOOL)
	$(BPFTOOL) gen object $@ $<

$(OUT)/fixture.skel.h: $(OUT)/fixture.bpf.o
	$(BPFTOOL) gen skeleton $< > $@

$(OUT)/prefetch_safety: loader.c fixture.h $(OUT)/fixture.skel.h $(LIBBPF)
	$(CC) -g -O2 -Wall -Wextra $(INCLUDES) -I$(OUT) loader.c $(LIBBPF) \
	  -lelf -lz -Wl,--build-id=none -o $@
