BPFTIME_ROOT ?= ../../../../bpftime-table1-575
BPFTIME_BUILD ?= $(BPFTIME_ROOT)/build-table1-575-strict
CLANG ?= clang
LIBBPF_INCLUDE := $(BPFTIME_BUILD)/libbpf
LIBBPF_STATIC := $(BPFTIME_BUILD)/libbpf/libbpf/libbpf.a
VMLINUX_INCLUDE := $(BPFTIME_ROOT)/third_party/vmlinux/x86
BPF_INCLUDES := -I$(LIBBPF_INCLUDE) -I$(VMLINUX_INCLUDE) \
	-I/usr/include/x86_64-linux-gnu

.PHONY: all parent-bench test
all: parent-bench .output/uniform-probe.bpf.o .output/uniform-probe

test:
	python3 -m unittest -v test_strict_uniform_map.py

parent-bench:
	$(MAKE) -C .. \
		BPFTIME_ROOT=$(abspath $(BPFTIME_ROOT)) \
		BPFTIME_BUILD=$(abspath $(BPFTIME_BUILD)) \
		.output/map-bench .output/map-bench.ptx

.output:
	mkdir -p $@

.output/uniform-probe.bpf.o: uniform_probe.bpf.c | .output
	$(CLANG) -g -O2 -Wall -Werror -target bpf -D__TARGET_ARCH_x86 \
		$(BPF_INCLUDES) -c $< -o $@

.output/uniform-probe: uniform_probe.c | .output
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -I$(LIBBPF_INCLUDE) \
		$< $(LIBBPF_STATIC) -lelf -lz -o $@
