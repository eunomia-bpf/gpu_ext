BPFTIME_ROOT ?= ../../../bpftime-table1-575
BPFTIME_BUILD ?= $(BPFTIME_ROOT)/build-table1-575
CLANG ?= clang
NVCC ?= /usr/local/cuda-12.9/bin/nvcc
LIBBPF := $(BPFTIME_BUILD)/libbpf/libbpf
BPF_INCLUDES := -I$(LIBBPF)/include -I$(BPFTIME_ROOT)/third_party/vmlinux/x86 -I/usr/include/x86_64-linux-gnu
C_INCLUDES := -I$(LIBBPF)/include -I$(BPFTIME_ROOT)/runtime/include

.PHONY: all test-offline clean
all: .output/raw_map_probe .output/raw_map.bpf.o .output/raw_map_target

.output:
	mkdir -p $@

.output/raw_map.bpf.o: raw_map.bpf.c | .output
	$(CLANG) -g -O2 -target bpf -D__TARGET_ARCH_x86 $(BPF_INCLUDES) -c $< -o $@

.output/raw_map_probe: raw_map_probe.c | .output
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror $(C_INCLUDES) $< $(LIBBPF)/libbpf.a -lelf -lz -ldl -o $@

.output/raw_map_target: raw_map_target.cu | .output
	$(NVCC) -std=c++17 -O2 -cudart shared \
		-gencode arch=compute_120,code=sm_120 \
		-gencode arch=compute_120,code=compute_120 \
		$< -o $@ -Xlinker --build-id=none

test-offline: all
	python3 -m unittest -v test_offline.py
	python3 run_raw_map.py dry-run --plan-mode preflight --output /does/not/exist/preflight \
		--runtime-build /does/not/exist/runtime >/dev/null
	python3 run_raw_map.py dry-run --plan-mode full --output /does/not/exist/full \
		--runtime-build /does/not/exist/runtime --preflight /does/not/exist/preflight >/dev/null

clean:
	rm -rf .output __pycache__
