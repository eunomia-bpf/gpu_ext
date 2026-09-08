#!/usr/bin/env bash
set -euo pipefail
cd /home/yunwei37/workspace/gpu/gpu_ext/workloads/llama.cpp/observability_overhead/revision-rq4/raw/full-record-device-buffer-32bank-20260908.gaSN8I
mkdir -p .output
clang -g -O2 -target bpf -D__TARGET_ARCH_x86 \
  -I/home/yunwei37/workspace/gpu/bpftime-auto-warp/third_party/vmlinux/x86 \
  -I../table1-original-ring-encoded-20260908.iVdbES/gpubpf_tool_build/kernelretsnoop/.output \
  -I/usr/include/x86_64-linux-gnu \
  -c ../../full-record-device-buffer/full-record-device-buffer.bpf.c \
  -o .output/full-record-device-buffer.bpf.o
../table1-original-ring-encoded-20260908.iVdbES/gpubpf_tool_build/kernelretsnoop/.output/bpftool/bootstrap/bpftool \
  gen skeleton .output/full-record-device-buffer.bpf.o > .output/full-record-device-buffer.skel.h
cc -g -Wall -I. -I../../full-record-device-buffer \
  -I/usr/local/cuda-12.9/include \
  -I../table1-original-ring-encoded-20260908.iVdbES/gpubpf_tool_build/kernelretsnoop/.output \
  ../../full-record-device-buffer/full-record-device-buffer.c \
  ../table1-original-ring-encoded-20260908.iVdbES/gpubpf_tool_build/kernelretsnoop/.output/libbpf.a \
  -lcuda -lelf -lz -o kernelretsnoop
../table1-original-ring-encoded-20260908.iVdbES/gpubpf_tool_build/kernelretsnoop/.output/bpftool/bootstrap/bpftool \
  btf dump file .output/full-record-device-buffer.bpf.o format raw |
  sed -n '/STRUCT '\''frdb_value'\''/,+5p'
