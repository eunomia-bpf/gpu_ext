#!/bin/bash
# cflow 示例脚本 - 生成各种调用图

cd "$(dirname "$0")"

echo "=== Checking if cflow is installed ==="
if ! command -v cflow &> /dev/null; then
    echo "ERROR: cflow is not installed"
    echo "Please run: sudo apt-get install cflow"
    exit 1
fi

echo "cflow version: $(cflow --version | head -1)"
echo ""

# 创建输出目录
mkdir -p cflow_output

echo "=== Example 1: Fault Service Call Graph ==="
cflow --tree --number --depth=5 \
      --main=uvm_parent_gpu_service_replayable_faults \
      uvm_gpu_replayable_faults.c \
      > cflow_output/fault_service.txt
echo "Generated: cflow_output/fault_service.txt"
echo "Preview:"
head -30 cflow_output/fault_service.txt
echo ""

echo "=== Example 2: Reverse Call Graph (Who calls replay?) ==="
cflow --reverse --tree \
      uvm_gpu_replayable_faults.c | \
      grep -A 15 "push_replay_on_parent_gpu" \
      > cflow_output/replay_callers.txt
echo "Generated: cflow_output/replay_callers.txt"
echo "Preview:"
head -20 cflow_output/replay_callers.txt
echo ""

echo "=== Example 3: Thrashing Detection ==="
cflow --tree --number --depth=4 \
      --main=uvm_perf_thrashing_get_hint \
      uvm_perf_thrashing.c \
      > cflow_output/thrashing.txt 2>/dev/null
echo "Generated: cflow_output/thrashing.txt"
echo "Preview:"
head -20 cflow_output/thrashing.txt
echo ""

echo "=== Example 4: Simple Function List ==="
cflow --omit-arguments \
      uvm_gpu_replayable_faults.c | \
      head -50 \
      > cflow_output/function_list.txt
echo "Generated: cflow_output/function_list.txt"
echo "Preview:"
head -20 cflow_output/function_list.txt
echo ""

echo "=== Example 5: Policy-related Functions ==="
cflow --omit-arguments *.c | \
      grep -i "policy\|replay\|thrash\|prefetch" | \
      sort -u \
      > cflow_output/policy_functions.txt
echo "Generated: cflow_output/policy_functions.txt"
echo "Preview:"
head -20 cflow_output/policy_functions.txt
echo ""

echo "=== All outputs saved to cflow_output/ ==="
ls -lh cflow_output/

echo ""
echo "=== View files with: ==="
echo "less cflow_output/fault_service.txt"
echo "less cflow_output/replay_callers.txt"
echo "less cflow_output/thrashing.txt"
