#!/bin/bash
# Generate call graphs for GPU scheduling related functions
# Usage: ./gen_callgraph.sh
#
# Generates complete call graphs for each source file without specifying main

cd /home/yunwei37/workspace/gpu/open-gpu-kernel-modules

OUTPUT_DIR="docs/sched/callgraphs"
mkdir -p "$OUTPUT_DIR"

echo "=== GPU Scheduling Call Graph Generator ==="
echo ""

# Common cflow options
# -d 10: depth 10 levels
# -n: print line numbers
CFLOW_OPTS="-d 10 -n"

echo "Generating call graphs (no main specified, full module analysis)..."
echo ""

# 1. kernel_channel_group.c - TSG management (contains task_init, task_destroy hooks)
echo "1. kernel_channel_group.c (TSG init/destroy, task_init & task_destroy hooks)..."
cflow $CFLOW_OPTS \
    src/nvidia/src/kernel/gpu/fifo/kernel_channel_group.c \
    kernel-open/nvidia/nv-gpu-sched-hooks.c \
    2>/dev/null > "$OUTPUT_DIR/01_kernel_channel_group.txt"

# 2. kernel_channel_group_api.c - TSG API (contains schedule hook)
echo "2. kernel_channel_group_api.c (TSG API, schedule hook, timeslice, interleave)..."
cflow $CFLOW_OPTS \
    src/nvidia/src/kernel/gpu/fifo/kernel_channel_group_api.c \
    kernel-open/nvidia/nv-gpu-sched-hooks.c \
    2>/dev/null > "$OUTPUT_DIR/02_kernel_channel_group_api.txt"

# 3. kernel_channel.c - Channel management (contains token_request hook)
echo "3. kernel_channel.c (channel management, token_request hook)..."
cflow $CFLOW_OPTS \
    src/nvidia/src/kernel/gpu/fifo/kernel_channel.c \
    kernel-open/nvidia/nv-gpu-sched-hooks.c \
    2>/dev/null > "$OUTPUT_DIR/03_kernel_channel.txt"

# 4. kernel_fifo_ctrl.c - FIFO control (disable channels, idle)
echo "4. kernel_fifo_ctrl.c (FIFO control, disable channels)..."
cflow $CFLOW_OPTS \
    src/nvidia/src/kernel/gpu/fifo/kernel_fifo_ctrl.c \
    2>/dev/null > "$OUTPUT_DIR/04_kernel_fifo_ctrl.txt"

# 5. kernel_fifo.c - Core FIFO
echo "5. kernel_fifo.c (core FIFO, timeslice, runlist)..."
cflow $CFLOW_OPTS \
    src/nvidia/src/kernel/gpu/fifo/kernel_fifo.c \
    2>/dev/null > "$OUTPUT_DIR/05_kernel_fifo.txt"

# 6. Ampere FIFO (preemption)
echo "6. kernel_fifo_ga100.c (Ampere FIFO, hardware preempt)..."
cflow $CFLOW_OPTS \
    src/nvidia/src/kernel/gpu/fifo/arch/ampere/kernel_fifo_ga100.c \
    2>/dev/null > "$OUTPUT_DIR/06_kernel_fifo_ga100.txt"

# 7. Turing FIFO
echo "7. kernel_fifo_tu102.c (Turing FIFO)..."
cflow $CFLOW_OPTS \
    src/nvidia/src/kernel/gpu/fifo/arch/turing/kernel_fifo_tu102.c \
    2>/dev/null > "$OUTPUT_DIR/07_kernel_fifo_tu102.txt"

# 8. Hook implementation
echo "8. nv-gpu-sched-hooks.c (eBPF hook implementations)..."
cflow $CFLOW_OPTS \
    kernel-open/nvidia/nv-gpu-sched-hooks.c \
    2>/dev/null > "$OUTPUT_DIR/08_nv-gpu-sched-hooks.txt"

# 9. Channel descendant
echo "9. channel_descendant.c (channel object management)..."
cflow $CFLOW_OPTS \
    src/nvidia/src/kernel/gpu/fifo/channel_descendant.c \
    2>/dev/null > "$OUTPUT_DIR/09_channel_descendant.txt"

# 10. Idle channels
echo "10. kernel_idle_channels.c (idle channel management)..."
cflow $CFLOW_OPTS \
    src/nvidia/src/kernel/gpu/fifo/kernel_idle_channels.c \
    2>/dev/null > "$OUTPUT_DIR/10_kernel_idle_channels.txt"

echo ""
echo "=== Generated call graphs ==="
ls -lh "$OUTPUT_DIR"/*.txt

echo ""
echo "=== Line counts ==="
wc -l "$OUTPUT_DIR"/*.txt | sort -n

echo ""
echo "=== Functions containing 'nv_gpu_sched' (hook calls) ==="
grep -l "nv_gpu_sched" "$OUTPUT_DIR"/*.txt | while read f; do
    echo "--- $f ---"
    grep -n "nv_gpu_sched" "$f"
done
