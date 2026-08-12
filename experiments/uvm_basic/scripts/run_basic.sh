#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "$0")/common.sh"

SIZE="${UVM_BASIC_BYTES:-256M}"
ALLOCATION="${UVM_BASIC_ALLOCATION:-managed}"
ITERATIONS="${UVM_BASIC_ITERATIONS:-1}"
CPU_RETOUCH="${UVM_BASIC_CPU_RETOUCH:-none}"
GPU_PREFETCH="${UVM_BASIC_GPU_PREFETCH:-no}"
CPU_PREFETCH="${UVM_BASIC_CPU_PREFETCH_BEFORE_RETOUCH:-no}"
OUTPUT="${UVM_BASIC_OUTPUT:-${RESULTS_DIR}/basic_$(timestamp_utc).jsonl}"

command -v nvidia-smi >/dev/null
command -v nvcc >/dev/null
safe_size_or_skip "${SIZE}"
build_uvm_basic

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
timeout --signal=TERM --kill-after=10s "${UVM_BASIC_TIMEOUT_SECONDS:-300}" \
    "${PROGRAM}" \
    --bytes "${SIZE}" \
    --allocation "${ALLOCATION}" \
    --iterations "${ITERATIONS}" \
    --cpu-retouch "${CPU_RETOUCH}" \
    --gpu-prefetch "${GPU_PREFETCH}" \
    --cpu-prefetch-before-retouch "${CPU_PREFETCH}" \
    --verify yes \
    --output "${OUTPUT}"

echo "Result: ${OUTPUT}"
