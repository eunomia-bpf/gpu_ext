#!/usr/bin/env bash
set -euo pipefail
cd /home/yunwei37/workspace/gpu/gpu_ext
date --iso-8601=seconds
export UV_CACHE_DIR
UV_CACHE_DIR=$(mktemp -d /tmp/gpubpf-table1-uv.XXXXXX)
uv run --directory /home/yunwei37/workspace/gpu/gpu_ext/workloads/llama.cpp --no-sync python -B \
  /home/yunwei37/workspace/gpu/gpu_ext/workloads/llama.cpp/observability_overhead/revision-rq4/run_table1_perf.py \
  --build-only --output-dir /tmp/gpubpf-table1-components-20260909.JPYk6U \
  --bpftime-root /home/yunwei37/workspace/gpu/bpftime-table1-hostfix-plt \
  --bpftime-build-dir /home/yunwei37/workspace/gpu/bpftime-table1-hostfix-plt/build-table1-575-warp \
  --nvbit-root /home/yunwei37/workspace/gpu/gpu_ext/workloads/llama.cpp/observability_overhead/revision-rq4/deps/nvbit_release_x86_64
date --iso-8601=seconds
