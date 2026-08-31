#!/usr/bin/env bash
# =============================================================================
# RQ4: llama.cpp device-side observability overhead
#
# Runs llama-bench prefill with no probe, then with kernelretsnoop, threadhist,
# and launchlate attached to a selected llama.cpp CUDA kernel.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"
uv run python observability_overhead/run_observability_overhead.py "$@"
