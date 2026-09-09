#!/usr/bin/env bash
set -euo pipefail
cd /home/yunwei37/workspace/gpu/gpu_ext
date --iso-8601=seconds
bash scripts/artifact/build_table1_runtime.sh \
  --output-dir /tmp/gpubpf-table1-runtime-20260909.X8d8Ho \
  --source-rev eef8a51abaf2ca1f0cdca9f2425af3bd535da1b7 \
  --jobs 2
date --iso-8601=seconds
