#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "$0")/stage4_common.sh"
for kind in timing trace; do
    runs=20
    for index in $(seq 1 "${runs}"); do
        STAGE3_RESULTS="${STAGE4_RESULTS}" GPU_EXT_RUN_EVIDENCE_CLASS=GPU_EXT_STAGE4_TRACE_OVERHEAD \
            bash "$(dirname "$0")/run_stage3_case.sh" --experiment trace_overhead \
            --policy custom_no_policy --kind "${kind}" --ratio 256M --index "${index}"
    done
done
python3 "${UVM_BASIC_DIR}/analysis/summarize_stage4.py" --results "${STAGE4_RESULTS}"
