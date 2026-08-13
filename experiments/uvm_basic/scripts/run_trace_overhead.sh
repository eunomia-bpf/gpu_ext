#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
for run in $(seq 1 "${STAGE3_OVERHEAD_RUNS:-10}"); do
    bash "${HERE}/run_stage3_case.sh" --experiment trace_overhead \
        --policy custom_no_policy --kind timing --index "${run}"
done
for run in $(seq 1 "${STAGE3_OVERHEAD_RUNS:-10}"); do
    bash "${HERE}/run_stage3_case.sh" --experiment trace_overhead \
        --policy custom_no_policy --kind trace --index "${run}"
done
