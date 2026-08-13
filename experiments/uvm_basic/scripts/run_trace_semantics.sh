#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
POLICIES=(custom_no_policy prefetch_none prefetch_always_max prefetch_adaptive_sequential)

for policy in "${POLICIES[@]}"; do
    for run in $(seq 1 "${STAGE3_TIMING_RUNS:-5}"); do
        bash "${HERE}/run_stage3_case.sh" --experiment trace_semantics \
            --policy "${policy}" --kind timing --index "${run}"
    done
    for run in $(seq 1 "${STAGE3_TRACE_RUNS:-3}"); do
        bash "${HERE}/run_stage3_case.sh" --experiment trace_semantics \
            --policy "${policy}" --kind trace --index "${run}"
    done
    for run in $(seq 1 "${STAGE3_NSYS_RUNS:-1}"); do
        bash "${HERE}/run_stage3_case.sh" --experiment trace_semantics \
            --policy "${policy}" --kind nsys --index "${run}"
    done
done
