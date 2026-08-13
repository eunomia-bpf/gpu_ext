#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
POLICIES=(custom_no_policy prefetch_none prefetch_always_max prefetch_adaptive_sequential)
CONDITIONS=(full:no page:no full:yes)

for policy in "${POLICIES[@]}"; do
    for condition in "${CONDITIONS[@]}"; do
        pattern="${condition%%:*}"
        prefetch="${condition##*:}"
        for run in $(seq 1 "${STAGE3_FIRST_TOUCH_RUNS:-10}"); do
            bash "${HERE}/run_stage3_case.sh" --experiment cpu_first_touch \
                --policy "${policy}" --kind trace --index "${run}" \
                --first-touch "${pattern}" --prefetch-cpu "${prefetch}"
        done
    done
done
