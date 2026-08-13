#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
POLICIES=(custom_no_policy prefetch_none prefetch_always_max prefetch_adaptive_sequential)
MODES=(read-a read-b write-c vector-add)

for policy in "${POLICIES[@]}"; do
    for mode in "${MODES[@]}"; do
        bash "${HERE}/run_stage3_case.sh" --experiment array_migration \
            --policy "${policy}" --kind nsys --index 1 --kernel-mode "${mode}"
        bash "${HERE}/run_stage3_case.sh" --experiment array_migration \
            --policy "${policy}" --kind trace --index 1 --kernel-mode "${mode}"
    done
done
