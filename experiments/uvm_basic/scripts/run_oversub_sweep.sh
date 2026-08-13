#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
POLICIES=(custom_no_policy prefetch_none prefetch_always_max prefetch_adaptive_sequential)
read -r -a RATIOS <<<"${STAGE3_OVERSUB_RATIOS:-0.95 1.05 1.10}"

for ratio in "${RATIOS[@]}"; do
    python3 - "${ratio}" <<'PY'
import os, sys
ratio = float(sys.argv[1])
if ratio > 1.25:
    raise SystemExit("absolute oversubscription ceiling is 1.25x")
if ratio > 1.15 and os.environ.get("GPU_EXT_ALLOW_HIGH_OVERSUB") != "1":
    raise SystemExit("ratios above 1.15x require GPU_EXT_ALLOW_HIGH_OVERSUB=1")
PY
    bash "${HERE}/check_oversub_safety.sh" "${ratio}"
    for policy in "${POLICIES[@]}"; do
        for run in $(seq 1 "${STAGE3_OVERSUB_TIMING_RUNS:-3}"); do
            bash "${HERE}/run_stage3_case.sh" --experiment oversub \
                --policy "${policy}" --kind timing --ratio "${ratio}" --index "${run}"
        done
        bash "${HERE}/run_stage3_case.sh" --experiment oversub \
            --policy "${policy}" --kind trace --ratio "${ratio}" --index 1
        if [[ "${ratio}" == "${STAGE3_NSYS_RATIO:-1.10}" ]]; then
            bash "${HERE}/run_stage3_case.sh" --experiment oversub \
                --policy "${policy}" --kind nsys --ratio "${ratio}" --index 1
        fi
    done
done
