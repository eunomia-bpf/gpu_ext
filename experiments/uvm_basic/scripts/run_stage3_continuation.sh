#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

[[ "${I_UNDERSTAND_STAGE3_CONTINUATION:-}" == YES ]] || {
    echo "Set I_UNDERSTAND_STAGE3_CONTINUATION=YES after reviewing the recorded prefetch_none 1.05x runtime limit." >&2
    exit 2
}

# This is bounded characterization after a recorded stop condition. It does not
# retry prefetch_none, relax the 300 second limit, or unlock Stage 3D.
run_case() {
    local ratio="$1" policy="$2" kind="$3" index="$4"
    bash "${HERE}/run_stage3_case.sh" --experiment oversub \
        --policy "${policy}" --kind "${kind}" --ratio "${ratio}" --index "${index}"
}

bash "${HERE}/check_oversub_safety.sh" 1.05
for policy in prefetch_always_max prefetch_adaptive_sequential; do
    for run in 1 2 3; do
        run_case 1.05 "${policy}" timing "${run}"
    done
    run_case 1.05 "${policy}" trace 1
done

bash "${HERE}/check_oversub_safety.sh" 1.10
for policy in custom_no_policy prefetch_always_max prefetch_adaptive_sequential; do
    for run in 1 2 3; do
        run_case 1.10 "${policy}" timing "${run}"
    done
    run_case 1.10 "${policy}" trace 1
    run_case 1.10 "${policy}" nsys 1
done

echo "PASS_STAGE3_BOUNDED_CONTINUATION"
