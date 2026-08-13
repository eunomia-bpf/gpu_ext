#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-initial}"

case "${MODE}" in
initial)
    bash "${HERE}/run_trace_semantics.sh"
    bash "${HERE}/run_cpu_first_touch_diagnosis.sh"
    bash "${HERE}/run_array_migration_diagnosis.sh"
    ;;
oversub)
    bash "${HERE}/run_oversub_sweep.sh"
    ;;
joint)
    [[ "${GPU_EXT_STAGE3C_STABLE:-}" == YES ]] || {
        echo "Set GPU_EXT_STAGE3C_STABLE=YES only after 1.05x and 1.10x pass without Xid or residue." >&2
        exit 2
    }
    POLICIES=(custom_no_policy prefetch_always_max prefetch_always_max_cycle_moe)
    for policy in "${POLICIES[@]}"; do
        for run in 1 2 3; do
            bash "${HERE}/run_stage3_case.sh" --experiment joint_policy \
                --policy "${policy}" --kind timing --ratio 1.10 --index "${run}"
        done
        bash "${HERE}/run_stage3_case.sh" --experiment joint_policy \
            --policy "${policy}" --kind trace --ratio 1.10 --index 1
        bash "${HERE}/run_stage3_case.sh" --experiment joint_policy \
            --policy "${policy}" --kind nsys --ratio 1.10 --index 1
    done
    ;;
*) echo "Usage: $0 initial|oversub|joint" >&2; exit 2 ;;
esac
