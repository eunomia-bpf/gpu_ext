#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SAFE="${ROOT}/scripts/SAFE_STAGE4_COMMANDS.sh"
[[ -f "${SAFE}" ]] || { echo "Missing SAFE_STAGE4_COMMANDS.sh" >&2; exit 1; }
[[ ! -x "${SAFE}" ]] || { echo "SAFE_STAGE4_COMMANDS.sh must remain non-executable" >&2; exit 1; }

EXECUTABLES=(
    "${ROOT}/scripts/stage4_common.sh"
    "${ROOT}/scripts/check_stage4.sh"
    "${ROOT}/scripts/check_stage4_runtime_safety.sh"
    "${ROOT}/scripts/run_stage4_case.sh"
    "${ROOT}/scripts/run_reduced_capacity_calibration.sh"
    "${ROOT}/scripts/run_reduced_capacity_prefetch_matrix.sh"
    "${ROOT}/scripts/run_eviction_smoke.sh"
    "${ROOT}/scripts/run_joint_policy_matrix.sh"
    "${ROOT}/scripts/run_natural_capacity_confirmation.sh"
    "${ROOT}/scripts/measure_trace_disabled_overhead.sh"
)
for file in "${EXECUTABLES[@]}"; do
    if rg -n '(^|[;&|[:space:]])(sudo|rmmod|insmod|modprobe)([[:space:]]|$)|pkill[[:space:]]+-f|make[[:space:]]+modules_install|/lib/modules/.+\.ko' "${file}"; then
        echo "Forbidden privileged or broad-cleanup command in ${file}" >&2
        exit 1
    fi
    if rg -n 'timeout[^\n]*(30[1-9]|3[1-9][0-9]|[4-9][0-9]{2,})s' "${file}"; then
        echo "Stage 4 timeout exceeds 300 seconds in ${file}" >&2
        exit 1
    fi
done

if rg -n 'pkill[[:space:]]+-f|make[[:space:]]+modules_install|cp[^\n]+/lib/modules' "${SAFE}"; then
    echo "Forbidden operation in SAFE_STAGE4_COMMANDS.sh" >&2
    exit 1
fi

echo "PASS_STAGE4_STATIC_SAFETY"
