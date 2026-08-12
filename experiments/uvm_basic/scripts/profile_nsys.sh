#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "$0")/common.sh"
command -v nsys >/dev/null || { echo "Nsight Systems is unavailable" >&2; exit 2; }
build_uvm_basic

SIZE="${UVM_BASIC_PROFILE_BYTES:-256M}"
safe_size_or_skip "${SIZE}"
STAMP="$(timestamp_utc)"
PREFIX="${RESULTS_DIR}/uvm_basic_${STAMP}"
JSONL="${RESULTS_DIR}/profile_${STAMP}.jsonl"
HELP="${RESULTS_DIR}/nsys_profile_help_${STAMP}.txt"

nsys profile --help >"${HELP}" 2>&1
PROFILE_OPTIONS=(--trace=cuda,nvtx,osrt --force-overwrite=true --output="${PREFIX}")
FAULT_OPTIONS_SUPPORTED=no
if grep -q -- '--cuda-um-cpu-page-faults=' "${HELP}" &&
   grep -q -- '--cuda-um-gpu-page-faults=' "${HELP}"; then
    PROFILE_OPTIONS+=(--cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true)
    FAULT_OPTIONS_SUPPORTED=yes
fi
if grep -q -- '--cuda-memory-usage=' "${HELP}"; then
    PROFILE_OPTIONS+=(--cuda-memory-usage=true)
fi

printf 'fault_options_supported=%s\n' "${FAULT_OPTIONS_SUPPORTED}" >"${RESULTS_DIR}/nsys_capability_${STAMP}.txt"
timeout --signal=TERM --kill-after=20s "${UVM_BASIC_PROFILE_TIMEOUT_SECONDS:-600}" \
    nsys profile "${PROFILE_OPTIONS[@]}" \
    "${PROGRAM}" --bytes "${SIZE}" --allocation managed --iterations 1 \
    --cpu-retouch page --gpu-prefetch yes --cpu-prefetch-before-retouch no \
    --verify yes --output "${JSONL}"

REP="${PREFIX}.nsys-rep"
[[ -s "${REP}" ]] || { echo "Nsight report was not created: ${REP}" >&2; exit 1; }
if [[ "${FAULT_OPTIONS_SUPPORTED}" == yes ]]; then
    nsys stats --force-export=true --force-overwrite=true --format csv \
        --report um_sum,um_total_sum,um_cpu_page_faults_sum,cuda_gpu_kern_sum,nvtx_sum \
        --output "${PREFIX}_stats" "${REP}" \
        >"${PREFIX}_stats.stdout" 2>"${PREFIX}_stats.stderr"
    SQLITE="${PREFIX}.sqlite"
    [[ -s "${SQLITE}" ]] || { echo "Nsight SQLite export is missing: ${SQLITE}" >&2; exit 1; }
    for phase in cpu_first_touch kernel_1_demand kernel_2_hot cpu_retouch \
                 kernel_3_after_cpu_touch explicit_gpu_prefetch kernel_4_after_gpu_prefetch; do
        nsys stats --force-overwrite=true --format csv --report um_total_sum \
            --filter-nvtx "${phase}" --output "${PREFIX}_phase_${phase}" "${SQLITE}" \
            >>"${PREFIX}_stats.stdout" 2>>"${PREFIX}_stats.stderr"
    done
else
    echo "Unified Memory fault options are unavailable in this Nsight Systems version." \
        >"${PREFIX}_stats.stderr"
fi

python3 "${UVM_BASIC_DIR}/analysis/summarize.py" --experiment-dir "${UVM_BASIC_DIR}"
echo "Nsight report: ${REP}"
