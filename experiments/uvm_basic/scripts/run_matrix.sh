#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "$0")/common.sh"
build_uvm_basic

STAMP="$(timestamp_utc)"
OUTPUT="${RESULTS_DIR}/basic_${STAMP}.jsonl"
LOG="${RESULTS_DIR}/basic_${STAMP}.log"
SIZES=(256M 1G)
CASES=(
    "managed_none_no_prefetch|managed|none|no|no"
    "managed_page_retouch|managed|page|no|no"
    "managed_cpu_prefetch_page_retouch|managed|page|no|yes"
    "managed_gpu_prefetch|managed|none|yes|no"
    "device_baseline|device|none|no|no"
)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
: >"${OUTPUT}"
: >"${LOG}"

run_case() {
    local size="$1" spec="$2"
    local name allocation retouch gpu_prefetch cpu_prefetch case_log
    IFS='|' read -r name allocation retouch gpu_prefetch cpu_prefetch <<<"${spec}"
    case_log="${RESULTS_DIR}/${name}_${size}_${STAMP}.log"
    echo "===== ${name} ${size} =====" | tee -a "${LOG}"
    set +e
    timeout --signal=TERM --kill-after=10s "${UVM_BASIC_TIMEOUT_SECONDS:-300}" \
        "${PROGRAM}" --bytes "${size}" --allocation "${allocation}" --iterations 1 \
        --cpu-retouch "${retouch}" --gpu-prefetch "${gpu_prefetch}" \
        --cpu-prefetch-before-retouch "${cpu_prefetch}" --verify yes --output "${OUTPUT}" \
        2>&1 | tee "${case_log}" | tee -a "${LOG}"
    local rc=${PIPESTATUS[0]}
    set -e
    printf '%s\n' "${rc}" >"${case_log%.log}.exit"
    if (( rc != 0 )); then
        echo "FAIL ${name} ${size}: exit ${rc}" | tee -a "${LOG}"
        return "${rc}"
    fi
}

for size in "${SIZES[@]}"; do
    if ! safe_size_or_skip "${size}" 2>>"${LOG}"; then
        continue
    fi
    for spec in "${CASES[@]}"; do
        run_case "${size}" "${spec}"
    done
done

python3 "${UVM_BASIC_DIR}/analysis/summarize.py" --experiment-dir "${UVM_BASIC_DIR}"
echo "Matrix result: ${OUTPUT}"
