#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "$0")/common.sh"
command -v nsys >/dev/null || { echo "Nsight Systems is unavailable" >&2; exit 2; }
build_uvm_basic

SIZE="${UVM_BASIC_AB_PROFILE_BYTES:-256M}"
safe_size_or_skip "${SIZE}"
OUT_DIR="${RESULTS_DIR}/prefetch_ab"
mkdir -p "${OUT_DIR}"
HELP="${OUT_DIR}/nsys_profile_help_$(timestamp_utc).txt"
nsys profile --help >"${HELP}" 2>&1

profile_one() {
    local case_name="$1" stamp prefix jsonl rep sqlite phase
    stamp="$(timestamp_utc)"
    prefix="${OUT_DIR}/${case_name}_${stamp}"
    jsonl="${prefix}.jsonl"
    capture_kernel_log "${prefix}.dmesg_before.txt"
    local options=(--trace=cuda,nvtx,osrt --force-overwrite=true --output="${prefix}")
    if grep -q -- '--cuda-um-cpu-page-faults=' "${HELP}" &&
       grep -q -- '--cuda-um-gpu-page-faults=' "${HELP}"; then
        options+=(--cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true)
    fi
    if grep -q -- '--cuda-memory-usage=' "${HELP}"; then
        options+=(--cuda-memory-usage=true)
    fi
    printf '%q ' nsys profile "${options[@]}" "${PROGRAM}" --bytes "${SIZE}" \
        --allocation managed --iterations 1 --cpu-retouch page --after-retouch "${case_name}" \
        --stop-after-hot no --gpu-prefetch no --cpu-prefetch-before-retouch no \
        --verify yes --output "${jsonl}" >"${prefix}.command.txt"
    printf '\n' >>"${prefix}.command.txt"
    timeout --signal=TERM --kill-after=20s "${UVM_BASIC_PROFILE_TIMEOUT_SECONDS:-600}" \
        nsys profile "${options[@]}" "${PROGRAM}" --bytes "${SIZE}" \
        --allocation managed --iterations 1 --cpu-retouch page --after-retouch "${case_name}" \
        --stop-after-hot no --gpu-prefetch no --cpu-prefetch-before-retouch no \
        --verify yes --output "${jsonl}" >"${prefix}.stdout" 2>"${prefix}.stderr"
    rep="${prefix}.nsys-rep"
    [[ -s "${rep}" ]] || { echo "Missing Nsight report: ${rep}" >&2; return 1; }
    nsys stats --force-export=true --force-overwrite=true --format csv \
        --report um_sum,um_total_sum,um_cpu_page_faults_sum,cuda_gpu_kern_sum,nvtx_sum \
        --output "${prefix}_stats" "${rep}" \
        >"${prefix}_stats.stdout" 2>"${prefix}_stats.stderr"
    sqlite="${prefix}.sqlite"
    [[ -s "${sqlite}" ]] || { echo "Missing Nsight SQLite export: ${sqlite}" >&2; return 1; }
    for phase in cpu_first_touch kernel_1_demand kernel_2_hot cpu_prefetch_to_cpu \
                 cpu_retouch gpu_prefetch_after_retouch \
                 kernel_after_retouch_demand kernel_after_retouch_prefetch; do
        nsys stats --force-overwrite=true --format csv --report um_total_sum \
            --filter-nvtx "${phase}" --output "${prefix}_phase_${phase}" "${sqlite}" \
            >>"${prefix}_stats.stdout" 2>>"${prefix}_stats.stderr" || true
    done
    capture_kernel_log "${prefix}.dmesg_after.txt"
    local before after
    before="$(xid_count_or_unavailable "${prefix}.dmesg_before.txt")"
    after="$(xid_count_or_unavailable "${prefix}.dmesg_after.txt")"
    printf 'xid_before=%s\nxid_after=%s\n' "${before}" "${after}" >"${prefix}.xid_status"
    if [[ "${before}" != UNAVAILABLE && "${after}" != UNAVAILABLE ]] && (( after > before )); then
        echo "New NVIDIA Xid detected after ${case_name} profile." >&2
        return 1
    fi
}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
profile_one demand
sleep 1
profile_one prefetch
python3 "${UVM_BASIC_DIR}/analysis/summarize_prefetch_ab.py" --experiment-dir "${UVM_BASIC_DIR}"
echo "Nsight A/B reports: ${OUT_DIR}/demand_*.nsys-rep ${OUT_DIR}/prefetch_*.nsys-rep"
