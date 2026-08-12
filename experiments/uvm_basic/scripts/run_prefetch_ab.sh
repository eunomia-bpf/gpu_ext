#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "$0")/common.sh"

command -v nvidia-smi >/dev/null
command -v nvcc >/dev/null
build_uvm_basic

OUT_DIR="${RESULTS_DIR}/prefetch_ab"
mkdir -p "${OUT_DIR}"
STAMP="$(timestamp_utc)"
MANIFEST="${OUT_DIR}/run_${STAMP}.tsv"
printf 'size\tcase\trepetition\tresult\texit_code\txid_before\txid_after\n' >"${MANIFEST}"

run_one() {
    local size="$1" case_name="$2" repetition="$3"
    local run_dir="${OUT_DIR}/${size}/${case_name}"
    local base="${run_dir}/${case_name}_${STAMP}_r${repetition}"
    local output="${base}.jsonl" rc
    mkdir -p "${run_dir}"
    capture_kernel_log "${base}.dmesg_before.txt"
    printf '%q ' "${PROGRAM}" --bytes "${size}" --allocation managed --iterations 1 \
        --cpu-retouch page --after-retouch "${case_name}" --stop-after-hot no \
        --gpu-prefetch no --cpu-prefetch-before-retouch no --verify yes --output "${output}" \
        >"${base}.command.txt"
    printf '\n' >>"${base}.command.txt"
    set +e
    timeout --signal=TERM --kill-after=15s "${UVM_BASIC_TIMEOUT_SECONDS:-300}" \
        "${PROGRAM}" --bytes "${size}" --allocation managed --iterations 1 \
        --cpu-retouch page --after-retouch "${case_name}" --stop-after-hot no \
        --gpu-prefetch no --cpu-prefetch-before-retouch no --verify yes --output "${output}" \
        >"${base}.stdout" 2>"${base}.stderr"
    rc=$?
    set -e
    printf '%s\n' "${rc}" >"${base}.exit_code"
    capture_kernel_log "${base}.dmesg_after.txt"
    local xid_before xid_after
    xid_before="$(xid_count_or_unavailable "${base}.dmesg_before.txt")"
    xid_after="$(xid_count_or_unavailable "${base}.dmesg_after.txt")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${size}" "${case_name}" "${repetition}" "${output}" "${rc}" \
        "${xid_before}" "${xid_after}" >>"${MANIFEST}"
    if (( rc != 0 )); then
        echo "A/B run failed: size=${size} case=${case_name} repetition=${repetition}" >&2
        return "${rc}"
    fi
    if [[ "${xid_before}" != UNAVAILABLE && "${xid_after}" != UNAVAILABLE ]] &&
       (( xid_after > xid_before )); then
        echo "New NVIDIA Xid detected; stopping A/B matrix." >&2
        return 1
    fi
}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
for specification in "256M:${UVM_BASIC_AB_256_REPEATS:-5}" "1G:${UVM_BASIC_AB_1G_REPEATS:-3}"; do
    size="${specification%%:*}"
    repeats="${specification##*:}"
    if ! safe_size_or_skip "${size}"; then
        printf '%s\tSKIPPED\t0\tUNAVAILABLE\t0\tUNAVAILABLE\tUNAVAILABLE\n' "${size}" >>"${MANIFEST}"
        continue
    fi
    for case_name in demand prefetch; do
        for ((repetition = 1; repetition <= repeats; ++repetition)); do
            run_one "${size}" "${case_name}" "${repetition}"
        done
    done
done

python3 "${UVM_BASIC_DIR}/analysis/summarize_prefetch_ab.py" --experiment-dir "${UVM_BASIC_DIR}"
echo "A/B manifest: ${MANIFEST}"
