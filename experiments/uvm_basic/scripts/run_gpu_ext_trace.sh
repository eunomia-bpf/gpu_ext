#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "$0")/common.sh"
EXTENSION_DIR="${GPU_EXT_ROOT}/extension"
POLICIES=(baseline prefetch_none prefetch_always_max prefetch_adaptive_sequential)
TRACE_PIDS=()
POLICY_PID=""

cleanup_case() {
    local pid
    if [[ -n "${POLICY_PID}" ]] && kill -0 "${POLICY_PID}" 2>/dev/null; then
        kill -TERM "${POLICY_PID}" 2>/dev/null || true
        wait "${POLICY_PID}" 2>/dev/null || true
    fi
    POLICY_PID=""
    for pid in "${TRACE_PIDS[@]:-}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill -TERM "${pid}" 2>/dev/null || true
            wait "${pid}" 2>/dev/null || true
        fi
    done
    TRACE_PIDS=()
}
trap cleanup_case EXIT INT TERM

if ! grep -q 'uvm_bpf_call_gpu_page_prefetch' /proc/kallsyms 2>/dev/null; then
    echo "The loaded nvidia_uvm module does not expose the gpu_ext prefetch hook." >&2
    echo "Loaded module: $(modinfo -n nvidia_uvm 2>/dev/null || echo UNAVAILABLE)" >&2
    echo "No policy or trace was attached." >&2
    echo "Inspect first: bash ${UVM_BASIC_DIR}/scripts/SAFE_GPU_EXT_COMMANDS.sh inspect" >&2
    exit 2
fi
if [[ ${EUID} -ne 0 ]]; then
    echo "The gpu_ext hook is visible, but root BPF capability is required to attach tracing." >&2
    echo "No policy or trace was attached." >&2
    exit 2
fi
if bpftool map show 2>/dev/null | grep -q 'struct_ops'; then
    echo "An existing struct_ops map is present; refusing to detach or replace an unrelated policy." >&2
    exit 2
fi

build_uvm_basic
make -C "${EXTENSION_DIR}" -j"${JOBS:-4}" \
    prefetch_trace chunk_trace prefetch_none prefetch_always_max prefetch_adaptive_sequential

for policy in "${POLICIES[@]}"; do
    STAMP="$(timestamp_utc)"
    PREFETCH_CSV="${RESULTS_DIR}/prefetch_trace_${policy}_${STAMP}.csv"
    CHUNK_CSV="${RESULTS_DIR}/chunk_trace_${policy}_${STAMP}.csv"
    PROGRAM_JSON="${RESULTS_DIR}/program_${policy}_${STAMP}.jsonl"
    cleanup_case
    "${EXTENSION_DIR}/prefetch_trace" >"${PREFETCH_CSV}" 2>"${PREFETCH_CSV%.csv}.stderr" &
    TRACE_PIDS+=("$!")
    "${EXTENSION_DIR}/chunk_trace" >"${CHUNK_CSV}" 2>"${CHUNK_CSV%.csv}.stderr" &
    TRACE_PIDS+=("$!")
    if [[ "${policy}" != baseline ]]; then
        "${EXTENSION_DIR}/${policy}" >"${RESULTS_DIR}/policy_${policy}_${STAMP}.stdout" \
            2>"${RESULTS_DIR}/policy_${policy}_${STAMP}.stderr" &
        POLICY_PID="$!"
        sleep 1
        kill -0 "${POLICY_PID}"
    fi
    timeout --signal=TERM --kill-after=10s "${UVM_BASIC_TIMEOUT_SECONDS:-300}" \
        "${PROGRAM}" --bytes "${UVM_BASIC_TRACE_BYTES:-256M}" --allocation managed \
        --iterations 1 --cpu-retouch page --gpu-prefetch yes \
        --cpu-prefetch-before-retouch no --verify yes --output "${PROGRAM_JSON}"
    cleanup_case
done

python3 "${UVM_BASIC_DIR}/analysis/summarize.py" --experiment-dir "${UVM_BASIC_DIR}"
