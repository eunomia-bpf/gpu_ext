#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

STAGE3_RESULTS="${RESULTS_DIR}/stage3"
EXTENSION_DIR="${GPU_EXT_ROOT}/extension"
CUSTOM_UVM="${GPU_EXT_CUSTOM_UVM:-/home/peng/workspace/gpu_ext_private/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko}"
mkdir -p "${STAGE3_RESULTS}"

resolve_stage3_bpftool() {
    local candidate
    if [[ -n "${GPU_EXT_BPFTOOL:-}" && -x "${GPU_EXT_BPFTOOL}" ]]; then
        STAGE3_BPFTOOL="$(readlink -f "${GPU_EXT_BPFTOOL}")"
        return
    fi
    for candidate in \
        "${GPU_EXT_ROOT}/tools/bpftool-stage2/bpftool" \
        /usr/lib/linux-hwe-*-tools-*/bpftool \
        /usr/lib/linux-tools/*/bpftool; do
        if [[ -x "${candidate}" ]] && "${candidate}" -j map show >/dev/null 2>&1; then
            STAGE3_BPFTOOL="$(readlink -f "${candidate}")"
            return
        fi
    done
    echo "No usable bpftool found." >&2
    return 2
}

stage3_struct_ops_snapshot() {
    "${STAGE3_BPFTOOL}" -j map show >"$1"
}

stage3_has_struct_ops() {
    grep -Eiq '"type"[[:space:]]*:[[:space:]]*"struct_ops"' "$1"
}

stage3_custom_module_loaded() {
    [[ -r "${CUSTOM_UVM}" && -r /sys/module/nvidia_uvm/srcversion ]] || return 1
    [[ "$(cat /sys/module/nvidia_uvm/srcversion)" == "$(modinfo -F srcversion "${CUSTOM_UVM}")" ]] || return 1
    grep -q 'uvm_bpf_trace_gpu_page_prefetch_decision' /proc/kallsyms 2>/dev/null
}

stage3_require_runtime() {
    [[ ${EUID} -eq 0 ]] || { echo "Stage 3 BPF runtime requires the reviewed SAFE_STAGE3_COMMANDS.sh path." >&2; return 2; }
    stage3_custom_module_loaded || { echo "Enhanced custom nvidia_uvm is not loaded." >&2; return 2; }
    resolve_stage3_bpftool
}

stage3_build() {
    build_uvm_basic
    make -C "${EXTENSION_DIR}" -j"${JOBS:-4}" \
        prefetch_trace chunk_trace prefetch_none prefetch_always_max \
        prefetch_adaptive_sequential eviction_fifo \
        prefetch_always_max_cycle_moe prefetch_cooperative
}

stage3_policy_binary() {
    case "$1" in
        custom_no_policy) printf '\n' ;;
        prefetch_none|prefetch_always_max|prefetch_adaptive_sequential|eviction_fifo|prefetch_always_max_cycle_moe|prefetch_cooperative)
            printf '%s/%s\n' "${EXTENSION_DIR}" "$1" ;;
        *) echo "Unsupported Stage 3 policy: $1" >&2; return 2 ;;
    esac
}

stage3_monotonic_ns() {
    python3 -c 'import time; print(time.monotonic_ns())'
}
