#!/usr/bin/env bash
set -Eeuo pipefail

cat <<'WARNING'
WARNING: Stage 3 temporarily replaces only nvidia_uvm and attaches BPF struct_ops.
It can terminate CUDA processes or destabilize the GPU. Review each action.
This script is intentionally non-executable and is never called by a runner.
WARNING

ACTION="${1:-inspect}"
HERE="$(cd "$(dirname "$0")" && pwd)"
GPU_EXT_ROOT="$(cd "${HERE}/../../.." && pwd)"
MODULE_REPO="${GPU_EXT_CUSTOM_MODULE_REPO:-/home/peng/workspace/gpu_ext_private/kernel-module/nvidia-module}"
CUSTOM_UVM="${MODULE_REPO}/kernel-open/nvidia-uvm.ko"
BPFTool="${GPU_EXT_BPFTOOL:-${GPU_EXT_ROOT}/tools/bpftool-stage2/bpftool}"

inspect() {
    uname -a
    nvidia-smi
    cat /proc/driver/nvidia/version
    lsmod | grep '^nvidia' || true
    echo "loaded_srcversion=$(cat /sys/module/nvidia_uvm/srcversion 2>/dev/null || true)"
    if [[ -r "${CUSTOM_UVM}" ]]; then
        modinfo "${CUSTOM_UVM}" | grep -E '^(version|srcversion|vermagic|depends):'
        sha256sum "${CUSTOM_UVM}"
    fi
    grep -E 'uvm_bpf_(call_gpu_page_prefetch|trace_gpu_page_prefetch_decision)|gpu_mem_ops' \
        /proc/kallsyms 2>/dev/null || true
    bash "${HERE}/check_stage3.sh"
}

check_users() {
    nvidia-smi
    sudo fuser -v /dev/nvidia* || true
    systemctl status nvidia-persistenced --no-pager || true
    systemctl status gdm3 --no-pager || true
}

require_reload_ack() {
    [[ "${I_UNDERSTAND_GPU_DRIVER_RELOAD:-}" == YES ]] || {
        echo "Set I_UNDERSTAND_GPU_DRIVER_RELOAD=YES after checking GPU users." >&2
        exit 2
    }
}

require_trace_ack() {
    [[ "${I_UNDERSTAND_STAGE3_BPF:-}" == YES ]] || {
        echo "Set I_UNDERSTAND_STAGE3_BPF=YES after reviewing Stage 3 scripts." >&2
        exit 2
    }
}

require_idle_uvm() {
    [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d')" ]] || {
        echo "Active GPU compute process found." >&2; exit 2;
    }
    devices=()
    [[ ! -e /dev/nvidia-uvm ]] || devices+=(/dev/nvidia-uvm)
    [[ ! -e /dev/nvidia-uvm-tools ]] || devices+=(/dev/nvidia-uvm-tools)
    if ((${#devices[@]})) && sudo fuser "${devices[@]}" >/dev/null 2>&1; then
        sudo fuser -v "${devices[@]}" || true
        echo "An active process owns a UVM device." >&2
        exit 2
    fi
    [[ "$(awk '$1=="nvidia_uvm" {print $3}' /proc/modules)" == 0 ]] || {
        echo "nvidia_uvm use count is nonzero." >&2; exit 2;
    }
}

validate_custom() {
    [[ -r "${CUSTOM_UVM}" ]] || { echo "Missing ${CUSTOM_UVM}" >&2; exit 2; }
    local loaded_version
    loaded_version="$(sed -nE 's/.* ([0-9]+\.[0-9]+\.[0-9]+) .*/\1/p' /proc/driver/nvidia/version | head -n1)"
    [[ "$(modinfo -F version "${CUSTOM_UVM}")" == "${loaded_version}" ]] || exit 2
    [[ "$(modinfo -F vermagic "${CUSTOM_UVM}")" == "$(uname -r) "* ]] || exit 2
    [[ "$(modinfo -F depends "${CUSTOM_UVM}")" == nvidia ]] || exit 2
}

run_root_stage3() {
    require_trace_ack
    validate_custom
    grep -q 'uvm_bpf_trace_gpu_page_prefetch_decision' /proc/kallsyms || {
        echo "Enhanced decision hook is not visible." >&2; exit 2;
    }
    sudo --preserve-env=CUDA_VISIBLE_DEVICES,GPU_EXT_BPFTOOL,GPU_EXT_CUSTOM_UVM,JOBS,STAGE3_TIMING_RUNS,STAGE3_TRACE_RUNS,STAGE3_NSYS_RUNS,STAGE3_FIRST_TOUCH_RUNS,STAGE3_OVERHEAD_RUNS,STAGE3_OVERSUB_RATIOS,STAGE3_OVERSUB_TIMING_RUNS,STAGE3_NSYS_RATIO,STAGE3_MIN_FREE_GIB,GPU_EXT_ALLOW_HIGH_OVERSUB,GPU_EXT_STAGE3C_STABLE,I_UNDERSTAND_STAGE3_CONTINUATION \
        env GPU_EXT_BPFTOOL="${BPFTool}" GPU_EXT_CUSTOM_UVM="${CUSTOM_UVM}" \
        bash "$1" "${@:2}"
}

case "${ACTION}" in
inspect)
    inspect
    ;;
build-custom)
    make -C "${MODULE_REPO}" -j"${JOBS:-8}" modules \
        NV_GPU_EXT_CFLAGS=-DNV_GPU_EXT_ENABLE_STRUCT_OPS=1
    validate_custom
    ;;
check-users)
    inspect
    check_users
    ;;
switch-uvm-only)
    require_reload_ack
    validate_custom
    require_idle_uvm
    sudo rmmod nvidia_uvm
    sudo insmod "${CUSTOM_UVM}"
    [[ "$(cat /sys/module/nvidia_uvm/srcversion)" == "$(modinfo -F srcversion "${CUSTOM_UVM}")" ]]
    grep -q 'uvm_bpf_trace_gpu_page_prefetch_decision' /proc/kallsyms
    ;;
verify-hooks)
    validate_custom
    grep -E 'uvm_bpf_(call_gpu_page_prefetch|trace_gpu_page_prefetch_decision)|gpu_mem_ops' /proc/kallsyms
    sudo "${BPFTool}" prog show
    sudo "${BPFTool}" -j map show
    ;;
run-trace-overhead)
    run_root_stage3 "${HERE}/run_trace_overhead.sh"
    ;;
run-trace-semantics)
    run_root_stage3 "${HERE}/run_trace_semantics.sh"
    ;;
run-first-touch)
    run_root_stage3 "${HERE}/run_cpu_first_touch_diagnosis.sh"
    ;;
run-array-migration)
    run_root_stage3 "${HERE}/run_array_migration_diagnosis.sh"
    ;;
run-oversub)
    [[ "${I_UNDERSTAND_OVERSUBSCRIPTION:-}" == YES ]] || {
        echo "Set I_UNDERSTAND_OVERSUBSCRIPTION=YES after reviewing safety limits." >&2; exit 2;
    }
    run_root_stage3 "${HERE}/run_oversub_sweep.sh"
    ;;
run-continuation)
    [[ "${I_UNDERSTAND_OVERSUBSCRIPTION:-}" == YES ]] || {
        echo "Set I_UNDERSTAND_OVERSUBSCRIPTION=YES after reviewing safety limits." >&2; exit 2;
    }
    [[ "${I_UNDERSTAND_STAGE3_CONTINUATION:-}" == YES ]] || {
        echo "Set I_UNDERSTAND_STAGE3_CONTINUATION=YES after reviewing the resource-limit result." >&2; exit 2;
    }
    run_root_stage3 "${HERE}/run_stage3_continuation.sh"
    ;;
run-joint)
    [[ "${I_UNDERSTAND_OVERSUBSCRIPTION:-}" == YES ]] || exit 2
    run_root_stage3 "${HERE}/run_stage3_policy_matrix.sh" joint
    ;;
verify-detached)
    sudo "${BPFTool}" prog show
    sudo "${BPFTool}" -j map show
    pgrep -a -x prefetch_trace || true
    pgrep -a -x chunk_trace || true
    ;;
restore-distribution-uvm)
    require_reload_ack
    require_idle_uvm
    sudo rmmod nvidia_uvm
    sudo modprobe nvidia_uvm
    ;;
verify-restoration)
    lsmod | grep '^nvidia'
    modinfo nvidia_uvm
    nvidia-smi
    echo "loaded_srcversion=$(cat /sys/module/nvidia_uvm/srcversion)"
    echo "distribution_srcversion=$(modinfo -F srcversion nvidia_uvm)"
    grep -E 'uvm_bpf_trace_gpu_page_prefetch_decision|gpu_mem_ops' /proc/kallsyms 2>/dev/null || true
    ;;
*)
    echo "Usage: bash $0 inspect|build-custom|check-users|switch-uvm-only|verify-hooks|run-trace-overhead|run-trace-semantics|run-first-touch|run-array-migration|run-oversub|run-continuation|run-joint|verify-detached|restore-distribution-uvm|verify-restoration" >&2
    exit 2
    ;;
esac
