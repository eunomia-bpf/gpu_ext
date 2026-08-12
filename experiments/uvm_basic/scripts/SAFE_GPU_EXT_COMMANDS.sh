#!/usr/bin/env bash
set -Eeuo pipefail

cat <<'WARNING'
WARNING: manual NVIDIA module switching can terminate GPU users or the display session.
This file is never called by an experiment runner and is intentionally non-executable.
Review the selected action before invoking it with `bash`.
WARNING

ACTION="${1:-inspect}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CUSTOM_MODULE_DIR="${GPU_EXT_CUSTOM_MODULE_DIR:-/home/peng/workspace/gpu_ext_private/kernel-module/nvidia-module/kernel-open}"
EXPECTED_KERNEL="$(uname -r)"
CUSTOM_UVM="${CUSTOM_MODULE_DIR}/nvidia-uvm.ko"
CUSTOM_CORE="${CUSTOM_MODULE_DIR}/nvidia.ko"
CUSTOM_MODESET="${CUSTOM_MODULE_DIR}/nvidia-modeset.ko"
CUSTOM_DRM="${CUSTOM_MODULE_DIR}/nvidia-drm.ko"
BPFTool="${GPU_EXT_BPFTOOL:-}"

resolve_bpftool() {
    if [[ -n "${BPFTool}" && -x "${BPFTool}" ]]; then return 0; fi
    local candidate
    for candidate in \
        "$(cd "${HERE}/../../.." && pwd)/tools/bpftool-stage2/bpftool" \
        /usr/lib/linux-hwe-*-tools-*/bpftool \
        /usr/lib/linux-tools/*/bpftool \
        /usr/sbin/bpftool; do
        if [[ -x "${candidate}" ]] \
            && "${candidate}" struct_ops help >/dev/null 2>&1 \
            && "${candidate}" version 2>/dev/null | grep -Eq 'bpftool v7\.[6-9]|bpftool v[89]\.'; then
            BPFTool="$(readlink -f "${candidate}")"
            return 0
        fi
    done
    echo "No bpftool with struct_ops support was found." >&2
    return 2
}

# A. Inspection only: no root operation.
inspect() {
    echo "kernel=${EXPECTED_KERNEL}"
    cat /proc/driver/nvidia/version 2>/dev/null || true
    nvidia-smi 2>/dev/null || true
    lsmod | grep '^nvidia' || true
    systemctl status nvidia-persistenced --no-pager 2>/dev/null || true
    systemctl status gdm3 --no-pager 2>/dev/null || true
    grep -E 'uvm_bpf_call_gpu_page_prefetch|gpu_mem_ops' /proc/kallsyms 2>/dev/null || true
    echo "distribution_uvm=$(modinfo -n nvidia_uvm 2>/dev/null || echo UNAVAILABLE)"
    echo "distribution_depends=$(modinfo -F depends nvidia_uvm 2>/dev/null || echo UNAVAILABLE)"
    for module in "${CUSTOM_CORE}" "${CUSTOM_MODESET}" "${CUSTOM_DRM}" "${CUSTOM_UVM}"; do
        if [[ -r "${module}" ]]; then
            echo "--- ${module}"
            modinfo -F version "${module}" || true
            modinfo -F vermagic "${module}" || true
            modinfo -F depends "${module}" || true
            sha256sum "${module}"
        else
            echo "MISSING ${module}"
        fi
    done
}

# B. Check active GPU users. Root improves fuser attribution but does not mutate state.
check_users() {
    nvidia-smi
    sudo fuser -v /dev/nvidia* || true
    lsmod | grep '^nvidia' || true
    systemctl status nvidia-persistenced --no-pager || true
    systemctl status gdm3 --no-pager || true
}

require_authorization() {
    [[ "${I_UNDERSTAND_GPU_DRIVER_RELOAD:-}" == YES ]] || {
        echo "Set I_UNDERSTAND_GPU_DRIVER_RELOAD=YES after reviewing active GPU users." >&2
        exit 2
    }
}

require_idle_gpu() {
    [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d')" ]] || {
        echo "Active GPU compute processes detected; refusing module switch." >&2
        exit 2
    }
    ! sudo fuser /dev/nvidia* >/dev/null 2>&1 || {
        echo "A process has an NVIDIA device open; refusing module switch." >&2
        exit 2
    }
    [[ "$(systemctl is-active gdm3 2>/dev/null || true)" != active ]] || {
        echo "gdm3 is active; stop it manually before switching modules." >&2
        exit 2
    }
}

require_idle_uvm() {
    [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d')" ]] || {
        echo "Active GPU compute processes detected; refusing UVM switch." >&2
        exit 2
    }
    local devices=()
    [[ ! -e /dev/nvidia-uvm ]] || devices+=(/dev/nvidia-uvm)
    [[ ! -e /dev/nvidia-uvm-tools ]] || devices+=(/dev/nvidia-uvm-tools)
    if ((${#devices[@]})) && sudo fuser "${devices[@]}" >/dev/null 2>&1; then
        echo "A process has a UVM device open; refusing UVM-only switch." >&2
        sudo fuser -v "${devices[@]}" || true
        exit 2
    fi
    [[ "$(awk '$1=="nvidia_uvm" {print $3}' /proc/modules)" == 0 ]] || {
        echo "nvidia_uvm has a non-zero module use count; refusing UVM-only switch." >&2
        exit 2
    }
}

wait_for_uvm_release() {
    local attempt use_count
    for attempt in {1..50}; do
        use_count="$(awk '$1 == "nvidia_uvm" {print $3}' /proc/modules)"
        if [[ -z "${use_count}" || "${use_count}" == 0 ]]; then
            return 0
        fi
        sleep 0.1
    done
    echo "nvidia_uvm still has ${use_count:-unknown} references after 5 seconds." >&2
    return 1
}

validate_custom_module() {
    local loaded_version
    [[ -r "${CUSTOM_UVM}" ]] || { echo "Missing ${CUSTOM_UVM}" >&2; exit 2; }
    loaded_version="$(sed -nE 's/.* ([0-9]+\.[0-9]+\.[0-9]+) .*/\1/p' /proc/driver/nvidia/version | head -n1)"
    [[ "$(modinfo -F version "${CUSTOM_UVM}")" == "${loaded_version}" ]] || {
        echo "Custom UVM driver version mismatch." >&2; exit 2;
    }
    [[ "$(modinfo -F vermagic "${CUSTOM_UVM}")" == "${EXPECTED_KERNEL} "* ]] || {
        echo "Custom UVM vermagic mismatch." >&2; exit 2;
    }
    [[ "$(modinfo -F depends "${CUSTOM_UVM}")" == nvidia ]] || {
        echo "Custom UVM has dependencies beyond nvidia; use the reviewed full-stack path." >&2
        exit 2
    }
}

unload_full_stack() {
    local module
    for module in nvidia_uvm nvidia_drm nvidia_modeset nvidia; do
        if lsmod | awk '{print $1}' | grep -qx "${module}"; then sudo rmmod "${module}"; fi
    done
}

case "${ACTION}" in
inspect) # A
    inspect
    ;;
check-users) # B
    inspect
    check_users
    ;;
switch-uvm-only) # C. Preferred because custom nvidia-uvm.ko depends only on nvidia.
    require_authorization
    validate_custom_module
    require_idle_uvm
    sudo rmmod nvidia_uvm
    sudo insmod "${CUSTOM_UVM}"
    [[ "$(cat /sys/module/nvidia_uvm/srcversion)" == "$(modinfo -F srcversion "${CUSTOM_UVM}")" ]] || {
        echo "Loaded UVM srcversion does not match the custom module." >&2; exit 1;
    }
    grep -q 'uvm_bpf_call_gpu_page_prefetch' /proc/kallsyms || {
        echo "Custom UVM loaded but gpu_ext hook is not visible." >&2; exit 1;
    }
    echo "Custom UVM-only switch verified. Next: bash $0 verify-hooks"
    ;;
switch-full-stack) # D. Use only after proving UVM-only symbol-version incompatibility.
    require_authorization
    validate_custom_module
    require_idle_gpu
    for module in "${CUSTOM_CORE}" "${CUSTOM_UVM}"; do [[ -r "${module}" ]] || exit 2; done
    unload_full_stack
    sudo insmod "${CUSTOM_CORE}"
    [[ ! -r "${CUSTOM_MODESET}" ]] || sudo insmod "${CUSTOM_MODESET}"
    [[ ! -r "${CUSTOM_DRM}" ]] || sudo insmod "${CUSTOM_DRM}"
    sudo insmod "${CUSTOM_UVM}"
    ;;
verify-hooks) # E
    resolve_bpftool
    grep -E 'uvm_bpf_call_gpu_page_prefetch|gpu_mem_ops' /proc/kallsyms
    sudo "${BPFTool}" prog show
    sudo "${BPFTool}" -j map show
    ;;
run-stage2) # F
    [[ "${I_UNDERSTAND_GPU_EXT_TRACE:-}" == YES ]] || {
        echo "Set I_UNDERSTAND_GPU_EXT_TRACE=YES to authorize Stage 2 BPF attachments." >&2
        exit 2
    }
    grep -q 'uvm_bpf_call_gpu_page_prefetch' /proc/kallsyms || exit 2
    resolve_bpftool
    sudo --preserve-env=CUDA_VISIBLE_DEVICES,UVM_BASIC_TIMEOUT_SECONDS,UVM_BASIC_STAGE2_RUN_1G,UVM_BASIC_STAGE2_POLICIES,UVM_BASIC_STAGE2_TIMING_RUNS,UVM_BASIC_STAGE2_TRACE_RUNS,UVM_BASIC_STAGE2_NSYS_RUNS,UVM_BASIC_STAGE2_TIMING_RUNS_1G,UVM_BASIC_STAGE2_TRACE_RUNS_1G,GPU_EXT_BPFTOOL \
        env GPU_EXT_BPFTOOL="${BPFTool}" bash "${HERE}/run_gpu_ext_trace.sh"
    ;;
verify-detached) # G. Runner owns exact PIDs; this action only verifies no residue.
    resolve_bpftool
    sudo "${BPFTool}" prog show
    sudo "${BPFTool}" -j map show
    ;;
restore-distribution-uvm) # H. Preferred restoration after UVM-only switch.
    require_authorization
    if lsmod | awk '{print $1}' | grep -qx nvidia_uvm; then
        require_idle_uvm
        wait_for_uvm_release
        sudo rmmod nvidia_uvm
    fi
    sudo modprobe nvidia_uvm
    ;;
restore-distribution-full) # H. Restoration after an explicitly chosen full-stack switch.
    require_authorization
    require_idle_gpu
    unload_full_stack
    sudo modprobe nvidia
    sudo modprobe nvidia_uvm
    ;;
verify-restoration) # I
    lsmod | grep '^nvidia'
    modinfo nvidia_uvm
    nvidia-smi
    grep -E 'uvm_bpf_call_gpu_page_prefetch|gpu_mem_ops' /proc/kallsyms 2>/dev/null || true
    ;;
*)
    echo "Usage: bash $0 inspect|check-users|switch-uvm-only|switch-full-stack|verify-hooks|run-stage2|verify-detached|restore-distribution-uvm|restore-distribution-full|verify-restoration" >&2
    exit 2
    ;;
esac
