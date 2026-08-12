#!/usr/bin/env bash
set -Eeuo pipefail

echo "WARNING: this file can unload the NVIDIA driver only when explicitly invoked with reload-custom."
echo "It is never called by the UVM experiment runners. Review every path before use."

ACTION="${1:-inspect}"
CUSTOM_MODULE_DIR="${GPU_EXT_CUSTOM_MODULE_DIR:-/home/peng/workspace/gpu_ext_private/kernel-module/nvidia-module/kernel-open}"
EXPECTED_KERNEL="$(uname -r)"
CUSTOM_UVM="${CUSTOM_MODULE_DIR}/nvidia-uvm.ko"
CUSTOM_CORE="${CUSTOM_MODULE_DIR}/nvidia.ko"
CUSTOM_MODESET="${CUSTOM_MODULE_DIR}/nvidia-modeset.ko"
CUSTOM_DRM="${CUSTOM_MODULE_DIR}/nvidia-drm.ko"

inspect() {
    echo "kernel=${EXPECTED_KERNEL}"
    cat /proc/driver/nvidia/version 2>/dev/null || true
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || true
    systemctl is-active display-manager 2>/dev/null || true
    lsmod | grep '^nvidia' || true
    for module in "${CUSTOM_CORE}" "${CUSTOM_MODESET}" "${CUSTOM_DRM}" "${CUSTOM_UVM}"; do
        if [[ -r "${module}" ]]; then
            echo "--- ${module}"
            modinfo -F version "${module}" || true
            modinfo -F vermagic "${module}" || true
            sha256sum "${module}"
        else
            echo "MISSING ${module}"
        fi
    done
}

inspect
[[ "${ACTION}" == inspect ]] && exit 0

require_idle_gpu() {
    [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d')" ]] || {
        echo "Active GPU compute processes detected; refusing driver reload." >&2
        exit 2
    }
    [[ "$(systemctl is-active display-manager 2>/dev/null || true)" != active ]] || {
        echo "Display manager is active; refusing driver reload." >&2
        exit 2
    }
}

unload_nvidia_modules() {
    local loaded_module
    for loaded_module in nvidia_uvm nvidia_drm nvidia_modeset nvidia; do
        if lsmod | awk '{print $1}' | grep -qx "${loaded_module}"; then
            sudo rmmod "${loaded_module}"
        fi
    done
}

case "${ACTION}" in
reload-custom)
    [[ "${I_UNDERSTAND_GPU_DRIVER_RELOAD:-}" == YES ]] || {
        echo "Set I_UNDERSTAND_GPU_DRIVER_RELOAD=YES only after stopping all GPU/display users." >&2
        exit 2
    }
    require_idle_gpu
    loaded_version="$(sed -nE 's/.* ([0-9]+\.[0-9]+\.[0-9]+) .*/\1/p' /proc/driver/nvidia/version | head -n1)"
    for module in "${CUSTOM_CORE}" "${CUSTOM_UVM}"; do
        [[ -r "${module}" ]] || { echo "Missing ${module}" >&2; exit 2; }
        [[ "$(modinfo -F version "${module}")" == "${loaded_version}" ]] || {
            echo "Driver version mismatch: ${module}" >&2; exit 2;
        }
        [[ "$(modinfo -F vermagic "${module}")" == "${EXPECTED_KERNEL} "* ]] || {
            echo "Kernel vermagic mismatch: ${module}" >&2; exit 2;
        }
    done
    # Manual, temporary module swap. This script never installs modules.
    unload_nvidia_modules
    sudo insmod "${CUSTOM_CORE}"
    [[ ! -r "${CUSTOM_MODESET}" ]] || sudo insmod "${CUSTOM_MODESET}"
    [[ ! -r "${CUSTOM_DRM}" ]] || sudo insmod "${CUSTOM_DRM}"
    sudo insmod "${CUSTOM_UVM}"
    cat <<'EOF'
Temporary custom modules loaded. Run the trace with:
I_UNDERSTAND_GPU_EXT_TRACE=YES bash scripts/SAFE_GPU_EXT_COMMANDS.sh run-trace

Restore the distribution driver with:
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia
sudo modprobe nvidia_uvm
EOF
    ;;
run-trace)
    [[ "${I_UNDERSTAND_GPU_EXT_TRACE:-}" == YES ]] || {
        echo "Set I_UNDERSTAND_GPU_EXT_TRACE=YES to authorize a temporary BPF attach." >&2
        exit 2
    }
    grep -q 'uvm_bpf_call_gpu_page_prefetch' /proc/kallsyms || {
        echo "gpu_ext hook is not visible; load the matching custom module first." >&2
        exit 2
    }
    sudo --preserve-env=CUDA_VISIBLE_DEVICES,UVM_BASIC_TRACE_BYTES,UVM_BASIC_TIMEOUT_SECONDS \
        bash "$(dirname "$0")/run_gpu_ext_trace.sh"
    ;;
restore-system)
    [[ "${I_UNDERSTAND_GPU_DRIVER_RELOAD:-}" == YES ]] || {
        echo "Set I_UNDERSTAND_GPU_DRIVER_RELOAD=YES before restoring modules." >&2
        exit 2
    }
    require_idle_gpu
    unload_nvidia_modules
    sudo modprobe nvidia
    sudo modprobe nvidia_uvm
    ;;
*)
    echo "Usage: bash $0 inspect|reload-custom|run-trace|restore-system" >&2
    exit 2
    ;;
esac
