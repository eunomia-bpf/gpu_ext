#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "$0")/common.sh"
OUT="${RESULTS_DIR}/environment.txt"
CUSTOM_MODULE_DIR="${GPU_EXT_CUSTOM_MODULE_DIR:-/home/peng/workspace/gpu_ext_private/kernel-module/nvidia-module/kernel-open}"
TMP="$(mktemp)"
trap 'rm -f "${TMP}"' EXIT

run_section() {
    local title="$1"
    shift
    printf '\n===== %s =====\n' "${title}"
    set +e
    "$@" 2>&1
    local rc=$?
    set -e
    printf '[exit_code=%d]\n' "${rc}"
}

{
    echo "evidence_class=SYSTEM_NVIDIA_DRIVER_USERSPACE_UVM"
    echo "collected_utc=$(date -u --iso-8601=seconds)"
    echo "gpu_ext_root=${GPU_EXT_ROOT}"
    run_section "uname -a" uname -a
    run_section "uname -r" uname -r
    run_section "nvidia-smi" nvidia-smi
    run_section "nvidia-smi query" nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
    run_section "proc driver version" cat /proc/driver/nvidia/version
    run_section "nvcc version" nvcc --version
    run_section "cmake version" cmake --version
    run_section "git status" git -C "${GPU_EXT_ROOT}" status --short
    run_section "git submodules" git -C "${GPU_EXT_ROOT}" submodule status --recursive
    run_section "loaded NVIDIA modules" bash -c "lsmod | grep nvidia"
    run_section "loaded nvidia_uvm metadata" modinfo nvidia_uvm

    echo
    echo "===== custom module compatibility ====="
    echo "custom_module_dir=${CUSTOM_MODULE_DIR}"
    custom_uvm="${CUSTOM_MODULE_DIR}/nvidia-uvm.ko"
    if [[ -r "${custom_uvm}" ]]; then
        loaded_version="$(sed -nE 's/.* ([0-9]+\.[0-9]+\.[0-9]+) .*/\1/p' /proc/driver/nvidia/version | head -n1)"
        custom_version="$(modinfo -F version "${custom_uvm}" 2>/dev/null || true)"
        custom_vermagic="$(modinfo -F vermagic "${custom_uvm}" 2>/dev/null || true)"
        loaded_uvm="$(modinfo -n nvidia_uvm 2>/dev/null || true)"
        echo "loaded_driver_version=${loaded_version:-UNAVAILABLE}"
        echo "custom_driver_version=${custom_version:-UNAVAILABLE}"
        echo "custom_vermagic=${custom_vermagic:-UNAVAILABLE}"
        echo "loaded_uvm_path=${loaded_uvm:-UNAVAILABLE}"
        [[ ! -r "${loaded_uvm}" ]] || sha256sum "${loaded_uvm}"
        sha256sum "${custom_uvm}"
        if [[ "${loaded_version}" == "${custom_version}" && "${custom_vermagic}" == "$(uname -r) "* ]]; then
            echo "custom_module_identity=MATCHES_VERSION_AND_KERNEL"
            if [[ -r "${loaded_uvm}" ]] && cmp -s "${loaded_uvm}" "${custom_uvm}"; then
                echo "custom_module_binary=LOADED"
            else
                echo "custom_module_binary=NOT_LOADED"
            fi
            echo "risk=Version/vermagic match, but a manual reload remains disruptive and runtime compatibility is not proven."
        else
            echo "custom_module_identity=MISMATCH"
            echo "risk=Do not load this custom module on the current kernel/driver stack."
        fi
    else
        echo "custom_module_identity=UNAVAILABLE"
        echo "risk=No readable custom nvidia-uvm.ko was found."
    fi
} >"${TMP}"

sed -E 's/[[:space:]]+$//' "${TMP}" >"${OUT}"

cat "${OUT}"
