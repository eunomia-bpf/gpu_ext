#!/usr/bin/env bash
# INTENTIONALLY NON-EXECUTABLE. Review each subcommand before invoking it with bash.
set -Eeuo pipefail

ROOT=/home/peng/workspace/gpu_ext
EXP="${ROOT}/experiments/uvm_basic"
CUSTOM_UVM="${GPU_EXT_CUSTOM_UVM:-/home/peng/workspace/gpu_ext_private/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko}"
ACTION="${1:-help}"

inspect() {
    uname -r
    nvidia-smi
    fuser -v /dev/nvidia* || true
    lsmod | grep nvidia || true
    modinfo nvidia_uvm
    modinfo "${CUSTOM_UVM}"
    grep -E 'uvm_bpf_trace_gpu_page_prefetch_decision|gpu_mem_ops' /proc/kallsyms || true
    systemctl status nvidia-persistenced --no-pager || true
    systemctl status gdm3 --no-pager || true
}

switch_uvm_only() {
    [[ -f "${CUSTOM_UVM}" ]]
    [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^$/d')" ]]
    [[ "$(modinfo -F vermagic "${CUSTOM_UVM}")" == "$(uname -r) "* ]]
    sudo rmmod nvidia_uvm
    sudo insmod "${CUSTOM_UVM}"
    grep -E 'uvm_bpf_trace_gpu_page_prefetch_decision|gpu_mem_ops' /proc/kallsyms
}

restore_distribution_uvm() {
    sudo rmmod nvidia_uvm
    sudo modprobe nvidia_uvm
    lsmod | grep nvidia
    modinfo nvidia_uvm
    nvidia-smi
}

run_calibration() {
    cd "${ROOT}"
    sudo --preserve-env=PATH,CUDA_HOME,LD_LIBRARY_PATH,GPU_EXT_CUSTOM_UVM \
        bash "${EXP}/scripts/run_reduced_capacity_calibration.sh"
}

run_prefetch_matrix() {
    cd "${ROOT}"
    sudo --preserve-env=PATH,CUDA_HOME,LD_LIBRARY_PATH,GPU_EXT_CUSTOM_UVM \
        bash "${EXP}/scripts/run_reduced_capacity_prefetch_matrix.sh"
}

run_eviction_smoke() {
    cd "${ROOT}"
    sudo --preserve-env=PATH,CUDA_HOME,LD_LIBRARY_PATH,GPU_EXT_CUSTOM_UVM \
        bash "${EXP}/scripts/run_eviction_smoke.sh"
}

run_joint_matrix() {
    cd "${ROOT}"
    sudo --preserve-env=PATH,CUDA_HOME,LD_LIBRARY_PATH,GPU_EXT_CUSTOM_UVM \
        bash "${EXP}/scripts/run_joint_policy_matrix.sh"
}

run_natural_confirmation() {
    cd "${ROOT}"
    sudo --preserve-env=PATH,CUDA_HOME,LD_LIBRARY_PATH,GPU_EXT_CUSTOM_UVM,STAGE4_NATURAL_POLICIES \
        bash "${EXP}/scripts/run_natural_capacity_confirmation.sh"
}

run_trace_overhead() {
    cd "${ROOT}"
    sudo --preserve-env=PATH,CUDA_HOME,LD_LIBRARY_PATH,GPU_EXT_CUSTOM_UVM \
        bash "${EXP}/scripts/measure_trace_disabled_overhead.sh"
}

case "${ACTION}" in
    inspect) inspect ;;
    switch-uvm-only) switch_uvm_only ;;
    calibration) run_calibration ;;
    prefetch-matrix) run_prefetch_matrix ;;
    eviction-smoke) run_eviction_smoke ;;
    joint-matrix) run_joint_matrix ;;
    natural-confirmation) run_natural_confirmation ;;
    trace-overhead) run_trace_overhead ;;
    restore) restore_distribution_uvm ;;
    help)
        cat <<'EOF'
Manual sequence (review after each step):
  bash scripts/SAFE_STAGE4_COMMANDS.sh inspect
  bash scripts/SAFE_STAGE4_COMMANDS.sh switch-uvm-only
  bash scripts/SAFE_STAGE4_COMMANDS.sh calibration
  bash scripts/SAFE_STAGE4_COMMANDS.sh prefetch-matrix
  bash scripts/SAFE_STAGE4_COMMANDS.sh eviction-smoke
  bash scripts/SAFE_STAGE4_COMMANDS.sh joint-matrix
  bash scripts/SAFE_STAGE4_COMMANDS.sh natural-confirmation
  bash scripts/SAFE_STAGE4_COMMANDS.sh trace-overhead
  bash scripts/SAFE_STAGE4_COMMANDS.sh restore

Stop immediately on Xid, CUDA/correctness failure, detach failure, or GPU loss.
EOF
        ;;
    *) echo "Unknown action: ${ACTION}" >&2; exit 2 ;;
esac
