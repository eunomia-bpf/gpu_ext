#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "$0")/stage3_common.sh"
OUTPUT="${STAGE3_RESULTS}/preflight.json"
mkdir -p "${STAGE3_RESULTS}"

stage3_build

MODULE_SOURCE="/home/peng/workspace/gpu_ext_private/kernel-module/nvidia-module"
SOURCE_HOOK=0
TRACE_BINARY=0
CUDA_BINARIES=0
CUSTOM_LOADED=0
HOOK_VISIBLE=0
CUSTOM_SYMBOLS=0
CUSTOM_NM=""

rg -q 'uvm_bpf_trace_gpu_page_prefetch_decision' \
    "${MODULE_SOURCE}/kernel-open/nvidia-uvm/uvm_perf_prefetch.c" \
    "${MODULE_SOURCE}/kernel-open/nvidia-uvm/uvm_bpf_struct_ops.c" && SOURCE_HOOK=1
[[ -x "${EXTENSION_DIR}/prefetch_trace" && -x "${EXTENSION_DIR}/chunk_trace" ]] && TRACE_BINARY=1
[[ -x "${PROGRAM}" && -x "${PHASE_SCAN_PROGRAM}" ]] && CUDA_BINARIES=1
grep -q 'uvm_bpf_trace_gpu_page_prefetch_decision' /proc/kallsyms 2>/dev/null && HOOK_VISIBLE=1
stage3_custom_module_loaded && CUSTOM_LOADED=1 || true
CUSTOM_NM="$(nm -n "${CUSTOM_UVM}" 2>/dev/null || true)"
if grep -q 'uvm_bpf_trace_gpu_page_prefetch_decision' <<<"${CUSTOM_NM}" && \
   grep -q 'uvm_bpf_trace_gpu_eviction_selected' <<<"${CUSTOM_NM}"; then
    CUSTOM_SYMBOLS=1
fi

LOADED_SRCVERSION="$(cat /sys/module/nvidia_uvm/srcversion 2>/dev/null || true)"
CUSTOM_SRCVERSION="$(modinfo -F srcversion "${CUSTOM_UVM}" 2>/dev/null || true)"
CUSTOM_VERSION="$(modinfo -F version "${CUSTOM_UVM}" 2>/dev/null || true)"
CUSTOM_VERMAGIC="$(modinfo -F vermagic "${CUSTOM_UVM}" 2>/dev/null || true)"
CUSTOM_SHA256="$(sha256sum "${CUSTOM_UVM}" 2>/dev/null | awk '{print $1}')"

python3 - "${OUTPUT}" "${SOURCE_HOOK}" "${TRACE_BINARY}" "${CUDA_BINARIES}" \
    "${CUSTOM_LOADED}" "${HOOK_VISIBLE}" "${LOADED_SRCVERSION}" "${CUSTOM_SRCVERSION}" \
    "${CUSTOM_VERSION}" "${CUSTOM_VERMAGIC}" "${CUSTOM_UVM}" "${CUSTOM_SYMBOLS}" \
    "${CUSTOM_SHA256}" <<'PY'
import json, os, sys
from pathlib import Path
data = {
    "evidence_class": "GPU_EXT_STAGE3_PREFLIGHT",
    "source_decision_hook_present": sys.argv[2] == "1",
    "trace_binaries_ready": sys.argv[3] == "1",
    "cuda_binaries_ready": sys.argv[4] == "1",
    "custom_module_loaded": sys.argv[5] == "1",
    "decision_hook_visible": sys.argv[6] == "1",
    "loaded_srcversion": sys.argv[7] or None,
    "custom_srcversion": sys.argv[8] or None,
    "custom_version": sys.argv[9] or None,
    "custom_vermagic": sys.argv[10] or None,
    "custom_module": sys.argv[11],
    "custom_binary_trace_symbols": sys.argv[12] == "1",
    "custom_module_sha256": sys.argv[13] or None,
    "kernel_release": os.uname().release,
    "privileged_operation_executed": False,
}
ready = all(data[key] for key in (
    "source_decision_hook_present", "trace_binaries_ready", "cuda_binaries_ready",
    "custom_binary_trace_symbols"))
data["status"] = "READY_FOR_MANUAL_STAGE3" if ready else "BLOCKED_STAGE3_BUILD"
Path(sys.argv[1]).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
print(json.dumps(data, indent=2, sort_keys=True))
PY

python3 "${UVM_BASIC_DIR}/analysis/audit_eviction_policies.py" \
    --gpu-ext-root "${GPU_EXT_ROOT}" --experiment-dir "${UVM_BASIC_DIR}"
python3 "${UVM_BASIC_DIR}/analysis/analyze_prefetch_decisions.py" \
    --experiment-dir "${UVM_BASIC_DIR}"
python3 "${UVM_BASIC_DIR}/analysis/analyze_eviction_refault.py" \
    --experiment-dir "${UVM_BASIC_DIR}"
python3 "${UVM_BASIC_DIR}/analysis/analyze_array_migrations.py" \
    --experiment-dir "${UVM_BASIC_DIR}"
python3 "${UVM_BASIC_DIR}/analysis/summarize_stage3.py" \
    --experiment-dir "${UVM_BASIC_DIR}"
bash "${UVM_BASIC_DIR}/scripts/check_stage3_safety.sh"
