#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "$0")/stage4_common.sh"

OUTPUT="${1:-${STAGE4_RESULTS}/preflight.json}"
mkdir -p "$(dirname "${OUTPUT}")"
stage4_require_disk
stage4_build
python3 "${UVM_BASIC_DIR}/analysis/audit_eviction_policies.py" \
    --extension-dir "${EXTENSION_DIR}" \
    --json "${STAGE4_RESULTS}/eviction_policy_audit.json" \
    --markdown "${UVM_BASIC_DIR}/docs/EVICTION_POLICY_SAFETY_AUDIT_STAGE4.md"

CUSTOM_SRCVERSION="$(modinfo -F srcversion "${CUSTOM_UVM}" 2>/dev/null || true)"
LOADED_SRCVERSION="$(cat /sys/module/nvidia_uvm/srcversion 2>/dev/null || true)"
HOOK_VISIBLE=0
grep -q 'uvm_bpf_trace_gpu_page_prefetch_decision' /proc/kallsyms 2>/dev/null && HOOK_VISIBLE=1 || true
ACTIVE="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | paste -sd, -)"
DISK_KIB="$(df -Pk "${STAGE4_RESULTS}" | awk 'NR == 2 {print $4}')"
MEM_KIB="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"

python3 - "${OUTPUT}" "${CUSTOM_UVM}" "${CUSTOM_SRCVERSION}" "${LOADED_SRCVERSION}" \
    "${HOOK_VISIBLE}" "${ACTIVE}" "${DISK_KIB}" "${MEM_KIB}" \
    "$(bytes_from_size "${STAGE4_TARGET_EFFECTIVE}")" \
    "$(bytes_from_size "${STAGE4_GUARD_DEVICE_BYTES}")" <<'PY'
import json, sys
from pathlib import Path
custom = Path(sys.argv[2])
custom_src, loaded_src = sys.argv[3], sys.argv[4]
hook = sys.argv[5] == "1"
data = {
    "evidence_class": "GPU_EXT_STAGE4_PREFLIGHT",
    "custom_module": str(custom),
    "custom_module_exists": custom.is_file(),
    "custom_srcversion": custom_src or None,
    "loaded_srcversion": loaded_src or None,
    "custom_module_loaded": bool(custom_src and custom_src == loaded_src and hook),
    "custom_hook_visible": hook,
    "active_compute_pids": sys.argv[6].split(",") if sys.argv[6] else [],
    "result_disk_free_bytes": int(sys.argv[7]) << 10,
    "host_mem_available_bytes": int(sys.argv[8]) << 10,
    "nonprivileged_build": "PASS",
    "audit_generated": True,
    "capacity_model": "PHYSICALLY_RESERVED_GUARD_MODEL",
    "target_effective_gpu_bytes": int(sys.argv[9]),
    "guard_device_bytes": int(sys.argv[10]),
}
data["status"] = ("READY_FOR_STAGE4A_RECALIBRATION" if data["custom_module_loaded"]
                  else "READY_FOR_MANUAL_STAGE4A_RECALIBRATION")
Path(sys.argv[1]).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
print(json.dumps(data, indent=2, sort_keys=True))
PY
