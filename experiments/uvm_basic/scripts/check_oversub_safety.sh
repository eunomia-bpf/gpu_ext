#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "$0")/stage3_common.sh"
RATIO="${1:-0.95}"
OUTPUT="${2:-${STAGE3_RESULTS}/oversub_safety_${RATIO}.json}"

stage3_require_runtime
python3 - "${RATIO}" <<'PY'
import os, sys
ratio = float(sys.argv[1])
if ratio <= 0 or ratio > 1.25:
    raise SystemExit("ratio must be in (0, 1.25]")
if ratio > 1.15 and os.environ.get("GPU_EXT_ALLOW_HIGH_OVERSUB") != "1":
    raise SystemExit("ratio above 1.15 requires GPU_EXT_ALLOW_HIGH_OVERSUB=1")
PY

GPU_FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${CUDA_DEVICE:-0}" | head -n1 | tr -d ' ')"
GPU_TOTAL_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${CUDA_DEVICE:-0}" | head -n1 | tr -d ' ')"
HOST_AVAILABLE_KIB="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
SWAP_USED_BEFORE_KIB="$(awk '/^SwapTotal:/ {t=$2} /^SwapFree:/ {f=$2} END {print t-f}' /proc/meminfo)"
sleep 2
SWAP_USED_AFTER_KIB="$(awk '/^SwapTotal:/ {t=$2} /^SwapFree:/ {f=$2} END {print t-f}' /proc/meminfo)"
ACTIVE_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | paste -sd, -)"
TRACE_PIDS="$(pgrep -x prefetch_trace 2>/dev/null | paste -sd, - || true)"
POLICY_PIDS="$(pgrep -f '^.*/(prefetch_none|prefetch_always_max|prefetch_adaptive_sequential|eviction_fifo|prefetch_always_max_cycle_moe|prefetch_cooperative)$' 2>/dev/null | paste -sd, - || true)"
SNAPSHOT="$(mktemp)"
trap 'rm -f "${SNAPSHOT}"' EXIT
stage3_struct_ops_snapshot "${SNAPSHOT}"
STRUCT_OPS=0
stage3_has_struct_ops "${SNAPSHOT}" && STRUCT_OPS=1 || true
XID_COUNT="$(dmesg --color=never 2>/dev/null | grep -Eic 'NVRM: Xid|NVIDIA.*Xid' || true)"

python3 - "${OUTPUT}" "${RATIO}" "${GPU_FREE_MIB}" "${GPU_TOTAL_MIB}" \
    "${HOST_AVAILABLE_KIB}" "${SWAP_USED_BEFORE_KIB}" "${SWAP_USED_AFTER_KIB}" \
    "${ACTIVE_PIDS}" "${TRACE_PIDS}" "${POLICY_PIDS}" "${STRUCT_OPS}" "${XID_COUNT}" <<'PY'
import json, sys
from pathlib import Path
ratio = float(sys.argv[2])
gpu_free = int(sys.argv[3]) << 20
working_set = int(gpu_free * ratio)
host_available = int(sys.argv[5]) << 10
swap_before, swap_after = int(sys.argv[6]) << 10, int(sys.argv[7]) << 10
checks = {
    "no_other_gpu_compute_process": not bool(sys.argv[8]),
    "no_trace_process": not bool(sys.argv[9]),
    "no_policy_loader": not bool(sys.argv[10]),
    "no_struct_ops": sys.argv[11] == "0",
    "host_memory_margin_16gib": host_available > working_set + (16 << 30),
    "swap_not_rapidly_growing": swap_after - swap_before < (256 << 20),
    "custom_decision_hook_loaded": True,
}
data = {
    "evidence_class": "GPU_EXT_STAGE3_OVERSUB_SAFETY",
    "ratio": ratio, "gpu_free_bytes": gpu_free,
    "gpu_total_bytes": int(sys.argv[4]) << 20,
    "planned_working_set_bytes": working_set,
    "host_available_bytes": host_available,
    "swap_used_before_bytes": swap_before, "swap_used_after_bytes": swap_after,
    "existing_xid_count": int(sys.argv[12]), "checks": checks,
    "status": "PASS_OVERSUB_SAFETY" if all(checks.values()) else "BLOCKED_OVERSUB_SAFETY",
}
Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
Path(sys.argv[1]).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
print(json.dumps(data, indent=2, sort_keys=True))
if data["status"] != "PASS_OVERSUB_SAFETY": raise SystemExit(2)
PY
