#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "$0")/stage4_common.sh"

RATIO="${1:?ratio is required}"
TARGET="${2:?target effective capacity is required}"
OUTPUT="${3:?output path is required}"

[[ ${EUID} -eq 0 ]] || {
    echo "Stage 4 runtime is restricted to the reviewed SAFE_STAGE4_COMMANDS.sh path." >&2
    exit 2
}
stage3_require_runtime
stage4_require_disk
stage4_require_host_memory "${TARGET}" "${RATIO}"
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d')" ]] || {
    echo "Another GPU compute process is active." >&2
    exit 2
}

SNAPSHOT="$(mktemp)"
trap 'rm -f "${SNAPSHOT}"' EXIT
stage3_struct_ops_snapshot "${SNAPSHOT}"
stage3_has_struct_ops "${SNAPSHOT}" && { echo "Residual struct_ops exists." >&2; exit 2; }
[[ -z "$(pgrep -x prefetch_trace 2>/dev/null || true)$(pgrep -x chunk_trace 2>/dev/null || true)" ]] || {
    echo "Residual trace process." >&2
    exit 2
}
[[ -z "$(pgrep -f '^.*/(prefetch_none|prefetch_always_max|prefetch_adaptive_sequential|eviction_fifo|prefetch_always_max_cycle_moe|prefetch_cooperative)$' 2>/dev/null || true)" ]] || {
    echo "Residual policy loader." >&2
    exit 2
}

GPU_FREE="$(gpu_free_bytes)"
TARGET_BYTES="$(bytes_from_size "${TARGET}")"
HEADROOM_BYTES="$(bytes_from_size "${STAGE4_SAFETY_HEADROOM}")"
((GPU_FREE > TARGET_BYTES + HEADROOM_BYTES)) || {
    echo "Cannot reserve enough memory for target effective capacity ${TARGET}." >&2
    exit 2
}

python3 - "${OUTPUT}" "${RATIO}" "${TARGET_BYTES}" "${HEADROOM_BYTES}" "${GPU_FREE}" <<'PY'
import json, sys
from pathlib import Path
ratio = float(sys.argv[2])
target, headroom, free = map(int, sys.argv[3:6])
data = {
    "evidence_class": "GPU_EXT_STAGE4_RUNTIME_SAFETY",
    "ratio": ratio,
    "target_effective_gpu_bytes": target,
    "safety_headroom_bytes": headroom,
    "gpu_free_before_bytes": free,
    "planned_managed_working_set_bytes": int(target * ratio),
    "checks": {
        "custom_module_loaded": True,
        "no_active_compute_process": True,
        "no_residual_struct_ops": True,
        "no_residual_trace": True,
        "disk_headroom_32gib": True,
        "host_memory_margin_16gib": True,
        "reserve_possible": True,
    },
    "status": "PASS_STAGE4_RUNTIME_SAFETY",
}
Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
Path(sys.argv[1]).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
print(json.dumps(data, indent=2, sort_keys=True))
PY
