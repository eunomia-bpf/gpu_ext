#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "$0")/stage4_common.sh"

RATIO=""
TARGET="${STAGE4_TARGET_EFFECTIVE}"
EXPERIMENT="reduced_capacity"
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; ++i)); do
    case "${ARGS[i]}" in
        --ratio) RATIO="${ARGS[i + 1]}" ;;
        --target-effective) TARGET="${ARGS[i + 1]}" ;;
        --experiment) EXPERIMENT="${ARGS[i + 1]}" ;;
    esac
done
[[ -n "${RATIO}" ]] || { echo "--ratio is required" >&2; exit 2; }

if [[ "${EXPERIMENT}" != natural_stage4 ]]; then
    SAFETY_FILE="${STAGE4_RESULTS}/preflight/runtime_${EXPERIMENT}_${TARGET}_${RATIO}_$(timestamp_utc).json"
    bash "$(dirname "$0")/check_stage4_runtime_safety.sh" "${RATIO}" "${TARGET}" "${SAFETY_FILE}"
else
    stage4_require_disk
    NATURAL_FREE="$(gpu_free_bytes)"
    NATURAL_PLANNED="$(python3 - "${NATURAL_FREE}" "${RATIO}" <<'PY'
import sys
print(int(int(sys.argv[1]) * float(sys.argv[2])))
PY
)"
    HOST_AVAILABLE="$(( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo) * 1024 ))"
    ((HOST_AVAILABLE > NATURAL_PLANNED + (16 << 30))) || {
        echo "Natural-capacity run lacks the host-memory plus 16 GiB margin." >&2
        exit 2
    }
fi

export STAGE3_RESULTS="${STAGE4_RESULTS}"
export GPU_EXT_RUN_EVIDENCE_CLASS="GPU_EXT_STAGE4_RUN"
exec bash "$(dirname "$0")/run_stage3_case.sh" "$@"
