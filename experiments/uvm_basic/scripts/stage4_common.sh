#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/stage3_common.sh"

STAGE4_RESULTS="${RESULTS_DIR}/stage4"
STAGE4_TARGET_EFFECTIVE="${STAGE4_TARGET_EFFECTIVE:-8G}"
STAGE4_SAFETY_HEADROOM="${STAGE4_SAFETY_HEADROOM:-1G}"
STAGE4_MIN_DISK_GIB="${STAGE4_MIN_DISK_GIB:-32}"
STAGE4_TIMEOUT_SECONDS=300
mkdir -p "${STAGE4_RESULTS}"

stage4_build() {
    stage3_build
}

stage4_require_disk() {
    local available_kib required_kib
    available_kib="$(df -Pk "${STAGE4_RESULTS}" | awk 'NR == 2 {print $4}')"
    required_kib="$((STAGE4_MIN_DISK_GIB * 1024 * 1024))"
    ((available_kib >= required_kib)) || {
        echo "Stage 4 needs ${STAGE4_MIN_DISK_GIB} GiB free in the result filesystem." >&2
        return 2
    }
}

stage4_require_host_memory() {
    local target="$1" ratio="$2" target_bytes planned available gpu_free headroom reserve required
    target_bytes="$(bytes_from_size "${target}")"
    planned="$(python3 - "${target_bytes}" "${ratio}" <<'PY'
import sys
print(int(int(sys.argv[1]) * float(sys.argv[2])))
PY
)"
    gpu_free="$(gpu_free_bytes)"
    headroom="$(bytes_from_size "${STAGE4_SAFETY_HEADROOM}")"
    reserve=$((gpu_free > target_bytes + headroom ? gpu_free - target_bytes - headroom : 0))
    required=$((planned + reserve + (16 << 30)))
    available="$(( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo) * 1024 ))"
    ((available > required)) || {
        echo "Host MemAvailable does not cover managed working set + reserve + 16 GiB." >&2
        return 2
    }
}

stage4_latest_case_exit() {
    local root="$1" file
    file="$(find "${root}" -type f -name exit_code -printf '%T@ %p\n' 2>/dev/null |
        sort -nr | head -n1 | cut -d' ' -f2-)"
    [[ -n "${file}" ]] || return 1
    cat "${file}"
}
