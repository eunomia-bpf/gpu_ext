#!/usr/bin/env bash
set -Eeuo pipefail

UVM_BASIC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_EXT_ROOT="$(cd "${UVM_BASIC_DIR}/../.." && pwd)"
BUILD_DIR="${UVM_BASIC_DIR}/build"
RESULTS_DIR="${UVM_BASIC_DIR}/results"
PROGRAM="${BUILD_DIR}/uvm_vector_add"

mkdir -p "${BUILD_DIR}" "${RESULTS_DIR}"

timestamp_utc() {
    date -u +%Y%m%dT%H%M%SZ
}

bytes_from_size() {
    python3 - "$1" <<'PY'
import re
import sys

match = re.fullmatch(r"([0-9]+)([KMG]?(?:i?B)?)?", sys.argv[1], re.I)
if not match:
    raise SystemExit(f"invalid size: {sys.argv[1]}")
value = int(match.group(1))
suffix = (match.group(2) or "").upper()
factor = {"": 1, "K": 1 << 10, "KB": 1 << 10, "KIB": 1 << 10,
          "M": 1 << 20, "MB": 1 << 20, "MIB": 1 << 20,
          "G": 1 << 30, "GB": 1 << 30, "GIB": 1 << 30}[suffix]
print(value * factor)
PY
}

gpu_free_bytes() {
    local free_mib
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${CUDA_DEVICE:-0}" | head -n1 | tr -d ' ')"
    [[ "${free_mib}" =~ ^[0-9]+$ ]] || return 1
    printf '%s\n' "$((free_mib * 1024 * 1024))"
}

safe_size_or_skip() {
    local size="$1"
    local bytes free required limit
    bytes="$(bytes_from_size "${size}")"
    free="$(gpu_free_bytes)"
    required=$((bytes * 3))
    limit=$((free / 5))
    if (( bytes > 1073741824 )); then
        echo "SKIP ${size}: default bytes per array exceeds 1 GiB" >&2
        return 1
    fi
    if (( required > limit )); then
        echo "SKIP ${size}: three arrays need ${required} bytes, above 20% of free GPU memory (${limit})" >&2
        return 1
    fi
}

build_uvm_basic() {
    cmake -S "${UVM_BASIC_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=RelWithDebInfo
    cmake --build "${BUILD_DIR}" --parallel "${JOBS:-4}"
}

capture_kernel_log() {
    local output="$1"
    if dmesg --color=never >"${output}" 2>"${output}.stderr"; then
        printf 'DMESG\n' >"${output}.source"
        return 0
    fi
    if journalctl -k --no-pager >"${output}" 2>>"${output}.stderr" && [[ -s "${output}" ]] &&
       ! grep -q 'not seeing messages from other users' "${output}.stderr" &&
       ! grep -qx -- '-- No entries --' "${output}"; then
        printf 'JOURNALCTL\n' >"${output}.source"
        return 0
    fi
    printf 'KERNEL_LOG_UNAVAILABLE\n' >"${output}"
    printf 'UNAVAILABLE\n' >"${output}.source"
    return 0
}

xid_count_or_unavailable() {
    local log="$1"
    if grep -qx 'KERNEL_LOG_UNAVAILABLE' "${log}"; then
        printf 'UNAVAILABLE\n'
    else
        grep -Eic 'NVRM: Xid|NVIDIA.*Xid' "${log}" || true
    fi
}
