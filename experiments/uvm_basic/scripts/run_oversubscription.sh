#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "$0")/common.sh"
[[ "${ALLOW_UVM_OVERSUBSCRIPTION:-}" == YES ]] || {
    echo "Oversubscription is disabled by default. Set ALLOW_UVM_OVERSUBSCRIPTION=YES after stopping other GPU work." >&2
    exit 2
}
build_uvm_basic

free_gpu="$(gpu_free_bytes)"
available_host="$(awk '/MemAvailable:/{print $2 * 1024}' /proc/meminfo)"
start=$((free_gpu / 3 / 2))
step=$((free_gpu / 3 / 20))
limit=$((free_gpu / 3 + free_gpu / 3 / 2))
host_limit=$((available_host / 6))
(( limit > host_limit )) && limit=${host_limit}
(( step < 67108864 )) && step=67108864

STAMP="$(timestamp_utc)"
OUTPUT="${RESULTS_DIR}/oversubscription_${STAMP}.jsonl"
for ((bytes=start; bytes<=limit; bytes+=step)); do
    before_xid="$(dmesg 2>/dev/null | grep -c 'NVRM: Xid' || true)"
    if ! timeout --signal=TERM --kill-after=15s 600 \
        "${PROGRAM}" --bytes "${bytes}" --allocation managed --iterations 1 \
        --cpu-retouch page --gpu-prefetch no --cpu-prefetch-before-retouch no \
        --verify yes --output "${OUTPUT}"; then
        echo "Stopping after first failure at ${bytes} bytes per array." >&2
        exit 1
    fi
    after_xid="$(dmesg 2>/dev/null | grep -c 'NVRM: Xid' || true)"
    if (( after_xid > before_xid )); then
        echo "Stopping after a new NVIDIA Xid." >&2
        exit 1
    fi
done
