#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "$0")/stage3_common.sh"

EXPERIMENT=""
POLICY="custom_no_policy"
KIND="timing"
RATIO="na"
INDEX="1"
FIRST_TOUCH="full"
PREFETCH_CPU="no"
KERNEL_MODE="vector-add"

while (($#)); do
    case "$1" in
        --experiment) EXPERIMENT="$2"; shift 2 ;;
        --policy) POLICY="$2"; shift 2 ;;
        --kind) KIND="$2"; shift 2 ;;
        --ratio) RATIO="$2"; shift 2 ;;
        --index) INDEX="$2"; shift 2 ;;
        --first-touch) FIRST_TOUCH="$2"; shift 2 ;;
        --prefetch-cpu) PREFETCH_CPU="$2"; shift 2 ;;
        --kernel-mode) KERNEL_MODE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -n "${EXPERIMENT}" ]] || { echo "--experiment is required" >&2; exit 2; }
[[ "${KIND}" =~ ^(timing|trace|nsys)$ ]] || { echo "invalid --kind" >&2; exit 2; }
stage3_require_runtime
stage3_build

if [[ "${EXPERIMENT}" == oversub || "${EXPERIMENT}" == joint_policy ]]; then
    AVAILABLE_KIB="$(df -Pk "${STAGE3_RESULTS}" | awk 'NR == 2 {print $4}')"
    REQUIRED_KIB="$(( ${STAGE3_MIN_FREE_GIB:-32} * 1024 * 1024 ))"
    if (( AVAILABLE_KIB < REQUIRED_KIB )); then
        echo "Insufficient result-disk headroom: $((AVAILABLE_KIB / 1024 / 1024)) GiB available; ${STAGE3_MIN_FREE_GIB:-32} GiB required." >&2
        exit 2
    fi
fi

if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d')" ]]; then
    echo "Another GPU compute process is active." >&2
    exit 2
fi

RUN_ID="$(timestamp_utc)_${KIND}_r${INDEX}"
RUN_DIR="${STAGE3_RESULTS}/${EXPERIMENT}/${POLICY}/${RATIO}/${RUN_ID}"
mkdir -p "${RUN_DIR}"
TRACE_PIDS=()
POLICY_PID=""
CLEANUP_OK=1

cleanup_stage3_case() {
    local pid
    for pid in "${TRACE_PIDS[@]:-}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill -TERM "${pid}" 2>/dev/null || true
            wait "${pid}" 2>/dev/null || true
        fi
    done
    TRACE_PIDS=()
    if [[ -n "${POLICY_PID}" ]] && kill -0 "${POLICY_PID}" 2>/dev/null; then
        kill -TERM "${POLICY_PID}" 2>/dev/null || true
        wait "${POLICY_PID}" 2>/dev/null || true
    fi
    POLICY_PID=""
    if [[ -n "${STAGE3_BPFTOOL:-}" ]]; then
        if ! stage3_struct_ops_snapshot "${RUN_DIR}/bpftool_after.txt"; then
            CLEANUP_OK=0
        elif stage3_has_struct_ops "${RUN_DIR}/bpftool_after.txt"; then
            CLEANUP_OK=0
        fi
    fi
}
trap cleanup_stage3_case EXIT
trap 'cleanup_stage3_case; exit 130' INT
trap 'cleanup_stage3_case; exit 143' TERM

capture_kernel_log "${RUN_DIR}/kernel_log_before.txt"
nvidia-smi >"${RUN_DIR}/nvidia_smi_before.txt"
GPU_USED_BEFORE="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${CUDA_DEVICE:-0}" | head -n1 | tr -d ' ')"
printf '%s\n' "${GPU_USED_BEFORE}" >"${RUN_DIR}/gpu_memory_used_before_mib.txt"
if [[ "${EXPERIMENT}" == oversub || "${EXPERIMENT}" == joint_policy ]]; then
    bash "$(dirname "$0")/check_oversub_safety.sh" "${RATIO}" \
        "${RUN_DIR}/oversub_safety_before.json"
fi
stage3_struct_ops_snapshot "${RUN_DIR}/bpftool_before.txt"
"${STAGE3_BPFTOOL}" prog show >"${RUN_DIR}/bpftool_prog_before.txt"
if stage3_has_struct_ops "${RUN_DIR}/bpftool_before.txt"; then
    echo "Unknown or residual struct_ops map exists; refusing attach." >&2
    exit 2
fi

POLICY_BINARY="$(stage3_policy_binary "${POLICY}")"
if [[ -n "${POLICY_BINARY}" ]]; then
    "${POLICY_BINARY}" >"${RUN_DIR}/policy.log" 2>&1 &
    POLICY_PID=$!
    printf '%s\n' "${POLICY_PID}" >"${RUN_DIR}/policy.pid"
    attached=0
    for _ in {1..50}; do
        sleep 0.1
        kill -0 "${POLICY_PID}" 2>/dev/null || break
        if stage3_struct_ops_snapshot "${RUN_DIR}/bpftool_attached.txt" && \
           stage3_has_struct_ops "${RUN_DIR}/bpftool_attached.txt"; then
            attached=1
            break
        fi
    done
    (( attached == 1 )) || { echo "Policy attach failed: ${POLICY}" >&2; exit 1; }
    "${STAGE3_BPFTOOL}" prog show >"${RUN_DIR}/bpftool_prog_attached.txt"
else
    printf 'custom module, no policy attached\n' >"${RUN_DIR}/policy.log"
    cp "${RUN_DIR}/bpftool_before.txt" "${RUN_DIR}/bpftool_attached.txt"
    cp "${RUN_DIR}/bpftool_prog_before.txt" "${RUN_DIR}/bpftool_prog_attached.txt"
fi

TRACE_START_NS=""
if [[ "${KIND}" != timing ]]; then
    TRACE_START_NS="$(stage3_monotonic_ns)"
    "${EXTENSION_DIR}/prefetch_trace" >"${RUN_DIR}/prefetch_decision_trace.csv" \
        2>"${RUN_DIR}/prefetch_decision_trace.stderr" &
    TRACE_PIDS+=("$!")
    "${EXTENSION_DIR}/chunk_trace" >"${RUN_DIR}/chunk_trace.csv" \
        2>"${RUN_DIR}/chunk_trace.stderr" &
    TRACE_PIDS+=("$!")
    printf '%s\n' "${TRACE_PIDS[@]}" >"${RUN_DIR}/trace.pids"
    sleep 0.3
    for pid in "${TRACE_PIDS[@]}"; do
        kill -0 "${pid}" 2>/dev/null || { echo "Trace helper exited early." >&2; exit 1; }
    done
else
    : >"${RUN_DIR}/prefetch_decision_trace.csv"
    : >"${RUN_DIR}/chunk_trace.csv"
fi

case "${EXPERIMENT}" in
    trace_semantics|trace_overhead)
        WORKLOAD=("${PROGRAM}" --bytes 256M --allocation managed --iterations 1
            --cpu-retouch none --stop-after-hot yes --gpu-prefetch no
            --cpu-prefetch-before-retouch no --verify yes
            --kernel-mode vector-add --output "${RUN_DIR}/program.jsonl")
        ;;
    cpu_first_touch)
        WORKLOAD=("${PROGRAM}" --bytes 256M --allocation managed --iterations 1
            --cpu-retouch none --stop-after-cpu-first-touch yes
            --cpu-first-touch "${FIRST_TOUCH}"
            --prefetch-cpu-before-first-touch "${PREFETCH_CPU}"
            --gpu-prefetch no --cpu-prefetch-before-retouch no --verify yes
            --output "${RUN_DIR}/program.jsonl")
        ;;
    array_migration)
        WORKLOAD=("${PROGRAM}" --bytes 256M --allocation managed --iterations 1
            --cpu-retouch none --stop-after-hot yes --gpu-prefetch no
            --cpu-prefetch-before-retouch no --verify yes
            --kernel-mode "${KERNEL_MODE}" --output "${RUN_DIR}/program.jsonl")
        ;;
    oversub|joint_policy)
        WORKLOAD=("${PHASE_SCAN_PROGRAM}" --total-bytes auto --working-set-ratio "${RATIO}"
            --region-a-ratio 0.5 --cycles 1 --gpu-id "${CUDA_DEVICE:-0}"
            --verify yes --output "${RUN_DIR}/program.jsonl")
        ;;
    *) echo "Unknown experiment: ${EXPERIMENT}" >&2; exit 2 ;;
esac

printf '%q ' "${WORKLOAD[@]}" >"${RUN_DIR}/command.txt"
printf '\n' >>"${RUN_DIR}/command.txt"
WORKLOAD_START_NS="$(stage3_monotonic_ns)"
set +e
if [[ "${KIND}" == nsys ]]; then
    timeout --signal=TERM --kill-after=15s 300s nsys profile \
        --trace=cuda,nvtx,osrt --cuda-um-cpu-page-faults=true \
        --cuda-um-gpu-page-faults=true --force-overwrite=true \
        --output="${RUN_DIR}/representative" "${WORKLOAD[@]}" \
        >"${RUN_DIR}/program.log" 2>&1
    RC=$?
    if ((RC == 0)); then
        nsys stats --force-export=true --force-overwrite=true --format csv \
            --report um_sum,um_total_sum,um_cpu_page_faults_sum,cuda_gpu_kern_sum,nvtx_sum \
            --output "${RUN_DIR}/nsys_stats" "${RUN_DIR}/representative.nsys-rep" \
            >"${RUN_DIR}/nsys_stats.stdout" 2>"${RUN_DIR}/nsys_stats.stderr"
    fi
elif [[ "${EXPERIMENT}" == cpu_first_touch && -x /usr/bin/time ]]; then
    timeout --signal=TERM --kill-after=15s 300s /usr/bin/time -v \
        -o "${RUN_DIR}/resource_usage.txt" "${WORKLOAD[@]}" \
        >"${RUN_DIR}/program.log" 2>&1
    RC=$?
else
    timeout --signal=TERM --kill-after=15s 300s "${WORKLOAD[@]}" \
        >"${RUN_DIR}/program.log" 2>&1
    RC=$?
fi
set -e
WORKLOAD_END_NS="$(stage3_monotonic_ns)"

cleanup_stage3_case
trap - EXIT INT TERM
(( CLEANUP_OK == 1 )) || { echo "Policy or trace cleanup failed." >&2; RC=1; }
"${STAGE3_BPFTOOL}" prog show >"${RUN_DIR}/bpftool_prog_after.txt" || RC=1

capture_kernel_log "${RUN_DIR}/kernel_log_after.txt"
nvidia-smi >"${RUN_DIR}/nvidia_smi_after.txt" || RC=1
GPU_USED_AFTER="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${CUDA_DEVICE:-0}" | head -n1 | tr -d ' ')"
printf '%s\n' "${GPU_USED_AFTER}" >"${RUN_DIR}/gpu_memory_used_after_mib.txt"
ACTIVE_AFTER="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | paste -sd, -)"
[[ -z "${ACTIVE_AFTER}" ]] || RC=1
(( GPU_USED_AFTER <= GPU_USED_BEFORE + 16 )) || RC=1
if [[ "${EXPERIMENT}" == oversub || "${EXPERIMENT}" == joint_policy ]]; then
    bash "$(dirname "$0")/check_oversub_safety.sh" "${RATIO}" \
        "${RUN_DIR}/oversub_safety_after.json" || RC=1
fi
awk 'NR == 1 || tolower($0) ~ /evict/' "${RUN_DIR}/chunk_trace.csv" \
    >"${RUN_DIR}/eviction_trace.csv" 2>/dev/null || : >"${RUN_DIR}/eviction_trace.csv"
printf '%s\n' "${RC}" >"${RUN_DIR}/exit_code"

python3 - "${RUN_DIR}" "${EXPERIMENT}" "${POLICY}" "${KIND}" "${RATIO}" \
    "${INDEX}" "${RC}" "${TRACE_START_NS}" "${WORKLOAD_START_NS}" "${WORKLOAD_END_NS}" \
    "${FIRST_TOUCH}" "${PREFETCH_CPU}" "${KERNEL_MODE}" "${CLEANUP_OK}" \
    "${GPU_USED_BEFORE}" "${GPU_USED_AFTER}" "${ACTIVE_AFTER}" <<'PY'
import json, re, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for line in (root / "program.jsonl").read_text(errors="replace").splitlines() if (root / "program.jsonl").exists() else []:
    try: rows.append(json.loads(line))
    except json.JSONDecodeError: pass
before = (root / "kernel_log_before.txt").read_text(errors="replace")
after = (root / "kernel_log_after.txt").read_text(errors="replace")
xid = lambda text: len(re.findall(r"NVRM:\s*Xid|NVIDIA.*Xid", text, re.I))
data = {
    "evidence_class": "GPU_EXT_STAGE3_RUN",
    "experiment": sys.argv[2], "policy": sys.argv[3], "run_kind": sys.argv[4],
    "ratio": sys.argv[5], "repetition": int(sys.argv[6]), "exit_code": int(sys.argv[7]),
    "trace_start_monotonic_ns": int(sys.argv[8]) if sys.argv[8] else None,
    "workload_start_monotonic_ns": int(sys.argv[9]),
    "workload_end_monotonic_ns": int(sys.argv[10]),
    "first_touch_pattern": sys.argv[11], "prefetch_cpu_before_first_touch": sys.argv[12],
    "kernel_mode": sys.argv[13], "struct_ops_detached": sys.argv[14] == "1",
    "gpu_memory_used_before_mib": int(sys.argv[15]),
    "gpu_memory_used_after_mib": int(sys.argv[16]),
    "gpu_memory_released": int(sys.argv[16]) <= int(sys.argv[15]) + 16,
    "active_compute_after": sys.argv[17] or None,
    "correct": int(sys.argv[7]) == 0 and bool(rows)
               and all(row.get("correct", False) for row in rows),
    "completed": int(sys.argv[7]) == 0,
    "xid_before": xid(before), "xid_after": xid(after),
    "xid_delta": max(0, xid(after) - xid(before)),
    "trace_pid_attribution": "OWNER_TGID_AND_ISOLATED_WINDOW",
    "trace_limit": "Events without a usable owner_tgid retain isolated-window attribution only.",
}
(root / "manifest.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
if (data["exit_code"] or not data["correct"] or not data["struct_ops_detached"]
        or data["xid_delta"] or not data["gpu_memory_released"] or data["active_compute_after"]):
    raise SystemExit(1)
PY

echo "${RUN_DIR}"
exit "${RC}"
