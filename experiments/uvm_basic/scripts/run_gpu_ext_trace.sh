#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "$0")/common.sh"
EXTENSION_DIR="${GPU_EXT_ROOT}/extension"
STAGE2_DIR="${RESULTS_DIR}/stage2"
read -r -a POLICIES <<<"${UVM_BASIC_STAGE2_POLICIES:-custom_no_policy prefetch_none prefetch_always_max prefetch_adaptive_sequential}"
TIMING_RUNS="${UVM_BASIC_STAGE2_TIMING_RUNS:-10}"
TRACE_RUNS="${UVM_BASIC_STAGE2_TRACE_RUNS:-3}"
NSYS_RUNS="${UVM_BASIC_STAGE2_NSYS_RUNS:-1}"
TIMING_RUNS_1G="${UVM_BASIC_STAGE2_TIMING_RUNS_1G:-5}"
TRACE_RUNS_1G="${UVM_BASIC_STAGE2_TRACE_RUNS_1G:-1}"
TRACE_PIDS=()
POLICY_PID=""
RUN_POLICY_PID=""
CLEANUP_FAILED=0
mkdir -p "${STAGE2_DIR}"

resolve_bpftool() {
    local candidate
    if [[ -n "${GPU_EXT_BPFTOOL:-}" && -x "${GPU_EXT_BPFTOOL}" ]]; then
        BPFTOOL="${GPU_EXT_BPFTOOL}"
        return
    fi
    for candidate in \
        "${GPU_EXT_ROOT}/tools/bpftool-stage2/bpftool" \
        /usr/lib/linux-hwe-*-tools-*/bpftool \
        /usr/lib/linux-tools/*/bpftool; do
        if [[ -x "${candidate}" ]] \
            && "${candidate}" struct_ops help >/dev/null 2>&1 \
            && "${candidate}" version 2>/dev/null | grep -Eq 'bpftool v7\.[6-9]|bpftool v[89]\.'; then
            BPFTOOL="$(readlink -f "${candidate}")"
            return
        fi
    done
    echo "No usable bpftool with struct_ops support." >&2
    exit 2
}

struct_ops_snapshot() {
    local output="$1"
    # `struct_ops list` crashes against this custom kernel implementation.
    # Map enumeration still exposes every registered BPF_MAP_TYPE_STRUCT_OPS map.
    if ! "${BPFTOOL}" -j map show >"${output}" 2>&1; then
        echo "bpftool map show failed; refusing to infer an empty struct_ops registry." >&2
        return 1
    fi
}

has_uvm_struct_ops() {
    local input="$1"
    grep -Eiq '"type"[[:space:]]*:[[:space:]]*"struct_ops"' "${input}"
}

cleanup_case() {
    local pid snapshot
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
    if [[ -n "${CURRENT_RUN_DIR:-}" ]]; then
        snapshot="${CURRENT_RUN_DIR}/bpftool_after.txt"
        if ! struct_ops_snapshot "${snapshot}"; then
            CLEANUP_FAILED=1
        elif has_uvm_struct_ops "${snapshot}"; then
            echo "gpu_mem_ops remains after exact-PID cleanup; stopping matrix." >&2
            CLEANUP_FAILED=1
        fi
    fi
}
trap cleanup_case EXIT
trap 'cleanup_case; exit 130' INT
trap 'cleanup_case; exit 143' TERM

if ! grep -q 'uvm_bpf_call_gpu_page_prefetch' /proc/kallsyms 2>/dev/null; then
    echo "Custom gpu_ext prefetch hook is not visible; no policy or trace was attached." >&2
    echo "Run: bash ${UVM_BASIC_DIR}/scripts/SAFE_GPU_EXT_COMMANDS.sh inspect" >&2
    exit 2
fi
bash "${UVM_BASIC_DIR}/scripts/check_gpu_ext_stage2.sh" >/dev/null
python3 - "${RESULTS_DIR}/gpu_ext_stage2_preflight.json" <<'PY'
import json, sys
data=json.load(open(sys.argv[1]))
if not data.get("custom_binary_loaded"):
    raise SystemExit("preflight does not prove that the loaded nvidia_uvm is the custom binary")
if not data.get("custom_module", {}).get("version_kernel_match"):
    raise SystemExit("custom module driver version or kernel vermagic mismatch")
if not data.get("all_binaries_ready"):
    raise SystemExit("one or more Stage 2 extension binaries is unavailable")
PY
if [[ ${EUID} -ne 0 ]]; then
    echo "Root BPF capability is required. Use only the reviewed SAFE_GPU_EXT_COMMANDS.sh path." >&2
    exit 2
fi
resolve_bpftool
INITIAL_STRUCT_OPS="${STAGE2_DIR}/bpftool_initial_$(timestamp_utc).txt"
struct_ops_snapshot "${INITIAL_STRUCT_OPS}"
if has_uvm_struct_ops "${INITIAL_STRUCT_OPS}"; then
    echo "An existing gpu_mem_ops policy is attached; refusing to replace it." >&2
    exit 2
fi

build_uvm_basic
make -C "${EXTENSION_DIR}" -j"${JOBS:-4}" \
    prefetch_trace chunk_trace prefetch_none prefetch_always_max prefetch_adaptive_sequential

monotonic_ns() { python3 -c 'import time; print(time.monotonic_ns())'; }

start_policy() {
    local policy="$1" run_dir="$2"
    if [[ "${policy}" == custom_no_policy ]]; then return 0; fi
    "${EXTENSION_DIR}/${policy}" >"${run_dir}/policy.log" 2>&1 &
    POLICY_PID=$!
    printf '%s\n' "${POLICY_PID}" >"${run_dir}/policy.pid"
    for _ in {1..30}; do
        sleep 0.1
        kill -0 "${POLICY_PID}" 2>/dev/null || return 1
        if struct_ops_snapshot "${run_dir}/bpftool_attached.txt" \
            && has_uvm_struct_ops "${run_dir}/bpftool_attached.txt"; then
            return 0
        fi
    done
    return 1
}

start_traces() {
    local run_dir="$1"
    "${EXTENSION_DIR}/prefetch_trace" >"${run_dir}/prefetch_trace.csv" \
        2>"${run_dir}/prefetch_trace.stderr" &
    TRACE_PIDS+=("$!")
    "${EXTENSION_DIR}/chunk_trace" >"${run_dir}/chunk_trace.csv" \
        2>"${run_dir}/chunk_trace.stderr" &
    TRACE_PIDS+=("$!")
    printf '%s\n' "${TRACE_PIDS[@]}" >"${run_dir}/trace.pids"
    sleep 0.3
    local pid
    for pid in "${TRACE_PIDS[@]}"; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "A Stage 2 trace helper exited before the workload started." >&2
            return 1
        fi
    done
}

write_manifest() {
    local run_dir="$1" policy="$2" kind="$3" size="$4" index="$5" rc="$6"
    python3 - "${run_dir}" "${policy}" "${kind}" "${size}" "${index}" "${rc}" \
        "${TRACE_START_NS:-}" "${WORKLOAD_START_NS:-}" "${WORKLOAD_END_NS:-}" \
        "${TRACE_STOP_NS:-}" "${RUN_POLICY_PID:-}" <<'PY'
import hashlib, json, re, sys
from pathlib import Path
root = Path(sys.argv[1])
program = []
path = root / "program.jsonl"
if path.exists():
    for line in path.read_text(errors="replace").splitlines():
        try: program.append(json.loads(line))
        except json.JSONDecodeError: pass
def digest(path):
    if not path.is_file(): return None
    return hashlib.sha256(path.read_bytes()).hexdigest()
data = {
    "evidence_class": "GPU_EXT_STAGE2_RUN",
    "policy": sys.argv[2], "run_kind": sys.argv[3], "size": sys.argv[4],
    "repetition": int(sys.argv[5]), "exit_code": int(sys.argv[6]),
    "trace_start_monotonic_ns": int(sys.argv[7]) if sys.argv[7] else None,
    "workload_start_monotonic_ns": int(sys.argv[8]) if sys.argv[8] else None,
    "workload_end_monotonic_ns": int(sys.argv[9]) if sys.argv[9] else None,
    "trace_stop_monotonic_ns": int(sys.argv[10]) if sys.argv[10] else None,
    "policy_loader_pid": int(sys.argv[11]) if sys.argv[11] else None,
    "trace_process_pids": [int(value) for value in (root / "trace.pids").read_text().split()]
                           if (root / "trace.pids").exists() else [],
    "workload_pid": program[0].get("process_id") if program else None,
    "trace_has_pid_fields": True,
    "attribution": "PID_AND_ISOLATED_WINDOW" if program else "UNAVAILABLE",
    "correct": bool(program) and all(row.get("correct") for row in program if not row.get("skipped")),
    "struct_ops_detached": not re.search(
        r"gpu_mem_ops|uvm_ops", (root / "bpftool_after.txt").read_text(errors="replace"), re.I
    ) if (root / "bpftool_after.txt").exists() else False,
    "files": {p.name: digest(p) for p in root.iterdir() if p.is_file()},
}
(root / "manifest.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
}

run_one() {
    local policy="$1" kind="$2" size="$3" index="$4"
    local run_id="$(timestamp_utc)_${size}_${kind}_r${index}"
    local run_dir="${STAGE2_DIR}/${policy}/${run_id}" rc before_xid after_xid
    CURRENT_RUN_DIR="${run_dir}"
    RUN_POLICY_PID=""
    if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d')" ]]; then
        echo "Another CUDA workload is active; refusing an ambiguous trace window." >&2
        return 1
    fi
    mkdir -p "${run_dir}"
    cleanup_case
    (( CLEANUP_FAILED == 0 )) || return 1
    dmesg --color=never >"${run_dir}/dmesg_before.txt"
    before_xid="$(grep -Eic 'NVRM: Xid|NVIDIA.*Xid' "${run_dir}/dmesg_before.txt" || true)"
    struct_ops_snapshot "${run_dir}/bpftool_before.txt"
    if has_uvm_struct_ops "${run_dir}/bpftool_before.txt"; then return 1; fi
    if ! start_policy "${policy}" "${run_dir}"; then
        echo "Policy attach failed: ${policy}" >&2
        cleanup_case
        return 1
    fi
    RUN_POLICY_PID="${POLICY_PID}"
    if [[ "${policy}" == custom_no_policy ]]; then
        printf 'custom driver baseline: no policy attached\n' >"${run_dir}/policy.log"
        cp "${run_dir}/bpftool_before.txt" "${run_dir}/bpftool_attached.txt"
    fi
    TRACE_START_NS="" WORKLOAD_START_NS="" WORKLOAD_END_NS="" TRACE_STOP_NS=""
    if [[ "${kind}" == trace || "${kind}" == nsys ]]; then
        TRACE_START_NS="$(monotonic_ns)"
        if ! start_traces "${run_dir}"; then
            cleanup_case
            return 1
        fi
    else
        : >"${run_dir}/prefetch_trace.csv"
        : >"${run_dir}/chunk_trace.csv"
    fi
    printf '%q ' "${PROGRAM}" --bytes "${size}" --allocation managed --iterations 1 \
        --cpu-retouch none --stop-after-hot yes --gpu-prefetch no \
        --cpu-prefetch-before-retouch no --verify yes --output "${run_dir}/program.jsonl" \
        >"${run_dir}/command.txt"
    printf '\n' >>"${run_dir}/command.txt"
    WORKLOAD_START_NS="$(monotonic_ns)"
    set +e
    if [[ "${kind}" == nsys ]]; then
        timeout --signal=TERM --kill-after=20s "${UVM_BASIC_PROFILE_TIMEOUT_SECONDS:-600}" \
            nsys profile --trace=cuda,nvtx,osrt --cuda-um-cpu-page-faults=true \
            --cuda-um-gpu-page-faults=true --force-overwrite=true \
            --output="${run_dir}/representative" "${PROGRAM}" --bytes "${size}" \
            --allocation managed --iterations 1 --cpu-retouch none --stop-after-hot yes \
            --gpu-prefetch no --cpu-prefetch-before-retouch no --verify yes \
            --output "${run_dir}/program.jsonl" >"${run_dir}/program.log" 2>&1
        rc=$?
        if (( rc == 0 )); then
            nsys stats --force-export=true --force-overwrite=true --format csv \
                --report um_sum,um_total_sum,um_cpu_page_faults_sum,cuda_gpu_kern_sum,nvtx_sum \
                --output "${run_dir}/nsys_stats" "${run_dir}/representative.nsys-rep" \
                >"${run_dir}/nsys_stats.stdout" 2>"${run_dir}/nsys_stats.stderr"
            for phase in cpu_first_touch kernel_1_demand kernel_2_hot; do
                nsys stats --force-overwrite=true --format csv --report um_total_sum \
                    --filter-nvtx "${phase}" --output "${run_dir}/nsys_phase_${phase}" \
                    "${run_dir}/representative.sqlite" \
                    >>"${run_dir}/nsys_stats.stdout" 2>>"${run_dir}/nsys_stats.stderr"
            done
        fi
    else
        timeout --signal=TERM --kill-after=15s "${UVM_BASIC_TIMEOUT_SECONDS:-300}" \
            "${PROGRAM}" --bytes "${size}" --allocation managed --iterations 1 \
            --cpu-retouch none --stop-after-hot yes --gpu-prefetch no \
            --cpu-prefetch-before-retouch no --verify yes --output "${run_dir}/program.jsonl" \
            >"${run_dir}/program.log" 2>&1
        rc=$?
    fi
    set -e
    WORKLOAD_END_NS="$(monotonic_ns)"
    if [[ "${kind}" == trace || "${kind}" == nsys ]]; then TRACE_STOP_NS="$(monotonic_ns)"; fi
    cleanup_case
    dmesg --color=never >"${run_dir}/dmesg_after.txt"
    nvidia-smi >"${run_dir}/nvidia_smi_after.txt" 2>&1 || {
        echo "GPU disappeared from nvidia-smi; stopping Stage 2." >&2
        return 1
    }
    after_xid="$(grep -Eic 'NVRM: Xid|NVIDIA.*Xid' "${run_dir}/dmesg_after.txt" || true)"
    printf '%s\n' "${rc}" >"${run_dir}/exit_code"
    "${BPFTOOL}" prog show >"${run_dir}/bpftool_prog_after.txt" 2>&1 || true
    write_manifest "${run_dir}" "${policy}" "${kind}" "${size}" "${index}" "${rc}"
    (( CLEANUP_FAILED == 0 )) || return 1
    (( rc == 0 )) || return "${rc}"
    python3 - "${run_dir}/program.jsonl" <<'PY'
import json, sys
rows=[json.loads(line) for line in open(sys.argv[1]) if line.strip()]
assert rows and all(row.get("correct") for row in rows if not row.get("skipped"))
assert {row["phase"] for row in rows if not row.get("skipped")} >= {"kernel_1_demand", "kernel_2_hot"}
PY
    if (( after_xid > before_xid )); then
        echo "New NVIDIA Xid detected; stopping Stage 2." >&2
        return 1
    fi
    if grep -Eiq 'UVM.*fatal|GPU has fallen off|NVRM.*GPU.*lost|kernel BUG|Oops:' \
        "${run_dir}/dmesg_after.txt"; then
        echo "Fatal GPU/UVM kernel evidence detected; stopping Stage 2." >&2
        return 1
    fi
}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
for policy in "${POLICIES[@]}"; do
    case "${policy}" in
        custom_no_policy|prefetch_none|prefetch_always_max|prefetch_adaptive_sequential) ;;
        *) echo "Unsupported Stage 2 policy: ${policy}" >&2; exit 2 ;;
    esac
    safe_size_or_skip 256M
    for index in $(seq 1 "${TIMING_RUNS}"); do run_one "${policy}" timing 256M "${index}" || exit 1; done
    for index in $(seq 1 "${TRACE_RUNS}"); do run_one "${policy}" trace 256M "${index}" || exit 1; done
    if command -v nsys >/dev/null && (( NSYS_RUNS > 0 )); then
        for index in $(seq 1 "${NSYS_RUNS}"); do run_one "${policy}" nsys 256M "${index}" || exit 1; done
    else
        printf 'Nsight Systems unavailable\n' >"${STAGE2_DIR}/${policy}/SKIPPED_NSYS.txt"
    fi
    if [[ "${UVM_BASIC_STAGE2_RUN_1G:-yes}" == yes ]] && safe_size_or_skip 1G; then
        for index in $(seq 1 "${TIMING_RUNS_1G}"); do run_one "${policy}" timing 1G "${index}" || exit 1; done
        for index in $(seq 1 "${TRACE_RUNS_1G}"); do run_one "${policy}" trace 1G "${index}" || exit 1; done
    else
        printf '1 GiB skipped by safety or configuration\n' >"${STAGE2_DIR}/${policy}/SKIPPED_1G.txt"
    fi
done

python3 "${UVM_BASIC_DIR}/analysis/summarize_stage2.py" --experiment-dir "${UVM_BASIC_DIR}"
echo "Stage 2 completed with exact-PID cleanup and no residual gpu_mem_ops."
