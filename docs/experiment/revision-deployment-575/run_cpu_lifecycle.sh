#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 BPFTIME_SOURCE BPFTIME_BUILD [REPETITIONS]" >&2
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
    usage
    exit 64
fi

bpftime_source=$(realpath "$1")
bpftime_build=$(realpath "$2")
repetitions=${3:-5}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
raw_dir="$script_dir/raw"
target="$raw_dir/lifecycle-target"
cli="$bpftime_build/tools/cli/bpftime"
agent="$bpftime_build/runtime/agent/libbpftime-agent.so"
server="$bpftime_build/runtime/syscall-server/libbpftime-syscall-server.so"

case "$repetitions" in
    ''|*[!0-9]*) echo "repetitions must be a positive integer" >&2; exit 64 ;;
esac
if (( repetitions < 1 )); then
    echo "repetitions must be a positive integer" >&2
    exit 64
fi
if [[ -n ${LD_PRELOAD:-} || -n ${BPFTIME_USED:-} ]]; then
    echo "refusing inherited LD_PRELOAD or BPFTIME_USED state" >&2
    exit 65
fi

for required in "$bpftime_source/CMakeLists.txt" \
                "$bpftime_build/CMakeCache.txt" "$cli" "$agent" "$server"; do
    if [[ ! -e "$required" ]]; then
        echo "missing required path: $required" >&2
        exit 66
    fi
done

if ! grep -Fxq 'BPFTIME_ENABLE_CUDA_ATTACH:BOOL=OFF' \
        "$bpftime_build/CMakeCache.txt"; then
    echo "refusing lifecycle run: build is not explicitly CPU-only" >&2
    exit 65
fi
if ldd "$agent" | grep -Eiq 'libcuda|libcudart|libnvptxcompiler'; then
    echo "refusing lifecycle run: agent links a CUDA library" >&2
    exit 65
fi

mkdir -p "$raw_dir"
cc -std=c11 -O2 -Wall -Wextra -Werror "$script_dir/lifecycle_target.c" \
   -ldl -o "$target"

{
    printf 'key\tvalue\n'
    printf 'kernel\t%s\n' "$(uname -srmo)"
    printf 'uid\t%s\n' "$(id -u)"
    printf 'cpu_model\t%s\n' \
        "$(awk -F': ' '/^model name/{print $2; exit}' /proc/cpuinfo)"
    printf 'cc\t%s\n' "$(cc --version | sed -n '1p')"
    printf 'ptrace_scope\t%s\n' \
        "$(sed -n '1p' /proc/sys/kernel/yama/ptrace_scope)"
    printf 'cuda_attach_build_option\tOFF\n'
    printf 'target_ptrace_opt_in\tPR_SET_PTRACER_ANY\n'
} > "$raw_dir/run-environment.tsv"

summary="$raw_dir/lifecycle.tsv"
printf 'route\trepetition\troute_to_ready_ms\tcommand_ms\ttarget_alive\tbefore_agent_mapping\tafter_agent_or_frida_mapping\tagent_named\tbpftime_used\tipc_socket\tcommand_status\n' > "$summary"

server_pid=''
target_pid=''
shm_path=''
current_shm_name=''

cleanup_processes() {
    if [[ -n "$target_pid" ]] && kill -0 "$target_pid" 2>/dev/null; then
        kill -TERM "$target_pid" 2>/dev/null || true
        wait "$target_pid" 2>/dev/null || true
    fi
    target_pid=''
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    server_pid=''
    if [[ -n "$shm_path" ]]; then
        case "$shm_path" in
            /dev/shm/gpubpf_deploy_audit_*) rm -f -- "$shm_path" ;;
            *) echo "refusing unexpected shared-memory cleanup path: $shm_path" >&2; exit 70 ;;
        esac
    fi
    shm_path=''
}
trap cleanup_processes EXIT INT TERM

wait_for_line() {
    local file=$1
    local pattern=$2
    local attempts=${3:-1000}
    local i
    for ((i = 0; i < attempts; i++)); do
        if grep -q "$pattern" "$file" 2>/dev/null; then
            return 0
        fi
        sleep 0.01
    done
    return 1
}

start_server() {
    local route=$1
    local repetition=$2
    current_shm_name="gpubpf_deploy_audit_${$}_${route}_${repetition}"
    local server_stdout="$raw_dir/${route}-${repetition}-server.stdout"
    local server_log="$raw_dir/${route}-${repetition}-server.log"

    shm_path="/dev/shm/$current_shm_name"
    if [[ -e "$shm_path" ]]; then
        echo "refusing pre-existing shared-memory path: $shm_path" >&2
        exit 73
    fi
    : > "$server_log"
    BPFTIME_GLOBAL_SHM_NAME="$current_shm_name" \
    BPFTIME_LOG_OUTPUT="$server_log" \
    LD_PRELOAD="$server" \
        "$target" --server >"$server_stdout" 2>&1 &
    server_pid=$!
    if ! wait_for_line "$server_stdout" '^SERVER_READY ' 1000; then
        echo "server did not become ready for $route repetition $repetition" >&2
        exit 1
    fi
    if [[ ! -e "$shm_path" ]]; then
        echo "shared memory not created for $route repetition $repetition" >&2
        exit 1
    fi
}

wait_for_ipc_socket() {
    local pid=$1
    local i
    for ((i = 0; i < 1000; i++)); do
        if grep -q "@bpftime-agent-$pid" /proc/net/unix; then
            return 0
        fi
        sleep 0.01
    done
    return 1
}

finish_iteration() {
    cleanup_processes
}

run_preload() {
    local repetition=$1
    local shm_name
    local target_stdout="$raw_dir/preload-${repetition}-target.stdout"
    local start_ns end_ns ready_ns command_ms route_ms
    local after_mapping target_alive agent_named bpftime_used ipc_socket status

    start_server preload "$repetition"
    shm_name=$current_shm_name
    start_ns=$(date +%s%N)
    BPFTIME_GLOBAL_SHM_NAME="$shm_name" \
    LD_PRELOAD="$agent" \
        "$target" >"$target_stdout" 2>&1 &
    target_pid=$!
    if wait_for_line "$target_stdout" '^AGENT_READY ' 1000; then
        status=0
    else
        status=1
    fi
    end_ns=$(date +%s%N)
    ready_ns=$(awk -F'realtime_ns=' '/^AGENT_READY / {split($2,a," "); print a[1]; exit}' "$target_stdout")
    if [[ -z "$ready_ns" ]]; then
        ready_ns=$end_ns
    fi
    route_ms=$(awk -v a="$start_ns" -v b="$ready_ns" 'BEGIN {printf "%.3f", (b-a)/1000000}')
    command_ms=$(awk -v a="$start_ns" -v b="$end_ns" 'BEGIN {printf "%.3f", (b-a)/1000000}')
    after_mapping=$(grep -Ec 'libbpftime-agent\.so|memfd:.*frida' "/proc/$target_pid/maps" 2>/dev/null || true)
    if kill -0 "$target_pid" 2>/dev/null; then target_alive=1; else target_alive=0; fi
    agent_named=$(awk -F'agent_named=' '/^AGENT_READY / {split($2,a," "); print a[1]; exit}' "$target_stdout")
    bpftime_used=$(awk -F'bpftime_used=' '/^AGENT_READY / {split($2,a," "); print a[1]; exit}' "$target_stdout")
    if wait_for_ipc_socket "$target_pid"; then ipc_socket=1; else ipc_socket=0; fi
    printf 'preload\t%d\t%s\t%s\t%d\t0\t%s\t%s\t%s\t%s\t%d\n' \
        "$repetition" "$route_ms" "$command_ms" "$target_alive" \
        "$after_mapping" "${agent_named:-0}" "${bpftime_used:-0}" \
        "$ipc_socket" "$status" >> "$summary"
    finish_iteration
}

run_attach() {
    local repetition=$1
    local shm_name
    local target_stdout="$raw_dir/attach-${repetition}-target.stdout"
    local cli_stdout="$raw_dir/attach-${repetition}-cli.stdout"
    local start_ns end_ns ready_ns command_ms route_ms
    local before_mapping after_mapping target_alive agent_named bpftime_used ipc_socket status

    start_server attach "$repetition"
    shm_name=$current_shm_name
    BPFTIME_GLOBAL_SHM_NAME="$shm_name" \
        "$target" >"$target_stdout" 2>&1 &
    target_pid=$!
    if ! wait_for_line "$target_stdout" '^TARGET_READY ' 1000; then
        echo "target did not become ready for attach repetition $repetition" >&2
        exit 1
    fi
    before_mapping=$(grep -Ec 'libbpftime-agent\.so|memfd:.*frida' "/proc/$target_pid/maps" 2>/dev/null || true)
    start_ns=$(date +%s%N)
    set +e
    BPFTIME_GLOBAL_SHM_NAME="$shm_name" \
        "$cli" --install-location "$bpftime_build" attach "$target_pid" \
        >"$cli_stdout" 2>&1
    status=$?
    set -e
    end_ns=$(date +%s%N)
    if ! wait_for_line "$target_stdout" '^AGENT_READY ' 1000; then
        status=1
    fi
    ready_ns=$(awk -F'realtime_ns=' '/^AGENT_READY / {split($2,a," "); print a[1]; exit}' "$target_stdout")
    if [[ -z "$ready_ns" ]]; then
        ready_ns=$end_ns
    fi
    route_ms=$(awk -v a="$start_ns" -v b="$ready_ns" 'BEGIN {printf "%.3f", (b-a)/1000000}')
    command_ms=$(awk -v a="$start_ns" -v b="$end_ns" 'BEGIN {printf "%.3f", (b-a)/1000000}')
    after_mapping=$(grep -Ec 'libbpftime-agent\.so|memfd:.*frida' "/proc/$target_pid/maps" 2>/dev/null || true)
    if kill -0 "$target_pid" 2>/dev/null; then target_alive=1; else target_alive=0; fi
    agent_named=$(awk -F'agent_named=' '/^AGENT_READY / {split($2,a," "); print a[1]; exit}' "$target_stdout")
    bpftime_used=$(awk -F'bpftime_used=' '/^AGENT_READY / {split($2,a," "); print a[1]; exit}' "$target_stdout")
    if wait_for_ipc_socket "$target_pid"; then ipc_socket=1; else ipc_socket=0; fi
    printf 'attach\t%d\t%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%d\n' \
        "$repetition" "$route_ms" "$command_ms" "$target_alive" \
        "$before_mapping" "$after_mapping" "${agent_named:-0}" \
        "${bpftime_used:-0}" "$ipc_socket" "$status" >> "$summary"
    finish_iteration
}

for ((rep = 1; rep <= repetitions; rep++)); do
    run_preload "$rep"
    run_attach "$rep"
done

lifecycle_summary="$raw_dir/lifecycle-summary.tsv"
awk -F'\t' 'NR == 1 {next} {
    count[$1]++;
    ready_total[$1] += $3;
    command_total[$1] += $4;
    if (count[$1] == 1 || $3 < ready_min[$1]) ready_min[$1] = $3;
    if (count[$1] == 1 || $3 > ready_max[$1]) ready_max[$1] = $3;
    if ($5 != 1 || $6 != 0 || $7 < 1 || $9 != 1 || $10 != 1 || $11 != 0)
        failed[$1]++;
} END {
    print "route\trepetitions\tready_mean_ms\tready_min_ms\tready_max_ms\tcommand_mean_ms\tfailures";
    routes[1] = "preload";
    routes[2] = "attach";
    for (i = 1; i <= 2; i++) {
        route = routes[i];
        printf "%s\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%d\n", route,
               count[route], ready_total[route]/count[route], ready_min[route],
               ready_max[route], command_total[route]/count[route], failed[route]+0;
    }
}' "$summary" > "$lifecycle_summary"
column -t -s $'\t' "$lifecycle_summary"

if awk -F'\t' 'NR > 1 && ($5 != 1 || $6 != 0 || $7 < 1 || $9 != 1 || $10 != 1 || $11 != 0) {exit 1}' "$summary"; then
    exit 0
else
    echo "one or more lifecycle repetitions failed closed; inspect $summary" >&2
    exit 1
fi
