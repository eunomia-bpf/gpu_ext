#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 BPFTIME_SOURCE BPFTIME_BUILD" >&2
    exit 64
fi

bpftime_source=$(realpath "$1")
bpftime_build=$(realpath "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
raw_dir="$script_dir/raw"
target="$raw_dir/lifecycle-target"
cli="$bpftime_build/tools/cli/bpftime"
agent="$bpftime_build/runtime/agent/libbpftime-agent.so"
server="$bpftime_build/runtime/syscall-server/libbpftime-syscall-server.so"
trace_file="$raw_dir/attach-ptrace-syscalls.txt"
shm_name="gpubpf_deploy_audit_${$}_ptrace"
shm_path="/dev/shm/$shm_name"
server_pid=''
target_pid=''

for required in "$bpftime_source/CMakeLists.txt" "$bpftime_build/CMakeCache.txt" \
                "$cli" "$agent" "$server"; do
    if [[ ! -e "$required" ]]; then
        echo "missing required path: $required" >&2
        exit 66
    fi
done
if ! grep -Fxq 'BPFTIME_ENABLE_CUDA_ATTACH:BOOL=OFF' \
        "$bpftime_build/CMakeCache.txt"; then
    echo "refusing diagnostic: build is not explicitly CPU-only" >&2
    exit 65
fi
if ! command -v strace >/dev/null; then
    echo "strace is required for this diagnostic" >&2
    exit 69
fi
if [[ -n ${LD_PRELOAD:-} || -n ${BPFTIME_USED:-} || -e "$shm_path" ]]; then
    echo "refusing inherited state or pre-existing shared-memory object" >&2
    exit 65
fi

mkdir -p "$raw_dir"
cc -std=c11 -O2 -Wall -Wextra -Werror "$script_dir/lifecycle_target.c" \
   -ldl -o "$target"
: > "$raw_dir/ptrace-server.log"

cleanup() {
    if [[ -n "$target_pid" ]] && kill -0 "$target_pid" 2>/dev/null; then
        kill -TERM "$target_pid" 2>/dev/null || true
        wait "$target_pid" 2>/dev/null || true
    fi
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    rm -f -- "$shm_path"
}
trap cleanup EXIT INT TERM

wait_for_line() {
    local file=$1
    local pattern=$2
    local i
    for ((i = 0; i < 1000; i++)); do
        if grep -q "$pattern" "$file" 2>/dev/null; then
            return 0
        fi
        sleep 0.01
    done
    return 1
}

BPFTIME_GLOBAL_SHM_NAME="$shm_name" \
BPFTIME_LOG_OUTPUT="$raw_dir/ptrace-server.log" \
LD_PRELOAD="$server" \
    "$target" --server > "$raw_dir/ptrace-server.stdout" 2>&1 &
server_pid=$!
if ! wait_for_line "$raw_dir/ptrace-server.stdout" '^SERVER_READY '; then
    echo "syscall server did not become ready" >&2
    exit 1
fi

BPFTIME_GLOBAL_SHM_NAME="$shm_name" \
    "$target" > "$raw_dir/ptrace-target.stdout" 2>&1 &
target_pid=$!
if ! wait_for_line "$raw_dir/ptrace-target.stdout" '^TARGET_READY '; then
    echo "target did not become ready" >&2
    exit 1
fi
before_mapping=$(grep -Ec 'libbpftime-agent\.so|memfd:.*frida' \
    "/proc/$target_pid/maps" 2>/dev/null || true)

set +e
BPFTIME_GLOBAL_SHM_NAME="$shm_name" \
strace -f -qq -e trace=ptrace -o "$trace_file" \
    "$cli" --install-location "$bpftime_build" attach "$target_pid" \
    > "$raw_dir/ptrace-cli.stdout" 2>&1
command_status=$?
set -e

if wait_for_line "$raw_dir/ptrace-target.stdout" '^AGENT_READY '; then
    ready=1
else
    ready=0
fi
after_mapping=$(grep -Ec 'libbpftime-agent\.so|memfd:.*frida' \
    "/proc/$target_pid/maps" 2>/dev/null || true)
if kill -0 "$target_pid" 2>/dev/null; then target_alive=1; else target_alive=0; fi
if grep -q "@bpftime-agent-$target_pid" /proc/net/unix; then ipc_socket=1; else ipc_socket=0; fi
if [[ -f "$trace_file" ]]; then
    ptrace_calls=$(awk '/ptrace\(/{count++} END {print count+0}' "$trace_file")
    attach_calls=$(awk '/PTRACE_ATTACH|PTRACE_SEIZE/{count++} END {print count+0}' "$trace_file")
else
    ptrace_calls=0
    attach_calls=0
fi

printf 'command_status\ttarget_alive\tready\tbefore_mapping\tafter_mapping\tipc_socket\tptrace_calls\tattach_or_seize_calls\ttarget_opt_in\n' \
    > "$raw_dir/ptrace-diagnostic.tsv"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tPR_SET_PTRACER_ANY\n' \
    "$command_status" "$target_alive" "$ready" "$before_mapping" \
    "$after_mapping" "$ipc_socket" "$ptrace_calls" "$attach_calls" \
    >> "$raw_dir/ptrace-diagnostic.tsv"

if [[ $command_status -ne 0 || $target_alive -ne 1 || $ready -ne 1 || \
      $before_mapping -ne 0 || $after_mapping -lt 1 || $ipc_socket -ne 1 || \
      $ptrace_calls -lt 1 || $attach_calls -lt 1 ]]; then
    echo "ptrace diagnostic failed closed" >&2
    exit 1
fi

column -t -s $'\t' "$raw_dir/ptrace-diagnostic.tsv"
