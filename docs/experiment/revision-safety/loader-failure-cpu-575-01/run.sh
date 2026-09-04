#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 BPFTIME_BUILD_DIR" >&2
    exit 64
fi

experiment_dir=$(cd "$(dirname "$0")" && pwd)
build_dir=$(cd "$1" && pwd)
preload="$build_dir/runtime/syscall-server/libbpftime-syscall-server.so"
cache="$build_dir/CMakeCache.txt"
raw="$experiment_dir/raw"
probe="$experiment_dir/load_probe"

[[ -f "$preload" && -f "$cache" ]]
source_dir=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$cache")
git -C "$source_dir" rev-parse --git-dir >/dev/null
grep -Eq '^ENABLE_EBPF_VERIFIER:BOOL=(ON|YES)$' "$cache"
grep -q '^BPFTIME_ENABLE_CUDA_ATTACH:BOOL=OFF$' "$cache"

mkdir -p "$raw"
cc -std=c11 -O2 -Wall -Wextra -Werror "$experiment_dir/load_probe.c" -o "$probe"

{
    date -u +'%Y-%m-%dT%H:%M:%SZ'
    uname -srmo
    sed -n -E '/^(ENABLE_EBPF_VERIFIER|BPFTIME_ENABLE_CUDA_ATTACH):BOOL=/p' "$cache"
    awk '/CapEff/ {print}' /proc/self/status
    printf 'source_revision=%s\n' "$(git -C "$source_dir" rev-parse --short=12 HEAD)"
    printf 'preload=%s\n' "$preload"
} >"$raw/environment.log"

run_cell() (
    local repetition=$1
    local arm=$2
    local level=$3
    local sequence=$4
    local shm_name="gpubpf_loader_safety_${$}_${repetition}_${arm}"
    local stdout="$raw/rep-${repetition}-${arm}.stdout"
    local stderr="$raw/rep-${repetition}-${arm}.stderr"
    local shm_path="/dev/shm/$shm_name"

    trap 'if [[ -e "$shm_path" ]]; then rm -- "$shm_path"; fi' EXIT

    [[ ! -e "$shm_path" ]]
    if [[ "$level" == "UNSET" ]]; then
        env -u BPFTIME_VERIFIER_LEVEL \
            BPFTIME_GLOBAL_SHM_NAME="$shm_name" \
            BPFTIME_SHM_MEMORY_MB=8 \
            BPFTIME_LOG_OUTPUT=console \
            SPDLOG_LEVEL=debug \
            LD_PRELOAD="$preload" \
            "$probe" "$sequence" >"$stdout" 2>"$stderr"
    else
        BPFTIME_VERIFIER_LEVEL="$level" \
            BPFTIME_GLOBAL_SHM_NAME="$shm_name" \
            BPFTIME_SHM_MEMORY_MB=8 \
            BPFTIME_LOG_OUTPUT=console \
            SPDLOG_LEVEL=debug \
            LD_PRELOAD="$preload" \
            "$probe" "$sequence" >"$stdout" 2>"$stderr"
    fi

    [[ -f "$shm_path" ]]
    rm -- "$shm_path"
    [[ ! -e "$shm_path" ]]
    trap - EXIT
)

for repetition in 1 2 3; do
    run_cell "$repetition" strict-invalid STRICT invalid-then-valid
    run_cell "$repetition" strict-control STRICT valid-only
    run_cell "$repetition" warning WARNING invalid-then-valid
    run_cell "$repetition" no-verify NO_VERIFY invalid-then-valid
    run_cell "$repetition" default UNSET invalid-then-valid
done

python3 - "$raw" <<'PY' | tee "$raw/summary.log"
import pathlib
import re
import sys

raw = pathlib.Path(sys.argv[1])
result_re = re.compile(r"^(invalid|valid)_rc=(-?\d+) \1_errno=(\d+)$")

def outcomes(path):
    parsed = {}
    for line in path.read_text().splitlines():
        match = result_re.fullmatch(line)
        if match:
            parsed[match.group(1)] = (int(match.group(2)), int(match.group(3)))
    return parsed

cells = 0
for repetition in range(1, 4):
    strict = outcomes(raw / f"rep-{repetition}-strict-invalid.stdout")
    control = outcomes(raw / f"rep-{repetition}-strict-control.stdout")
    assert strict["invalid"] == (-1, 22), strict
    assert strict["valid"][0] >= 0 and strict["valid"][1] == 0, strict
    assert control["valid"] == strict["valid"], (strict, control)
    strict_log = (raw / f"rep-{repetition}-strict-invalid.stderr").read_text()
    assert "Failed to verify program `unsafe_stack`" in strict_log
    assert "Loaded program `unsafe_stack`" not in strict_log
    cells += 2

    for arm in ("warning", "no-verify", "default"):
        observed = outcomes(raw / f"rep-{repetition}-{arm}.stdout")
        assert observed["invalid"][0] >= 0 and observed["invalid"][1] == 0, observed
        assert observed["valid"][0] >= 0 and observed["valid"][1] == 0, observed
        assert observed["invalid"][0] != observed["valid"][0], observed
        cells += 1

    warning_log = (raw / f"rep-{repetition}-warning.stderr").read_text()
    default_log = (raw / f"rep-{repetition}-default.stderr").read_text()
    no_verify_log = (raw / f"rep-{repetition}-no-verify.stderr").read_text()
    assert "Userspace verifier warning for program `unsafe_stack`" in warning_log
    assert "Userspace verifier warning for program `unsafe_stack`" in default_log
    assert "Userspace verifier warning for program `unsafe_stack`" not in no_verify_log

print(f"PASS cells={cells} repetitions=3")
print("strict=reject-invalid-without-slot-allocation")
print("warning=admit-invalid-after-warning")
print("no_verify=admit-invalid-without-verifier-warning")
print("default=warning")
PY

rm -- "$probe"
test ! -e "$probe"
