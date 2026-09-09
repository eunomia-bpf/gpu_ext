#!/usr/bin/env bash
# build_table1_runtime.sh
#
# Fresh, reproducible build of the two bpftime runtime shared libraries the
# RTX 5090 Table 1 gpubpf arms preload, from the published bpftime source
# revision:
#
#   <out>/build/runtime/agent/libbpftime-agent.so
#   <out>/build/runtime/syscall-server/libbpftime-syscall-server.so
#
# The historical measurements preloaded these from the untracked local tree
# build-table1-575-warp. This script instead prepares a brand-new checkout
# under --output-dir at the explicit published revision and builds only the
# two targets Table 1 needs, with the exact recorded CMake configuration
# (Debug, CUDA attach, LLVM JIT + uBPF JIT, userspace verifier, LLVM 15).
#
# Guarantees:
#   - writes only under --output-dir; refuses an existing nonempty directory
#   - never modifies or deletes the recorded checkout or build tree
#   - logs every command and its output to <out>/build.log
#   - records the checked-out commit IDs, configuration, and artifacts in
#     <out>/build-report.json (Git commit IDs only; no file hashes or
#     checksums are computed or printed)
#   - no GPU use, no driver load
#
# Submodules, including the nested uBPF dependency (third_party/ubpf) and the
# libbpf nested in third_party/bpftool, are gitlinks of the checked-out
# revision; `git submodule update --init --recursive` checks them out at the
# recorded commits and the resulting commit IDs are recorded, not gated.

set -euo pipefail

DEFAULT_SOURCE_REV="eef8a51abaf2ca1f0cdca9f2425af3bd535da1b7"
DEFAULT_BPFTIME_URL="https://github.com/eunomia-bpf/bpftime.git"
DEFAULT_CUDA_HOME="/usr/local/cuda-12.9"
DEFAULT_LLVM_CMAKE_DIR="/usr/lib/llvm-15/cmake"
DEFAULT_C_COMPILER="/usr/bin/cc"
DEFAULT_CXX_COMPILER="/usr/bin/c++"

usage() {
  cat <<'EOF'
Usage: build_table1_runtime.sh --output-dir DIR [options]

Fresh build of the Table 1 bpftime runtime (agent + syscall-server) from the
published bpftime revision, into a new output tree.

Required:
  --output-dir DIR          New build output directory (absent or empty;
                            relative paths are accepted and resolved)

Options:
  --source-rev REV          bpftime commit to build (default: the recorded
                            revision/table1-host-plt-fix tip eef8a51...)
  --bpftime-url URL         Source repository (default: the public bpftime
                            repository)
  --cuda-home DIR           CUDA installation root
                            (default: /usr/local/cuda-12.9, the recorded one)
  --llvm-cmake-dir DIR      Directory containing LLVMConfig.cmake
                            (default: /usr/lib/llvm-15/cmake, the recorded one)
  --c-compiler PATH         C compiler (default: /usr/bin/cc, recorded)
  --cxx-compiler PATH       C++ compiler (default: /usr/bin/c++, recorded)
  --jobs N                  Parallel build jobs (default: nproc)
  -h, --help                Show this help

Outputs: DIR/bpftime-source, DIR/build, DIR/build.log, DIR/build-report.json
Refuses an existing nonempty DIR; deletes nothing; no GPU or driver use.
EOF
}

die() { echo "build_table1_runtime: $*" >&2; exit 1; }

OUTPUT_DIR=""
SOURCE_REV="$DEFAULT_SOURCE_REV"
BPFTIME_URL="$DEFAULT_BPFTIME_URL"
CUDA_HOME="$DEFAULT_CUDA_HOME"
LLVM_CMAKE_DIR="$DEFAULT_LLVM_CMAKE_DIR"
C_COMPILER="$DEFAULT_C_COMPILER"
CXX_COMPILER="$DEFAULT_CXX_COMPILER"
JOBS="$(nproc 2>/dev/null || echo 4)"

while [ $# -gt 0 ]; do
  case "$1" in
    --output-dir)     OUTPUT_DIR="${2:?--output-dir needs a value}"; shift 2 ;;
    --source-rev)     SOURCE_REV="${2:?--source-rev needs a value}"; shift 2 ;;
    --bpftime-url)    BPFTIME_URL="${2:?--bpftime-url needs a value}"; shift 2 ;;
    --cuda-home)      CUDA_HOME="${2:?--cuda-home needs a value}"; shift 2 ;;
    --llvm-cmake-dir) LLVM_CMAKE_DIR="${2:?--llvm-cmake-dir needs a value}"; shift 2 ;;
    --c-compiler)     C_COMPILER="${2:?--c-compiler needs a value}"; shift 2 ;;
    --cxx-compiler)   CXX_COMPILER="${2:?--cxx-compiler needs a value}"; shift 2 ;;
    --jobs)           JOBS="${2:?--jobs needs a value}"; shift 2 ;;
    -h|--help)        usage; exit 0 ;;
    *)                usage >&2; die "unknown argument: $1" ;;
  esac
done

[ -n "$OUTPUT_DIR" ] || { usage >&2; die "--output-dir is required"; }
case "$JOBS" in
  ''|*[!0-9]*) die "--jobs must be a positive integer, got '$JOBS'" ;;
esac
[ "$JOBS" -ge 1 ] || die "--jobs must be >= 1, got '$JOBS'"

# Resolve all user-supplied paths to absolute (relative inputs are accepted).
OUTPUT_DIR="$(realpath -m -- "$OUTPUT_DIR")"
CUDA_HOME="$(realpath -m -- "$CUDA_HOME")"
LLVM_CMAKE_DIR="$(realpath -m -- "$LLVM_CMAKE_DIR")"
C_COMPILER="$(realpath -m -- "$C_COMPILER")"
CXX_COMPILER="$(realpath -m -- "$CXX_COMPILER")"

# Dependency checks (reported precisely, before anything is written).
command -v git   >/dev/null || die "git not found in PATH"
command -v cmake >/dev/null || die "cmake not found in PATH"
command -v jq    >/dev/null || die "jq not found in PATH (used only for build-report.json)"
[ -d "$CUDA_HOME" ] || die "CUDA home not found: $CUDA_HOME (install CUDA 12.9 or pass --cuda-home)"
[ -f "$LLVM_CMAKE_DIR/LLVMConfig.cmake" ] || die "LLVM CMake config not found: $LLVM_CMAKE_DIR/LLVMConfig.cmake (install the LLVM 15 development package or pass --llvm-cmake-dir)"
[ -x "$C_COMPILER" ] || die "C compiler missing or not executable: $C_COMPILER"
[ -x "$CXX_COMPILER" ] || die "C++ compiler missing or not executable: $CXX_COMPILER"

# Output directory: must be absent or empty; nothing is ever deleted.
if [ -e "$OUTPUT_DIR" ]; then
  [ -d "$OUTPUT_DIR" ] || die "--output-dir exists and is not a directory: $OUTPUT_DIR"
  if [ -n "$(ls -A -- "$OUTPUT_DIR")" ]; then
    die "refusing existing nonempty output directory: $OUTPUT_DIR (nothing is deleted)"
  fi
fi
mkdir -p -- "$OUTPUT_DIR"

LOG="$OUTPUT_DIR/build.log"
: > "$LOG"
trap 'echo "build_table1_runtime: failed; see $LOG" >&2' ERR

run() {
  printf '+ %s\n' "$*" >>"$LOG"
  printf '+ %s\n' "$*"
  "$@" >>"$LOG" 2>&1
}

{
  echo "=== build_table1_runtime $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "output_dir=$OUTPUT_DIR source_rev=$SOURCE_REV"
  echo "cuda_home=$CUDA_HOME llvm_cmake_dir=$LLVM_CMAKE_DIR jobs=$JOBS"
  echo "c_compiler=$C_COMPILER cxx_compiler=$CXX_COMPILER"
} >>"$LOG"

SRC="$OUTPUT_DIR/bpftime-source"
BUILD="$OUTPUT_DIR/build"

echo "build_table1_runtime: checking out $SOURCE_REV into $SRC"
run git clone --no-checkout "$BPFTIME_URL" "$SRC"
run git -C "$SRC" checkout --detach "$SOURCE_REV"
run git -C "$SRC" submodule update --init --recursive
git -C "$SRC" log --oneline -1 >>"$LOG"
head_rev="$(git -C "$SRC" rev-parse HEAD)"
pins="$(git -C "$SRC" submodule status --recursive)"
echo "$pins" >>"$LOG"
echo "checked out HEAD $head_rev; submodule commit IDs recorded in $LOG"

echo "build_table1_runtime: configuring (recorded Debug / CUDA-attach / LLVM+uBPF JIT / verifier config)"
run cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER="$C_COMPILER" \
  -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
  -DBPFTIME_ENABLE_CUDA_ATTACH=ON \
  -DBPFTIME_CUDA_ROOT="$CUDA_HOME" \
  -DBPFTIME_LLVM_JIT=ON \
  -DBPFTIME_UBPF_JIT=ON \
  -DENABLE_EBPF_VERIFIER=ON \
  -DLLVM_DIR="$LLVM_CMAKE_DIR"

echo "build_table1_runtime: building bpftime-agent and bpftime-syscall-server"
run cmake --build "$BUILD" --config Debug \
  --target bpftime-agent --target bpftime-syscall-server -j "$JOBS"

AGENT="$BUILD/runtime/agent/libbpftime-agent.so"
SERVER="$BUILD/runtime/syscall-server/libbpftime-syscall-server.so"
[ -s "$AGENT" ]  || die "expected artifact missing or empty: $AGENT"
[ -s "$SERVER" ] || die "expected artifact missing or empty: $SERVER"
agent_bytes="$(stat -c %s -- "$AGENT")"
server_bytes="$(stat -c %s -- "$SERVER")"

pins_json="$(printf '%s\n' "$pins" | jq -R -s '
  split("\n")
  | map(select(length > 0) | {path: (.[42:] | split(" ")[0]), commit: .[1:41]})
')"

jq -n \
  --arg timestamp_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg url "$BPFTIME_URL" \
  --arg requested_rev "$SOURCE_REV" \
  --arg head "$head_rev" \
  --arg srcdir "$SRC" \
  --arg builddir "$BUILD" \
  --argjson pins "$pins_json" \
  --arg cuda "$CUDA_HOME" \
  --arg llvm "$LLVM_CMAKE_DIR" \
  --arg cc "$C_COMPILER" \
  --arg cxx "$CXX_COMPILER" \
  --arg agent_path "$AGENT" \
  --arg server_path "$SERVER" \
  --argjson agent_bytes "$agent_bytes" \
  --argjson server_bytes "$server_bytes" \
  '{
    kind: "table1_runtime_fresh_build",
    status: "completed",
    timestamp_utc: $timestamp_utc,
    bpftime_url: $url,
    bpftime_source_rev_requested: $requested_rev,
    bpftime_head: $head,
    source_dir: $srcdir,
    build_dir: $builddir,
    submodule_pins: $pins,
    cmake_options: {
      CMAKE_BUILD_TYPE: "Debug",
      BPFTIME_ENABLE_CUDA_ATTACH: "ON",
      BPFTIME_CUDA_ROOT: $cuda,
      BPFTIME_LLVM_JIT: "ON",
      BPFTIME_UBPF_JIT: "ON",
      ENABLE_EBPF_VERIFIER: "ON",
      LLVM_DIR: $llvm,
      CMAKE_C_COMPILER: $cc,
      CMAKE_CXX_COMPILER: $cxx
    },
    build_targets: ["bpftime-agent", "bpftime-syscall-server"],
    artifacts: {
      agent: {path: $agent_path, bytes: $agent_bytes},
      syscall_server: {path: $server_path, bytes: $server_bytes}
    },
    note: "fresh component build from the published revision; Table 1 measured numbers remain the historical records and are unchanged"
  }' >"$OUTPUT_DIR/build-report.json"

echo
echo "build_table1_runtime: build completed"
echo "  agent:          $AGENT ($agent_bytes bytes)"
echo "  syscall-server: $SERVER ($server_bytes bytes)"
echo "  log:            $LOG"
echo "  report:         $OUTPUT_DIR/build-report.json"
echo "Pass --bpftime-build-dir $BUILD to the Table 1 perf runner."
echo "Runtime execution (probe loaders and benchmarks) remains untested;"
echo "this script compiled only the two Table 1 runtime libraries."
