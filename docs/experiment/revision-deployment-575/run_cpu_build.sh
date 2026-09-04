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
mkdir -p "$raw_dir"

if [[ ! -f "$bpftime_source/CMakeLists.txt" || \
      ! -f "$bpftime_build/CMakeCache.txt" ]]; then
    echo "source or configured build tree is missing" >&2
    exit 66
fi
if ! grep -Fxq 'BPFTIME_ENABLE_CUDA_ATTACH:BOOL=OFF' \
        "$bpftime_build/CMakeCache.txt"; then
    echo "refusing build: BPFTIME_ENABLE_CUDA_ATTACH is not explicitly OFF" >&2
    exit 65
fi

cmake --build "$bpftime_build" \
    --target bpftime-cli-cpp bpftime-agent bpftime-syscall-server \
    --parallel "${BPFTIME_BUILD_JOBS:-4}" 2>&1 | tee "$raw_dir/cpu-build.log"

cli="$bpftime_build/tools/cli/bpftime"
agent="$bpftime_build/runtime/agent/libbpftime-agent.so"
server="$bpftime_build/runtime/syscall-server/libbpftime-syscall-server.so"
for artifact in "$cli" "$agent" "$server"; do
    if [[ ! -f "$artifact" ]]; then
        echo "expected build artifact absent: $artifact" >&2
        exit 1
    fi
done

ldd "$agent" > "$raw_dir/agent-ldd.txt"
if grep -Eiq 'libcuda|libcudart|libnvptxcompiler' "$raw_dir/agent-ldd.txt"; then
    echo "CPU-only agent unexpectedly links a CUDA library" >&2
    exit 1
fi

printf 'artifact\tbytes\telf_type\tmachine\n' > "$raw_dir/bpftime-artifacts.tsv"
for artifact in "$cli" "$agent" "$server"; do
    elf_type=$(readelf -h "$artifact" | \
        awk -F: '/^[[:space:]]*Type:/{sub(/^[[:space:]]+/, "", $2); print $2; exit}')
    machine=$(readelf -h "$artifact" | \
        awk -F: '/^[[:space:]]*Machine:/{sub(/^[[:space:]]+/, "", $2); print $2; exit}')
    printf '%s\t%s\t%s\t%s\n' "${artifact#"$bpftime_build/"}" \
        "$(stat -c '%s' "$artifact")" "$elf_type" "$machine" \
        >> "$raw_dir/bpftime-artifacts.tsv"
done

readelf -Ws "$agent" | \
    awk '$8 == "bpftime_agent_main" || $8 == "__libc_start_main" {print}' \
    > "$raw_dir/agent-entry-symbols.txt"
if ! grep -q 'bpftime_agent_main' "$raw_dir/agent-entry-symbols.txt" || \
   ! grep -q '__libc_start_main' "$raw_dir/agent-entry-symbols.txt"; then
    echo "agent entry symbols absent" >&2
    exit 1
fi

"$cli" --version > "$raw_dir/bpftime-version.txt"
echo "CPU-only bpftime build audit passed"
