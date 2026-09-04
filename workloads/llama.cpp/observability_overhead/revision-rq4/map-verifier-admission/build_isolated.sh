#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
bpftime_root=/home/yunwei37/workspace/gpu/bpftime-table1-575
build_dir=/home/yunwei37/workspace/gpu/bpftime-map-verifier-admission-build

while (($#)); do
    case "$1" in
        --bpftime-root)
            bpftime_root=$2
            shift 2
            ;;
        --build-dir)
            build_dir=$2
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 64
            ;;
    esac
done

bpftime_root=$(realpath "$bpftime_root")
build_dir=$(realpath -m "$build_dir")
if [[ ! -f "$bpftime_root/bpftime-verifier/include/gpu_verifier.hpp" ]]; then
    echo "not a bpftime verifier source tree: $bpftime_root" >&2
    exit 66
fi
case "$build_dir" in
    "$bpftime_root"|"$bpftime_root"/*)
        echo "isolated build directory must be outside the bpftime source tree" >&2
        exit 64
        ;;
esac

verifier_status=$(git -C "$bpftime_root" status --short -- \
    bpftime-verifier/src bpftime-verifier/include)
if [[ -n "$verifier_status" ]]; then
    echo "refusing to build from an in-progress verifier source tree" >&2
    echo "$verifier_status" >&2
    exit 65
fi

mkdir -p "$build_dir"
cmake -S "$script_dir" -B "$build_dir" -G Ninja \
    -DBPFTIME_ROOT="$bpftime_root" \
    -DCMAKE_BUILD_TYPE=Release \
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
    -DFETCHCONTENT_SOURCE_DIR_CATCH2="$bpftime_root/third_party/Catch2"
CMAKE_BUILD_PARALLEL_LEVEL=2 cmake --build "$build_dir" \
    --target map_verifier_admission --parallel 2

if [[ ! -x "$build_dir/map_verifier_admission" ]]; then
    echo "build did not produce map_verifier_admission" >&2
    exit 67
fi
printf 'built %s\n' "$build_dir/map_verifier_admission"
