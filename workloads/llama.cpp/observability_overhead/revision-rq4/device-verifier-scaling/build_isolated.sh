#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
bpftime_root=/home/yunwei37/workspace/gpu/bpftime-table1-575
build_dir=/home/yunwei37/workspace/gpu/bpftime-device-verifier-scaling-build

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
build_parent=$(dirname "$build_dir")
mkdir -p "$build_parent"
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
if [[ "$build_dir" == *build-table1-575-strict* ]]; then
    echo "refusing the existing strict runtime build tree" >&2
    exit 64
fi

mkdir -p "$build_dir"

cmake -S "$script_dir" -B "$build_dir" -G Ninja \
    -DBPFTIME_ROOT="$bpftime_root" \
    -DCMAKE_BUILD_TYPE=Release \
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
    -DFETCHCONTENT_SOURCE_DIR_CATCH2="$bpftime_root/third_party/Catch2"

CMAKE_BUILD_PARALLEL_LEVEL=2 cmake --build "$build_dir" \
    --target verifier_scaling_probe --parallel 2

"$build_dir/verifier_scaling_probe" --describe \
    --family linear --instructions 16
"$build_dir/verifier_scaling_probe" --describe \
    --family diamonds --instructions 4096

printf 'built %s\n' "$build_dir/verifier_scaling_probe"
