#!/usr/bin/env bash
# Source preparation only: no compiler jobs and no CUDA initialization.
set -euo pipefail
workload_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
tvm_dir="$workload_dir/deps/tvm"
revision=513c2be0c3b853a3b77de729f0ea75d448ee3c37
if [[ ! -d "$tvm_dir/.git" ]]; then
    mkdir -p "$tvm_dir"
    git -C "$tvm_dir" init
    git -C "$tvm_dir" remote add origin https://github.com/apache/tvm.git
    git -C "$tvm_dir" fetch --depth 1 origin "$revision"
    git -C "$tvm_dir" checkout --detach FETCH_HEAD
fi
if [[ $(git -C "$tvm_dir" rev-parse HEAD) != "$revision" ]]; then
    echo "Unexpected TVM revision; refusing to alter the checkout" >&2
    exit 1
fi
patch_file="$workload_dir/deps/upstream/patch/tvm.patch"
if git -C "$tvm_dir" apply --reverse --check "$patch_file" 2>/dev/null; then
    echo "Original GPreempt TVM recording patch already applied"
else
    git -C "$tvm_dir" apply --check "$patch_file"
    git -C "$tvm_dir" apply "$patch_file"
fi
git -C "$tvm_dir" submodule update --init --depth 1 --jobs 4 \
    3rdparty/dmlc-core 3rdparty/dlpack 3rdparty/rang
echo "Prepared original patched TVM $revision; configure/build remain separate"
