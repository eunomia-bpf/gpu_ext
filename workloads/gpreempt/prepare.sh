#!/usr/bin/env bash
set -euo pipefail
workload_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir="$workload_dir/deps/upstream"
source_cache="$workload_dir/../../docs/driver_docs/sched/GPreempt"
revision=249ee3e
if [[ ! -d "$source_dir/.git" ]]; then
    mkdir -p "$workload_dir/deps"
    if [[ -d "$source_cache/.git" ]]; then
        git clone --no-hardlinks "$source_cache" "$source_dir"
    else
        git clone https://github.com/thustorage/GPreempt.git "$source_dir"
    fi
    git -C "$source_dir" checkout --detach "$revision"
fi
actual_revision=$(git -C "$source_dir" rev-parse --short=7 HEAD)
if [[ "$actual_revision" != "$revision" ]]; then
    echo "Expected upstream revision $revision, found $actual_revision" >&2
    exit 1
fi
patch_files=(compatibility.patch)
if [[ ${1:-} == --bridge ]]; then
    patch_files+=(policy-bridge.patch measurement.patch)
elif [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--bridge]" >&2
    exit 1
fi
for patch_name in "${patch_files[@]}"; do
    patch_file="$workload_dir/$patch_name"
    if git -C "$source_dir" apply --recount --unidiff-zero --reverse --check "$patch_file" 2>/dev/null; then
        echo "$patch_name already present"
    else
        git -C "$source_dir" apply --recount --unidiff-zero --check "$patch_file"
        git -C "$source_dir" apply --recount --unidiff-zero "$patch_file"
    fi
done
# Override only this clone's transport; preserve upstream .gitmodules and pins.
git -C "$source_dir" submodule init
git -C "$source_dir" config submodule.third_party/jsoncpp.url https://github.com/open-source-parsers/jsoncpp.git
git -C "$source_dir" config submodule.third_party/glog.url https://github.com/google/glog.git
git -C "$source_dir" submodule update --init --recursive --jobs 4
echo "Prepared upstream $revision with scoped compatibility changes"
