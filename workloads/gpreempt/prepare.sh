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
if [[ ${1:-} == --bridge || ${1:-} == --load-study ]]; then
    patch_files+=(policy-bridge.patch measurement.patch flag-transport.patch)
    if [[ ${1:-} == --load-study ]]; then
        patch_files+=(load-study.patch)
    fi
elif [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--bridge|--load-study]" >&2
    exit 1
fi
# Later patches intentionally replace earlier patch context. When the complete
# load-study patch is present, do not try to reapply its already stacked parents.
# Its build switch remains OFF for ordinary builds, which keep the legacy path.
if [[ -f "$workload_dir/load-study.patch" ]] &&
    git -C "$source_dir" apply --recount --unidiff-zero --reverse --check \
        "$workload_dir/load-study.patch" 2>/dev/null; then
    echo "Complete load-study patch stack already present"
    patch_files=()
else
    # Also support upgrading an existing bridge/measurement stack. Earlier
    # reverse checks need not apply once a later patch has replaced their lines.
    for ((patch_index=${#patch_files[@]}-1; patch_index>=0; patch_index--)); do
        if git -C "$source_dir" apply --recount --unidiff-zero --reverse --check \
            "$workload_dir/${patch_files[patch_index]}" 2>/dev/null; then
            echo "Patch stack through ${patch_files[patch_index]} already present"
            patch_files=("${patch_files[@]:$((patch_index+1))}")
            break
        fi
    done
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
