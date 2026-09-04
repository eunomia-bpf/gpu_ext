#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 BPFTIME_SOURCE BPFTIME_BUILD DRIVER_575_SOURCE UPSTREAM_575_SOURCE" >&2
}

if [[ $# -ne 4 ]]; then
    usage
    exit 64
fi

bpftime_source=$(realpath "$1")
bpftime_build=$(realpath "$2")
driver_source=$(realpath "$3")
upstream_source=$(realpath "$4")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
raw_dir="$script_dir/raw"
mkdir -p "$raw_dir"

for required in \
    "$bpftime_source/attach/nv_attach_impl/nv_attach_impl.cpp" \
    "$bpftime_source/benchmark/gpu/nvbit/nvbit_vec_add.cu" \
    "$bpftime_source/benchmark/gpu/nvbit/nvbit_timing_funcs.cu" \
    "$bpftime_source/tools/cli/main.cpp" \
    "$bpftime_source/runtime/agent/agent.cpp" \
    "$bpftime_build/CMakeCache.txt" \
    "$driver_source/version.mk" \
    "$upstream_source/version.mk"; do
    if [[ ! -f "$required" ]]; then
        echo "missing required source: $required" >&2
        exit 66
    fi
done

version_from_tree() {
    awk '/^NVIDIA_VERSION[[:space:]]*=/{print $3; exit}' "$1/version.mk"
}

driver_version=$(version_from_tree "$driver_source")
upstream_version=$(version_from_tree "$upstream_source")
if [[ "$driver_version" != "575.57.08" || "$upstream_version" != "575.57.08" ]]; then
    echo "both driver trees must identify exactly version 575.57.08" >&2
    exit 65
fi
if ! grep -Fxq 'BPFTIME_ENABLE_CUDA_ATTACH:BOOL=OFF' \
        "$bpftime_build/CMakeCache.txt"; then
    echo "bpftime build must be explicitly configured without CUDA attach" >&2
    exit 65
fi

resolve_source_ref() {
    local source=$1
    local git_entry="$source/.git"
    local git_dir common_dir head ref revision

    if [[ -d "$git_entry" ]]; then
        git_dir=$git_entry
    elif [[ -f "$git_entry" ]]; then
        git_dir=$(sed -n 's/^gitdir: //p' "$git_entry")
        git_dir=$(realpath "$git_dir")
    else
        echo "none none"
        return
    fi
    head=$(sed -n '1p' "$git_dir/HEAD")
    if [[ "$head" == ref:\ * ]]; then
        ref=${head#ref: }
        if [[ -f "$git_dir/commondir" ]]; then
            common_dir=$(realpath "$git_dir/$(sed -n '1p' "$git_dir/commondir")")
        else
            common_dir=$git_dir
        fi
        if [[ -f "$common_dir/$ref" ]]; then
            revision=$(sed -n '1p' "$common_dir/$ref")
        else
            revision=$(awk -v wanted="$ref" '$2 == wanted {print $1; exit}' \
                "$common_dir/packed-refs")
        fi
        if [[ -z "$revision" ]]; then
            echo "could not resolve source ref $ref" >&2
            exit 65
        fi
        echo "$ref $revision"
    else
        echo "detached $head"
    fi
}

read -r bpftime_ref bpftime_revision < <(resolve_source_ref "$bpftime_source")
read -r driver_ref driver_revision < <(resolve_source_ref "$driver_source")
{
    printf 'component\tref\trevision_or_release\tsource\n'
    printf 'bpftime\t%s\t%s\t%s\n' "$bpftime_ref" "$bpftime_revision" \
        "$bpftime_source"
    printf 'driver-local\t%s\t%s\t%s\n' "$driver_ref" "$driver_revision" \
        "$driver_source"
    printf 'driver-reference\trefs/tags/575.57.08\t575.57.08\t%s\n' \
        'https://github.com/NVIDIA/open-gpu-kernel-modules/archive/refs/tags/575.57.08.tar.gz'
} > "$raw_dir/revisions.tsv"

printf 'component\tpath\tbytes\tphysical_lines\n' > "$raw_dir/source-inventory.tsv"
record_source_file() {
    local component=$1
    local root=$2
    local relative=$3
    local path="$root/$relative"
    local bytes lines

    if [[ ! -f "$path" ]]; then
        echo "inventory source disappeared: $path" >&2
        exit 66
    fi
    bytes=$(stat -c '%s' "$path")
    lines=$(wc -l < "$path")
    printf '%s\t%s\t%s\t%s\n' "$component" "$relative" "$bytes" "$lines" \
        >> "$raw_dir/source-inventory.tsv"
}

record_source_file bpftime "$bpftime_source" attach/nv_attach_impl/nv_attach_impl.cpp
record_source_file bpftime "$bpftime_source" attach/nv_attach_impl/ptx_compiler/ptx_compiler.cpp
record_source_file bpftime "$bpftime_source" benchmark/gpu/nvbit/nvbit_vec_add.cu
record_source_file bpftime "$bpftime_source" benchmark/gpu/nvbit/nvbit_timing_funcs.cu
record_source_file bpftime "$bpftime_source" tools/cli/main.cpp
record_source_file bpftime "$bpftime_source" runtime/agent/agent.cpp

printf 'check\tobservation\tstatus\n' > "$raw_dir/semantic-checks.tsv"
record_check() {
    printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$raw_dir/semantic-checks.tsv"
}

if rg -q 'nvPTXCompilerCreate|compile_ebpf_to_ptx_from_words' \
        "$bpftime_source/attach/nv_attach_impl" \
        --glob '*.{c,cc,cpp,h,hpp}'; then
    record_check product_gpu_backend 'PTX compiler and eBPF-to-PTX markers present' pass
else
    record_check product_gpu_backend 'PTX implementation markers absent' fail
    exit 1
fi
if rg -qi 'nvbit|sass' "$bpftime_source/attach/nv_attach_impl" \
        --glob '*.{c,cc,cpp,h,hpp,cu,cuh}'; then
    record_check product_sass_backend 'NVBit or SASS marker found in product attach implementation' fail
    exit 1
else
    record_check product_sass_backend 'no NVBit or SASS marker in product attach implementation' pass
fi
if rg -q 'nvbit_get_instrs' \
        "$bpftime_source/benchmark/gpu/nvbit/nvbit_vec_add.cu" && \
   rg -q 'nvbit_insert_call' \
        "$bpftime_source/benchmark/gpu/nvbit/nvbit_vec_add.cu"; then
    record_check nvbit_location 'audited NVBit call sites are present in the selected benchmark example' pass
else
    record_check nvbit_location 'expected NVBit benchmark calls absent' fail
    exit 1
fi
if rg -q '^[[:space:]]*[^/[:space:]].*clock64[[:space:]]*\(' \
        "$bpftime_source/benchmark/gpu/nvbit/nvbit_timing_funcs.cu"; then
    record_check nvbit_device_timing 'active device clock statement found' fail
    exit 1
else
    record_check nvbit_device_timing 'device clock statements are commented; injected functions are no-op' pass
fi
if rg -q 'frida_injector_inject_library_file_sync' \
        "$bpftime_source/tools/cli/main.cpp"; then
    record_check running_pid_attach 'CLI delegates running-process injection to Frida' pass
else
    record_check running_pid_attach 'Frida injection call absent' fail
    exit 1
fi
if rg -q 'ptrace[[:space:]]*\(' "$bpftime_source/tools" \
        "$bpftime_source/runtime/agent" --glob '*.{c,cc,cpp,h,hpp}'; then
    record_check direct_ptrace_api 'direct ptrace call found in product CLI or agent' fail
    exit 1
else
    record_check direct_ptrace_api 'no direct ptrace call in product CLI or agent; mechanism belongs to injector dependency' pass
fi
if rg -q 'LD_PRELOAD=' "$bpftime_source/tools/cli/main.cpp" && \
   rg -q '__libc_start_main' "$bpftime_source/runtime/agent/agent.cpp" && \
   rg -q 'setenv\("BPFTIME_USED"' "$bpftime_source/runtime/agent/agent.cpp" && \
   rg -q 'start_agent_ipc_server_once' "$bpftime_source/runtime/agent/agent.cpp"; then
    record_check preload_route 'CLI preload, startup wrapper, agent marker, and IPC initialization are present' pass
else
    record_check preload_route 'one or more preload lifecycle markers absent' fail
    exit 1
fi
record_check cpu_build_configuration 'BPFTIME_ENABLE_CUDA_ATTACH is OFF' pass

added_files=(
    kernel-open/common/inc/nv-gpreempt-transport.h
    kernel-open/common/inc/nv-gpu-rpc-diagnostic.h
    kernel-open/common/inc/nv-gpu-sched-init-diagnostic.h
    kernel-open/common/inc/nv-gpu-timeslice-control.h
    kernel-open/common/inc/nv-gpu-transition-validator.h
    kernel-open/nvidia-uvm/uvm_bpf_struct_ops.c
    kernel-open/nvidia-uvm/uvm_bpf_struct_ops.h
    kernel-open/nvidia/nv-gpu-sched-hooks.c
    kernel-open/nvidia/nv-gpu-sched-hooks.h
)
modified_files=(
    kernel-open/nvidia-uvm/nvidia-uvm-sources.Kbuild
    kernel-open/nvidia-uvm/uvm.c
    kernel-open/nvidia-uvm/uvm_migrate.c
    kernel-open/nvidia-uvm/uvm_migrate.h
    kernel-open/nvidia-uvm/uvm_perf_prefetch.c
    kernel-open/nvidia-uvm/uvm_pmm_gpu.c
    kernel-open/nvidia-uvm/uvm_pmm_gpu.h
    kernel-open/nvidia/nv.c
    kernel-open/nvidia/nvidia-sources.Kbuild
)

temporary_delta=$(mktemp "$raw_dir/source-delta.XXXXXX")
cleanup_delta() {
    rm -f -- "$temporary_delta"
}
trap cleanup_delta EXIT INT TERM

printf 'class\tpath\tlocal_physical_lines\tlocal_nonblank_lines\tadded_lines\tdeleted_lines\n' \
    > "$raw_dir/open-module-delta.tsv"
for relative in "${added_files[@]}"; do
    local_file="$driver_source/$relative"
    upstream_file="$upstream_source/$relative"
    if [[ ! -f "$local_file" || -e "$upstream_file" ]]; then
        echo "added-file classification failed: $relative" >&2
        exit 1
    fi
    physical=$(wc -l < "$local_file")
    nonblank=$(awk 'NF {count++} END {print count+0}' "$local_file")
    printf 'added\t%s\t%s\t%s\t%s\t0\n' \
        "$relative" "$physical" "$nonblank" "$physical" \
        >> "$raw_dir/open-module-delta.tsv"
done

for relative in "${modified_files[@]}"; do
    local_file="$driver_source/$relative"
    upstream_file="$upstream_source/$relative"
    if [[ ! -f "$local_file" || ! -f "$upstream_file" ]]; then
        echo "modified-file classification failed: $relative" >&2
        exit 1
    fi
    delta_status=0
    diff -U0 -- "$upstream_file" "$local_file" > "$temporary_delta" || delta_status=$?
    if [[ $delta_status -ne 1 ]]; then
        echo "expected a real source difference for $relative; diff status=$delta_status" >&2
        exit 1
    fi
    physical=$(wc -l < "$local_file")
    nonblank=$(awk 'NF {count++} END {print count+0}' "$local_file")
    additions=$(awk '/^\+/{if ($0 !~ /^\+\+\+/) count++} END {print count+0}' "$temporary_delta")
    deletions=$(awk '/^-/{if ($0 !~ /^---/) count++} END {print count+0}' "$temporary_delta")
    printf 'modified\t%s\t%s\t%s\t%s\t%s\n' \
        "$relative" "$physical" "$nonblank" "$additions" "$deletions" \
        >> "$raw_dir/open-module-delta.tsv"
done

awk -F'\t' 'NR > 1 {
    files[$1]++;
    added[$1] += $5;
    deleted[$1] += $6;
} END {
    print "scope\tfiles\tadded_lines\tdeleted_lines";
    printf "added_files\t%d\t%d\t%d\n", files["added"], added["added"], deleted["added"];
    printf "modified_files\t%d\t%d\t%d\n", files["modified"], added["modified"], deleted["modified"];
    printf "audited_production_scope\t%d\t%d\t%d\n", files["added"]+files["modified"], added["added"]+added["modified"], deleted["added"]+deleted["modified"];
}' "$raw_dir/open-module-delta.tsv" > "$raw_dir/open-module-delta-summary.tsv"

printf 'module\tbytes\tversion\tvermagic\trelevant_defined_symbols\n' \
    > "$raw_dir/module-artifacts.tsv"
: > "$raw_dir/module-symbols.txt"
for module in nvidia nvidia-uvm nvidia-modeset nvidia-drm nvidia-peermem; do
    module_path="$driver_source/kernel-open/$module.ko"
    if [[ ! -f "$module_path" ]]; then
        echo "missing built module: $module_path" >&2
        exit 1
    fi
    module_version=$(modinfo -F version "$module_path")
    module_vermagic=$(modinfo -F vermagic "$module_path")
    if [[ "$module_version" != "575.57.08" ]]; then
        echo "unexpected module version for $module_path: $module_version" >&2
        exit 1
    fi
    relevant_symbols=$(nm -g --defined-only "$module_path" 2>/dev/null | \
        awk '$3 ~ /(nv_gpu_sched|gpu_sched|gpreempt|timeslice|uvm_bpf)/ {count++} END {print count+0}')
    nm -g --defined-only "$module_path" 2>/dev/null | \
        awk -v module="$module" \
            '$3 ~ /(nv_gpu_sched|gpu_sched|gpreempt|timeslice|uvm_bpf)/ {print module ": " $0}' \
        >> "$raw_dir/module-symbols.txt"
    printf '%s\t%s\t%s\t%s\t%s\n' "$module.ko" \
        "$(stat -c '%s' "$module_path")" "$module_version" \
        "$module_vermagic" "$relevant_symbols" \
        >> "$raw_dir/module-artifacts.tsv"
done

if ! awk -F'\t' '$1 == "nvidia.ko" && $5 > 0 {found=1} END {exit !found}' \
        "$raw_dir/module-artifacts.tsv"; then
    echo "nvidia.ko does not expose the expected scheduler boundary symbols" >&2
    exit 1
fi

echo "source/build audit passed"
