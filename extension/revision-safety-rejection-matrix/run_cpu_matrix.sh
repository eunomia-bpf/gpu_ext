#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
bpftime_root=${BPFTIME_ROOT:-"$repo_root/../bpftime-r5"}
bpftime_build=${BPFTIME_BUILD:-"$bpftime_root/build-r5-v2"}
output_dir=${1:?usage: run_cpu_matrix.sh OUTPUT_DIR}

if [[ -e "$output_dir" ]]; then
    printf 'output directory already exists: %s\n' "$output_dir" >&2
    exit 2
fi

verifier_tests="$bpftime_build/bpftime-verifier/bpftime_verifier_tests"
verifier_lib="$bpftime_build/bpftime-verifier/libbpftime-verifier.a"
base_verifier_lib="$bpftime_build/bpftime-verifier/ebpf-verifier/libebpfverifier.a"
btf_lib="$bpftime_build/bpftime-verifier/ebpf-verifier/external/libbtf/libbtf/liblibbtf.a"
libbpf_lib="$bpftime_build/libbpf/libbpf.a"

for required in "$verifier_tests" "$verifier_lib" "$base_verifier_lib" \
                "$btf_lib" "$libbpf_lib"; do
    if [[ ! -f "$required" ]]; then
        printf 'required verifier artifact is absent: %s\n' "$required" >&2
        exit 2
    fi
done

mkdir -p "$output_dir/raw"
build_dir=$(mktemp -d /tmp/gpubpf-safety-matrix.XXXXXX)

cc -O2 -std=c11 -Wall -Wextra -Werror \
    -I"$repo_root/kernel-module/nvidia-module/kernel-open/common/inc" \
    "$script_dir/transition_outcome_matrix.c" \
    -o "$build_dir/transition_outcome_matrix"

cc -O2 -std=c11 -Wall -Wextra -Werror \
    -I"$repo_root/kernel-module/nvidia-module/kernel-open/common/inc" \
    "$repo_root/kernel-module/nvidia-module/kernel-open/tests/transition-validator/transition_validator_test.c" \
    -o "$build_dir/transition_validator_test"

c++ -O2 -std=gnu++20 -Wall -Wextra -Werror \
    -DBPFTIME_BUILD_WITH_LIBBPF=1 \
    -DLLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1 \
    -I"$bpftime_root/bpftime-verifier/include" \
    -I"$bpftime_root/bpftime-verifier/src/gpu" \
    -I"$bpftime_root/bpftime-verifier/ebpf-verifier/src" \
    -I"$bpftime_build/libbpf" \
    "$script_dir/simt_additional_matrix.cpp" \
    "$verifier_lib" "$base_verifier_lib" "$btf_lib" "$libbpf_lib" \
    -lyaml-cpp -lz -lelf \
    -o "$build_dir/simt_additional_matrix"

{
    uname -r
    cc --version | sed -n '1p'
    c++ --version | sed -n '1p'
    awk '/^CapEff:/ {print}' /proc/self/status
} | tee "$output_dir/raw/environment.log"

"$build_dir/transition_outcome_matrix" \
    | tee "$output_dir/raw/transition-outcomes.log"
"$build_dir/transition_validator_test" \
    | tee "$output_dir/raw/transition-existing-regression.log"
"$build_dir/simt_additional_matrix" \
    | tee "$output_dir/raw/simt-additional.log"
"$verifier_tests" '[gpu][revision-safety]' --rng-seed 1 \
    | tee "$output_dir/raw/simt-existing-regression.log"

printf 'PASS cpu rejection matrix\n' | tee "$output_dir/raw/summary.log"
