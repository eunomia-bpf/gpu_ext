#!/usr/bin/env bash
set -euo pipefail
pod_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pod_source="$pod_root/deps/vattention/pod_attn"
pod_revision=71a0e91aa46ff8fa985bcca3327efe0ab9929a39
if [[ ! -d "$pod_root/deps/vattention" ]]; then
    mkdir -p "$pod_root/deps/vattention"
    curl --fail --location --silent --show-error \
        "https://codeload.github.com/microsoft/vattention/tar.gz/$pod_revision" \
        | tar -xz --strip-components=1 -C "$pod_root/deps/vattention"
fi
for pod_file in setup.py pod_attn/fused_api.cpp pod_attn/flash_api.cpp \
                pod_attn/fused_fwd_kernel.h pod_attn/fused_fwd_launch_template.h \
                csrc/cutlass/include/cute/tensor.hpp; do
    [[ -f "$pod_source/$pod_file" ]] || { echo "Missing official source: $pod_file" >&2; exit 1; }
done
if patch --dry-run --silent --forward -p1 -d "$pod_source" < "$pod_root/pod-compat.patch"; then
    patch --forward -p1 -d "$pod_source" < "$pod_root/pod-compat.patch"
elif patch --dry-run --silent --reverse -p1 -d "$pod_source" < "$pod_root/pod-compat.patch"; then
    echo 'POD compatibility patch already applied (reverse application check passed).'
else
    echo 'POD source differs from the pinned source/patch; preserving it and stopping.' >&2
    exit 1
fi
