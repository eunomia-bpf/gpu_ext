#!/bin/bash
# Merge the BPF .func PTX and the CUDA trampoline PTX into a single
# ptxas-ready translation unit:
#   - one .version/.target/.address_size header (from the wrapper)
#   - bpf_exit definition (BPF exporter output, headers stripped)
#   - sk_bpf_trampoline definition (wrapper, headers stripped,
#     the matching .extern .func bpf_exit declaration removed so the
#     call resolves inside the module)
# Args: BPF_PTX WRAPPER_PTX OUT
set -euo pipefail
bpf_ptx=$1
wrap_ptx=$2
out=$3

{
	awk '/^\.version /{print; exit}' "$wrap_ptx"
	awk '/^\.target /{print; exit}' "$wrap_ptx"
	awk '/^\.address_size /{print; exit}' "$wrap_ptx"
	awk '!/^\.version / && !/^\.target / && !/^\.address_size /' "$bpf_ptx"
	awk '
		{ line = $0; gsub(/^[ \t]+|[ \t\r]+$/, "", line) }
		inext { if (line == ";") inext = 0; next }
		line ~ /^\.extern[ \t]+\.func/ { inext = 1; next }
		line ~ /^\.version / || line ~ /^\.target / || line ~ /^\.address_size / { next }
		{ print }
	' "$wrap_ptx"
} > "$out"
