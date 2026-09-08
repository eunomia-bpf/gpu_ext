#!/bin/bash
# Merge the BPF .func PTX and the trusted trampoline PTX into a single
# ptxas-ready translation unit (adapted from the verified sass-kretprobe
# pipeline, tool/merge_ptx.sh):
#   - one .version/.target/.address_size header (from the trampoline TU)
#   - xsched_guardian definition (BPF exporter output, headers stripped)
#   - xg_tramp / decision helpers (trampoline, headers stripped, the
#     matching .extern .func xsched_guardian declaration removed so the
#     call resolves inside the module)
# Args: BPF_PTX TRAMP_PTX OUT
set -euo pipefail
bpf_ptx=$1
tramp_ptx=$2
out=$3

{
	awk '/^\.version /{print; exit}' "$tramp_ptx"
	awk '/^\.target /{print; exit}' "$tramp_ptx"
	awk '/^\.address_size /{print; exit}' "$tramp_ptx"
	awk '!/^\.version / && !/^\.target / && !/^\.address_size /' "$bpf_ptx"
	awk '
		{ line = $0; gsub(/^[ \t]+|[ \t\r]+$/, "", line) }
		inext { if (line == ";") inext = 0; next }
		line ~ /^\.extern[ \t]+\.func/ { inext = 1; next }
		line ~ /^\.version / || line ~ /^\.target / || line ~ /^\.address_size / { next }
		{ print }
	' "$tramp_ptx"
} > "$out"
