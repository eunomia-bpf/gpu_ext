# CPU-only SPIR-V build audit — 2026-09-04

## Outcome

The existing source-native `spirv_opencl_test.cpp` configures and builds in a
new directory with LLVM 20.1.8's native SPIR-V target and the system OpenCL 3.0
development interface.  The initial build completed all 15 compile/link steps;
the retained follow-up configure log reconfirms the same LLVM/SPIR-V/OpenCL
selection, and the retained target build reports that the executable is up to
date.  No OpenCL program or GPU workload was executed in this audit.

The 136,907,048-byte executable is newer than its 12,765-byte demo source,
loads an available `libOpenCL.so.1`, and contains the `spirv64` target and
result-oracle literals.  The relevant tracked source inputs are unmodified.
The only source-submodule status entry is the pre-existing untracked
`cli/libbpf_project/`, which this work neither reads as evidence nor changes.

## Evidence boundary

This resolves only the dependency/build question.  It does not show that the
code generator emits a valid module, that OpenCL accepts it, or that the device
computes 142.  Those checks remain frozen in `run_spirv_opencl_preflight.py`
and must wait for the shared GPU leases.  Even a passing device preflight will
show only a standalone LLVM-SPIR-V/OpenCL path on NVIDIA, not a gpubpf attach
backend, SIMT-verifier integration, cross-layer maps, or AMD/Intel execution.

## Retained records

- `raw/build-audit-575-01/configure.log`: LLVM 20, SPIR-V component, native
  backend, OpenCL include/library, and configure output.
- `raw/build-audit-575-01/build.log`: follow-up target build status.
- `raw/build-audit-575-01/tool-versions.txt`: tool versions and executable
  presence.
- `raw/build-audit-575-01/linked-libraries.txt`: dynamic dependency resolution.
- `raw/build-audit-575-01/file-metadata.tsv`: ordinary size/time/mode inventory.
- `raw/build-audit-575-01/source-state.txt`: source revision and the retained
  untracked status entry.
- `raw/build-audit-575-01/semantic-gates.tsv`: eight fail-closed build checks.

All configure/build commands were pinned to CPUs 16–23 with at most two build
jobs for the retained pass, leaving the active S0 client cores 8–15 untouched.
