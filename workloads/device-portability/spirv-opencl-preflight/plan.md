# SPIR-V standalone device preflight plan

## Admission and scope

- **Reviewer-facing uncertainty.** The response says the device JIT has a
  SPIR-V backend path, but the retained deployment audit has only source
  inspection.  It is unknown whether the existing source-native demo still
  builds and executes in the current LLVM/OpenCL environment.
- **Expected result.** LLVM 20 emits a validator-accepted OpenCL SPIR-V module
  from the demo's eBPF bytecode, and the OpenCL device computes 100 + 42 = 142.
- **Contradictory result.** Configure, code generation, validation, OpenCL IL
  ingestion, kernel creation, execution, or result checking fails.  Such a
  result requires narrowing the paper to an unexecuted backend path.
- **Independent evidence.** The existing PTX/Table 1 experiments do not execute
  SPIR-V.  This preflight exercises a different LLVM target and OpenCL IL path.
- **Largest credible story.** The checked-in standalone bpftime path from eBPF
  through SPIR-V to a device remains runnable.  It cannot show that gpubpf can
  attach SPIR-V policies to application kernels or that another vendor works.
- **Role.** Supporting implementation evidence only.  This is one correctness
  preflight, not a performance experiment or a new portability case study.

This narrow check is worth running because success and failure change the
wording of the already-promised portability discussion.  A broader multi-vendor
campaign would have higher ultimate value but is not runnable on the only
OpenCL device currently exposed by this host.

## Frozen path

- Source: bpftime's existing
  `vm/llvm-jit/example/spirv/spirv_opencl_test.cpp`; no copied or rewritten
  generator is substituted.
- Toolchain: LLVM 20.1.8 native SPIR-V backend, system OpenCL 3.0 headers and
  loader, and SPIRV-Tools `spirv-val`/`spirv-dis`.
- Build: a new directory outside the Table 1 strict build and source submodule,
  with `LLVMBPF_ENABLE_SPIRV=ON` and `LLVM_DIR` fixed to LLVM 20.
- Positive oracle: one source-native demo process exits zero; reports the
  SPIR-V target, generation and patching; selects the RTX 5090 OpenCL GPU;
  loads, builds and launches `bpf_main`; and reports input 100, expected 142,
  actual 142 and `Test PASSED`.
- Structure oracle: emitted `bpf_program.spv` is word-aligned, has the SPIR-V
  magic word, has the byte count printed by the demo, passes `spirv-val`, and
  disassembles with one `OpEntryPoint Kernel ... "bpf_main"`, an OpenCL memory
  model and a function body.
- Negative control: copy the emitted bytes, replace only the magic word with
  zero, require `spirv-val` to reject it, and never load that copy with OpenCL.
- Safety: acquire the existing GPU-0 and struct-ops leases via read-only file
  descriptors; reject ambient injection variables; require the 575.57.08
  driver, active 400 W limit service, idle device, zero UVM reference count,
  empty struct-ops inventory and clean current-boot GPU/kernel diagnostics;
  require the same conditions after process-group cleanup and unchanged boot.
- Completion: exactly one positive execution, one positive validation and
  disassembly, one rejected tampered validation, no owned process survivors,
  and `result.json` status `complete`.
- Raw output: `raw/spirv-opencl-575-01/`; analysis output:
  `raw/spirv-opencl-575-01/analysis.json`.
- Retry rule: no automatic retry or relabeling.  A failed directory remains
  retained; a repaired attempt uses a new directory.

## Interpretation

- Positive: say only that the standalone eBPF-to-SPIR-V OpenCL demo built and
  executed correctly on the RTX 5090 in this environment.
- Negative: remove runnable/prototype wording and describe SPIR-V solely as an
  architectural direction until repaired.
- Mixed: if code generation/validation passes but OpenCL execution fails,
  distinguish a runnable code generator from an unvalidated execution path.
- This result never supports performance, gpubpf attach integration,
  cross-layer-map integration, verifier coverage, or cross-vendor claims.

