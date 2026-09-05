# eBPF-to-SASS AOT readiness note

Date: 2026-09-04 (updated 2026-09-05)

## Status

The bpftime `revision/sass-backend` branch at commit `fd976ea` establishes
verified standalone live cubin execution. A real clang-built BPF ELF section
`cuda__/sass_aot` writes 42 through the complete pipeline: the existing
strict GPU verifier (explicit 8-byte PREVAIL context plus SIMT verification),
the existing `ptxpass::compile_ebpf_to_ptx_from_words` eBPF-to-NVPTX
compiler, CUDA 12.9 `ptxas` assembly for `sm_120`, and CUDA Driver API
module load, entry-point lookup, 1x1x1 launch, synchronize, and DtoH
transfer.

The live command shown below exited zero and printed
`verified SASS result: 42`. Post-run driver 575.57.08, 15 MiB, zero percent GPU
utilization, P0. CPU build targets `bpftime_verifier_tests`,
`bpftime_sass_aot_tests`, and `bpftime_sass_aot_live` passed. The focused
explicit-context verifier test passed 3 assertions. Both CTest verifier and
AOT suites passed. The invalid lane-varying SIMT case is rejected before
PTX, cubin, or ptxas.

This result updates the implementation-readiness state recorded in the
[earlier SASS-only admission audit](revision-sass-only-stop-20260904.md),
which predated the committed AOT path. It replaces the prior build-time-only
feasibility record with verified standalone live cubin execution on the GPU.

## Reproduction record

From the bpftime checkout on `revision/sass-backend` at `fd976ea`:

```sh
cmake -S . -B build-spike \
  -DBPFTIME_ENABLE_CUDA_ATTACH=1 \
  -DBPFTIME_ENABLE_SASS_AOT_SPIKE=1 \
  -DENABLE_EBPF_VERIFIER=1 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-spike \
  --target bpftime_verifier_tests bpftime_sass_aot_tests bpftime_sass_aot_live -j2
ctest --test-dir build-spike -R '^bpftime_(verifier|sass_aot)_tests$' \
  --output-on-failure
# Live cubin execution:
./build-spike/attach/nv_attach_impl/sass_aot/bpftime_sass_aot_live \
  /tmp/bpftime_sass_aot_live-20260905 0
```

On 2026-09-05, configuration and compilation completed successfully. CTest
ran both verifier and AOT suites with no failures. The live command exited
zero and printed the verified SASS result 42 on the RTX 5090 with driver
575.57.08.

A separate fresh configuration with
`-DBPFTIME_ENABLE_SASS_AOT_SPIKE=ON` and
`-DBPFTIME_ENABLE_CUDA_ATTACH=OFF` stopped with the expected requirement that
CUDA attachment be enabled.

## Claim boundary

This result establishes verified standalone live cubin execution: a real
BPF-ELF section passes through strict GPU verification (PREVAIL plus SIMT),
PTX compilation, PTX assembly, CUDA Driver API module load, entry-point
lookup, kernel launch, and DtoH readback, producing the expected constant on
real GPU hardware. It does not insert the generated code into an existing
PTX-free application binary, execute an application hook, validate
application/helper/map semantics, measure insertion and runtime overhead, or
use NVBit to patch an existing SASS binary. The boundary is standalone
generated cubin only; no instrumentation or injection into arbitrary
PTX-free existing-application SASS/fatbin is claimed, and the result is not
performance evidence or full historical NVBit claim validation.
