# eBPF-to-SASS AOT readiness note

Date: 2026-09-04

## Status

The bpftime `revision/sass-backend` branch at commit `bf048fa` completes a
narrow build-time AOT feasibility path. The test reads a real clang-built
ELF64 BPF object section, runs the existing strict GPU verifier, passes the
verified instructions to the existing
`ptxpass::compile_ebpf_to_ptx_from_words` compiler, assembles the resulting
PTX for `sm_120` with CUDA 12.9 `ptxas`, and inspects the cubin with
`cuobjdump`. The inspection confirms `code for sm_120`, the
`sass_aot_probe` SASS function, and the global entry symbol
`sass_aot_probe`. The helper-free fixture returns a constant so this test
exercises the real lowering and assembler path without substituting
hand-written policy semantics.

The negative case uses a correctly encoded lane-varying branch. The strict
verifier rejects it with `Warp-Uniform Branch Conditions` before the compiler
or `ptxas` runs and before PTX or cubin artifacts are created. The spike is
default-OFF, and configuring it with CUDA attachment disabled fails explicitly.

This evidence updates the implementation-readiness state recorded in the
[earlier SASS-only admission audit](revision-sass-only-stop-20260904.md), which
predated the committed AOT path. It does not change that audit's requirement
for live application integration before making a paper claim.

## Reproduction record

From the bpftime checkout on `revision/sass-backend` at `bf048fa`:

```sh
cmake -S . -B build-spike \
  -DBPFTIME_ENABLE_CUDA_ATTACH=1 \
  -DBPFTIME_ENABLE_SASS_AOT_SPIKE=1 \
  -DENABLE_EBPF_VERIFIER=1 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-spike --target bpftime_sass_aot_tests -j2
ctest --test-dir build-spike \
  -R '^bpftime_sass_aot_tests$' \
  --output-on-failure
```

On 2026-09-04, configuration and compilation completed successfully. CTest
ran `bpftime_sass_aot_tests`: 1/1 passed, 0 failed, in 0.24 seconds.

A separate fresh configuration with
`-DBPFTIME_ENABLE_SASS_AOT_SPIKE=ON` and
`-DBPFTIME_ENABLE_CUDA_ATTACH=OFF` stopped with the expected requirement that
CUDA attachment be enabled.

## Claim boundary

This result establishes real BPF-ELF-to-cubin/SASS artifact generation and
strict-verifier rejection ordering. It does not insert the generated code into
a live PTX-free application binary, execute an application hook, validate
application/helper/map semantics, or measure insertion and runtime overhead.
It also does not use NVBit to patch an existing SASS binary. Therefore it does
not fully validate the rebuttal's historical claim of a working SASS-level
patching prototype or support a PTX-free application-overhead claim. The
defensible current statement is that an AOT artifact-feasibility prototype now
exists; live PTX-free application insertion and its overhead campaign remain
open.
