# eBPF-to-SASS AOT readiness note

Date: 2026-09-04 (updated 2026-09-07)

## Existing-application injection — completed 2026-09-07

Main `9d823ede` publishes the [live EXIT-injection result](../../workloads/sass-kretprobe/results/sass-exit-575-20260907-01/results.md).
A real BPF ELF is compiled into a device function and embedded in an NVBit
tool. In a cubin-only vector-add application on RTX 5090, two EXIT sites
execute the callback on all 100,352 launched threads; every per-thread
slot contains 42, and the target's expected result is retained. Both
baseline and instrumented processes return zero. This extends the earlier
standalone/companion boundary to real in-body application instrumentation,
but only for one context, one selected kernel and bounded context writes.
The historical sections below retain their original scope; they do not
establish general helpers/maps, late attach or a full performance comparison.

## Recovered later branch evidence — 2026-09-07

The current published `revision/sass-backend` head is `8e4e64d`, not
`fd976ea`. The latter remains the standalone result described below.
Subsequent commits `31b9fdc` and `4e5110f` add a same-context companion
kernel and first-party launch interposition; `8e4e64d` publishes the
[five-run report](https://github.com/eunomia-bpf/bpftime/blob/8e4e64d/results/sass-aot-interpose-575-01/results.md)
and [raw log](https://github.com/eunomia-bpf/bpftime/blob/8e4e64d/results/sass-aot-interpose-575-01/raw.log).

The root re-read the existing 87,811-byte raw log without rerunning the GPU:
all ten phase summaries have return code zero, and there are 320 callback
records (five interposed runs, 64 iterations each). Median steady per-iteration
total time is **5.1375 us uninstrumented / 32.1015 us interposed**, approximately
6.25x or +524.8%. Cold totals are 16.205 us / 25,007.647 us, including the
first compilation on the interposed path. These are medians over five
per-run summaries; steady values divide each run's 63-iteration sum by 63.
They are not llama.cpp prefill throughput or a paired overhead estimate.

The companion executes a separate BPF-derived kernel before the original
application launch. It does **not** modify the target kernel's own SASS and
does not close the missing in-body instrumentation path. The
[current application-injection task](sass-existing-application-next-20260907.md)
therefore reuses NVBit's EXIT instrumentation rather than repeating either
the standalone or interposition study. All earlier source and results remain.

For the current implementation, a source-only worktree at
`/home/yunwei37/workspace/gpu/bpftime-sass-existing-application` (branch
`revision/sass-existing-application`, `fd976ea`) provides the committed
explicit-context verifier source. The existing Table1 verifier archive lacks
`verify_gpu_program_with_context`; local GLM compiled the required source
against reusable build dependencies. Main `08e1692e` publishes the working
exporter, build file and ABI notes. The real four-instruction BPF input now
exports a 316-byte device function, without promotion to a kernel entry.
The active Table1 worktree is unchanged. Application-injection execution
remains the next step, not a consequence of successful code generation.

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
