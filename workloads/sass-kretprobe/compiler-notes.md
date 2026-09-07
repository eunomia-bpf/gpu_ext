# bpf_to_ptx compiler notes (GLM-owned: bpf_to_ptx.cpp, compiler.mk)

Status: done. Built via compiler.mk and exported the real input;
root's integrated make (build/ptx/bpf_exit.ptx) succeeded and the
inspected output is 316 B with the expected `.visible .func` and two
`.param .b64` parameters. No commits (root owns commits).

## CLI

    make -f compiler.mk                 # builds build/bpf_to_ptx (relative)
    make -f compiler.mk clean-compiler  # removes only exporter artifacts

    build/bpf_to_ptx INPUT_BPF_OBJECT SECTION OUTPUT_PTX [SYMBOL=bpf_exit] [SM=sm_120]

Example (real root-provided input):

    build/bpf_to_ptx build/kretprobe_sass.bpf.o cuda__/kretprobe_sass build/bpf_exit.ptx

Exit codes: 0 ok, 1 usage, 2 BPF-ELF load error, 3 GPU verifier rejected,
4 eBPF-to-PTX compilation failure, 5 output-write failure. Diagnostics on
stderr; stdout stays clean. PTX is only written for verifier-accepted
programs (rejected programs never reach the compiler).

## Pipeline (existing components only, no policy hand-coding)

1. Load the named section of a real clang-built BPF ELF object
   (libelf; requires 64-bit EM_BPF, PROGBITS, ALLOC|EXECINSTR, 8-byte
   multiples) -> vector<uint64_t> words.
2. GPU verifier with rejection enforced by the exporter, fd976ea sources:
   `bpftime::verifier::gpu::verify_gpu_program_with_context(words, section, 8)`
   - PREVAIL with an 8-byte read/write context descriptor {8,-1,-1,-1} and
     check_termination, assume_assertions and no_simplify enabled. PREVAIL's
     internal `strict` option remains false; enforced rejection does not
     mean that option is enabled.
   - Uniformity analysis + SIMT safety check run inside the same call
     chain; a SIMT violation is reported as a rejection, never bypassed.
3. Existing ptxpass compiler:
   `ptxpass::compile_ebpf_to_ptx_from_words(words, target, symbol,
   /*guard*/false, /*with_arguments*/true)`.

## Generated PTX ABI (compiler output kept byte-identical)

    .visible .func SYMBOL(
        .param .b64 SYMBOL_param_0,   // context pointer (the 8-byte R1 context)
        .param .b64 SYMBOL_param_1    // context length in bytes (always 8 here)
    )

- Two parameters: the injected device wrapper calls this function using
  the PTX device-call ABI, passing the context address and length 8.
  A `.func` cannot be launched using `cuLaunchKernel`; unlike the earlier
  standalone AOT executor, this exporter never promotes it to `.entry`
  and never edits the body.
- The verifier context and parameter 0 are the same 8-byte buffer.

## PTX target split (forward-compatibility; do not "fix")

- LLVM 15 NVPTX codegen target passed to the compiler: `sm_86` fixed
  (LLVM 15 cannot codegen sm_120).
- Standalone header emitted by the exporter: `.version 8.7`,
  `.target SM` (CLI arg, default `sm_120`), `.address_size 64`
  (CUDA 12.9 ptxas; PTX is forward-compatible).

## Source path (owned)

- workloads/sass-kretprobe/bpf_to_ptx.cpp
- workloads/sass-kretprobe/compiler.mk
- workloads/sass-kretprobe/compiler-notes.md
- Generated outputs under workloads/sass-kretprobe/build/
  (bpf_to_ptx, *.o, exported .ptx). Everything else in the directory is
  Qwen-owned and untouched.

## Reused source/build dependencies (read-only)

- SASS_TREE=/home/yunwei37/workspace/gpu/bpftime-sass-existing-application
  (fd976ea). Only one fresh TU is compiled:
  bpftime-verifier/src/gpu/gpu_verifier.cpp (adds
  verify_gpu_program_with_context). Include paths: SASS_TREE verifier
  include first (fd976ea gpu_verifier.hpp), then
  MAIN_TREE/bpftime-verifier/ebpf-verifier/src, then
  build-table1-575-warp/libbpf. Flags copied verbatim from
  bpftime-verifier/CMakeFiles/bpftime-verifier.dir/flags.make so the TU is
  ABI-compatible with the archived objects (simt/uniformity/gpu_platform
  sources are byte-identical between fd976ea and the built tree).
- Everything else links prebuilt archives from
  MAIN_TREE/build-table1-575-warp (no runtime rebuild):
  - bpftime-verifier/libbpftime-verifier.a    (verifier core; old TU of
    verify_gpu_program stays but is shadowed by the fresh object)
  - bpftime-verifier/ebpf-verifier/libebpfverifier.a      (PREVAIL)
  - bpftime-verifier/ebpf-verifier/external/libbtf/libbtf/liblibbtf.a
    (needed by PREVAIL asm_files.cpp.o; was the only missing dep)
  - attach/nv_attach_impl/pass/ptxpass_core/libptxpass_core.a
    (compile_ebpf_to_ptx_from_words; with_arguments=true ABI above)
  - vm/compat/llvm-vm/libllvmbpf_vm.a + /usr/lib/llvm-15 static libLLVM*.a
    (link set mirrored from the proven maps-example link.txt),
    third_party/spdlog/libspdlogd.a
  - system: -lelf -lrt -ldl -lm -lz -lzstd -ltinfo -lpthread; linker uses
    --start-group/--end-group; LLVM wildcard via
    `$(wildcard /usr/lib/llvm-15/lib/libLLVM*.a)`.
- compiler.mk contract: `BUILD ?= build`, `BIN := $(BUILD)/bpf_to_ptx`
  (relative so the main Makefile can depend on build/bpf_to_ptx); clean
  target is named clean-compiler to avoid clashing when included.
