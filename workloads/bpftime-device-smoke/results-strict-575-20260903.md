# Strict device-callback validation on RTX 5090

Two fresh-process positive/negative pairs passed on 2026-09-03, driver
575.57.08 and Linux 6.15.11. This closes the **narrow strict counter path**,
not general verifier soundness, the POD pointer/ticket ABI, or performance.

| Retained run | Positive | Negative | Cleanup |
| --- | --- | --- | --- |
| `raw/575-r5-strict-02` | Actual strict admission of 13 instructions; 4,096 threads each execute eight callbacks, total 32,768 | Lane-varying branch rejected at instruction 1, before creating its policy hook; fresh readback of all counters remains zero | Both private segments removed, no owned process survivors, clean pre/post GPU and kernel checks |
| `raw/575-r5-strict-03` | Same complete positive result | Same complete rejection and zero-counter result | Same complete cleanup and safety result |

Every native and instrumented target checks all 32,768 vector outputs exactly.
The negative target may execute its original CUDA kernel after policy rejection;
application correctness alone is not the rejection oracle. Full logs retain
`mode=STRICT, hook_created=0`, propagated initialization failure and the later
zero-counter observation. The rejected program was never executed in warning
or disabled mode.

## Build and repairs

Both agent and syscall server come from the independent R5 build
`bpftime-r5/build-r5-strict-device`, with verifier, CUDA attachment and LLVM JIT
all enabled. Source is revision `ea9907d`, based on `b4b0ba8`; LLVM JIT is the
pinned submodule `9ea0180d`. CUDA 12.9, LLVM 15, GCC 13.3 and the existing Frida
16.1.2 Gum archive were used. Nothing was installed over the performance
runtime. The 277-step build ran on CPUs 0–7 with eight jobs, after POD timing
ended; no formal performance cell overlapped it. Build logs retain existing
C++ ODR/header warnings, which this scoped repair does not resolve.

Three initial defects and their failed evidence are retained:

1. The new C++ fixture wrongly assumed an empty interceptor starts disabled.
   It actually starts enabled. The corrected test checks **unchanged** state
   and absence of hook ID 1; runtime rejection behavior was not weakened.
2. Strict run 01 matched the long ELF function name against the syscall name.
   `BPF_PROG_LOAD` retains only 15 characters, `cuda__count_ret`. The parser
   now requires that exact ABI name, with negative tests for other names.
3. Run 01 also exposed truncated host readback: the map allocated 4,096
   thread entries but advertised the default 1,024, returning only 8,192 of
   the expected 32,768 callback counts. R5 now stores the actual configured
   thread count in the map metadata after allocation. The unchanged complete
   4,096-thread engagement requirement passes in both subsequent runs.

Run 01 failed and did not proceed to the negative cell. It is not relabelled
as a pass. No count threshold was reduced, no verifier rule was changed and
no dirty main-runtime source was copied into R5.

The fresh build passes five public-verifier cases / 28 assertions, the
corrected strict counter fixture / six assertions, and mode-control fixture /
five assertions. Twelve Python evidence/cleanup tests pass. C++ fixtures are
not the compiled BPF object: the real pairs separately verify and execute the
actual instructions and map descriptor. Build/test logs and instruction
listings are in `docs/experiment/revision-safety/strict-build-575-01/`.

## Claim boundary

An admitted per-thread map callback executes correctly on a real return event;
the tested lane-varying callback is rejected before its policy hook is created,
without preventing the original application from producing correct output.
This is not a claim that all CUDA interception is absent on rejection, that
arbitrary callbacks are safe, or that the verification-disabled POD and
observability performance runtime was retrospectively verified. Native
scheduler-init and invalid-prefetch transition oracles remain separate work.
