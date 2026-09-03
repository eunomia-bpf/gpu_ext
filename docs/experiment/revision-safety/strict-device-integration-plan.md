# Q2: minimal strict-verifier/device integration

Status, 2026-09-03: the minimal integration code and 11 offline Python tests
are complete; all 11 tests passed on CPU 17, including a finite CPU orphan
cleanup test. The C++/BPF changes have **not been compiled or run**, and no
strict GPU evidence exists yet. No configuration, installation, GPU run or
worktree creation was performed. Root reviewed the scoped changes, independently
reran all 11 Python tests, and committed/pushed the R5 changes as `b4b0ba8` on
`revision/r5-safety-evidence`. This extends the existing
[Q2 safety work](../../revision-safety-design.md), not the performance
protocol. HB retains exclusive use of the GPU; the commands below await the
root agent's GPU/build queue slot.

The R5 edits start from commit `36610ee` on the existing branch
`revision/r5-safety-evidence`, initially clean. Only the three R5 files listed
in Section 6 were changed. The main `../bpftime` worktree, vendored code,
and frozen POD/HB sources were not changed by this task.

## 1. Claim to establish and current boundary

Establish one narrow end-to-end claim: a verifier-enabled runtime in strict
mode rejects an unsupported device callback before creating its hook, while
the **same admitted callback instruction stream** actually executes on a
real device event and preserves the workload's full numerical output.
This is dependency validation for Q2, not a scheduling result, a performance
comparison, a proof of verifier soundness, or validation of the POD interface.

The existing performance runtime is `../bpftime/build-cuda-pr503`; its cache
has `ENABLE_EBPF_VERIFIER=OFF`. The existing R5 source at `../bpftime-r5`
contains the verifier and strict attach path. However, its `build-r5-v2`
cache enables the verifier but disables CUDA attach **and LLVM JIT**. Neither
build currently establishes the proposed combined path. Preserve both,
including the dirty development runtime and all frozen POD sources.

R5 source references below are relative to `../bpftime-r5`; line references
in this table describe the pre-edit source at `36610ee`:

| Existing component | Relevant source and implication |
| --- | --- |
| Public GPU verification | `bpftime-verifier/src/gpu/gpu_verifier.cpp:267`: PREVAIL, uniformity analysis, SIMT checks; `:309`: public `verify_gpu_program` entry |
| Strict rejection before hook creation | `attach/nv_attach_impl/nv_attach_impl.cpp:228`: verify `data.instructions` using real map descriptors; `:245`: return `GPU_VERIFIER_REJECTED`; only then `:256` enables attachment and `:259` copies those instructions into the hook |
| Verifier linkage | `attach/nv_attach_impl/CMakeLists.txt:58`: `ENABLE_EBPF_VERIFIER` links `bpftime-verifier` and defines `ENABLE_BPFTIME_VERIFIER` |
| Mode selection and propagation | `runtime/src/bpftime_config.cpp:133`: `BPFTIME_VERIFIER_LEVEL=STRICT`; `runtime/src/attach/bpf_attach_ctx.cpp:360`: passes mode to CUDA attachment |
| Fail-closed link initialization | `runtime/src/attach/bpf_attach_ctx.cpp:130`: this rejection class destroys created links and returns the error; `runtime/agent/agent.cpp:931` checks initialization result |
| Existing attach-mode test, not CPU-only | `attach/nv_attach_impl/test/nv_attach_impl_tests.cpp:91`: strict rejects helper 7; warning and disabled modes can attach; successful attachment bootstraps a CUDA context, so do not run this test while HB owns the GPU |
| Existing CPU positive/negative pairs | `bpftime-verifier/test/gpu_revision_safety_test.cpp:146`: bounds, termination, branch, map side effects, atomic target, helper restrictions |

The negative claim concerns the unsafe **BPF callback**, not all GPU work:
the original application kernel may still execute without instrumentation
after agent initialization fails. Application failure, an attach-ready
message, or absence of a crash alone is not rejection evidence.

## 2. Reuse the finite device-event workload

Reuse [bpftime-device-smoke](../../../workloads/bpftime-device-smoke/), not
another vector-add harness or vendor example:

- `vector.cu:29`: eight launches, 4096 values checked after every launch,
  all CUDA API results checked; exactly 32768 correct outputs. No change is
  needed to its kernel, dimensions, launch count, or numerical gate.
- `probe.bpf.c:17`: real `kretprobe/_Z9vectorAddPKfS0_Pfi`, named
  `cuda__count_return`, with a constant map key and ordinary per-thread
  counter increment. Keep the program and map names aligned with the loader.
- `probe.c:38`: reads all 4096 per-thread counters; success requires every
  counter to equal eight, total 32768. This is independent of the vector's
  numerical result. There are no host increments to manufacture engagement.
- `run_smoke.py`, `run`: retained leases, safety snapshots, private shared
  memory and deadlines; cleanup now checks surviving owned process groups
  and the recorded shared-memory identity before deletion.
- Existing [canary evidence](../../../workloads/bpftime-device-smoke/raw/canary-evidence.json)
  records three failed attempts. The last achieved full numerical correctness
  and loaded a patched module, but did not prove callback execution. Added
  counter snapshots now make the next attempt more diagnostic.

The counter is a plausible positive for the existing model: its constant
key gives a uniform map-lookup null predicate; the per-thread value is
varying but is written only to a per-thread map, without atomic operations.
This follows `uniformity_analysis.cpp:419,630` and
`simt_safety_check.cpp:205,225`; **actual admission is still untested**.
Do not relax the required count to one callback per warp: `call.uni` is not
a one-lane invocation.

## 3. Implemented changes; build pending

After the current exclusive experiment, use the existing R5 checkout and a
separate build directory, for example `../bpftime-r5/build-r5-strict-device`.
Do not create another source worktree, install system-wide, modify vendored
dependencies, or repoint the performance runtime. Required configuration is
`ENABLE_EBPF_VERIFIER=ON`, `BPFTIME_ENABLE_CUDA_ATTACH=ON`,
`BPFTIME_LLVM_JIT=ON`, and `BPFTIME_CUDA_ROOT=/usr/local/cuda-12.9`.
Reuse the present Catch2 source through `FETCHCONTENT_SOURCE_DIR_CATCH2`;
retain the existing toolchain choices. Build only the needed
`bpftime-agent`, `bpftime-syscall-server`, `bpftime_verifier_tests`, and
`bpftime_nv_attach_tests` targets and their dependencies. CUDA/LLVM combined
build readiness remains to be checked; this note has not built them.

The source changes are limited to:

1. R5 `create_attach_with_ebpf_callback` now emits an explicit successful
   verification record immediately after the checked call, including strict
   mode, program/attach-point name, instruction count, and map types/sizes.
   The existing copy into the hook uses that same checked vector. Rejection
   and return behavior are unchanged; failed strict verification records
   `hook_created=0`. Warning-bypass and disabled modes emit no success record.
2. A new `GPU strict counter admission and rejection` test uses map type
   1502, key size 4, value size 8, and one entry. It checks positive admission
   through the public verifier and passes only the lane-branch negative to
   attachment, asserting the intended diagnostic, rejection, disabled state,
   and absence of hook ID 1. It does not make a successful CUDA attach.
   The fixture is reconstructed and is not called the device artifact;
   this C++ test has not yet been built or executed.
3. One separate negative object is defined in the existing smoke BPF
   source/Makefile: helper 511 (lane ID) controls whether the counter is
   incremented. The positive behavior, section, map, and program name remain
   unchanged. Before execution, inspect compiled instructions and require the intended
   branch-uniformity diagnostic; compiler elimination or a different rejection
   is not a passed SIMT-branch test. Do not execute this negative in warning
   or disabled mode on a GPU.
4. `run_smoke.py --strict` now inserts `STRICT` into the actual clean child
   environment, checks the three build flags, and uses both preload libraries
   from the selected R5 build. It runs the positive cell first and starts the
   negative cell only after all positive gates and cleanup pass. Default
   non-strict behavior remains a single positive smoke.
5. The negative branch requires the specific SIMT diagnostic and fail-closed
   initialization logs, then obtains a fresh all-zero observer snapshot after
   target exit, within five seconds. Every counter field must be an integer,
   not a Boolean. Missing evidence or timeout fails the test. The original
   numerical gate is retained even for the uninstrumented negative target;
   `probe.c` and `vector.cu` are unchanged.
6. Cleanup locally reuses GPreempt's owned PGID/SID survivor rule without
   importing its colliding `run_smoke` module. It stops target, probe, then
   baseline groups, including descendants of an exited leader, attempts each
   group despite earlier errors, and checks that no live owned group remains.
   `lstat` records the private segment's regular-file type, current UID,
   device and inode after READY. Only that same file is deleted; unknown,
   replaced, wrong-owner or symlink paths are preserved and fail cleanup.
   A pre-existing dangling symlink is also rejected. Lease release remains
   guaranteed if result-file writing fails.

Use ordinary source revisions, explicit file paths/sizes, build flags,
instruction listings, logs, and test outcomes as evidence. Preserve all
earlier raw runs; put each new attempt in its own output directory.

## 4. Execution and decision gates

Inspect the combined build's cache and compiler definitions, then run the
existing public-verifier `[gpu][revision-safety]` tests and the new counter
case. The old `GPU verifier mode controls attach rejection` test is **not
CPU-only**: its successful warning/disabled attaches call
`start_late_bootstrap_async`, which initializes a CUDA context. Schedule it
only when the GPU is exclusively available. Do not run the entire CUDA test
suite as a substitute. The same program/device condition is established by the
real strict attach call verifying the exact instruction vector that the
hook stores, rather than by equating a hand-written CPU fixture with an ELF.

Once HB is finished and the existing exclusive leases and safety checks pass,
run one finite positive/negative pair, each with fresh processes and private
shared memory:

The following commands are prepared, **not executed**. Run from
`/home/yunwei37/workspace/gpu/gpu_ext`, after HB and root review, using a new
output directory for every attempt. No install or performance-runtime
replacement is involved:

```bash
taskset -c 17 cmake -S ../bpftime-r5 -B ../bpftime-r5/build-r5-strict-device -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBPFTIME_ENABLE_UNIT_TESTING=ON \
  -DENABLE_EBPF_VERIFIER=ON -DBPFTIME_ENABLE_CUDA_ATTACH=ON \
  -DBPFTIME_LLVM_JIT=ON -DBPFTIME_CUDA_ROOT=/usr/local/cuda-12.9 \
  -DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG \
  -DFETCHCONTENT_SOURCE_DIR_CATCH2=/home/yunwei37/workspace/gpu/bpftime-r5/third_party/Catch2
taskset -c 17 cmake --build ../bpftime-r5/build-r5-strict-device --parallel 1 \
  --target bpftime-agent bpftime-syscall-server bpftime_verifier_tests bpftime_nv_attach_tests
taskset -c 17 make -C workloads/bpftime-device-smoke strict
taskset -c 17 llvm-objdump -d workloads/bpftime-device-smoke/.output/probe.bpf.o
taskset -c 17 llvm-objdump -d workloads/bpftime-device-smoke/.output/probe-negative.bpf.o
taskset -c 17 ../bpftime-r5/build-r5-strict-device/bpftime-verifier/bpftime_verifier_tests \
  '[gpu][revision-safety]'
taskset -c 17 ../bpftime-r5/build-r5-strict-device/attach/nv_attach_impl/test/bpftime_nv_attach_tests \
  '[strict-counter]'
taskset -c 17 ../bpftime-r5/build-r5-strict-device/attach/nv_attach_impl/test/bpftime_nv_attach_tests \
  'GPU verifier mode controls attach rejection'
taskset -c 17 python3 -B workloads/bpftime-device-smoke/run_smoke.py --strict \
  --runtime-build ../bpftime-r5/build-r5-strict-device \
  --output workloads/bpftime-device-smoke/raw/575-r5-strict-01
```

Require every command to succeed before continuing. The final command creates
fresh `positive/` and `negative/` cells, each with its own logs, result file,
private segment and safety checks. Build failures or a different verifier
diagnostic remain failures; neither has been ruled out by the offline tests.

| Cell | Required evidence |
| --- | --- |
| Strict positive | Explicit successful strict admission; native and instrumented 32768/32768 numerical checks; 4096 counters each eight, total 32768; clean owned teardown and post-run safety |
| Strict negative | Intended lane-varying-branch diagnostic; propagated `GPU_VERIFIER_REJECTED` before hook creation; no positive admission/attachment for that object; 32768/32768 numerical checks; all 4096 counters remain zero after rejection; clean owned teardown and post-run safety |

A valid strict positive must pass first. Otherwise all-zero negative counters
could simply reproduce the old instrumentation failure. One complete pair is
the minimum device closure for this narrow claim; a fresh-process repeat can
check lifecycle reproducibility, without adding a performance study.

If positive admission succeeds but counters fail, retain full counter
snapshots and existing debug launch-routing output in
`attach/nv_attach_impl/nv_attach_impl_frida_setup.cpp:171,205,517`.
Eight confirmed patched launches with zero counters direct diagnosis toward
map/code generation; absent patched routing plus zero counters supports a
routing problem. Neither conclusion follows from module-load messages alone.
Repair only the reproduced R5 routing/observer defect, if any; do not lower
the count, bypass verification, copy dirty runtime changes wholesale, or
switch to a direct-run toy and call the device-event test complete.

## 5. POD remains separate interface work

The frozen POD selector accepts a 72-byte context containing device pointers
and uses an atomic ticket/single-leader contract. The present generic SIMT
model does not encode that ABI's pointer bounds, lifetime/ownership, legal
target memory, or the leader-only execution predicate. Its atomic rule also
requires a warp-uniform target; it does not automatically justify POD's
ticket-based protocol. Merely enabling the verifier flag can reject a valid
POD program, and bypassing that rejection would establish nothing.

After the counter path closes, POD needs a separately reviewed context/helper
contract, corresponding verifier transfer rules, and matched bounds,
provenance, leader, and atomic-protocol positive/negative cases before a
strict POD device run. None of those changes is authorized or implemented by
this note. Until then, retain the paper's explicit distinction between the
strict verifier prototype and verification-disabled performance measurements.

## 6. Scope inventory and completed offline checks

| Root | Changed file | Scope |
| --- | --- | --- |
| `../bpftime-r5` | `attach/nv_attach_impl/nv_attach_impl.cpp` | Strict admission/map records and rejection-before-hook marker; no verifier rules or return semantics changed |
| `../bpftime-r5` | `attach/nv_attach_impl/test/CMakeLists.txt` | Link the existing verifier into the attach test target when enabled |
| `../bpftime-r5` | `attach/nv_attach_impl/test/nv_attach_impl_tests.cpp` | Counter positive/public verification and strict branch rejection fixture |
| `gpu_ext` | `workloads/bpftime-device-smoke/Makefile` | `strict` target and separate negative BPF object |
| `gpu_ext` | `workloads/bpftime-device-smoke/probe.bpf.c` | Compile-time negative lane guard; positive behavior unchanged |
| `gpu_ext` | `workloads/bpftime-device-smoke/run_smoke.py` | Strict ordered pair, evidence gates, owned group/segment cleanup |
| `gpu_ext` | `workloads/bpftime-device-smoke/test_runner.py` | 11 offline evidence, ordering, and cleanup tests |
| `gpu_ext` | This document | Current status, source lineage and unexecuted commands |

Completed: `taskset -c 17 python3 -B -m unittest discover -s
workloads/bpftime-device-smoke -p test_runner.py -v` passed all 11 tests;
`run_smoke.py --help`, `make -n ... strict`, and scoped `git diff --check`
passed. The make command was a dry run only. Root reviewed the scoped
admission/evidence and cleanup changes. C++ tests, BPF compilation, combined
runtime build, and the strict real-device pair remain pending; this is not
yet Q2 device-validation closure.
