# Native sm_120 guardian-stream extraction

The two stub kernels in this directory provide the guardian prefix and the
resume prefix that `GuardianSM120` copies into instruction memory:

- `check_preempt_port.cu` compiles to the check stub. It reads the
  preemption buffer (`c[0x0][0x1880]`) and the kernel arguments
  (`kernel_idx` at `+16`), and it retains a conditional exit behind the
  build-only `xg_check_exit_marker` store. The marker keeps the exit
  predicated after optimization; it writes an otherwise unused device
  marker and does not affect the extracted stream.
- `restore_exec_port.cu` compiles to the resume stub. It reloads the
  instrumented kernel entry point (`c[0x0][0x1888]`). The 0x1500-byte
  by-value pad parameter forces both stubs' parameter regions into the
  debugger window at `c[0x0][0x1880]..`, so no LDC immediate rewriting is
  needed. The extractor verifies the stub's compiled transfer tail
  (`HFMA2 R21=0`, `MOV R20` call-end preload, `CALL.REL.NOINC R2
  0xfffffffc`) on the actual instruction words, then rewrites the last
  three slots of the emitted resume stream to the transfer mechanism
  byte-identical in both runtime-proven upstream resume stubs (sm_70 and
  sm_86): `LDC R21, c[0x0][0x188c]`, `LDC R20, c[0x0][0x1888]`,
  `RET.ABS.NODEC R20 0x0`. The shared old-arch encodings are a candidate,
  not proof of sm_120 support on their own; the passing native run
  (raw `level2-native-retabs-20260909.oJDCbU`) is the evidence.

`ldc_patcher.cpp` consumes the two compiled cubins plus `nvdisasm` and emits
`xg_sm120_guardian_arrays.h`:

```
xg_ldc_patcher CHECK.cubin RESTORE.cubin NVDISASM OUT.h
```

For the current build layout (run from `level2-build`):

```
../level2/native/ldc_patcher \
    .output/<BUILD>/native/check_preempt_port.cubin \
    .output/<BUILD>/native/restore_exec_port.cubin \
    /usr/local/cuda-12.9/bin/nvdisasm \
    .output/<BUILD>/native/xg_sm120_guardian_arrays.h
```

The Makefile `ldc-patch` recipe passes these four arguments (the obsolete
probe cubin argument is gone; `probe.cu` is kept only as the historical
device of proof for the LDC immediate analysis).

## What the patcher verifies before emitting

- Both `.text.*` sections are 16-byte instruction streams and carry no main
  `.rela.text` relocations (the HAL copies the bytes raw; nothing could
  resolve a load-time relocation afterwards).
- The padded parameter ABI is intact: window at offset 0x380/0x1510-0x1518,
  ord 0 pad of 0x1500 bytes, and each real parameter lands exactly on its
  debugger slot (check: `preempt_buf` 0x1880, `_kernel_idx` 0x1890; restore:
  entry point 0x1888).
- The in-window consumers (decoded from `nvdisasm -c`) match the pinned
  prototype: check consumes {0x1880, 0x1890}, restore consumes {0x1880,
  0x1888}, all `LDC`/`LDCU` with width 8. Anything else fails loudly.
- The check blob cuts at `EIATTR_EXIT_INSTR_OFFSETS[0] + 16`, the retained
  predicated `@P0 EXIT` (encoding `0x...094d`). The marker store and the
  final standalone `EXIT` (`0x...794d`) are excluded; the extracted prefix
  falls through into the original kernel text that the HAL appends behind
  it. If the marker were removed from the stub, nvcc would collapse both
  exits into one unconditional `EXIT` and extraction fails (`exit_n != 2`).
- The restore blob cuts at the final `EXIT` and keeps, in order: the
  `@!P0 EXIT` (`0x...894d`), the entry-point load, and the rewritten
  transfer tail (`LDC R21/R20` from `0x188c/0x1888`, `RET.ABS.NODEC
  R20 0x0`). The compiled call tail - `HFMA2 R21=0`, `MOV R20` (call end),
  `CALL.REL.NOINC R2 0xfffffffc` - is still verified word-by-word before
  the rewrite (it pins the compiler-observed shape); the emitted stream
  replaces it. The transfer target is the full absolute entry address
  loaded from the debugger arguments; the transferred-to instrumented
  image ends with the original kernel `EXIT`, so nothing returns into the
  copied prefix. First run with this form (raw
  `level2-native-retabs-20260909.oJDCbU`) completed exit 0 with no repeat
  of the previous CUDA 700.

## Header contract

`xg_sm120_guardian_arrays.h` defines
`XG_SM120_CHECK_PREEMPT_BYTES` (currently 0x280) and
`XG_SM120_RESTORE_EXEC_BYTES` (currently 0x150) and the two
`static const unsigned long long` arrays. Each SASS instruction is two
`ull` entries: the 8-byte encoding word followed by its 8-byte
scheduling/control word, byte-identical to the cubin `.text` redaction the
HAL memcpy's. `static_assert`s pin `sizeof(array)` to the macros.

## Remaining unsupported forms (loud failures, not silent wrong output)

- Any change in consumer layout, parameter ABI, exit structure, or the
  pinned call/return-site encodings aborts extraction instead of emitting a
  wrong blob. Opcode constants (`0x094d`/`0x894d`/`0x794d`, `0x7344`,
  `0x7802`, `0x7431`) are pinned to the CUDA 12.9 (ptxas V12.9.86)
  specimens and must be re-derived for a newer toolkit.
- The REPLACEMENT transfer words (`0x00062300ff157b82`,
  `0x00062200ff147b82`, `0x0000000014007950` with control words
  `0x000fc00000000800` x2 and `0x001fea0003e00000`) are copied verbatim
  from the sm_70 and sm_86 upstream resume streams (the two archs carry
  identical words). sm_120 does not emit them itself; they are a shared
  old-arch candidate that the run validated under CUDA 12.9 / driver
  575.57.08.
- The compiled CALL `callinfo` immediate stays `0xfffffffc` exactly as
  compiled and the return site (`R20 = 0x150`) lies past the blob end; in
  the EMITTED stream both are replaced together with the call, so they
  survive only inside the cubin verification path, never in what the HAL
  copies.
- The marker literal is `HFMA2`-encoded; there is no token-search fallback
  for it (EIATTR exit offsets are the cut mechanism).
- Single-kernel cubins only; `.nv.merc.*` mirrored sections and
  `.nv.constant0` are not consumed; non-LDC/LDCU in-window consumers fail
  loudly.

## Reproducible source preparation from upstream (patch chain)

`xsched-native-meta-extend.patch` is a strict delta: it applies only on
top of the deploy XSched source pinned at upstream revision
`f49289f0220931df78de948ed841ecbaf960a919` (`deps/xsched` HEAD at publish
time) plus six preceding patches, and the chain ends with two further
steps, in this order:

1. `../xsched-level2-sm120.patch` - the sm_120 Level-2 port (arch file,
   guardian plumbing, `ToolPrepare`-era actuator, NVBit-era context
   push/pop in `Reactivate`, options, CMake).
2. `xsched-native-diag700-instrument.patch` - native-700 probes 1/2/3 in
   `instrument.cpp`. Applies BEFORE the resume-allocation padding: its
   probe-1 hunk context pins the un-rounded `Alloc(resume_size)` line.
3. `xsched-native-resume-allocation-padding.patch` - `ROUND_UP(resume_size, 256)`.
4. `xsched-native-original-entry-control.patch` - `XG_NATIVE_ORIGINAL_ENTRY_CONTROL`
   gate around `cuXtraSetEntryPoint`.
5. `xsched-native-window-args-relay.patch` - `AdoptWindowRelayExtra`/
   `RestoreWindowRelayExtra` implementations (cuda_command.{h,cpp}) and
   their instrument.cpp adoption call sites.
6. `xsched-native-launch-error-propagation.patch` - the `CUDA_ASSERT`
   fail-fast around `instrument_ctx_->Launch`.
7. `xsched-native-meta-extend.patch` - the shim meta extend (new
   `window_meta_extend.{h,cpp}`, four module-load hooks in
   `intercept.cpp`, grown-passthrough guard in `shim.cpp`), the HAL
   original-layout marshal over the relay implementation
   (`cuda_command.cpp`), and the one-line probe-2 hex fix
   (`instrument.cpp`, `entry=0x%llu` printed decimal; now `0x%llx`).
8. `xsched-native-level2-source-sync.patch` - the level2-era deltas
   that existed only in the successful live HAL source with no
   published patch: the OLD tool call was a plain `dlsym` of
   `xg_host_prepare` armed with `(ctx, func, dev)` plus a trailing
   zero-call; the level2 runs renamed it to `ToolPublish` exporting
   single-argument `xg_host_publish(dev)` (the form the successful runs
   used), with the `launch_mtx_` lock/unlock pairs around the actuator
   calls dropped; xg2/xg3 bounded debug counters and the `XG2
   Deactivate/Reactivate/Launch` logs removed; the at-create
   preempt-buffer header `MemsetD8Async`+sync removed; includes
   reordered (no `<atomic>`, `<cstdlib>` after `<cstring>`);
   `cuda_queue.cpp` `Reactivate()` replay restores the caller-context
   flow (NVBit-era `CtxPushCurrent_v2`/`CtxPopCurrent_v2` juggling
   removed). The matching gate tool itself is a repo-side artifact
   (level2/tool) and is not part of the patchable source tree.
9. `xsched-native-async-xqueue-audit.patch` - the suspend/resume audit
   counters and destructor `XSCHED_AUDIT` report that also existed only
   in the successful live PAD source (`preempt/async_xqueue.{h,cpp}`,
   outside `platforms/`, which the earlier platforms-scoped diffs
   missed): two relaxed atomics next to `suspended_`/`terminated_`, one
   `fetch_add` in `Suspend()` and one in `Resume()`, and the destructor
   `fprintf` of the audit line. No new audit logic - packaging exactly
   the measured-source diagnostics so a reproduced tree pays the same
   per-suspend/resume counter overhead.


Superseded content note: the previous published meta-extend form
re-carried the `cuda_command.h` relay hunks and the instrument.cpp
adoption/fail-fast hunks that patches 5 and 6 already contain; applying
the old form second makes `patch(1)` see previously-applied hunks. Those
hunk groups are removed from the current meta-extend patch; the
application order above is verified end-to-end (all sections apply
clean, no rejects).

`prepare_native_source.sh` automates the whole recipe:

```
level2/native/prepare_native_source.sh <xsched-git-checkout> \
    f49289f0220931df78de948ed841ecbaf960a919 \
    <new-output-dir> [<live-source-tree>]
```

It resolves all CLI paths before any `cd` (relative arguments work),
accepts normal checkouts and worktrees (`.git` file) via
`git rev-parse --git-dir`, runs with `set -euo pipefail` so a failed
`git archive` is never hidden by tar, extracts the revision with a
read-only `git archive` (the checkout's working tree, dirty or not, is
never read for content or modified; unrelated dirty edits are not
borrowed), fills the 3rdparty submodules before patching - `git
archive` outs gitlinks as empty directories, which leaves CMake unable
to configure - by archiving each submodule repo at the exact commit
pinned in the parent revision (recursive, so ipc's nested
boost-ipc/gtest land too) and extracting into the new tree; no
submodule working-tree files are copied and nothing is fetched, and a
missing pinned object fails with the needed `submodule update --init`
instruction, applies the nine patches noninteractively
with `patch -p1 --batch --no-backup-if-mismatch`, keeps every patch log,
every submodule tar, and the extracted upstream tar in
`<output>/logs/`, never deletes or
overwrites anything, refuses a non-empty output directory, and does no
build/GPU/driver work. Fuzz warnings are recorded in the log and
reported; they are tolerated because the chain's final file bytes are
the real check. The optional live-tree argument (the successful build's
`level2-build/.output/<build>/source`) adds an informational per-file
cmp report (`MATCH`/`DIFF`); the report does not gate the exit status,
and the run is not an audit framework - one clean application plus the
report is the deliverable.

### Honest status

With all nine steps applied the prepared tree reproduces the
successful live PAD source: the cmp report listed every chain-final
file as `MATCH`, the full-tree recursive diff found nothing outside
`3rdparty/` and the script's `logs/`, and the submodule extraction
matched the live vendored trees byte-for-byte. One known extra file
remains: `3rdparty/ipc/3rdparty/boost-ipc/container/build/Jamfile.v2`
is tracked at the pinned boost-ipc commit but was absent from the
successful live tree; it is inert build metadata for the boost tool,
not CMake input. Final-recipe build check (2026-09-10): the nine-step
preparation was re-run and a full cmake configure/build/install completed
against it with explicit `/usr/bin/gcc` and `/usr/bin/g++` (the machine's
default `c++` is a legacy shell wrapper that loses macro quoting, so the
first build attempt failed - its log is retained in the same
`.output` directory; that is a build-environment quirk, not a source
defect). The install at
`level2-build/.output/native-repro-supervisor-20260910.wEUkbn/install-gcc`
is wired to the same `20tGK7` arrays header and the normal
`libcuda.so.1 -> libshimcuda.so` shim links. Build completion alone is
not a performance result. What the chain does NOT include: build steps
(ldc-patch extraction of the sm_120 guardian arrays header, cmake
configure, install), the repo-side artifacts (stub `.cu` sources,
`ldc_patcher.cpp`, gate tool, run scripts, notes - all already in the
workloads repo), and any GPU/driver work; those remain the build
owner's.
