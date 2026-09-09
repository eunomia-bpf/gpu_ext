# Window repair evidence, 2026-09-08 session

Bounded local evidence toward the actual native parameter delivery for the
sm_120 Level-2 fused guardian/resume prefixes. No GPU was run by the agent
and no project build was run; the only built artifacts are the standalone
nvcc stub compilations of section 3 (CPU-only). This is a
records-and-feasibility note.

## 1. Fault model (raw-log evidence)

- Original-entry control (pn3UDn, change f6590e80) PASSED: 6 workers, 1200
  kernels launched with the original entry in control.
- The asynchronous CUDA_ERROR_ILLEGAL_ADDRESS (CUDA 700, "an illegal memory
  access was encountered") pattern appeared in runs whose guarded branch
  launched an instrumented image: every `probe 2: launch ret=0`, and then
  one or more `[XSCHED ERRO] cuda error 700: an illegal memory access was
  encountered @ .../cuda_command.cpp:31` surfaced at
  `CudaCommand::Synchronize` -> `EventSynchronize` (one `(null)` string
  variant seen; jOzEYp is the padding/alignment run in which the
  original-entry control env was unset). The earlier "immediate 700" belief
  was incomplete: launches validate, an instrumented image executes, and
  the fault surfaces asynchronously at the synchronize call.
- Relay config (cf39c397, UEtnu1): launches return 701 immediately (size
  re-validation rejection before execution). No crash, no running kernels.
- Hypothesis only, not observed causality: the exact faulting PC was never
  observed (no device debugger has located the fault). The reading that the
  check prefix's baked `LDC.64` consumers at `c[0x0][0x1880..0x1898]` fetch
  beyond compute_task's declared c[0x0] extent (0x3a0 bytes) is consistent
  with the evidence but is not proven; other in-image causes are not
  excluded by the logs.

## 2. Debugger-window mechanism (host side, driver 575.57.08)

- `libhalcuda.so` (568 KB, our shim+HAL+cuxtra bundle) --
  `cuXtraSetDebuggerParams` at 0x522e0 calls, through
  `cuxtra::etbl::GetEtblFunc(CU_ETID_ToolsModule, idx)` (UUID
  3fbe163e-4d4458b9-82e15c83-1e99f1af, resolved via the driver's
  `cuGetExportTable`):
  - idx 22: `EtblModule::FunctionSetDebuggerParamsSize(CUfunction, size_t
    size, uint8_t 0)` first, both results checked,
  - idx 20: `EtblModule::FunctionSetDebuggerParams(CUfunction, size_t
    offset=0, const void* params, size_t size)`.
  - `cuXtraGetDebuggerParams` (0x511e0) uses idx 21
    `(CUfunction, offset, void* buf, size_t)`.
- The wrapper therefore already attempts the size registration; it is not a
  bare host staging. Host-side round-trip through the window succeeded on
  every guarded launch in every run (`probe 2: readback equal=1`), so the
  etbl staging/state is consistent; what is unproven is the DEVICE
  placement/materialization of the window bytes at launch.
- The tools-module implementation lives in the real system driver
  (`/usr/lib/x86_64-linux-gnu/libcuda.so.575.57.08`, 92 MB, stripped). A
  full disassembly pass found no `0x1880` immediate, no UUID byte pattern,
  and the `cuGetExportTable` wrapper only binary-searches a 99-entry,
  24-byte-stride table before an internal resolver call. Placement semantics
  are not recoverable cheaply from the binary.

## 3. Embed-variant stubs, CPU-only builds (EXPERIMENTAL, NOT INTEGRATED)

Two parameter-free stub variants were written and compiled locally with
standalone nvcc on CPU (no GPU, no project build, not consumed by any HAL
build yet):

- `check_preempt_embed.cu`: identical guardian algorithm; the per-command
  values arrive as 64-bit placeholder constants. The compiler materializes
  each as a genuine single immediate instruction, e.g.
  `/*00e0*/ MOV.64 R2, 0xdf9f00010badc0de ;` and the two-exit retention
  structure survives (`@P0 EXIT`, marker store, final standalone `EXIT`).
- `restore_exec_embed.cu`: identical restore algorithm; placeholders plus
  the intact register-target transfer -- the listing keeps
  `MOV R20, 0x150` and `CALL.REL.NOINC R2` with the callee ending in the
  original kernel EXIT.

Feasibility of placeholder materialization: confirmed.

## 4. Why naive immediate patching is unsound (two-variant byte diff)

The placeholder low half was changed (0x0badc0de -> 0x0badc1de) and the
check cubin recompiled; the two `.text` streams were compared byte by byte.
The differing bytes land in FIVE 16-byte instruction slots, not one:

    instr pc=0e0 enc=010badc0de027402  (MOV.64 imm slot, diff byte 4)
    instr pc=140 enc=adc0de0606067980  (reassociated arithmetic, diff byte 6)
    instr pc=1b0 enc=adc0e60606047980  (reassociated arithmetic, diff byte 6)
    instr pc=1c0 enc=020badc0df087402  (MOV.64 imm slot, diff byte 4)
    instr pc=220 enc=adc0e60806007985  (reassociated arithmetic, diff byte 6)

The compiler splits a 64-bit pointer constant into a masked `MOV.64` high
part plus constant folding through `IADD.64` immediates and `LD.E`
displacements (load address reconstructed as sum of folded immediates).
Only the changed byte (0xc0) shows up; every byte of a patched value that
is not the changed byte stays propagated in unknown split forms. Patching
one slot for arbitrary runtime 64-bit values leaves stale copies behind.
This is why the HAL must not patch placeholder bytes without a mechanism
that re-derives every propagated form per value shape -- which is exactly
the kind of encoding assumption root excluded.

## 5. Route status

- Selected (root): compiler-declared extended-parameter/fused-module route
  for compute_task -- deliver the existing c[0x0][0x1880] window args as
  REAL launched parameters by extending the fused image's declared
  parameter metadata, keeping the original app arguments untouched and the
  same preempt/resume policy. Implemented this session as a sm_120 native
  actuator port (shim `window_meta_extend`), additive and unintegrated
  until root tests it; not claimable as an unchanged upstream artifact.
- Route A (window materialization GPU probe): deferred; no 4 KB-cap GPU
  probe is a gate for the selected route. Whether a rejection recurs at the
  extended sizes shows up in the run itself (the published launch-error
  propagation, patch commit 2b04ecaf attached in the diagnosis run, makes
  any launch failure fail fast instead of spinning on the started stamp).
- Route C (per-command embed): the materialization is sound (placeholders
  compile to real immediates) but general patching is not (section 4).

### 5.1 Implemented candidate: image metadata extension at module load

Isolated native HAL/shim sources touched (all inside
`level2-build/.output/hal-native-20260908.i8na51/source`):

- `platforms/cuda/shim/src/window_meta_extend.cpp` (new),
- `platforms/cuda/shim/include/xsched/cuda/shim/window_meta_extend.h` (new),
- `platforms/cuda/shim/src/intercept.cpp` (four load entry points
  instrumented: cuModuleLoadData, cuModuleLoadDataEx,
  cuModuleLoadFatBinary, cuLibraryLoadData), and
- `workloads/xsched/level2/native/run_native_blob.py`: the runner's worker
  environment previously scrubbed the variable via `rtp.clean_env()`; the
  opt-in variable is now forwarded exactly when the user sets it
  (presence-based, default off), the same explicit-forwarding pattern the
  original-entry control uses. For the test the control variable itself
  stays unset.

Mechanism (opt-in `XG_NATIVE_META_EXTEND`, gated at the shim):

- The image (fatbin container or bare ELF cubin) is copied into a raw
  malloc block carrying a 16-byte header (magic "XGEXTIMG" + data size);
  the handed-out pointer is the data pointer and
  `XgFreeExtendedImage(ptr)` validates the header and frees the block
  (foreign pointers are reported and deliberately not freed). The earlier
  std::vector-owner/data-pointer mismatch that crashed the CPU test
  harness is fixed by this raw-header ownership.
- Copy extents: bare cubin = max(ELF header, section table, program
  header table, all section bodies). The real 11136-byte specimen puts
  its five 56-byte program headers at 10856..11136, after the section
  table, so the earlier section-table-end framing (10856) was a real
  truncation and is fixed. Fatbin container extent = container header +
  fileSize field (the fileSize counts after the 0x10-byte container
  header in the nvcc 12.9 specimen), with a bounded legacy fallback.
- Per kernel `.nv.info.<mangled>` section of sm_120 kernels whose
  EIATTR_PARAM_CBANK declares param_start == 0x380: both compiler-written
  extent records are raised to 0x1520 --
  EIATTR_CBANK_PARAM_SIZE (id 0x19, fmt 3, 16-bit value at record
  offset +2) and EIATTR_PARAM_CBANK (id 0x0a, fmt 4, 16-bit param_size at
  payload offset +6). KPARAM_INFO (id 0x17, the app's own arguments) and
  every other record are byte-identical afterwards; all other sections
  (.nv.merc mirrors, module-wide .nv.info, .nv.constant0, code) are
  untouched. The relay blob built by the window-args relay for
  compute_task is exactly 0x1520 bytes (app params [0x380,0x3a0), window
  args at blob-relative 0x1500 = c[0x0][0x1880], span through 0x151c), so
  the declared extent now matches the delivered blob.
- Fail-open: every parse anomaly, unknown container, or oversized image
  logs a warning and loads the UNPATCHED copy; nothing aborts a load.
  Idempotent: an already-extended image passes through aligned (no
  rewrite).
- CPU verification of the shipped code, standalone test harness against
  the locally compiled workload cubin and fatbin (no GPU): exactly six
  bytes differ per image -- the two extent records of each of the two
  kernels (capture_globaltimer param extent 0x8 and compute_task 0x20,
  both raised to 0x1520); `nvdisasm -c` and `cuobjdump -elf` parse the
  patched cubin and patched fatbin and report EIATTR_CBANK_PARAM_SIZE
  0x1520 beside EIATTR_PARAM_CBANK "0x15200380"; repeated application is
  idempotent; freeing a foreign pointer is refused (reported, not freed) --
  note the helper validates the 16 header bytes preceding the pointer, so
  it is only safe for pointers the extender handed out. Session
  memory-safety runs did not retain a valgrind artifact file, so no
  valgrind-clean claim is recorded.

Explicit uncertainty to be resolved by the actual run, not a further
CPU-side prerequisite: `.nv.constant0` (the bank DATA image, 0x3a0 bytes
for compute_task) is deliberately NOT grown -- growing it would require
inserting bytes and rewriting all later section/table offsets. The current
implementation does not grow the data section; whether metadata-based
allocation suffices therefore remains unproven on the driver side and is
resolved by the actual run.
The run (root, XG_NATIVE_META_EXTEND=1, original-entry control unset,
same workload/50 tasks/340 blocks) decides between these; the publishable
failure mode is fail-fast via the launched-error propagation either way.

### 5.2 Runtime fatbin wrapper route (gdb diagnosis + NmlhJN run)

Root gdb (raw aY3rBY, debug-module-path.log / debug-module-image.log)
established the actual module-load dispatch: statically linked cudart
calls `cuLibraryLoadData` with a `__fatBinC_Wrapper_t` (fatbinary_section.h:
magic 0x466243b1, version 1), whose `data` field points at the real
container (root specimen: wrapper at `__fatDeviceText`, +8 ->
0x5555555efb88, +16 null). The pre-wrapper extender saw an unrecognized
magic and passed the image through unchanged silently -- which is why no
owned-copy log appeared. Wrapper interception, proc-address, and
export-table routes were explicitly closed off by root; the fix handled
the concrete missing format case.

Implementation (source freeze released, wrapper case added): in
`XgMetaExtendImage`, a wrapper branch (magic 0x466243b1 + version 1)
produces one owned block shaped
[16-byte XGEXTIMG header | 32-byte wrapper-copy slot (24 bytes used) |
patched nested container copy]; the copy's `data` field is retargeted at
the nested copy inside the same block, and `filename_or_fatbins` is
copied as-is, so the driver receives the original wrapper ABI except for
the owned data pointer. One `XgFreeExtendedImage(wrapper_ptr)` frees the
whole block. Wrapper versions other than 1 and nested payloads that are
neither fatbin nor ELF passthrough unchanged; nested cubins are sized by
the same phdr-aware extent rule.

Build-owner run (raw
`workloads/xsched/raw/level2-native-wrapped-meta-20260908.NmlhJN`): wrapped
candidate built and installed, real native_blob run completed but FAILED.
The four BE logs confirm wrapper coverage engages: both extent records
patched (0x8/0x20 -> 0x1520), owned wrapper copy 0x20 + nested container
0x4818 bytes, module load and binary guardian preparation succeed. The
first launch still returns 701 (CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES),
later launches 4, then exit-handler SIGSEGV. No numeric performance is
recorded. Conclusion: the metadata route is reached but is not
sufficient; the open failure is the driver-validated resource/parameter
delivery at launch time (KPARAM-declared param extent vs the 0x1520
relay blob).

Additive patch retained: `level2/native/xsched-native-meta-extend.patch`
regenerated (window_meta_extend.h/.cpp new-file hunks + intercept.cpp
diff), dry-run clean on a pristine shim tree and the applied tree
byte-matches the live source (HDR_MATCH/CPP_MATCH/IO_MATCH). The wrapper
path is compile-checked (g++ -O2 and -O3 -DRELEASE_MODE -Wall -Wextra
-Werror) but was not CPU-unit-tested before deployment.

Deployment note (from the run evidence): the shim/HAL install under
`level2-build/.output/hal-native-20260908.i8na51/install/lib` supplies
`libshimcuda.so` with softlinks `libcuda.so.1` / `libcuda.so` (existing
"File exists" softlink messages on rebuild are nonfatal because the
worker LD_LIBRARY_PATH points at exactly this directory and the shim
supplies the load entry points); the runner needs no further setup beyond
the forwarded variable.

Artifacts: native/check_preempt_embed.cu,
native/restore_exec_embed.cu (compiled + nvdisasm'd from
/tmp/opencode/embedstub, placeholders 0xdf9f0001XXXXXBAD...). Both remain
experimental and unintegrated: no HAL path consumes them.

Removals earlier this session: `xsched-native-window-relay-conform.patch`
(root review: contradictory adoption checks make relay adoption
impossible; kept as a rejected-candidate record only in the conversation
log). Preserved: `xsched-native-window-args-relay.patch` (cf39c397,
failed candidate: 16/16 immediate 701 under relay config),
`xsched-native-launch-error-propagation.patch` (approved; published as
patch commit 2b04ecaf in the diagnosis run lineage).

### 5.3 KPARAM ordinal growth + original-argument marshaling (HX4CV3 causal response)

Causal chain closed by root's driver-entry control (raw
`level2-relay-size-control-20260909.HX4CV3/debug-driver.log` + control
script): on the same installed binary (metadata=1, original entry), a
GDB change of only the relay `CU_LAUNCH_PARAM BUFFER_SIZE` 0x1520 -> 0x20
at the real driver `cuLaunchKernel` entry makes all 200 queued kernels
launch ret 0, the BE complete and exit cleanly (outputs_validated
17408000), while the unmodified relay 701s the first launch (kv7jHZ,
NmlhJN). So the rejected quantity is the relay buffer/declared-extent
relation, not entry redirection and not the CBANK patch. Conclusion:
the driver checks the launch parameter buffer against the KPARAM-declared
ordinal extent (0x20), which the CBANK-only patch did not raise.

Compiler specimen (nvcc 12.9 sm_120, /tmp/kpspec, kernel with a
0x1520-byte by-value parameter): large-parameter kernels carry
EIATTR_KPARAM_INFO_V2 (id 0x45, fmt 4, paylen 12: u32 index, u16 ordinal,
u16 offset, u16 plain_size, u16 attrs; attrs 0x0500 pointer / 0x0000
scalar), small kernels of the same compilation keep legacy
EIATTR_KPARAM_INFO (id 0x17: attrs 0xf500/0xf000, u16 size field ==
size*4+1 on 5 observed records). The compiler sets CBANK_PARAM_SIZE ==
PARAM_CBANK.psize == max ordinal end. The in-place rewrite therefore
converts each legacy record to the V2 shape (same 16-byte slot, attrs
high nibble cleared) and grows the max-end ordinal to 0x1520
blob-relative (timer ord0 0x8 -> 0x1520; compute ord4 0x8 -> 0x1508 at
offset 0x18; both fit the u16 plain-size field).

Original-argument marshaling (HAL): after growth the loaded param table
reports the grown size per ordinal, so the constructor deep copy and the
relay blob fill would copy ~5 KiB from the app's own 8-byte argument
pointer before any relay adoption. The extender now registers each grown
kernel's ORIGINAL per-ordinal layout (ordinal-indexed, density-checked,
keyed by the mangled name the section table carries, i.e. the string
cuFuncGetName reports) and the HAL (cuda_command.cpp) marshals:

  - constructor deep copy, kernelParams form: buffer size and per-ordinal
    memcpy from the original layout;
  - constructor deep copy, extra-buffer form: param pointer placement and
    the buffer-range assert from the original layout;
  - window-relay adoption: rel_param_end, blob fill and asserts from the
    original layout (blob still padded to 0x1520 via the window branch).

Plain non-relay launches marshal only original sizes. The lookup is
dlsym("XgGetRelayOriginalParams") on the already-loaded shim (no hard
link); without the shim or the knob the HAL falls back to loaded param
info, and a registry/loaded disagreement falls back with a warning
(count or last-offset cross-check). Registry is per-image memory, bounded
64 kernels x 32 params, no eviction claim.

Open dimensions for the build owner's run (not neutralized by this
wiring): (1) plain kernelParams launches of grown kernels (e.g. the
timer kernel outside the relay corridor) let the driver copy per-ordinal
at the grown extent from wherever the pointers land - CBANK growth was
already tolerated in NmlhJN, KPARAM growth of plain launches is the new
untested dimension; (2) whether the driver's acceptance additionally
depends on the .nv.constant0 section extent (compiler invariant
constant0 size == param_start + max ordinal end; ours stay 0x3a0) - if
the 701 persists after this repair, growing constant0 (section byte
insertion + shdr/phdr relocations) is the deferred next knob.

Verification this session (CPU only): both compile modes pass for the
shim file and the HAL file (g++ -O2 and -O3 -DRELEASE_MODE -DNDEBUG
-Wall -Wextra -Werror); local specimen test: both knobs patch the same
six extent bytes as before plus the V2 conversion/growth and the
registered layouts answer ordinal-dense {ordinal, offset, original size}
for compute (5 params) and timer (1 param); EXTEND-only keeps records
byte-identical (6-byte diff) and the query returns 0 (HAL fallback);
wrapper/fatbin route answers the same registry through the nested
container. Patch `xsched-native-meta-extend.patch` regenerated (h + c +
i + NEW cuda_command.cpp sections, 1437 lines), applies clean on a
pristine tree, applied bytes = live source for all four files, reverse
application restores pristine.

Delivery state: source READY for root build + real XSched native run under
worker env XG_NATIVE_META_EXTEND=1 XG_NATIVE_META_KPARAM=1 (+ the
corridor's existing relay/level2 flags). Build must re-run the cmake
configure step before --build --target install (GLOB_RECURSE note
above).

### 5.4 Non-relay driver-side overread closure (root follow-up before the run)

Defect: a deep copy with original sizes still leaves the ordinary
(non-relay) launch reading the GROWN KPARAM extent from an original-sized
allocation. In the deep-copy constructor the driver copies per-ordinal at
the loaded (grown) sizes from params_[i] buffers owned by the command:
the timer kernel's original 8-byte ordinal 0 would be read at the grown
0x1520 extent, ~5 KiB past the original-end malloc. A plain passthrough
launch is worse: the app's raw pointers go to the driver unchanged.

Closure 1 - command-owned params (cuda_command.cpp ctor): the params
buffer is now allocated at the LOADED (grown) extent (equal to the
original extent when nothing is registered) and zero-filled via calloc;
per-ordinal copies from the caller stay limited to the ORIGINAL sizes
(marshal truth = original layout, alloc truth = loaded layout). The
grown extent then reads back as zeroes in the pad: no out-of-bounds
driver-side read, deterministic upload content. The extra-buffer form is
unchanged (the driver copies the caller-declared BUFFER_SIZE bytes from
extra_data_, so there is no extent mismatch on the read side).

Closure 2 - plain passthrough corridor (shim.cpp): root's smaller option
"preserve original KPARAM metadata" is not implementable at image-patch
time - the extender cannot know which kernels will ride the relay and
which launch plainly, so the timer cannot be excluded from growth
there. The implemented closure is the other option, pad at the actual
corridor: XLaunchKernel{,_ptsz} and XLaunchKernelEx{,_ptsz} now detect a
grown kernel (registry name lookup, XgKernelGrown) and construct an
owned deep copy instead of passing the app's raw pointers, for both the
CHECK_STREAM fallback direct launch and the crafted command
(XLaunchKernelImpl gained a force_deep_copy argument). Non-grown kernels
keep the zero-cost passthrough; managed-stream launches were already
deep-copy and are covered by closure 1. Relay/instrumented launches are
unaffected (relay blob adoption under the launch mutex).

Bypassing the ctor entirely (true raw passthrough) now exists only where
no shim launch command is constructed at all: XLaunchKernelEx with a
null config (the driver rejects the null config before any parameter
copy) and legacy direct Driver:: calls outside the shim's launch
entries. Neither reaches a parameter copy of grown extents inside our
code.

Compile-checked after the changes: window_meta_extend.cpp,
cuda_command.cpp and shim.cpp all pass g++ -O2 and
-O3 -DRELEASE_MODE -DNDEBUG -Wall -Wextra -Werror in the isolated tree.
No new tests or gates were added.
