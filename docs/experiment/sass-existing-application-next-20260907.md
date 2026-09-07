# SASS-only existing-application BPF: next step after standalone AOT

Date: 2026-09-07. Bounded read-only source investigation. No builds, no GPU
runs, no downloads. Only this file was written.

Question: smallest REAL path to execute a verified BPF kernel-return
(kretprobe-style) program when the target application ships ONLY sm_120
cubin (SASS, no PTX), reusing existing instrumentation/injection machinery.
Not a standalone-cubin substitute, not a compile-only test, not a proxy hook.

Implementation follow-up: the root has assigned local OpenCode Qwen 27B the
NVBit tool, target and build-time embedding in `workloads/sass-kretprobe/`,
and local GLM the separate `bpf_to_ptx.cpp`, `compiler.mk` and compiler notes.
Both reuse the existing Table1 bpftime build and NVBit release. These are
active implementation tasks, not completed application-injection evidence.
The root will review the build and run the actual target; no new GPU matrix,
standalone-42 rerun, or additional verification campaign is requested.

Current build checkpoint (2026-09-07): the real BPF input compiles with
clang-15 and is published in main `596fefcc`. After fixing two ordinary CUDA
type/API compilation errors, the root successfully compiled the NVBit host
tool object and the device wrapper's PTX. The initial device-object command
`nvcc -dc ... -Xptxas -astoolspatch` fails because ptxas prohibits combining
`--compile-only` and `--compile-as-tools-patch`. Qwen is therefore changing
the build glue to merge the wrapper and compiler-generated BPF functions in
one PTX translation unit, then assemble and embed it without separable
tool-patch compilation. The exporter is now built and published in main
`08e1692e`: the actual four-instruction ELF section produces a 316-byte
two-parameter `.visible .func bpf_exit`, and the integrated Makefile export
command succeeds. The function remains device-callable, not a host-launched
`.entry`. PREVAIL rejection is enforced, while its internal `strict` option
remains false; the exporter does not change verifier options. Qwen's merged
PTX now assembles with `ptxas -arch=sm_120 -astoolspatch`, and the cubin
contains both `bpf_exit` and `sk_bpf_trampoline`. Tool embedding and the
existing application's live EXIT execution remain unfinished. GLM's next
bounded task assists with CPU-only embedding, without editing Qwen's files.
These are build steps only; no completed in-body SASS execution or performance result is
claimed, and no old GPU experiment is being repeated.

## Established facts (evidence)

### 1. The bpftime attach path is PTX-only; SASS-only fatbins produce no
### instrumentation

Table1 runtime `/home/yunwei37/workspace/gpu/bpftime-table1-hostfix-plt`
(branch `revision/table1-host-plt-fix` @ `89a1244`):

- Late attach scans each loaded module's `.nv_fatbin` / `.nvFatBinSegment`
  (`nv_attach_impl::bootstrap_existing_fatbins`, `attach/nv_attach_impl/
  nv_attach_impl.cpp:~1977-2130`), reads fatbin bytes, and runs
  `extract_ptxs(bytes)` which collects only `.ptx` entries. A fatbin that
  contains only an sm_120 ELF returns an empty map ("Got 0 PTX files").
- `fatbin_record::try_loading_ptxs` (`nv_attach_fatbin_record.cpp:241`) then
  runs `impl.hack_fatbin(original_ptx)` (`nv_attach_impl.cpp:1061`): for each
  PTX string, each hooked kernel is patched by the ptxpass executable, which
  consumes `runtime_request.input.full_ptx` (`ptxpass_kretprobe/main.cpp`
  `process_input`). No PTX string => no patch request => no recompiled
  module.
- Launch replacement works only if a replacement `CUfunction` exists:
  Frida-embedded interceptor hooks `cudaLaunchKernel(_ptsz)` and
  `cuLaunchKernel` (`nv_attach_impl_frida_setup.cpp:524-690`), resolves the
  host stub symbol to the device kernel name (`resolve_host_function_symbol`,
  `nv_attach_impl.cpp:1787-1833`), and calls
  `find_patched_kernel_function`; `prefill_patched_kernel_functions_from_
  loaded_fatbins` (`nv_attach_impl.cpp:1835`) fills that map from modules
  compiled from patched PTX. With a SASS-only target all maps are empty, the
  original kernel launches uninstrumented, and nothing fails loudly.

Conclusion: the missing interface on this path is *a source of target PTX*,
which does not exist for a PTX-free binary; ptxpass cannot consume SASS.
Patching the target's own SASS in place (adding a call before `EXIT` in the
target cubin, re-encoding relocations, reloading) is exactly the
"NVBit-patches-an-existing-SASS-binary" boundary that
`docs/experiment/revision-sass-aot-readiness-20260904.md` (Claim boundary)
explicitly does not claim.

### 2. The sass_aot branch proves the BPF-to-SASS half, in `.entry` form only

`revision/sass-backend` @ `fd976ea`
(`attach/nv_attach_impl/sass_aot/`): real clang BPF ELF section
`cuda__/sass_aot` -> strict GPU verifier (PREVAIL 8-byte context + SIMT) ->
`ptxpass::compile_ebpf_to_ptx_from_words` -> `ptxas -arch=sm_120` -> Driver
API load/launch/sync/DtoH. Verified live run printed `verified SASS result:
42` on RTX 5090, driver 575.57.08.

Key detail for reuse: `compile_ebpf_to_ptx_from_words` emits the BPF entry
as `.visible .func bpf_main` — a device-callable function. The standalone
AOT executable only upgrades it to `.visible .entry bpf_main` via
`patch_main_from_func_to_entry` (`nv_attach_utils.cpp:68-80`, also
`vm/llvm-jit/example/ptx/ptx_test.cpp:399`) so it can be launched as a
kernel. The kretprobe pass (`ptxpass_kretprobe/main.cpp:33-63`) instead
prepends that same `.func` form (`__retprobe_func__<kernel>`) to the target
kernel's PTX and inserts `call __retprobe_func__<kernel>;` before the
kernel's last `ret;`/`exit;` (predicate- and label-preserving regex).
So "kernel-return BPF" already has an exact code form in this codebase:
a no-argument PTX `.func` executed immediately before the target kernel
exits.

What the branch does NOT have: any API that hands out the device-function
SASS artifact (cubin bytes + entry symbol) for injection into a binary that
is not ours, and no SASS-side target to patch.

### 3. The pinned local NVBit release is a SASS-level DIB that supports the
### target class

Table1 NVBit: `workloads/llama.cpp/observability_overhead/revision-rq4/deps/
nvbit_release_x86_64` (the release the local adapters build against; the
older home-level release at `~/nvbit_release_x86_64` is NOT the one used).

From its `README.md` and `core/nvbit.h`:

- Dynamic binary instrumentation of **SASS**: "inspect and modify the
  assembly code (SASS) of a GPU application without requiring
  recompilation"; "inject one or more instrumentation calls to arbitrary
  device functions before (or after) a SASS instruction"; can read/write
  registers; works for `__global__` and `__device__` functions.
- Requirements: SM `>= 3.5 && <= 12.1` (sm_120 in range; the library ships a
  `gb12x_hal` object for the SM 12.x family), CUDA >= 12.0, driver
  `<= 575.xx` (machine driver is 575.57.08, in range), x86_64 Linux.
- Loaded via `LD_PRELOAD` or `CUDA_INJECTION64_PATH`; uses the same
  mechanism as nvprof/nsight, so those cannot be used together.
- Public API (`core/nvbit.h`):
  - lifecycle/inspection: `nvbit_at_init`, `nvbit_at_ctx_init`,
    `nvbit_tool_init`, `nvbit_at_cuda_event(ctx, is_exit, cbid, ...)`
    (cbids include `API_CUDA_cuLaunchKernel`, `..._Ex`, `_ptsz`),
    `nvbit_at_term`; `nvbit_get_instrs`, `nvbit_get_func_name`
    (mangled/demangled), `nvbit_is_func_kernel`, `nvbit_get_CFG`.
  - injection: `nvbit_insert_call(instr, dev_func_name,
    IPOINT_BEFORE|IPOINT_AFTER)` — "calls to device functions are
    identified by name ... declare the function as `extern "C" __device__
    __noinline__`"; argument pushers `nvbit_add_call_arg_const_val64`,
    `nvbit_add_call_arg_reg_val`, `nvbit_add_call_arg_ureg_val`,
    `nvbit_add_call_arg_launch_val32/64` (with `nvbit_set_at_launch`),
    predicate variants, `nvbit_add_call_arg_mref_addr64`,
    `nvbit_add_call_arg_cbank_val`; `nvbit_remove_orig`.
  - device helpers: `nvbit_read_reg`/`nvbit_write_reg`/`nvbit_read_pred_reg`
    and uniform/predicate variants (`__CUDACC__` only).
  - tool modules: `nvbit_load_tool_module(ctx, moduleBinary,
    &loadedModule)` ("Load a tool module into a program context"),
    `nvbit_find_function_by_name(ctx, mod, name, &func)` ("Load a tool
    kernel from a given tool module (to avoid lazy loading)"),
    `nvbit_launch_kernel(...)`.

### 4. The local gpu_ext NVBit integration already does EXIT-point
### (kernel-return) instrumentation of sm_120 SASS

`revision-rq4/nvbit_adapters/observability/` (prebuilt `observability.so`
present) is built with `ARCH=sm_120`,
`NVBIT_ROOT=$PWD/deps/nvbit_release_x86_64` (see
`nvbit_adapters/README.md`, "official NVBit 1.8 core"). Its `kernelretsnoop`
mode: in `nvbit_at_cuda_event` for launch cbids, matches the exact mangled
`OBS_TARGET_SYMBOL`, walks `nvbit_get_instrs`, and for every `EXIT`
instruction calls `nvbit_insert_call(instr, "observe_exit", IPOINT_BEFORE)`
with execution-predicate checks (see `observability.cu:395-465` and
`nvbit-exit-predicate-repair.md`), emits one record per actually-exiting
logical thread through NVBit's device-to-host channel, and validates on the
host. Runner pattern: `LD_PRELOAD=...so OBS_MODE=... OBS_TARGET_SYMBOL=<mang
led> ./app`.

Also demonstrated in-tree: embedding a separately compiled device binary
into the tool via `bin2c` (`tool_func/%.fatbin -> %.c` in the adapter
Makefile) and consuming it at runtime with `nvbit_load_tool_module` +
`nvbit_find_function_by_name` + `nvbit_launch_kernel`
(`observability.cu:540-545,665`; same pattern in the shipped
`tools/mem_trace/mem_trace.cu:318,524-553` and
`tools/record_reg_vals/record_reg_vals.cu:217,368-391`). The embedded
`flush_channel` in those examples is a `__global__` kernel launched with
`nvbit_launch_kernel`, not an `nvbit_insert_call` target.

## Candidate integration route (smallest real path)

Use the NVBit route, not the bpftime-Frida route. The NVBit tool is the
injection machinery; bpftime supplies only the verified BPF artifact.

1. **Verified BPF artifact (bpftime side, new small API).** Add
   `compile_ebpf_to_sass_device_func(words, out_cubin_bytes,
   out_entry_name, opts)` to the sass_aot area: existing strict GPU
   verifier -> `ptxpass::compile_ebpf_to_ptx_from_words(words, sm, name,
   add_register_guard, with_arguments)` (already emits `.visible .func`)
   -> `ptxas -arch=sm_120` -> raw cubin bytes + entry symbol. Do NOT call
   `patch_main_from_func_to_entry`: the entry must stay a device-callable
   `.func`. Reuses `attach/nv_attach_impl/sass_aot/sass_aot.cpp`
   (fd976ea) structure: ptxas invocation, context size, cuobjdump helpers.
   For context plumbing prefer `with_arguments=true` and pass the context
   pointer at injection time (item 4 below); if that ABI does not match
   NVBit's argument convention, fall back to the kretprobe no-arg form
   (`with_arguments=false`, as `ptxpass_kretprobe/main.cpp:55` already
   does) plus a fixed device global.
2. **NVBit tool (new `.cu`, skeleton copyable from
   `nvbit_adapters/observability`).** In `nvbit_at_cuda_event` for
   `API_CUDA_cuLaunchKernel*` cbids: match the target's mangled name with
   `nvbit_get_func_name(ctx, func, true)`; find each `EXIT` instruction
   (with the predicate treatment from `nvbit-exit-predicate-repair.md`);
   `nvbit_insert_call(exit_instr, <bpf_entry_symbol>, IPOINT_BEFORE)`;
   push the context pointer with `nvbit_add_call_arg_const_val64` (or
   `nvbit_add_call_arg_launch_val64` + `nvbit_set_at_launch`). Host side
   validates through NVBit's channel (or a device buffer read after
   `cudaDeviceSynchronize`) that the BPF program wrote its verified value
   (e.g. 42) at the target kernel's exit, once per exiting thread.
3. **Registering the BPF SASS symbol for `nvbit_insert_call` — two
   candidates, one is a known API, one is unresolved:**
   - **Glue A (based on the shipped tool-module pattern):** embed the BPF SASS into the
     tool's own device-linked fatbin at tool build time (the tool Makefile
     pattern already compiles `inject_funcs.cu` with
     `-Xptxas -astoolspatch --keep-device-functions`; adding the ptxas'd
     BPF cubin/PTX to the tool device-link, via `fatbinary`/`nvcc -dlink`,
     keeps its symbol in the embedded module). All shipped
     `nvbit_insert_call` examples (mov_replace, instr_count, the local
     adapters) resolve names from the tool's own module, so this is the
     conservative, demonstrated scope. Caveat: merging an externally
     ptxas-compiled cubin into the tool's device link is a build-technique
     inference, not demonstrated in-tree; the build spike must confirm the
     symbol survives in the embedded fatbin and resolves by name.
   - **Glue B (public API, resolution scope UNRESOLVED):** load the BPF
     cubin at runtime with `nvbit_load_tool_module` +
     `nvbit_find_function_by_name`, then `nvbit_insert_call` by the same
     name. `nvbit_find_function_by_name`'s doc ("to avoid lazy loading")
     implies tool-module functions are name-indexed by the core, but no
     shipped or local example calls `nvbit_insert_call` on a function from
     a `nvbit_load_tool_module` module, and the `nvbit_insert_call` doc
     does not state which module its name lookup covers (all examples use
     `__device__` functions from the tool's own compilation; local
     tool-module examples use `__global__` functions via
     `nvbit_launch_kernel` only). Do NOT promise Glue B works; a ~30-line
     probe tool (load a trivial `.func` cubin, `insert_call` it at an EXIT,
     check the marker) settles it with one build.
4. **Context/argument plumbing.** The verified kretprobe BPF program is
   no-arg in the PTX form today; the standalone AOT form gets its 8-byte
   context by launch argument. For NVBit injection the context device
   pointer must be supplied per call via the `nvbit_add_call_arg_*` APIs.
   Matching the exact register/predicate conventions between ptxpass's
   `with_arguments` ABI and NVBit's injected-call ABI is a dev-task
   verification item, not a design assumption.

Why this is the smallest real path: every other element already exists and
is locally demonstrated on this exact machine/target class (NVBit SASS DIB
on sm_120 with driver 575.57.08; EXIT-point kernel-return injection;
verified eBPF->PTX->sm_120 SASS pipeline; bin2c tool-module embedding). The
bpftime-Frida alternative would require building a SASS rewriter for the
target's own cubin (decode sm_120 SASS, insert a call, patch relocations and
ELF, reload the module) — strictly more new, unverified machinery than the
NVBit route, which NVIDIA already ships for this purpose.

## End-to-end workload command (to be run by the dev task; NOT run here)

- Target application: a small CUDA app built SASS-only, e.g. the NVBit
  `test-apps/vectoradd` (or a gpu_ext micro-bench) with
  `nvcc -gencode arch=compute_120,code=sm_120 ...` so the fatbin holds only
  an sm_120 ELF; confirm no PTX with `cuobjdump` (no `.ptx`/ELF type 5
  entry) before the run.
- BPF program: the existing spike program (`SEC("cuda__/sass_aot")`,
  8-byte PREVAIL context, writes 42) compiled via
  `compile_ebpf_to_sass_device_func` after passing the strict GPU verifier.
- Run (shape of the existing adapter runner):
  `LD_PRELOAD=./nvbit_bpf_kretprobe.so NVBIT_BPF_CUBIN=/path/bpf.cubin \
  NVBIT_BPF_ENTRY=<symbol> OBS_TARGET_SYMBOL=<mangled target kernel> \
  ./vectoradd`
- Success criteria: (a) NVBit banner and the tool's per-kernel
  instrumentation line name the target SASS-only kernel; (b) original
  application output is unchanged (vectoradd final sum correct); (c) the
  BPF marker (42) is observed in the host-validated record stream at the
  target kernel's EXIT, with per-exiting-thread multiplicity matching the
  kernel launch, proving the BPF code executed inside the target kernel's
  SASS, not in a standalone launch.

What it would prove: a strict-verifier-verified BPF kernel-return program
instrumented into and executed on the exit path of a real PTX-free sm_120
application — closing the `revision-sass-aot-readiness-20260904.md` boundary
("does not insert the generated code into an existing PTX-free application
binary ... or use NVBit to patch an existing SASS binary"). It would NOT
prove: coexistence of the NVBit tool with the bpftime Frida/PLT runtime
in one process, overhead numbers, graph-launch paths, or generality across
kernels/applications.

## Unresolved items (explicitly labeled)

1. `nvbit_insert_call` name-resolution scope across `nvbit_load_tool_module`
   modules: not documented, not demonstrated by any shipped or local example.
   Evidence leans toward a name index (the find-by-name API's stated
   purpose), but Glue B must be treated as unverified until the probe tool
   passes. Glue A is the supported fallback.
2. Whether an externally ptxas-compiled cubin merged into the tool's
   device-linked fatbin keeps its `.func` symbol visible to
   `nvbit_insert_call` (Glue A build-technique inference).
3. ABI match between ptxpass `with_arguments` PTX and NVBit's
   `nvbit_add_call_arg_*` injected-call convention (or use the no-arg
   kretprobe form).
4. Coexistence of NVBit (CUDA_INJECTION64_PATH/LD_PRELOAD) with bpftime's
   Frida-embedded driver hooks + host PLT patching in one target process:
   unknown; the dev task can validate standalone NVBit first and treat
   coexistence as a separate experiment.
5. NVBit's documented save/restore cost per injected call applies on every
   EXIT hit; overhead is a follow-up measurement, not a gate for this step.

## Actionable development task (single, bounded)

"NVBit BPF kretprobe tool for SASS-only sm_120 targets":

1. Reuse `sass_aot.cpp` (fd976ea) and existing ptxpass/verifier libraries
   for a small ELF-to-device-function exporter, keeping `.func` rather
   than promoting it to `.entry`. Integrate that compiler output into the
   tool build; a new bpftime public API or standalone launch is not required
   to establish the first application-injection path.
2. Probe tool (~30 lines, in the adapter tree): `nvbit_load_tool_module` +
   `nvbit_find_function_by_name` + `nvbit_insert_call` of a trivial
   `.func` cubin at one EXIT; settle unresolved item 1 (choose Glue B or A).
3. Real tool: target-symbol match, EXIT injection with predicate handling
   (copy `nvbit_adapters/observability` + `nvbit-exit-predicate-repair.md`),
   BPF entry symbol per the chosen glue, context pointer via
   `nvbit_add_call_arg_*`, host-side record validation.
4. End-to-end run on the SASS-only target with the criteria above.

Reusable files (read-only references):
- `revision-rq4/deps/nvbit_release_x86_64/core/{nvbit.h,libnvbit.a}` — pinned
  NVBit 1.8 (public API per header; no proprietary internals needed).
- `revision-rq4/nvbit_adapters/observability/{Makefile,observability.cu,
  inject_funcs.cu,common.h,tool_func/}` — sm_120 EXIT-injection tool
  skeleton, bin2c tool-module embedding, runner env-var pattern.
- `bpftime` branch `revision/sass-backend` @ fd976ea:
  `attach/nv_attach_impl/sass_aot/{sass_aot.cpp,sass_aot.hpp,live_sass_aot.
  cpp,sass_aot_spike.bpf.c}` — verifier->ptxpass->ptxas->Driver-API pipeline
  and the verified 42-program.
- `bpftime` (both trees): `attach/nv_attach_impl/pass/ptxpass_kretprobe/
  main.cpp` — kernel-return semantics (call before last `ret;`/`exit;`,
  no-arg `.func` form); `attach/nv_attach_impl/pass/ptxpass_core/`
  (`compile_ebpf_to_ptx_from_words`, `.visible .func bpf_main` emission);
  `attach/nv_attach_impl/nv_attach_utils.cpp:68` (`patch_main_from_func_to_
  entry`, to be avoided on this path).
- `revision-rq4/nvbit-exit-predicate-repair.md` — predicated-EXIT handling.

Out of scope / not touched: Fig13 code and runtime owned by the two other
local sessions; any bpftime Frida-runtime modification; standalone AOT
demo; compile-only or proxy-hook evidence.
