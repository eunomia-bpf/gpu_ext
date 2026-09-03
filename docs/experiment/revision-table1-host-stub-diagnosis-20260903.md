# Table 1 launchlate: host-stub diagnosis (2026-09-03)

Scope: read-only source/ELF inspection on CPU 17. No new GPU run, build,
runtime modification, or claim that launchlate now passes. The closed
`revision-rq4/raw/preflight-575-03` result remains invalid. Addresses below
are ELF-relative locations, not assumed process virtual addresses.

## Most useful finding

The selected host symbol in
`workloads/llama.cpp/build-ptx-1b/bin/libggml-cuda.so.0.9.4` is
`_Z9rope_normILb1ELb0Ef6__halfEvPKT1_PT2_iiiiiPKifff14rope_corr_dimsfPKfPKli`,
at `0x221730`. The actual `ggml_cuda_op_rope_impl<true>` call site prepares
that address as the **first argument** to `cudaLaunchKernel` at `0x227edb`,
then calls the CUDART launch entry at `0x227f27`. It does not call the selected
stub at that site. The stub also exists as a standalone launch wrapper.

The inspected disassembly's references to `0x221730` are CUDA registration
(`0x40b22`), the wrapper itself (`0x221913`), and the caller's launch argument
(`0x227edb`); no direct call/jump to it was found. This supports an inlined
launch-wrapper/bypassed-hook explanation. **Indirect calls have not been
excluded, and static inspection is not a runtime callback count.** Resolving
a real symbol and successfully attaching to it do not prove it executes.

## Three falsifiable checks, in order

1. In one untimed request, independently count the selected stub's entrance
   and actual `cudaLaunchKernel` entries whose first argument is that target
   host pointer. CUDART matches greater than zero with zero stub hits support
   bypass. Require per-link attach success before interpreting zero hits.
   Do not turn a count-only diagnostic into performance evidence.
2. Check installation and timing separately. Private runtime
   `runtime/src/attach/bpf_attach_ctx.cpp:106–130,390–397` continues past
   individual link failures, so `agent.cpp:999`'s `Attach successfully` is
   not a per-link guarantee. Relevant messages are `Failed to find module
   base address`, `Unable to attach: module name`, and Frida attach errors.
   Some failures are DEBUG-only. `frida_attach_utils.cpp:38–76` actively
   calls `gum_module_load`; a later application load alone is not proof of
   a missing module. LD_PRELOAD initialization supplies empty data
   (`agent.cpp:452`), disabling periodic refresh (`:483–486`).
3. Only if an independent host-entry counter is positive, distinguish map
   lookup failure from a queue update/visibility problem. The prepared
   `launchlate.bpf.c:65–68` returns before incrementing host count when its
   first queue lookup fails. GPU code at `:115–125` increments underflow
   before any successful dequeue. The actual old values are **0 host,
   0 successful dequeues, 220 underflows**; `launchlate.c:203–205` labels the
   dequeue count `Device entries`. They are not 220 successful device queue
   entries, and host count zero alone does not prove no callback ran.

CUDA graph replay is a secondary check: the runner sets
`GGML_CUDA_DISABLE_GRAPHS=1` (`run_revision_rq4.py:639`), and the selected
llama source checks that variable's presence before allowing graphs
(`ggml/src/ggml-cuda/ggml-cuda.cu:3618,3640`). A diagnostic can count
`cudaGraphLaunch` separately; do not assume replay caused this failure.

## Proposed two-probe pointer capture: feasible pieces, timing blocker

The real CUDA 12.9 declaration in
`/usr/local/cuda-12.9/targets/x86_64-linux/include/crt/host_runtime.h:242`
has zero-based argument 1 `hostFun`, and argument 3 `deviceName`. The current
CUDA runtime listener uses exactly these arguments
(`attach/nv_attach_impl/nv_attach_impl_frida_setup.cpp:325–331`). Thus a host
registration probe can match the full device name, save the actual host
pointer, and a launch probe can enqueue a timestamp only when argument 0
equals that pointer, without guessing an ASLR base.

Existing host-JIT helpers include `bpf_get_func_arg` (183,
`frida_uprobe_attach_impl.cpp:278–285`), `bpf_probe_read_user`, and
`bpf_strncmp` (`runtime/src/bpf_helper.cpp:1398–1403,1451–1456`). Bounded
name comparison must check helper errors. The current `probe_read_user_str`
implementation returns zero, not the kernel helper's string length
(`bpf_helper.cpp:612–617`), so do not depend on kernel return semantics.

However, **adding only these two ordinary BPF probes is not yet a complete
fix for this executable**. Its `libggml.so.0` has a direct `DT_NEEDED` on
`libggml-cuda.so.0`; CUDA registrations occur during shared-library
initialization before the application main wrapper installs ordinary BPF
links (`agent.cpp:448–453,894–923`). The earlier exported
`__cudaRegisterFatBinary` wrapper (`:718–737`) initializes logging/LLVM and
forwards the call; it does not install these links or retain
`__cudaRegisterFunction` events. The late-bootstrap path scans already
loaded fatbins (`nv_attach_impl.cpp:1798–1956`); the inspected path does not
replay past registrations through user host probes.

Consequently, first require a target registration hit before accepting the
two-probe route. Missing early events require early capture plus safe
replay/seeding, or a launch-side exact-symbol lookup using the existing
`nv_attach_impl::resolve_host_function_symbol` (`:1619`), which already uses
loaded ELF symbols/real addresses. The latter is a runtime facility, not an
already exposed BPF helper. Neither option was implemented or tested here.
Also verify composition with the existing `cudaLaunchKernel` replacement
(`nv_attach_impl.cpp:421–427`): the new listener must see the real incoming
host pointer once, without counting a replacement/trampoline twice. Other
launch APIs and graph replay are not covered merely by these two probes.

## Repeatable read-only checks

Run from the repository root; these print selected metadata/source or
disassembly only and do not execute CUDA code. Do not persist binary dumps.

```sh
taskset -c 17 nm -an workloads/llama.cpp/build-ptx-1b/bin/libggml-cuda.so.0.9.4 | rg '_Z9rope_normILb1ELb0Ef6__halfEv'
taskset -c 17 objdump -d --no-show-raw-insn --start-address=0x227e80 --stop-address=0x227f50 workloads/llama.cpp/build-ptx-1b/bin/libggml-cuda.so.0.9.4
taskset -c 17 objdump -d --no-show-raw-insn workloads/llama.cpp/build-ptx-1b/bin/libggml-cuda.so.0.9.4 | rg '221730'
taskset -c 17 readelf -d workloads/llama.cpp/build-ptx-1b/bin/libggml.so.0 | rg 'NEEDED|RUNPATH'
taskset -c 17 sed -n '448,453p;483,487p;718,737p;894,925p' ../bpftime-table1-575/runtime/agent/agent.cpp
taskset -c 17 sed -n '309,351p' ../bpftime-table1-575/attach/nv_attach_impl/nv_attach_impl_frida_setup.cpp
```

The two-probe suggestion is diagnostic planning, not evidence of corrected
launch latency, formal Table 1 completion, or GPU-verifier acceptance.
