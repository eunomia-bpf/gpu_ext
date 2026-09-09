# XSched Level-2 component build (bounded)

Current status (2026-09-09): the NVBit-actuator policy port has
[15 completed measurements](../raw/level2-device-policy-pair-20260909.buBjns/README.md).
The [canonical host-tool rebuild](../raw/canonical-tool-build-20260909.MnNeOe/README.md)
completed using this Makefile and the existing measured device carrier.
That isolated rebuild is not a fresh build of the entire device pipeline or
a new timing run. The original cuXtra path remains unfinished. Start with the
[source status](../level2/README.md) and [artifact guide](../../../ARTIFACT.md)
before following the original component notes below.

## Historical source-preparation notes

Status: **SOURCE PREPARATION / UNBUILT.** The root session has not built this
Makefile yet. Nothing in `.output/` exists until it does; no output, artifact,
or benchmark number is claimed or fabricated in this directory, and the
paper-scale performance protocol is deliberately out of scope for this build
task.

Scope: component build only. Sources are read from the sibling GLM-owned tree
`../level2` (`workloads/xsched/level2/`) and are never modified. All outputs
land in `.output/` inside this directory. The default `make` builds components
only and never runs GPU work, reloads a driver, loads the NVBit tool, or
executes any experiment.

## Component targets (explicit)

- `make bpf-ptx` — real BPF ELF to device-callable PTX of `xsched_guardian`:
  - `clang -g -O2 -target bpf -I../level2/tramp -c ../level2/bpf/xsched_guardian.bpf.c -o .output/xsched_guardian.bpf.o`
  - `bpf_to_ptx .output/xsched_guardian.bpf.o cuda__/xsched_guardian .output/ptx/xsched_guardian.ptx xsched_guardian sm_120`
  - Exporter CLI derived from the local source
    (`workloads/sass-kretprobe/bpf_to_ptx.cpp`): `OBJ SECTION OUT [SYMBOL] [SM]`.
    Resolution: copy the frozen exporter source into `.output/adapter/`,
    apply `../level2/bpf/bpf_to_ptx_ctx48.patch`, then build through its
    `compiler.mk` with local `PTX_EXPORTER_SRC`, `BUILD`, and `BIN` overrides.
    The original 8-byte exporter binary is not reused: this guardian needs
    the 48-byte scalar context. Outputs stay under `.output/`.
  - Exported ABI: `.visible .func xsched_guardian(.param .b64 context_ptr,
    .param .b64 context_length)`; the trampoline calls it as
    `xsched_guardian(ctx, sizeof(ctx))` and consumes the decision from the
    context write-back (`ctx->decision`), per
    `../level2/xsched_guardian_abi.h` and
    `../level2/tramp/xsched_guard_tramp.cu`.
- `make guardian-cubin` — shared trusted trampoline PTX merged with the BPF
  PTX, assembled for sm_120:
  - `nvcc -ptx -rdc=true -arch=sm_120 -O3 -std=c++14 --keep-device-functions ../level2/tramp/xsched_guard_tramp.cu -o .output/tramp/xsched_guard_tramp.ptx`
  - `bash ../level2/tool/merge_ptx.sh .output/ptx/xsched_guardian.ptx .output/tramp/xsched_guard_tramp.ptx .output/merged/xg_guardian.ptx`
    (existing GLM-owned script; one header, the `.extern .func` declaration
    removed so the call resolves inside the module)
  - `ptxas -arch=sm_120 -astoolspatch .output/merged/xg_guardian.ptx -o .output/xg_guardian.cubin`
  - Actuator semantics: `xg_tramp` compiles BOTH the native-C and the BPF
    decision arm into the same module; the on-device `decision_mode` argument
    (0 = native C, 1 = device eBPF) selects between them with the SAME
    actuator glue and no host decision fallback. The host tool only fixes the
    insertion-time constant (`XG_DECISION=bpf` in
    `../level2/tool/xsched_guard_tool.cu`).
- `make guard-tool` — NVBit xg guard shared library, linked exactly like the
  working `workloads/sass-kretprobe` example (tool object + embedded-fatbin
  carrier object, static NVBit):
  - `fatbinary -64 --embedded-fatbin .output/xg_guardian.fatbin.c --image3=kind=elf,sm=120,file=.output/xg_guardian.cubin`
  - `g++ -x c++ -fPIC -O2 -I<CUDA_INC> -c .output/xg_guardian.fatbin.c -o .output/xg_guardian_carrier.o`
  - `nvcc -c -arch=sm_120 -O3 -std=c++14 -Xcompiler -fPIC -I<NVBIT_ROOT>/core ../level2/tool/xsched_guard_tool.cu -o .output/xsched_guard_tool.o`
  - `nvcc -arch=sm_120 -O3 .output/xsched_guard_tool.o .output/xg_guardian_carrier.o -L<NVBIT_ROOT>/core -lnvbit -L<CUDA_LIBDIR> -lcuda -lcudart_static -shared -o .output/xsched_guard_tool.so`
- `make native` — sm_120 probe/stub cubins and the host LDC patcher:
  - `nvcc -cubin -O3 -gencode arch=compute_120,code=sm_120 ../level2/native/probe.cu -o .output/native/xg_ldc_probe.cubin`
  - `nvcc -cubin -O3 -gencode arch=compute_120,code=sm_120 ../level2/native/check_preempt_port.cu -o .output/native/check_preempt_port.cubin`
  - `nvcc -cubin -O3 -gencode arch=compute_120,code=sm_120 ../level2/native/restore_exec_port.cu -o .output/native/restore_exec_port.cubin`
  - `g++ -O2 -std=c++17 ../level2/native/ldc_patcher.cpp -o .output/native/xg_ldc_patcher`

## Non-default extras

- `make ldc-patch` — host-only (no GPU): runs the LDC re-encoder on the real
  generated cubins and emits the arrays header consumed by GLM's HAL
  integration:
  - `.output/native/xg_ldc_patcher .output/native/xg_ldc_probe.cubin .output/native/check_preempt_port.cubin .output/native/restore_exec_port.cubin /usr/local/cuda-12.9/bin/nvdisasm .output/xg_sm120_guardian_arrays.h`
- `make clean` — removes `.output/`.

## Defaults and overrides

- `CUDA_HOME ?= /usr/local/cuda-12.9` (the installed CUDA 12.9 root);
  `nvcc`/`ptxas`/`fatbinary`/`nvdisasm`/`cuobjdump` from its `bin/`,
  includes from `targets/x86_64-linux/include`, libs from
  `targets/x86_64-linux/lib`.
- `SM ?= sm_120`, `BPF_SYMBOL ?= xsched_guardian`,
  `BPF_SECTION ?= cuda__/xsched_guardian`, `BUILD ?= .output`,
  `CXX ?= g++`, `CLANG ?= clang`.
- `NVBIT_ROOT ?= ../../llama.cpp/observability_overhead/revision-rq4/deps/nvbit_release_x86_64`
  — taken from the existing `workloads/sass-kretprobe/Makefile` (same
  `workloads/` tree, re-based to this directory), not invented.
- Every one of these can be overridden with an environment variable or a
  make variable (e.g. `make CUDA_HOME=/opt/cuda-12.9 guard-tool`).

## Remaining source integration (GLM-owned; explicitly NOT provided here)

The exporter ABI source gap was addressed in `8d9e1a6c`: both native C and
BPF now consume the same six-word scalar snapshot, and the locally adapted
exporter declares its 48-byte context. BPF no longer receives a device
pointer in that snapshot. The exporter patch applies to the frozen source;
component compilation and execution remain pending. The original SASS
example source and binary are unchanged.

This build does NOT invent the completed original cuXtra/HAL integration:

- XSched HAL (`libhalcuda`, BPF-actuator mode): `cuXtraSetDebuggerParams`
  filling of the 28-byte argument block (`../level2/xsched_guardian_abi.h`),
  per-launch `xg_host_publish` context publication on the launching thread
  right before `cuLaunchKernel`, and check_preempt / restore_exec launch
  orchestration (`XG_LAUNCH_GUARDIAN` / `XG_LAUNCH_RESUME`).
- The sm_120 HAL consumer (`arch/sm120.cpp`) that includes
  `.output/xg_sm120_guardian_arrays.h` emitted by the non-default
  `make ldc-patch` host step.
- Target workload selection and instrumented-entry experiment wiring (the SASS
  example built its own `vectoradd` target; Level-2 target selection and
  replay belong to GLM's tree).

## Notes

- No new link helper was needed: the existing `../level2/tool/merge_ptx.sh`
  plus `fatbinary`/`ptxas`/`nvcc` recipes cover the whole pipeline.
- No file/content hashes, checksums, or fingerprints are generated or used
  anywhere in this build.
