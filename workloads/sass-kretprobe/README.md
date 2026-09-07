# sass-kretprobe

Verified eBPF -> ptxpass SASS -> NVBit EXIT injection into a SASS-only
sm_120 target kernel. The BPF body executes inside the target kernel's own
EXIT path: NVBit inserts a call (at every EXIT) into the target SASS; the
called `sk_bpf_trampoline` lives in this tool's embedded fatbin and calls
`bpf_exit`, the ptxpass-compiled SASS of the BPF program. Each logical
thread writes its result into its own 8-byte device slot, so concurrent
threads never race on one marker.

Live result (driver 575.57.08, sm_120): `results/sass-exit-575-20260907-01/
instrumented-attempt-01.log` — `instrumented_exits=2`, `threads=100352
expected_slots=100352 wrote_42=100352`, target output unchanged
(`sum100000` baseline match), `rc0`.

## Scope

- One CUDA context, one target kernel symbol (`SK_TARGET_SYMBOL`),
  launch APIs `cuLaunchKernel`, `cuLaunchKernel_ptsz`,
  `cuLaunchKernelEx(_ptsz)`.
- Per-launch slot pool: 1<<20 x 8 B device memory; launches exceeding the
  capacity simply skip BPF for the extra threads.
- This tool does not coexist with other injection tools (e.g. Frida) and
  no overhead numbers are claimed here.

## Ownership

- This agent: `bpf/`, `tool/` (incl. `merge_ptx.sh`), `Makefile`,
  `README.md`, `.gitignore`.
- GLM (do not edit from this tree): `bpf_to_ptx.cpp`, `compiler.mk`,
  `compiler-notes.md`.

## Build

```
make all          # BPF object, exporter (compiler.mk), BPF PTX, merged TU,
                  # astoolspatch SASS, fatbin carrier, tool .so, target app
make verify       # optional read-only artifact inspection
```

Pinned NVBit 1.8: `../llama.cpp/observability_overhead/revision-rq4/deps/
nvbit_release_x86_64`.

### GLM exporter contract

`compiler.mk` provides `$(BUILD)/bpf_to_ptx`. This Makefile invokes:

```
build/bpf_to_ptx build/kretprobe_sass.bpf.o cuda__/kretprobe_sass \
    build/ptx/bpf_exit.ptx bpf_exit sm_120
```

Output PTX: standalone `.visible .func bpf_exit(context_ptr, context_length)`
(two `.b64` params, 2-arg ABI; 8-byte PREVAIL context), no `.entry`,
bodies untouched.

### Embedding pipeline (verified)

1. `nvcc -ptx -rdc=true -arch=sm_120 -O3 -std=c++14 --keep-device-functions
   tool/sk_device.cu` — wrapper PTX (plain `-ptx` drops the unreferenced
   trampoline; NVVM needs `--keep-device-functions`).
2. `tool/merge_ptx.sh` — one header + both `.func` bodies, wrapper's
   `.extern .func bpf_exit` declaration removed (in-module resolution).
3. `ptxas -arch=sm_120 -astoolspatch` (NVBit tool-patch mode; must NOT run
   with compile-only) — SASS cubin with global symbols `bpf_exit`,
   `sk_bpf_trampoline`.
4. `fatbinary -64 --embedded-fatbin sk_bpf.fatbin.c
   --image3=kind=elf,sm=120,file=sk_bpf.cubin`.
5. `g++ -x c++ -fPIC` on the generated fatbin C — carrier object owning the
   `.nv_fatbin` section. (NVCC `-c` of a .cu that includes this C is NOT
   used: nvcc emits its own auto-generated fatbin in the same TU, causing
   duplicate `fatbinData`/`__fatDeviceText` declarations. Also not used:
   `nvcc -dlink` of a plain-ELF fatbin, which drops the image and its
   symbols from the linked fatbin.)
6. `nvcc -shared tool object + carrier object -lnvbit -lcudart_static`
   -> `build/sass_kretprobe.so`.

## Run (GPU)

```
SK_TARGET_SYMBOL=_Z6vecAddPdS_S_i \
LD_PRELOAD=$PWD/build/sass_kretprobe.so ./build/vectoradd 100000
```

Expected stderr lines:

```
SKRET tool loaded target=_Z6vecAddPdS_S_i
SKRET instrumented_exits=2 target=_Z6vecAddPdS_S_i
SKRET launch=1 threads=100352 expected_slots=100352 wrote_42=100352
SKRET done target_launches=1
```

plus the unmodified vectoradd `sum` output.

## Evidence checks (make verify)

- BPF PTX is a device `.func` (no `.entry`).
- `cuobjdump --dump-elf-symbols build/sass_kretprobe.so` lists
  `bpf_exit` / `sk_bpf_trampoline` in the tool fatbin.
- Target fatbin: no PTX, sm_120 SASS only.
