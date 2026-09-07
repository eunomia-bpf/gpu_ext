# Embedding notes — prebuilt sk_bpf.cubin into the NVBit tool .so (GLM assist)

Bounded assist for Qwen (owns tool/Makefile sources). Recorded from this
session only; conclusions marked (observed) come from actual commands run
by root/Qwen; final pipeline lives in Qwen's files.

## Working packaging path (observed, live-verified by root)

    # Optional standalone image used during the earlier dlink attempt
    fatbinary -64 --create build/tool/sk_bpf.fatbin \
        --image3=kind=elf,sm=120,file=build/tool/sk_bpf.cubin

    # Working build: emit C directly from the cubin (no prior fatbin needed)
    fatbinary -64 --embedded-fatbin build/tool/sk_bpf.fatbin.c \
        --image3=kind=elf,sm=120,file=build/tool/sk_bpf.cubin

    # 3. compile generated C with g++ (NOT nvcc) and link into the carrier
    g++ ... build/tool/sk_bpf.fatbin.c ... -> tool .so

Contents of the generated C: fatbin blob lands in section `.nv_fatbin`
and defines a wrapper `__fatBinC_Wrapper_t __fatDeviceText` for
`.nvFatBinSegment`. The compiled carrier was observed to retain `.nv_fatbin`;
the final tool exposes both SASS functions and NVBit successfully injects
them. The live run establishes discoverability, not which internal NVBit
module-loading or registration path provides it.

## Do NOT device-link the fatbin (observed)

    nvcc -dlink build/tool/sk_bpf.fatbin ...   # links, but ...

`nvcc -dlink` of the wrapped fatbin succeeds, yet the final .so loses
BOTH `bpf_exit` and `sk_bpf_trampoline` device symbols (observed).
The precise device-linker elimination mechanism was not traced. The
working pipeline bypasses that device-link step. `-rdc=true` remains useful
only when generating the wrapper PTX; it is not combined with tools-patch
assembly's incompatible compile-only mode.

## g++ vs nvcc for the generated fatbin C (observed)

Compiling `sk_bpf.fatbin.c` with plain g++ and linking it beside the
nvcc carrier works — tools-patch functions survive and are discoverable
(live run: SK_TARGET_SYMBOL=_Z6vecAddPdS_S_i LD_PRELOAD=tool on
vectoradd100000: rc=0, instrumented_exits=2, 100352/100352 slots wrote
the detector value; baseline rc=0 identical). Keep nvcc strictly for the
carrier device code and let g++ treat the generated fatbin C as opaque
data + host wrapper, so no nvcc device registration pass touches the
assembled SASS.

The failed NVCC carrier included this generated C in a `.cu` file. NVCC
then generated a second fatbin in the same translation unit, reporting
duplicate `__fatBinC_Wrapper_t`, `fatbinData` and `__fatDeviceText`
declarations. Ordinary g++ avoids that additional CUDA compilation pass.

## Status

Live run log:
workloads/sass-kretprobe/results/sass-exit-575-20260907-01/instrumented-attempt-01.log
Root owns report/commit. GLM assist concluded: packaging + discoverability
resolved; no further experiments, no extra tests, no commits here.
