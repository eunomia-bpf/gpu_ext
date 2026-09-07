# BPF execution inside an existing SASS-only application

Completed 2026-09-07 PDT on RTX 5090, driver 575.57.08, CUDA 12.9.86,
pinned NVBit 1.8. Root ran the existing NVBit vector-add application once
without instrumentation and once with the new tool. Both returned zero.
This closes the bounded application-injection path, not a performance study.

| Observation | Result |
| --- | --- |
| Target kernel | `_Z6vecAddPdS_S_i` |
| Target PTX | Absent; target contains sm_120 cubin |
| Inserted EXIT sites | 2 |
| Launched threads / BPF output slots written with 42 | 100,352 / 100,352 |
| Selected launches | 1 |
| Application sum, baseline / instrumented | 100000.000000 / 100000.000000 |
| Application sum/n, baseline / instrumented | 1.000000 / 1.000000 |

There are 98 blocks of 1,024 threads for 100,000 vector elements. The 352
padding threads also exit and execute the callback; all launched threads
therefore have an eight-byte BPF output slot. The guard predicate prevents
execution at an EXIT instruction that the lane does not take.

## Implementation and executed command

Local OpenCode Qwen 27B implemented the NVBit EXIT adapter and embedding;
local GLM implemented the BPF device-function exporter (main `08e1692e`).
The BPF input is a real clang-built ELF section with four instructions,
verified against an eight-byte context and compiled using the existing
ptxpass backend. Its body remains a `.func`, not a `.entry`. The wrapper
passes each logical thread's distinct slot address and length eight.

The generated BPF PTX and NVCC wrapper PTX are assembled together using
`ptxas -astoolspatch`. `fatbinary --embedded-fatbin` then emits a C carrier
which ordinary g++ compiles into the tool's `.nv_fatbin`. NVBit injects the
wrapper call into the target's own SASS. No extra BPF kernel is launched.
The failed separable-device-link route and NVCC duplicate-fatbin issue are
explained in the [build README](../../README.md); they are build failures,
not discarded GPU measurements.

Root copied the built binaries before the agent's clean rebuild to
`/var/tmp/sass-exit-live-20260907.XpMqrW/`: tool 3,136,624 bytes, target
1,025,320 bytes. The command actually executed was:

```sh
SK_TARGET_SYMBOL=_Z6vecAddPdS_S_i \
LD_PRELOAD=/var/tmp/sass-exit-live-20260907.XpMqrW/sass_kretprobe.so \
/var/tmp/sass-exit-live-20260907.XpMqrW/vectoradd 100000
```

Baseline command: `./build/vectoradd 100000`. Reproduction uses `make all`
and the relative build/run commands in the README. The final Makefile's
clean rebuild completed, and root's subsequent `make all` returned zero.
No additional build-inspection prerequisite is part of `all`.

## Retained evidence and boundaries

`baseline.log` and `instrumented-attempt-01.log` are complete process output.
`target-ptx.txt`, `target-symbols.txt` and `tool-symbols.txt` inspect the
actual executed binaries. The directory also retains the BPF ELF,
compiler-generated BPF PTX, wrapper PTX, merged PTX and assembled cubin.
The original standalone and companion-launch studies remain unchanged.
After this run the GPU was at 15 MiB and 0% utilization; no driver or
service changes were required.

The result covers one CUDA context, one selected kernel and one in-flight
launch, with a finite 1,048,576-slot buffer. It does not establish helper/map
coverage, arbitrary BPF programs, late attach, concurrent launches, or
general performance. The current adapter resets slots, synchronizes and
copies them to the host around a launch; none of that overhead is hidden
inside a claimed Table 1 result. This new path uses NVBit as its trusted
binary instrumentation mechanism, not an independent gpubpf SASS patcher.
