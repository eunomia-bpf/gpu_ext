# GPU-local array: build and measurement record

Local OpenCode/Qwen 27B implemented the BPF source, shared layout, collector
and Makefile. The root supplied small host build and initialization-order fixes
and runs the existing Table 1 performance harness. No shared bpftime runtime
or driver source is changed by this candidate.

## Build

The source was staged under
`/var/tmp/kernelretsnoop-onevalue-build.x1aICR/example/gpu/onevalue-array`.
The temporary root links `third_party` and `runtime` to the existing
`/home/yunwei37/workspace/gpu/bpftime-table1-hostfix-plt` tree. The staged
BPF section targets the same pp512 kernel as the completed Table 1 campaign:

```
kretprobe/_Z9rope_normILb1ELb0Ef6__halfEvPKT1_PT2_iiiiiPKifff14rope_corr_dimsfPKfPKli
```

`make -j8 CUDA_HOME=/usr/local/cuda-12.9 BPFTOOL=/usr/local/sbin/bpftool`
builds the BPF object, skeleton and host collector successfully. A
`kernelretsnoop -> onevalue-array` symlink in the staged directory lets the
existing `run_table1_perf.run_arm_cell` use the candidate without changing
the runner or original tool. The runtime remains `build-table1-575-warp`.

The first host build needed the shared-layout include and `uint32_t` instead
of the BPF-only `u32` alias. The producer does not update a shared committed
counter: the host sums the existing per-warp counters after the bulk copy.
Collector diagnostics do not impose a performance gate.

## Retained startup failures

Both attempts below returned numeric llama.cpp throughput, but the collector
was interrupted while initialization was stuck, returned -2, and produced
no event/drain output. They are **not instrumented-overhead results**.

| Attempt | Baseline token/s | Candidate-process token/s | Collector startup wait |
| --- | ---: | ---: | ---: |
| [Initial](../results-onevalue-array-575-20260907/cells.json) | 38097.432099 | 38220.107509 | 3 s |
| [Longer startup](../results-onevalue-array-startup20-575-20260907/cells.json) | 38045.068991 | 37690.974351 | 20 s |

A GDB backtrace on a separate owned diagnostic process identified recursion:
collector `cuInit` -> CUDA `fopen` -> syscall-server lazy startup -> its own
`cuInit`. The second initialization spun inside libcuda while the first was
still in progress. Increasing the delay did not fix this. The collector now
issues a read-only BPF map-ID query before calling CUDA, allowing syscall-server
startup to complete first; the query result is ignored, not an admission test.
Its CUDA context still exists before map allocation and outlives collection.
The diagnostic process and its temporary shared-memory segment were removed;
all experiment logs and numeric records are retained.

## First completed pair; remaining pairs running

[Raw campaign](../results-onevalue-array-bootstrap-575-20260907/cells.json):
RTX 5090, driver 575.57.08, TinyLlama-1.1B Q4_K_M, pp512, one repetition
with the existing warmup, CUDA graphs disabled, CPU affinity 8–15. Five
alternating-order baseline/candidate pairs are planned; block 1 is complete
and blocks 2–5 continue without rerunning block 1. No process timeout is set.

The first baseline is 38032.055080 token/s and the candidate is
35945.547270 token/s: **5.486182% overhead for this one pair**, not yet a
five-block aggregate. Both benchmark and collector exit zero. Collection
reports 720896 stored 32-byte records, 16384 active warp coordinates and no
reported overflow, out-of-range coordinates or coordinate mismatches.
The single 23199768-byte host lookup takes 10157582 ns (10.158 ms).

Prefill token/s excludes startup/JIT and final collection, as does the
original Table 1 measurement. Final collection is reported separately;
deferring it does not make its cost disappear. The arena is sized for this
finite pp512 run and is not an unbounded streaming collector. Existing P40
and RTX 5090 measurements, including 90.705% gpubpf and 99.621% NVBit overhead,
remain unchanged. No five-block optimized claim is made before those runs finish.
