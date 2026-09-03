# Q2 prefetch fixture: first CPU preparation

Recorded 2026-09-03 16:41 UTC. Root admitted this short CPU window after the
unrelated EB campaign stopped; no GPU, BPF attachment, module, service,
OpenCode, or Git operation was performed by this preparation.

Both commands ran on CPU 17 and exited **0**:

```sh
taskset -c 17 python3 -B extension/revision-prefetch/test_offline.py
taskset -c 17 make -C extension/revision-prefetch -j1
```

- [tests.log](tests.log): **3 synthetic tests passed**, covering the three
  functional controls, missing/missed observations, and input-only or invalid
  compute-mask evidence. These are not live driver observations.
- [build.log](build.log): first independent build completed without reported
  warnings/errors. It compiled only the new fixture/loader and reused the
  existing libbpf archive and bpftool. No driver or dependency was rebuilt;
  no source correction or second build attempt was needed.

Generated files, ordinary paths relative to the repository root:

| File | Bytes |
| --- | ---: |
| `extension/revision-prefetch/build/fixture.tmp.bpf.o` | 945,120 |
| `extension/revision-prefetch/build/fixture.bpf.o` | 27,744 |
| `extension/revision-prefetch/build/fixture.skel.h` | 92,852 |
| `extension/revision-prefetch/build/prefetch_safety` | 1,567,672 |

**Still unverified:** real fentry/fexit admission for the by-value region ABI,
actual wrapper-return/traversal accounting, native/BYPASS/invalid99 controls,
real signal/owned-cleanup behavior, and full target readback on 575. The observed
mask will be the pre-filter `compute_prefetch_mask` output, not the final hint
or DMA mask. No live-fallback success is claimed.
