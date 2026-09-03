# Private offloader build 01 — 2026-09-03

The actual private offloader build passes, exit 0: all **19 C++ translation
units**, link, exact-path import and runtime-interface checks. The fresh
`build/stage-check-02/finemoe/ops/prefetch/prefetch_op.cpython-312-x86_64-linux-gnu.so`
is **59,238,064 bytes**, exposes `section-vi-private-adapter-v1` and the snapshot
API, and does not initialize CUDA during import. The owned build process group
is empty on exit. Full compiler commands, warnings and checks are in
[offloader-build-01.log](adapter-cpu-02/offloader-build-01.log).

Root ran the already-reviewed [build wrapper](build_adapter.py) with a
1,200-second timeout, CPU 17, `MAX_JOBS=1`, the existing FineMoE environment
and CUDA devices hidden. This is C++/host-extension compilation, not a CUDA
kernel build. No source in the frozen FineMoE tree was changed. Part of this
build overlapped the non-performance Hummingbird preflight; it must not overlap
the upcoming formal GPU measurements.

The private tree now contains its built extension and cannot be treated as
a fresh build input by the wrapper. New build attempts use a new staged path.
No Section VI GPU request or performance cell has yet run. Remaining gates
are actual three-arm original-HF/logit agreement, whole-expert eviction and
copy accounting, real-input native/JIT decision parity, and the paired
performance experiment. This build alone does not establish policy reproduction.
