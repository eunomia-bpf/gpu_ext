# XSched CUDA build smoke

- Date: 2026-08-31
- Upstream: `https://github.com/XpuOS/xsched.git`
- Commit: `f49289f0220931df78de948ed841ecbaf960a919`
- Commit date: `2026-08-19T19:16:15+08:00`
- Host compiler: `/usr/bin/gcc` and `/usr/bin/g++` 13.3.0. The explicit
  paths avoid this host's nonstandard `/usr/bin/c++` Python wrapper.
- Host driver at smoke time: NVIDIA 610.43.02
- Command: `make PLATFORM=cuda`
- Result: success; installed `xserver`, `xcli`, `libpreempt.so`,
  `libhalcuda.so`, `libshimcuda.so`, and the `libcuda.so[.1]` shim links under
  `deps/xsched/output/`.
- Official example compile: `make cuda` in
  `examples/Linux/1_transparent_sched` also succeeds with nvcc 12.9. The
  example has not been executed because the GPU is occupied.
- The reviewed passive engagement patch applies and reverses cleanly against
  the pinned commit. A clean CUDA rebuild with the patch succeeds. Admission
  compares the exact small source diff and records ordinary path, size, inode,
  and modification/change-time metadata for required runtime files.
- Architecture scope verified from upstream `platforms/cuda/hal/src/arch/arch.cpp`:
  sm_120 takes `CudaQueueLv1`; `Guardian::Instance` and
  `TarpHandler::Instance` return null for sm_120.

This is a build-only smoke test, not evidence that interception, suspension,
or a paper workload works on the RTX 5090.
