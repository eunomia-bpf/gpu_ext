# Separate optimized runtime build for POD startup

Status: configured successfully; agent/server build is running. No Release
performance measurement exists yet. The completed Debug-runtime campaign
remains in `../results-current-runtime-575-20260907.md` (main `e768d559`).

## Reason and preserved control

The retained `build-table1-575-warp/CMakeCache.txt` says `Debug`. Its generated
`bpftime_nv_attach_impl.dir/flags.make` contains `-O0 -g3`; the PTX core/pass
flags have debug information and no optimization flag. This is a plausible
contributor to the 234.967-second median pre-Python time, not a measured cause.

The new build is a separate directory in the same worktree
`/home/yunwei37/workspace/gpu/bpftime-table1-hostfix-plt`, revision `89a1244`.
No tracked source is changed for this build. The existing dirty `vm/llvm-jit`
submodule is preserved, not reset; a worktree revision alone is not a complete
dependency-state claim. The old build, its agent/server/pass libraries and
all prior results remain untouched. No timing patch has been applied.

## Commands executed

From that bpftime worktree:

```sh
cmake -S . -B build-pod-release-575 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBPFTIME_ENABLE_CUDA_ATTACH=ON -DBPFTIME_CUDA_ROOT=/usr/local/cuda-12.9 \
  -DBPFTIME_BUILD_EXECUTABLE=OFF -DBPFTIME_BUILD_STATIC_LIB=OFF \
  -DBPFTIME_BUILD_WITH_LIBBPF=ON -DBPFTIME_BUILD_KERNEL_BPF=ON \
  -DBPFTIME_LLVM_JIT=ON -DBPFTIME_UBPF_JIT=ON \
  -DBPFTIME_ENABLE_UNIT_TESTING=OFF -DBPFTIME_USE_CATCH2=OFF \
  -DBPFTIME_ENABLE_CCACHE=OFF -DBPFTIME_ENABLE_ASAN=OFF \
  -DBPFTIME_ENABLE_LTO=OFF -DENABLE_EBPF_VERIFIER=ON \
  -DLLVM_DIR=/usr/lib/llvm-15/cmake
cmake --build build-pod-release-575 \
  --target bpftime-agent bpftime-syscall-server -j 8
```

Configuration completed with GNU 13.3, LLVM 15 and CUDA 12.9. Generated flags
for `ptxpass_core` and `ptxpass_kprobe_entry` contain `-O3 -DNDEBUG`. The
attach target appends `-O2 -flto=auto -ffat-lto-objects`, so its effective
optimization is not simply the global `-O3`. These existing target-specific
flags remain unchanged; do not describe this as a one-flag isolated ablation.

Once the build completes, use the existing current-runtime single-cell
launcher with `--bpftime-build` pointing to `build-pod-release-575` and a new
output directory. Keep workload, selector, adapter, PTX inputs and measurement
boundaries unchanged. Record Release results separately, including unfavorable
ones; do not relabel or overwrite the completed Debug cells. The GLM stage
timing patch remains a separate implementation task, not a prerequisite gate.
