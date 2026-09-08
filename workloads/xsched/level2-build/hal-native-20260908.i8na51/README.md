# Native GuardianSM120 HAL build, 2026-09-08

The native guardian arrays from `../native-extraction-20260908.fnVM6q/`
are now compiled into a separate installed HAL. Patched configuration and
`cmake --build ... --target install -j2` both return 0. This is a build,
not a completed native GPU workload or performance comparison.

Source/build/install live under ignored
`.output/hal-native-20260908.i8na51/{source,build,install}`. Sources were
copied from frozen `deps/xsched` (upstream `f49289f` plus the existing
passive async_xqueue engagement modifications), excluding `.git`, `build`,
`output`, and built example `app`. This is not a new Git worktree; the
dependency and running tool-actuator source copies were not changed.
The saved `level2/xsched-level2-sm120.patch` is the main-tree version at
`3a134a94`. Both builds held both shared experiment locks.

## Commands

From the repository root, after copying the source:

```sh
patch --batch --forward -p1 \
  -d workloads/xsched/level2-build/.output/hal-native-20260908.i8na51/source \
  -i /home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/level2/xsched-level2-sm120.patch
cmake -S workloads/xsched/level2-build/.output/hal-native-20260908.i8na51/source \
  -B workloads/xsched/level2-build/.output/hal-native-20260908.i8na51/build \
  -DPLATFORM_CUDA=ON -DBUILD_SERVICE=OFF -DBUILD_TEST=OFF -DSHIM_SOFTLINK=ON \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_INSTALL_PREFIX=/home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/level2-build/.output/hal-native-20260908.i8na51/install \
  -DXG_SM120_GENERATED_HEADER=/home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/level2-build/.output/native-extraction-20260908.fnVM6q/xg_sm120_guardian_arrays.h
cmake --build workloads/xsched/level2-build/.output/hal-native-20260908.i8na51/build \
  --target install -j2
```

The actual first `git -C <copied-source> apply` returned zero but skipped
all paths because the source directory was nested inside the main Git
worktree. Initial configuration warned that `XG_SM120_GENERATED_HEADER`
was unused; `git apply --stat` then reported zero files. The initial build
therefore was not the native integration. `configure.log` and `build.log`
retain it. No GPU execution used that installation.

The explicit `patch -p1` command applied nine files (`apply.log`), followed
by reconfiguration with the same cached options (`configure-patched.log`)
and the successful `build-patched.log`. CMake compiled `arch/sm120.cpp`;
`halcuda.dir/flags.make` contains the generated-header definition. Installed
`libhalcuda.so` exports both `GuardianSM120::GetGuardianInstructions` and
`GuardianSM120::GetResumeInstructions`.

Installed library sizes: `libhalcuda.so` 556,048 bytes; `libshimcuda.so`
822,960 bytes. Existing shim symlinks produced two `File exists` messages
during reinstall; the build/install command still returned 0. The small
logs are retained here; binaries and build caches are not committed.

Next: run the native cuXtra path with this HAL and the existing service-only
workload. The separate NVBit tool-route replay failure remains unresolved;
neither component build establishes that it is fixed.

## Next-run failure localization

Main `16a9aece` adds only a first-mismatch index/actual/expected print to
the worker's existing output-error branch. It changes neither computation,
timing, nor the existing comparison. CUDA 12.9 compilation returned 0 with
the same flags as `../service-output-20260908.66ppYl/README.md`, changing
only the output to `.output/service-mismatch-20260908.sHSYtE/priority_workload`
(1,058,280 bytes). The compiler produced no diagnostics. The previous
measured worker remains untouched. No GPU cell was rerun for this change;
the new binary is available for the next unfinished native/tool attempt.
