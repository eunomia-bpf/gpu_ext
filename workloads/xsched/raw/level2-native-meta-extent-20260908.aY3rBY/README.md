# Native metadata-extent candidate: build and execution record

2026-09-08 PDT, RTX 5090, NVIDIA 575.57.08, CUDA 12.9.
This retries the unfinished native Level-2 redirected-entry configuration;
no completed baseline or original-entry control is repeated. It is not yet
a performance result. Earlier failures remain in their original directories.

The local GLM candidate copies the module image and raises two parameter-bank
extent records to `0x1520`, retaining the existing window-argument relay.
It does not enlarge the constant-bank data section. Whether the driver's
allocation accepts this metadata-only change must be established by the
actual run, not inferred from a CPU parser accepting the image.

The isolated source/build/install are under
`workloads/xsched/level2-build/.output/hal-native-20260908.i8na51/`.
Root builds and runs under both shared GPU and struct-ops locks.
The runner forwards `XG_NATIVE_META_EXTEND=1`; the original-entry control
variable is unset. The unchanged worker shape is 2 LC + 4 BE processes,
4 streams each, 50 tasks per stream, 340 blocks, 256 threads, and
9,511,106 recurrence iterations.

## Initial build and failed initialization

`build.log` reaches installation, but the cached CMake source inventory
omits the new `window_meta_extend.cpp`. The installed shim therefore has
unresolved `XgMetaExtendImage` and `XgFreeExtendedImage` symbols, observed
with `nm -D`. In `cells/`, all six workers exit 2 before `ready`, reporting
`cudaSetDeviceFlags: CUDA driver version is insufficient for CUDA runtime
version`. The runner exits 1. No GPU service sample is produced. This is
a candidate-library loading failure, not evidence that the installed
driver needs downgrading or that the metadata changes execute successfully.

`reconfigure.log` records refreshing the existing CMake configuration.
The next build, `rebuild.log`, actually compiles the new source and exits 2:
three `if` statements become empty when `XDEBG` is disabled, triggering
`-Werror=empty-body`. The diagnostics were returned to the same GLM session
for a behavior-preserving repair; compiler warnings are not disabled globally.

GPU remains idle after the failed initialization. No driver reload, reset,
reboot, or local-model termination is performed for this attempt.
