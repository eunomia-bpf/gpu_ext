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

## Repaired build and real launch failure

The local model adds braces to the three debug-only conditional bodies.
`rebuild-fixed.log` records build/install exit 0. The same runner and
configuration then execute in `cells-fixed/`; all six processes reach
`ready`, but BE1 exits with SIGSEGV before `running`, and the runner exits 1.
Several stderr reader threads encounter non-UTF-8 error text, so their JSON
logs are incomplete. `runner-fixed.log` retains that limitation. No timing
sample is accepted or reconstructed from those fragments.

Root performs one single-worker debugger diagnosis, retaining raw stderr
bytes in `debug-backtrace-fixed.log`. Its first launch returns CUDA **701**
(too many resources requested). Later launch threads return **4** after
CUDA teardown starts, and multiple `CUDA_ASSERT` error paths enter
`exit(1)`; the observed SIGSEGV is in `__run_exit_handlers`. Thus the primary
observed launch error remains 701. The preceding debugger-parameter
readback reports equality, so a stack-overwrite diagnosis is not established.
No metadata-extender log is present; whether runtime module loading actually
traverses the new shim wrappers remains unresolved. This run does not prove
that the driver accepted and then rejected an extended parameter bank.

The first debugger invocation accidentally replaces the inferior's arguments
with redirection only; it exits at usage without launching anything. Its
`debug-backtrace.log` is retained separately, not counted as a workload run.
The corrected invocation supplies the original arguments explicitly and
feeds `debug-go.txt`; each diagnostic has its own short-lived xserver log.
The debugger's exit 0 is not application success: its inferior stops on
SIGSEGV. Both diagnostics use the shared leases and leave GPU idle.

The additive shim patch is
`../../level2/native/xsched-native-meta-extend.patch`. A reverse-application
check succeeds against the isolated source after the build repair; this
records source correspondence, not a successful runtime result. The next
implementation must resolve module-load coverage and launch 701. All
completed comparisons and earlier failed attempts remain unchanged.
