# Failed native-port loader diagnostic

## Subsequent localization

The trace contains two successive executable images under the same PID:
tool initialization and its loaded message precede transfer to taskset;
after exec into priority_workload there is no tool initialization/loaded
message, but the workload later loads HAL and fails to find the publication
function. Thus the duplicated HAL messages span exec, not demonstrated
duplicate instances within the workload. Main `42049461` delays LD_PRELOAD
until the real command via taskset followed by `/usr/bin/env`. No HAL
namespace bridge or driver modification is required by this launcher fix.
The actual failed/unstarted policy cells are being retried separately;
this trace itself remains a failed diagnostic, not performance evidence.

One diagnostic invocation retried only the failed native-port configuration
from `../level2-tool-native-bpf-20260908.sjyzim/`, with `LD_DEBUG=files`.
The completed baseline was not repeated. The runner and worker remain at
`c5f6e4ea` and `e9529745`; `cells/protocol.json` records the actual workload
and paths. This is startup-failure evidence, not a performance measurement.

The tool loads and reports the native decision mode. Loader output then
shows a second link-map generation and initialization for the CUDA shim,
HAL and preemption library when the executable opens libcuda.so.1. Each
process initializes the global scheduler twice. HAL subsequently fails to
resolve `xg_host_publish` despite its export from the loaded tool. The
printed loader IDs are all `[0]`: duplicate loading is observed, but a
separate linker namespace is not established by this trace. Earlier GLIBC
dlsym-version lookup errors are version probes, not the final assertion.

The invocation exits 1. All owned workers/server have ended; GPU utilization
is 0% with 1 MiB used, and both shared experiment leases are released. No
driver module was changed. Local Qwen owns the publication-path repair;
resume only failed/unstarted campaign cells after integration is repaired.

Inventory: six worker JSON logs (27,911–32,419 bytes), server JSON
(13,409 bytes), failure JSON (340 bytes), protocol JSON (5,109 bytes), and
runner log (24,126 bytes). No binary or large cache is included.
