# Failed native-port loader diagnostic

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
