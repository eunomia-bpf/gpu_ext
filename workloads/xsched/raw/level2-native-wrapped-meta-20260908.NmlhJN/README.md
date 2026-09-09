# Wrapped fatbin metadata retry: launch 701 remains

2026-09-08 PDT, RTX 5090 / driver 575.57.08 / CUDA 12.9.
This is one retry of the failed native Level-2 configuration after the local
GLM session implements the runtime fatbin wrapper identified in `fefe8898`.
No successful original-entry control or baseline is repeated.

The candidate recognizes CUDA's version-1 wrapper, copies its 24 bytes into
a 32-byte area, and retargets its data pointer to the copied nested image.
One allocation owns both copies until the driver load call returns. The
existing metadata extent changes, launch-argument relay, and error
propagation remain. The exact candidate source is `window_meta_extend.cpp`
here; use it in the isolated shim at
`platforms/cuda/shim/src/window_meta_extend.cpp` on top of the previously
published native patches. This source snapshot preserves the measured
candidate independently of the local session's subsequent edits.

The existing isolated native CMake build/install exits 0 (`build.log`).
Root then runs `run_native_blob.py` with `XG_NATIVE_META_EXTEND=1`, original
entry control unset, one repetition, 50 tasks per stream, 9,511,106
iterations, 340 blocks and 256 threads. Both shared leases cover build/run.
There are 2 LC + 4 BE processes and 4 streams per process.

All six workers reach ready. Each BE log now records both compiler extent
records changing to `0x1520` and the owned wrapper (`0x20` bytes) plus nested
image (`0x4818` bytes). This confirms the missing wrapper case is exercised;
it is no longer merely a CPU-side specimen result. However, the first native
launch still returns CUDA 701. Later error paths produce CUDA 4 and a host
SIGSEGV during teardown; BE1 exits -11 before running, and the runner exits 1.
Some stderr readers fail on non-UTF-8 teardown messages, so the JSON logs
are incomplete; `runner.log` preserves these reader failures. No complete
service or throughput sample is produced.

The constant-bank data section is still not enlarged. This result neither
establishes why the remaining launch is rejected nor proves that a larger
data section is the solution. The same GLM session has received the actual
error and continues parameter/resource delivery repair. No unchanged retry
is warranted. GPU is idle at 0% / 1 MiB / P8 after cleanup; no persistence
restart, module change, reset, or local-model termination is required.
