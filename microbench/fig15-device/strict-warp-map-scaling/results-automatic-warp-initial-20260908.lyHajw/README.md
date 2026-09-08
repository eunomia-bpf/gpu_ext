# Initial automatic-warp execution on the original device map probe

Date: 2026-09-08, approximately 09:27 UTC. This is an initial execution
attempt, not a completed paired performance campaign or a Table 1 result.
RTX 5090, driver 575.57.08. No driver/module change was made.

The existing `.output/warp-map-bench`, `.output/warp-map-loader` and
`.output/warp-map-probe.bpf.o` were used without changing the probe. Both
instrumented processes select `shared_update` / `cuda__shared` (14 BPF
instructions), shared GPU map type 1503, with
`BPFTIME_GPU_AUTO_WARP_EXECUTION` set to 0 or 1. The runtime is the isolated
`bpftime-auto-warp/build-auto-warp-575` build, from source checkpoint
`e801d28` plus the kernel-entry/idempotent-declaration work in progress.
The included `runtime-source.patch` was collected **after** the run: GLM had
already added the `.b32` correction by then, so that patch is a post-run
source checkpoint, not the exact source of the failed runtime. It uses CUDA 12.9,
LLVM 15, verifier enabled, and the build-only `-include cstdint` compatibility
option. No runtime binaries or shared-memory payloads are published.

Each application command uses `--threads 128 --warmup 8 --launches 128`;
run IDs are 90801, 90802, and 90803, in baseline/off/on order. The measurement
is CUDA-event elapsed milliseconds covering all 128 measured launches, not
per-launch latency. This is one sample per setting, not five or ten repeats.

| Setting | Elapsed ms | App exit | Loader exit | Observed execution |
| --- | ---: | ---: | ---: | --- |
| Baseline | 0.260127991 | 0 | n/a | Original application |
| Automatic warp off | 0.329535991 | 0 | 0 | Hook executes; one populated shared-map key |
| Automatic warp on | 0.258848011 | 0 | 0 | PTX compilation fails; application falls back without the hook |

The enabled path accepts the same probe for automatic leader execution,
but ptxas reports `Unexpected instruction types specified for 'activemask'`.
The emitted instruction omits its `.b32` type. The application still exits
zero and reports its own arithmetic result, while the loader reports zero
populated map keys. Thus the smaller enabled-path time is **not an
optimization benefit**. All raw times and the failure remain here. Root
sent the concrete assembly error to the existing GLM session for correction;
the failed enabled arm will be retried separately, without replacing this
attempt or repeating the completed baseline/off cells.

Common settings: 128 GPU threads, 256 MiB private bpftime shared-memory
segment, 1024 max FDs, `sm_120`, `STRICT` verifier mode. Loader preloads the
new syscall-server library; application preloads its agent library with
deferred PTX extraction and targeted late bootstrap enabled. The unchanged
loader's maximum supported lifetime argument is 3600 seconds; root signaled
its normal SIGINT/readback path immediately after each application finished.
No timeout expired. Loader PIDs were 2521058 and 2521984, and application
PIDs were 2521317 and 2522426. The GPU lock was held by the root shell only
and released after both runs. Both loaders and applications exited.

The two task-owned 256 MiB shared-memory files
`/dev/shm/fig15_auto_initial_20260908_2520348` and its `_on` variant were
removed after checking their exact paths and completed processes. These
temporary transport buffers are not archived; their small map readbacks
remain in the loader logs. Existing storage loaders and unrelated worktrees
were untouched.
