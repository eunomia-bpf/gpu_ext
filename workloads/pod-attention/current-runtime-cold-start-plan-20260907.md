# POD current-runtime cold-start comparison

Status: prepared, not measured. This is a new runtime comparison, not a
repeat or replacement of the completed shape and phase studies.

## Question and unchanged work

Does the retained current runtime still incur the historical 271.225-second
median interval before the first Python statement? Measure that interval,
loader readiness, total client duration, and the existing CUDA-event operator
time separately. A faster startup must not be described as faster attention.

Use the existing `bench.py --phase-study`: Llama-3-8B, decode batch 32,
10 warmups and 100 samples. Keep the existing selector, adapter, PTX packets,
and operator workload unchanged. Select
`/home/yunwei37/workspace/gpu/bpftime-table1-hostfix-plt/build-table1-575-warp`
using the runtime override in main commit `6b8d66fd`. The historical
`bpftime/build-cuda-pr503` binary is absent at its recorded path; therefore
historical/current differences cannot isolate a single source change.

## New measurements

Run three rotated blocks, each cell in a fresh process:

| Block | First | Second | Third |
| --- | --- | --- | --- |
| 1 | pod_inline | pod_cuda | pod_bpf |
| 2 | pod_cuda | pod_bpf | pod_inline |
| 3 | pod_bpf | pod_inline | pod_cuda |

Use the local-model single-cell launcher once available. Keep GPU execution
exclusive with the existing GPU and struct-ops locks. Apply CPU affinity
8–15 before setting target `LD_PRELOAD`; give each BPF cell its own loader
and private shared-memory segment. Do not add a preflight, clock calibration,
new correctness campaign, or artificial process timeout.

Retain each exact command/environment, loader/client logs, child exits,
monotonic spawn/exit/readiness timestamps, and the existing `operator.json`.
Compute pre-Python duration from client spawn to `process_main_ns` and client
wall duration from spawn to exit. Report final cleanup separately if captured.
All failed attempts remain visible; a failure is not a performance sample.
Do not rerun a completed cell to improve its number. If an actual runtime
failure requires implementation work, record the failure and source change
before starting a distinct attempt.

Report all three values and their median per arm, plus matched BPF/CUDA
operator ratios. These three blocks characterize this startup path; they do
not establish equivalence, a universal attach cost, or a general performance
bound. Preserve the original 15-cell phase report and every earlier result.

No paper files are edited by this coordinating session or its delegates.
