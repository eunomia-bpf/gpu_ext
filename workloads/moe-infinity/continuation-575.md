# MoE-Infinity four-cell continuation on 575

Current protocol: `proposal-3-revision-9-575-cuda129`, recorded before any 575
performance result. Revision 8's first warm-up and the following diagnostic
failure are preserved; see the compiler-load investigation below.

The user requested automatic completion of the actual MoE-Infinity, XSched,
and LMCache comparisons, explicitly including gpubpf. This continuation uses
Linux 6.15.11 and NVIDIA 575.57.08 for every cell; no 610 correctness or timing
sample is pooled. It reuses the existing request, exact-output, CPU-affinity,
policy-ownership, telemetry, and paired-analysis implementation rather than
creating an unrelated benchmark.

Unchanged: all four configurations, exact 120B MXFP4 source models, CPU 0–7,
512 input and 64 output tokens, eight measured prompts, one excluded warm-up,
new server/policy process per cell, frozen randomized configuration/prompt
orders, 60-second idle cooldown, five valid complete blocks with at most eight
attempts, and the paired geometric-mean/10,000-resample analysis. The MoE
artifact includes the already disclosed row-chunking and deterministic
accumulation correctness repairs. gpubpf is the existing safe per-CPU,
1/256-sampled host-stride/LFU ablation, not the full device-observed policy.

Execution corrections reflect facts established before this continuation:

- Each of the four cells is revalidated on 575 with two complete eight-prompt
  correctness passes. Exact equality is required within each configuration;
  cross-configuration equality remains diagnostic because the previous
  single-slot controls established differences between valid greedy outputs.
- MoE's seven-partition store is buffered-hydrated once into CPU memory by
  each new server. Expert dispatch/cache counters must engage; a measured
  positive disk-read delta is not required and no steady-state direct-I/O
  claim is made. A new 61 GiB derived disk store is created during preflight
  and reused by later fresh processes, so duplicate immutable model copies
  do not exhaust the disk. Model loading and hydration remain excluded from
  measured request throughput. No old evidence or user file is deleted.
- gpubpf requires positive page-fault, stride, prefetch, activation, access,
  sampled update/reorder, and eviction-prepare callback deltas. The earlier
  UVM Tools descriptor did not deliver type-14 events even during real
  oversubscription. This continuation therefore does not assert completed
  evictions from that monitor or rename prepare callbacks as completions.
- All cells keep the same enforced 400 W safety limit. Software power-cap
  activity is recorded rather than discarding the intended capped condition;
  thermal slowdown and hardware slowdown/brake conditions still invalidate
  the cell. Kernel/Xid, idle-GPU, empty struct_ops, and ownership-safe cleanup
  checks run before and after every cell. A failure stops the run instead of
  blindly spending replacement attempts on the same error.

Commands from this directory:

```bash
.venv/bin/python -m unittest -q test_offline.py test_575_head_to_head.py
.venv/bin/python run_575_head_to_head.py admit
.venv/bin/python run_575_head_to_head.py preflight \
  --output raw/head-to-head-575-cuda129/preflight \
  --expert-store raw/head-to-head-575/preflight/expert-store
.venv/bin/python run_575_head_to_head.py run \
  --preflight raw/head-to-head-575-cuda129/preflight \
  --output raw/head-to-head-575-cuda129/timing --max-blocks 2
# Continue the same frozen sequence, without rerunning accepted blocks:
.venv/bin/python run_575_head_to_head.py run \
  --preflight raw/head-to-head-575-cuda129/preflight \
  --output raw/head-to-head-575-cuda129/timing --max-blocks 5
```

The two-block stage is a time-budgeted preliminary checkpoint, not full
reproduction and not a replacement for the five-block objective. Historical
request durations for planning are 6.5–13.6 s for UVM, 6.4–7.3 s for N-CMoE32,
about 5 s for repaired MoE, and one 45.026 s gpubpf warm-up. These are not
current performance measurements. Startup, 575 performance, and any failures
may prevent the preliminary checkpoint from finishing within one hour.

## 575 compiler-load investigation

On 2026-09-02, revision 8's real MoE server loaded the model and answered
health/model requests, but its first 512-token warm-up lost the HTTP connection.
The kernel journal identifies a user-process segmentation fault in
`libcuda.so.575.57.08`, not an OOM kill or NVIDIA Xid. A subsequent owned GDB
diagnostic reproduced the failure and recorded the relevant native chain:

```text
libcuda.so.1 internal frames
cuModuleLoadData
Triton cuda_utils.loadBinary
PyTorch PythonKernelHolder / dispatcher
```

The active Triton is 3.7.1. Its `get_ptxas(arch)` selects `ptxas-blackwell` for
architecture 100 and later, including this sm_120 GPU. The actual bundled
`ptxas-blackwell --version` reports CUDA 13.1 / V13.1.80, whereas
`/usr/local/cuda-12.9/bin/ptxas --version` reports CUDA 12.9 / V12.9.86.
Thus the earlier CUDA 12.9 environment description did not cover the JIT
compiler selected for the first inference. Source inspection shows that
Triton derives PTX 9.1 versus PTX 8.8 from these compiler versions.

A second GDB diagnostic changed only the two explicit Triton assembler paths
to CUDA 12.9 and used a separate compilation cache. The identical warm-up then
returned HTTP 200 with 512 prompt and 64 completion tokens in 6.866 seconds;
its visible output exactly matched the historical repaired MoE warm-up. This
is strong evidence that the selected JIT toolchain/cache was the immediate
compatibility problem, not proof that every possible 575 workload is fixed.
Because compiler selection and cache isolation changed together, this evidence
does not isolate a stale-cache effect from an unsupported binary-toolchain
effect. The GDB duration is diagnostic, not a performance sample.

Revision 9 fixes both assembler variables to `/usr/local/cuda-12.9/bin/ptxas`
and uses `deps/triton-cache-cuda129` for every MoE correctness/timing process.
It records compiler version and file metadata and reruns all four correctness
cells before accepting timing. No model, expert policy, input, output length,
request count, or performance estimator changes. The already generated tensor
store is reused; every new process still recreates its live CPU/GPU caches.

Preserved local evidence:

- `raw/head-to-head-575/preflight`: initial real failure, numerical tests,
  source/runtime inventory, server log, and cleanup.
- `raw/head-to-head-575/gdb-warmup-01`: native backtrace of the repeated
  compiler-load fault; not a timing sample.
- `raw/head-to-head-575/gdb-warmup-cuda129-01`: successful same-request
  diagnostic, full response, compiler environment, and cleanup.

All three runs returned the GPU to 2 MiB, zero utilization, UVM refcount zero,
and empty struct_ops. No NVIDIA Xid occurred. The original CPU segmentation
fault is nevertheless a real failure and is not hidden by the no-Xid statement.

## Correctness continuation after the UVM connection failure

The CUDA-12.9 preflight completed both full MoE and gpubpf correctness cells.
The next UVM control returned the warm-up and nine correctness responses, then
closed the connection during pass 2, prompt 2. Its log ends after prompt
processing, about five seconds after the previous response. The available
kernel journal has no matching OOM, segmentation fault, or Xid; no new core
was found. Because the original cleanup helper did not record the server's
exit status before cleanup, the process exit cause is not established.

The original failed directory is retained. `--resume-preflight` rechecks the
unchanged runtime inventory and every saved response, output pair, and cleanup
record before retaining a previously complete cell; failed cells run their
entire two-pass workload in new retry directories. Each prior failed summary
is preserved separately. This avoids repeating the complete ten-minute BPF
correctness cell while retaining all failures. It does not skip any of the
four configurations or admit an incomplete correctness pass. Added diagnostics
record failed-request elapsed time and the server exit status before and after
owned cleanup; server arguments, requests, model, and policy are unchanged.
