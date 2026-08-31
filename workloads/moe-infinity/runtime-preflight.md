# MoE-Infinity runtime preflight evidence

Date: 2026-08-31

## Attempt 1: closed before correctness

The first real preflight is preserved locally at
`raw/correctness-preflight-610-20260831-01`. It used the frozen first schedule
order, so `moe_infinity_075` ran first. Before launch, full admission passed:

- no foreign compute process, 15 MiB residual GPU memory;
- NVIDIA 610.43.02 and no pre-existing struct_ops map or link;
- live custom UVM BTF with the exact six-member `gpu_mem_ops` and all three
  kfuncs used by the combined policy;
- all 15 HF weight shards, seven metadata files, and the 63.4 GB GGUF matched
  the expected inventory and sizes;
- the workspace NVMe, required executables, patches, and instrumentation files
  were present at their recorded paths.

MoE-Infinity loaded the exact checkpoint, created a roughly 61 GiB expert
store across seven logged storage partitions, moved the dense and sparse
parameters, and reached its healthy API. The first excluded 512-token warm-up
then terminated the native execution path with:

```text
batch_size should be (0, 256 ] , but got 353
```

The failure comes from the pinned upstream source
`core/parallel/expert_module.cpp`: `kMaxTokens` is 256 and sizes all reusable
expert buffers; the warm-up routed 353 prefill rows to one expert. No warm-up
response completed, and no correctness, engagement, O_DIRECT, or timing result
was produced.

The approved protocol freezes a 512-token input and upstream source, permitting
only the already disclosed load-only counter getter. Increasing `kMaxTokens`,
adding chunked prefill, or shortening the workload would therefore create a
different experiment rather than repair this execution. Repeating the same
frozen configuration cannot change this deterministic capacity failure. The
MoE-Infinity protocol is closed after this attempt and the revision's MoE axis
is routed to its named DeepSpeed ZeRO-Inference or PowerInfer fallback.

## Cleanup observation

The native fatal path left the owned strace/server process group stuck during
exit. The runner's SIGINT and SIGTERM budgets expired. After checking that PGID
209559 contained only the exact strace and revision-server commands from this
attempt, it was killed as a group. Post-cleanup checks showed no GPU compute
process, 15 MiB residual memory, no struct_ops map/link, and UVM reference count
zero. No foreign process was signaled.

The cleanup helper now escalates from SIGINT to SIGTERM and finally SIGKILL only
for the already verified owned process group. This execution-safety repair does
not alter any scientific configuration. The failed raw directory remains local
and is not a paper result.

## Repaired protocol attempt 1: oversized route completed, harness rejected

The independently approved repaired protocol's first attempt is preserved at
`raw/repaired-preflight/attempt-01`. Admission and the standalone numerical
gate passed, and `moe_infinity_075` again ran first. The exact GPT-OSS-120B
model and topology reached a healthy API with the active repaired Python 3.12
extensions.

Unlike the unmodified artifact, the identical frozen 512-token warm-up crossed
the previously failing oversized expert route and returned HTTP 200 with
exactly 512 prompt tokens and 64 completion tokens. The server log contains no
256-row fatal, CUDA error, or traceback. This is direct evidence that the
disclosed row-chunking repair reaches and completes the original failure path;
it is not a complete correctness or performance sample.

Immediately after the excluded warm-up, the CPU-affinity gate rejected the
owned process tree. The Python server had affinity CPU 0--7 as frozen, but the
outer `strace` process had CPU 0--23 because the launcher placed `strace`
outside the recorded `taskset` command. The gate therefore failed before the
two correctness passes:

```text
owned process tree escaped CPU 0-7
```

The runner recorded `status=failed` and `retry_allowed=false`, stopped the
owned process group, and returned the GPU and struct_ops state to idle/empty.
No timing ran. An unchanged attempt 2 is prohibited. The next proposal may
only move `taskset` outside the tracing wrapper so every owned process inherits
the frozen CPU set; it must retain attempt 1, increment the protocol revision,
count the next launch as attempt 2, and receive independent review first.

## Repaired protocol attempt 2: full smoke executed, output race detected

The independently approved launcher-only revision ran at
`raw/repaired-preflight/attempt-02`. The recorded command placed
`taskset -c 0-7` outside `strace`; both tracer and Python server retained the
frozen affinity, so attempt 1's harness defect did not recur.

The repaired MoE configuration completed the excluded 512+64-token warm-up and
both complete correctness passes: 16 further requests each returned HTTP 200
with exactly 512 prompt tokens and 64 completion tokens. The two output texts
matched for prompts 5 and 7 but differed for the other six prompts. The
unchanged exact-output gate therefore rejected the configuration before
engagement acceptance or the remaining three configurations:

```text
non-deterministic smoke output for prompt 1
```

The requests used `temperature=0.0` and MoE-Infinity's sampler took its greedy
`argmax` path, so sampling randomness does not explain the divergence. Source
inspection instead found that four expert workers enqueue in-place additions
to shared `final_hidden_states_` from separate CUDA streams. The host mutex
serializes enqueue calls but neither waits for each GPU write nor imposes a
fixed expert reduction order. This is a concrete upstream accumulation race,
not a reason to weaken the frozen correctness oracle.

Attempt 2 is preserved with `status=failed` and `retry_allowed=false`; cleanup
returned the GPU and struct_ops state to idle/empty, and no timing ran. An
unchanged attempt 3 is prohibited. Any final attempt requires a disclosed
deterministic accumulation repair, a GPU numerical/determinism gate, rebuild,
read-only admission, and independent review while preserving both earlier
attempts and every scientific setting.

## Repaired protocol attempt 3: deterministic execution passed, I/O gate rejected metadata

The follow-up-approved deterministic repair ran at
`raw/repaired-preflight/attempt-03`. It preserved four expert compute threads
while binding each worker's mask/input and forward path to its external CUDA
stream, checking output completion, propagating worker failures, and reducing
completed outputs in expert-index order.

The exact GPT-OSS-120B model again reached its healthy API. The excluded
512+64-token warm-up completed, followed by both complete eight-prompt smoke
passes. All 16 requests returned HTTP 200 with exactly 512 prompt tokens and
64 completion tokens. Every prompt's two greedy output strings matched
exactly. The server log contains no 256-row fatal, worker failure, CUDA error,
or traceback. Because execution reached the final storage-open check, the
preceding gates also accepted CPU affinity, 1,024 generated smoke tokens,
engine steps, expert-cache activity, 128 KV-cache blocks, and positive process
read bytes.

The preflight nevertheless failed closed in `validate_moe_odirect()`. The
tracer recorded seven successful `O_DIRECT` opens for the seven
`archer_param_*` expert-store partitions, but the gate required every open
under the offload root to use `O_DIRECT`. It therefore rejected 28,119 ordinary
metadata opens of `archer_index`, as well as initial metadata/partition creation
opens. The first reported error was:

```text
expert-store open without successful O_DIRECT: .../archer_index ... O_WRONLY|O_CREAT|O_TRUNC
```

This is a harness classification defect: `archer_index` is metadata rather
than an expert tensor partition. It does not invalidate the observed exact
model execution, but the approved protocol required every gate to pass, so the
attempt remains `status=failed`, `retry_allowed=false` and is not promoted to a
complete preflight. The fixed three-attempt budget is exhausted; no fourth
attempt or MoE timing run is authorized. Cleanup returned the GPU and
struct_ops state to idle/empty. The raw directory is preserved unchanged.
