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
