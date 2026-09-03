# Q2 invalid-prefetch attempt01: observer loading failed

**Result: failed before target release; zero completed functional controls and
no live invalid-action/fallback evidence.** The actual 575.57.08 loader exited
1 during the first (`native`) cell; the coordinator also terminated with exit 1.
All original logs are retained. This report only reads the closed attempt.

## Failure and execution scope

[loader.jsonl](native/loader.jsonl) reports `range_enter` loading with `-EINVAL`:

```text
The function uvm_perf_prefetch_bitmap_tree_iter_get_range return type STRUCT is unsupported.
processed 0 insns
```

This is rejection of the observation program's attachment target prototype,
not rejection of action 99, an invalid region, or an executed resource transition.
Existing BTF/function-entry instrumentation and a successful CPU build were
insufficient to establish live fentry support for this structure-returning
function. This failure also does not establish admission of all other intended
attachments.

The loader's object-load failure occurs before its attachment loop and before
any struct_ops attachment (`extension/revision-prefetch/loader.c:144`). It never
emitted `ready` or `final_metrics`. The
[target log](native/target.log) records PID **701397** at its post-initialization
pause. CUDA context/allocation and CPU initialization occurred, but the owned
target was not released to the fault-stream kernel or full correctness readback.

| Planned control | Actual status |
| --- | --- |
| Native, observers only | Loader admission attempted; target never released |
| Legal empty-region BYPASS | Not started |
| Invalid action 99 with legal empty region | Not started |

[execution.json](native/execution.json) has `complete=false`, no `released_ns`,
no validated target result and no policy metrics. There is no `target.json`,
`bypass/`, `invalid99/`, or campaign `summary.json`. Therefore there is no
131,072-value correctness result, callback count, native-traversal observation,
or compute-mask result from this attempt.

## Cleanup and safety evidence

The recorded loader group is **701416**. The frozen runner stops the target
group first, then the loader, before cleaning monitors; it requires both a
reaped leader and an empty owned group. The retained record has no
`cleanup_failure` or `safety_error`. The loader's exit 1 is recorded by the
readiness failure. The target's individual exit code was **not serialized**;
do not report target exit 0 or a successful numerical run. Target/loader cleanup
is supported by the completed cleanup path together with the final empty state,
not a separately recorded per-target exit-status audit.
The coordinator additionally reported a post-run
`ps -p 701393,701396,701397,701416 -o pid,pgid,comm` check that returned only
its header: all four recorded PIDs were gone. Its separate `bpftool struct_ops
show` check was empty. These additional checks were communicated to this audit,
not serialized as individual exit codes in the cell JSON.

Both monitors were alive before shutdown and both cleanup attempts are retained:

| Owned monitor PID | Exit status | Cleanup error |
| --- | ---: | --- |
| 701396 (kernel journal follower) | −2 (SIGINT) | none |
| 701393 (GPU telemetry) | 0 | none |

The before/after safety snapshots both show 575.57.08, 400 W, **no compute
clients, zero UVM references, and no struct_ops maps/links**. Final GPU memory is
2 MiB and utilization is 0%. Xid and recorded abnormal-kernel arrays are empty;
[kernel-follow.log](native/kernel-follow.log) is empty.
[Telemetry](native/gpu-telemetry.csv) contains 13 samples, reaching 46°C and
508 MiB during initialization, with no recorded throttle reason. These are
safety observations, not performance measurements or proof of fault-free GPU
execution in general. The
[launch log](../prefetch-invalid-575-01-launch/coordinator.log) shows both GDM
and nvidia-persistenced restored to `active` after the failure.

The pre-run path/size/mtime inventory is present. The successful-cell
`files_after` check was not reached, so this failed attempt is not evidence of
a completed before/after artifact-stability check.

## Next step: reviewed read-only diagnostic seam

Do not retry this unchanged fixture or bypass the return-type gate with offsets,
prototype misrepresentation, or relaxed tracing restrictions. No C/BPF or driver
source was changed during this result audit.

Prepare a separate, independently reviewed driver diagnostic with a **void
return and a pointer to a read-only, driver-filled context**, following the
existing Kbuild noinline/barrier diagnostic pattern. Emit one phase after
initial-effect selection and another after actual native traversal completes;
copy the action, validation/effect values, legal bounds and selected region,
plus sufficient invocation-correlation fields. It must neither dispatch a
policy nor change the validation, actuator, or return status. The minimal driver
review scope is `kernel-open/nvidia-uvm/uvm_bpf_struct_ops.h`,
`uvm_bpf_struct_ops.c`, and `uvm_perf_prefetch.c`; no such patch is made here.

Only after independent review, CPU checks and a separately admitted build/load
window can new native/BYPASS/invalid99 controls run into a **new directory**.
The intended compute-mask observation remains pre-filter, not the final hint
or DMA mask. The Q2 live-prefetch fallback requirement remains **open**.
