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

## Revision 5 continuation: MoE revalidated, gpubpf warm-up exposed overload

The reviewed read-only revision 5 classifier revalidated the preserved attempt
3 without launching MoE-Infinity again. The separate
`revalidation-result.json` passed all 17 saved response gates (one warm-up and
two eight-prompt passes), exact pairwise output equality, MoE-used runtime
continuity, server-log validation, and the exact seven-partition storage-open
grammar. The original revision 4 `preflight-result.json` remains failed and
unchanged. The observed deployment is one-time buffered NVMe hydration followed
by an activation-aware CPU expert offload/cache; it is not steady-state direct
NVMe I/O.

The fixed continuation then attempted only the three missing cells in their
original relative order. Three setup failures occurred before any warm-up or
request and are preserved separately: the admission checker named superseded
direct-list kfuncs, Linux 7.1 did not enumerate the otherwise map/PID-owned
struct_ops link, and llama-server exposed two owned UVM descriptors while the
old monitor path required exactly one. The corrected runner checks the safe
`bpf_gpu_request_reorder` transition API, proves map/PID ownership when the
kernel does not enumerate a link, and probes every owned UVM descriptor. It
records a descriptor as non-trackable only when the started monitor receives
the exact driver response `NV_STATUS 22`; all other monitor setup failures
remain fail-closed. Fifty offline tests and independent review accept these
repairs.

The next launch crossed all setup gates and reached the real 512-token gpubpf
warm-up. The exact GPT-OSS-120B model loaded, processed all 512 prompt tokens,
and then failed at CUDA synchronization before returning a response. The
server recorded `illegal memory access`; the kernel recorded Xid 109 context
switch timeout followed by Xid 31 MMU fault. During the warm-up, the combined
host-stride/LFU policy accumulated 14,871,351 page-fault callbacks, 1,993,700
LFU access callbacks, 685,225 activations, and 669,400 eviction-prepare calls.
This is a real policy/mechanism feasibility failure rather than another
preflight classifier defect: no correctness or timing sample was accepted, and
an unchanged retry is not justified. The owned server, monitors, and policy
were stopped; the GPU returned to idle and no struct_ops state remained.

## Revision 6 sampled-LFU canary: failed, no further tuning

The independently approved, single allowed sampled-LFU canary ran after a clean
custom-UVM reload. It retained the host-stride policy and counted every LFU
access, but performed the expensive LFU frequency update and tail-reorder
request only once per 256 callbacks on each CPU. The emitted final counters
prove the intended reduction: 1,994,365 LFU access callbacks produced 7,789
sampled updates and 7,789 reorder requests. The policy also observed 17,017,111
page-fault callbacks, 855,287 activations, and 839,464 eviction prepares.

Despite reducing the access-path frequency work by roughly 256x, the exact
512-token warm-up again reached `prompt done` and then returned no response.
CUDA synchronization reported an illegal memory access. The kernel again
recorded Xid 109 context-switch timeout first and Xid 31 MMU fault second. No
warm-up response, correctness sample, or performance sample was accepted. The
failure result explicitly keeps `full_correctness_authorized=false`.

The pre-registered stopping rule now applies: the sampling ratio will not be
tuned, full gpubpf correctness and timing remain unauthorized, and the exact
and sampled failures are both retained. Cleanup succeeded: GPU memory returned
to 15 MiB, utilization to zero, the UVM refcount to zero, and the struct_ops
inventory to empty. The remaining UVM and N-CMoE control correctness cells may
still be collected, but cannot promote this experiment to a complete
four-configuration result.

## Revision 6 control continuation: fixed seed remained nondeterministic

The frozen control-only continuation next launched `llama_uvm` with one server
slot, greedy decoding, `seed=42`, and EOS ignored so every response had the
required 64 completion tokens. The warm-up and both complete eight-prompt
passes finished, yielding all 16 requested responses. Exact two-pass equality
held for prompts 2, 3, 4, 6, 7, and 8, but failed for prompts 1 and 5. For
prompt 1, the first differing words were `issue` and `problem`; prompt 5
diverged from its first character. Thus fixing the request seed did not remove
the control-path output nondeterminism under the frozen runtime settings. The
evidence does not isolate whether the cause is a GPU numerical effect or
another llama.cpp runtime effect, so it is recorded only as observed output
nondeterminism.

The exact-output gate rejected `llama_uvm`, and the fail-closed protocol did
not launch `llama_ncmoe32`. The runner also could not establish continuity of
the Xid history; an empty `new_xids` field is therefore not evidence that no
new event occurred, and this gate remains failed. Cleanup returned GPU memory
to 15 MiB and utilization to zero, the UVM reference count to zero, and the
struct_ops inventory to empty.

No further seed, sampling, oracle, or harness tuning is authorized. The
control continuation is a negative correctness result, not a performance
sample: N-CMoE was not run, gpubpf remains infeasible, the four-configuration
preflight is incomplete, and timing remains unauthorized.

## Revision 7 repair: sampled-LFU canary and controls passed

The sampled-LFU failure had two independent causes. First, every activation
updated a shared frequency HASH map and unconditionally requested a list-head
reorder; sampling only the access hook therefore left the activation hot path
under severe contention. The repaired policy uses a bounded 16,384-entry
per-CPU approximate counter array, makes activation observation-only with a
DEFAULT return, and retains deterministic 1/256 access sampling with sampled
tail reorder. Second, an intervening llama build had `GGML_CUDA=OFF`; the
CUDA/no-VMM server was rebuilt and the runner now requires its log to identify
the RTX 5090 as CUDA0 before inference.

The final canary is preserved at
`raw/repaired-preflight/sampled-lfu-percpu-canary-06`. Its unchanged 512+64
request completed in 45.026 seconds. The policy observed 5,198,943 LFU access
callbacks, 20,307 sampled updates, 20,307 sampled reorder requests, 50,342
activations, and 34,533 eviction-prepare callbacks. The sample count satisfies
the per-CPU 1/256 rounding gate. The response, single-slot identity, CUDA
identity, policy cleanup, UVM refcount, struct_ops inventory, 400 W service,
journal, and Xid gates all passed; no new Xid was recorded.

The event-monitor check exposed a separate evidence-source mismatch rather
than a policy failure. The only event-trackable owned descriptor reported zero
type-14 events, as it did in preserved earlier runs, while the other descriptor
returned the already classified `NV_STATUS 22`. The canary now uses the direct
in-kernel `gpu_evict_prepare` callback counter for feasibility engagement. The
full timing protocol's completed-eviction requirement is unchanged and remains
out of scope.

The final repaired controls passed under
`raw/repaired-preflight/controls-single-slot-04`. Explicit `--kv-unified`
prevents llama-server build 7102 from rewriting `--parallel 1` to four slots.
The final run observed one slot for both `llama_uvm` and `llama_ncmoe32`;
each completed its warm-up and two full eight-prompt passes and reproduced all
eight outputs exactly between passes. Both per-server pre/post safety gates
passed with no Xid or kernel abnormality. Prompts 5, 7, and 8 differed across
configurations and are recorded diagnostically under the predeclared rule.

The predecessor `controls-single-slot-03` run had likewise observed one slot for both
`llama_uvm` and `llama_ncmoe32`; each completed its warm-up and two full
eight-prompt passes and reproduced every output exactly between passes. Its
separate pre/post safety gates passed with no Xid or kernel abnormality. Three
prompts differed across configurations despite greedy decoding. That directory
remains failed because the criterion was clarified only after observing it.

Both requested repairs are complete. Neither result creates a complete
four-configuration preflight or authorizes performance timing.
