# Two-context / real GSP timeslice canary

On 2026-09-03 UTC the original two-context path passed real GSP-completion,
numerical and negative-case checks. The BPF path passed those context checks
but **failed the final timeslice gate**: CUDA overwrote both requested values
with 2,048 us before kernel execution. This is a transport/policy correctness
canary, not GPReempt performance or its GDRCopy actuator test.

## Build

```sh
taskset -c 8-15 make -C extension -f gpreempt.mk -j2 context-smoke
cd extension
python3 -m unittest gpreempt_context_smoke_check_test.py
```

The C++ canary uses the actual 575 SDK envelope/channel types and the shared
owned-transport header. Compile-time sizes and offsets match the driver's
32-byte NVOS54 and 536-byte channel payload. Its tiny embedded PTX is JITed on
the GPU only when the canary is explicitly executed.

## Scope

Two different creator threads each create a CUDA context, query its owned GR
TSG, and either call the original narrow SET_TIMESLICE or register the actual
BPF decision. LC requests 1,000,000 us and BE 1 us. The BPF path does not call
the original C setter. Each context executes a 1,024-element integer fill
kernel whose every output is checked; role contexts stay alive while the main
thread runs the ownership and error checks.

The canary checks the following 17 negative cases:

- Eleven bad-ABI/target cases: query payload size, invalid/null query output
  pointer, nonzero query object, bad outer size, unknown control version,
  nonwhitelisted command, bad control payload size, invalid control input
  pointer, forbidden timeslice 2 us, and a child channel supplied as a GR TSG.
- A newly **execed** foreign process opens its own control FD and must be
  denied both query and timeslice control of the parent's live role context.
  This tests process ownership even when both processes run as root. No CUDA
  API is called in the foreign process.
- Two native, unmarked contexts created by the same main thread make QUERY
  explicitly ambiguous (`NV_ERR_INVALID_STATE`, no selected handles). Deleting
  one restores a unique query; deleting both makes the query fail again.
- The two destroyed role TSG handles must reject later timeslice controls.

It also confirms that a separate FD in the **same** process queries the same
owned role handles. Unmarked ambiguity-test contexts must not become BPF
policy targets. A non-TSG channel rejection is not a claim that every possible
CE TSG handle was enumerated. The existing BPF engine filter and counters
separately distinguish GR 1..8 from other engines; its CPU tests cover all GR
values, unknown zero, and ignored CE. Full hint semantics are tested as direct
JIT decisions, but this tiny canary does not execute the two GDR blocking
kernels; the full client and the separate GDRCopy checks still must do that.

## Real GSP observation, not host shadow state

The observer is read-only BPF tracing of the normal Kbuild-instrumented
`nv_gpu_sched_gsp_control_complete` hook added in driver `e7d46fa5`. Core RM
functions are notrace, and an initial direct attachment failed safely; no
notrace restriction is bypassed. The hook is emitted only after a real GSP
RPC wait. The observer restricts records to SET_TIMESLICE `0xa06c0103` and
captures the actual client, object, original input value/size, serialized size,
transport status, firmware status/validity and monotonic completion timestamp.
It never writes policy settings or changes the existing RPC return path.

The 575 source chain is explicit:

1. `src/nvidia/src/kernel/rmapi/rpc_common.c:69` installs
   `rpcRmApiControl_GSP` as physical RMAPI control for a GSP client.
2. `NV_RM_RPC_CONTROL` in `src/nvidia/inc/kernel/vgpu/rpc.h` invokes that
   physical control on the GSP branch.
3. `rpcRmApiControl_GSP` in `src/nvidia/src/kernel/vgpu/rpc.c:10361` builds the
   GSP control message and waits for it. After `_issueRpcAndWait`, it combines
   the transport result with the **actual GSP control handler's returned
   status**, before returning to its caller. The new kernel-open hook captures
   these statuses before the existing error handling and deserialization.

The offline checker requires actual completed RPC records and zero transport
and valid firmware statuses, matching queried role handles and timeslice. It rejects a later
different successful timeslice **before the kernel begins**, so a CUDA-side
overwrite cannot be hidden by an earlier successful BPF request. Event losses,
missing completions, and no-event observer runs are failures. This proves
firmware acceptance/status when run successfully; userspace additionally checks
its final ioctl/RM status. It does **not** measure the
physical scheduling quantum or replace contention performance measurements.
The observer currently checks **timeslice only**, not interleave or a combined
timeslice/interleave transaction.

## Execution prerequisites and commands

The coordinator must first grant the GPU slot, hold **both existing experiment
leases**, admit the updated 575 driver, and pass pre-run safety. Run all arms
with the same privileges; the current BPF bridge reads root-private maps.
Use the established process-group timeout/teardown and post-run safety checks.
Do not run these bare commands as a substitute for that wrapper.

Within the admitted wrapper, start the RPC observer, wait for its ready line,
then run the canary with a hard outer deadline (normally 60 seconds):

The provided wrapper holds both original lease paths (opens existing files
without `O_CREAT`), checks shared pre/post safety, supervises only its owned
process groups, captures all logs, and runs the offline correlation check:

```sh
sudo -n python3 extension/gpreempt_context_smoke_run.py --mode original \
  --output NEW_ORIGINAL_OUTPUT --timeout 60
sudo -n python3 extension/gpreempt_context_smoke_run.py --mode bpf \
  --output NEW_BPF_OUTPUT --timeout 60
```

Its child commands, shown for clarity rather than as a substitute for leases
and supervision, are:

```sh
extension/.output/gpreempt_context_smoke_rpc 120
GPREEMPT_POLICY=original CUDA_CACHE_DISABLE=1 \
  extension/.output/gpreempt_context_smoke
```

For BPF, also start the policy loader described in `gpreempt-policy.md`, wait
for its ready line, and run with `GPREEMPT_POLICY=bpf`, `GPREEMPT_HINT_CODE`, and
`GPREEMPT_BPF_MAPS` set. Preserve stdout/stderr from every process. After the
client exits, stop the policy and observer cleanly, retaining their final
statistics. The healthy BPF run must have exactly two role registrations,
scope entries/leaves, successful requested timeslices, and target destructions,
plus zero policy errors. Additional unmarked contexts do not count as targets.

Then analyze the saved outputs:

```sh
python3 extension/gpreempt_context_smoke_check.py \
  --mode original --client-log CLIENT_LOG --rpc-log RPC_LOG
python3 extension/gpreempt_context_smoke_check.py \
  --mode bpf --client-log CLIENT_LOG --rpc-log RPC_LOG --policy-log POLICY_LOG
```

The analyzer's seven CPU unit tests use explicitly synthetic records only; they
exercise missing/failed RPCs, wrong ownership, zero BPF engagement, dropped
events, and a later timeslice overwrite. They are not GPU measurements.
Finally require zero UVM references, no Xid, empty struct_ops, and removal of
only the loader's own pins; release both leases only after cleanup. A canary
or parser failure keeps its raw evidence and does not authorize performance.

## Actual canary record (2026-09-03 UTC)

Raw evidence is under `workloads/xsched/raw/`:

| Directory | Result |
| --- | --- |
| `gpreempt-context-original-20260903-0036` | Direct core-RM probe rejected as notrace before CUDA; safe failure. |
| `gpreempt-context-original-20260903-0041` | Explicit observer-off original context/control/numerical pass; no direct firmware evidence. |
| `gpreempt-context-bpf-20260903-0042` | Context/numerical pass, policy fails two shadow mismatches; grpID-only map could confuse GR and CE runlists. |
| `gpreempt-context-original-e7d46fa5-20260903-0106` | Full canary pass: each role default 2,048 us then explicit original 1,000,000 / 1 us, real GSP statuses zero. |
| `gpreempt-context-bpf-e7d46fa5-20260903-0107` | Corrected composite identity: two clean GR registrations/destructions, two matching binds, zero policy errors. **Fails firmware gate:** requested 1,000,000 / 1 us is subsequently overwritten by CUDA's 2,048 us for both roles. |

Each executed client checks all 2,048 outputs and all 17 negative cases. All
five attempts finish with zero UVM references, empty struct_ops and no Xid.
The failed BPF result is retained; successful bind shadow counters do not
override contradictory firmware evidence. Init-only timeslice actuation needs
a later validated BPF decision/actuator path before this arm can be timed as
an equivalent policy. No physical quantum, interleave, full hint actuator or
performance result is claimed here.

`--rpc-observer off` (wrapper) / `--context-only` (offline checker) is an explicit
diagnostic mode only; it marks firmware status unobserved. The default still
requires the real completion observer and the last-value-before-kernel check.
