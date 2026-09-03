# Two-context / real GSP timeslice canary

Preparation completed on 2026-09-03 UTC; **no GPU run or attachment is implied
by these build and parser tests**. This is a transport/policy correctness
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

The observer is read-only BPF tracing. It records entry/return from
`rpcRmApiControl_GSP`, restricted to SET_TIMESLICE `0xa06c0103`, and counts
`_issueRpcAndWait` entry/return while that control is active. It records the
actual client, TSG, input timeslice, parameters size, wait count, transport
status, final control status, and timestamps. It neither changes the driver
nor writes policy settings.

The 575 source chain is explicit:

1. `src/nvidia/src/kernel/rmapi/rpc_common.c:69` installs
   `rpcRmApiControl_GSP` as physical RMAPI control for a GSP client.
2. `NV_RM_RPC_CONTROL` in `src/nvidia/inc/kernel/vgpu/rpc.h` invokes that
   physical control on the GSP branch.
3. `rpcRmApiControl_GSP` in `src/nvidia/src/kernel/vgpu/rpc.c:10361` builds the
   GSP control message and waits for it. After `_issueRpcAndWait`, it combines
   the transport result with the **actual GSP control handler's returned
   status**, before returning to its caller. Both traced functions exist in
   the built 575 module symbol inventory.

The offline checker requires one observed wait/completion and zero statuses,
matching the queried role handles and timeslice. It also rejects a later
different successful timeslice **before the kernel begins**, so a CUDA-side
overwrite cannot be hidden by an earlier successful BPF request. Event losses,
missing completions, and no-event observer runs are failures. This proves
firmware acceptance/status when run successfully; it does **not** measure the
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

The analyzer's six CPU unit tests use explicitly synthetic records only; they
exercise missing/failed RPCs, wrong ownership, zero BPF engagement, dropped
events, and a later timeslice overwrite. They are not GPU measurements.
Finally require zero UVM references, no Xid, empty struct_ops, and removal of
only the loader's own pins; release both leases only after cleanup. A canary
or parser failure keeps its raw evidence and does not authorize performance.
