# RTX 5090 scheduler-init native commit-path candidate

Status: **CPU/design only; no module was built, loaded, or replaced, no BPF
program was attached, and no GPU workload was run.** This candidate extends the
already completed seven-fixture compile and production-validator CPU record in
[`extension/revision_init.md`](../../../../extension/revision_init.md). It does
not change or reinterpret any existing result.

## Admission and hypothesis

Reviewer F asks whether the safe mechanism can implement existing policies and
whether generality changes their behavior; Reviewers B/F also ask what happens
to invalid or pathological transitions. The remaining load-bearing uncertainty
for scheduler initialization is not whether the recorder and validator work in
a CPU test. It is whether their decisions reach, or are withheld from, the
native 575 scheduling setters and the completed GSP control path in a real CUDA
constructor.

Hypothesis: on the RTX 5090/575.57.08/6.15.11 path, an accepted typed BPF
request is committed once per requested field through the real host setter and
one successful post-wait GSP control; absent, out-of-range, or conflicting
fields retain the constructor defaults and issue no policy setter/control;
valid and invalid fields in one callback are decided independently. Every
constructor and the CUDA output remain successful.

The contradictory outcome is any accepted field without exactly one matching
successful host setter and GSP completion, any rejected/default field with a
policy setter/control, a final field mismatch, a failed constructor, or a CUDA
error/mismatch. Such an outcome bounds or contradicts this scheduler-init
mechanism path; it does not by itself challenge every gpubpf transition type.

Role: **supporting Q2 evidence**. It adds actual constructor commit/rejection
evidence beyond the existing verifier-load and CPU-validator results. It is a
functional experiment, not a latency or throughput comparison.

## Frozen matrix

Run each row in two fresh processes and use block-major order, for 16 cells.
Each process runs the existing `workloads/bpftime-device-smoke/vector` target:
8 launches and 32,768 checked values. No cell reuses a CUDA context, BPF map,
link, pin directory, diagnostic process, or output directory.

| Cell | Policy state | Timeslice decision / native calls | Interleave decision / native calls | Final fields |
| --- | --- | --- | --- | --- |
| `native_unattached` | no scheduler struct_ops | DEFAULT / 0 | DEFAULT / 0 | D / MEDIUM |
| `bpf_no_request` | attached, requests nothing | DEFAULT / 0 | DEFAULT / 0 | D / MEDIUM |
| `bpf_legal` | D; LOW | APPLY / 1 | APPLY / 1 | D / LOW |
| `bpf_duplicate` | D,D; LOW,LOW | APPLY / 1 | APPLY / 1 | D / LOW |
| `bpf_invalid_interleave` | none; 3 | DEFAULT / 0 | REJECT_RANGE / 0 | D / MEDIUM |
| `bpf_conflict` | D,D^1,D; LOW,HIGH,LOW | CONFLICT / 0 | CONFLICT / 0 | D / MEDIUM |
| `bpf_independent_interleave` | D; 3 | APPLY / 1 | REJECT_RANGE / 0 | D / MEDIUM |
| `bpf_independent_timeslice` | D,D^1; LOW | CONFLICT / 0 | APPLY / 1 | D / LOW |

`D` is the actual constructor default, and `D^1` is the existing overflow-safe
different value. MEDIUM=1 and LOW=0. A repeated request is recorded as
APPLY,REPEAT but validates once as APPLY. A conflict rejects the whole field.
The independent rows are required controls: rejecting one field must not veto
the other. Timeslice `D` intentionally exercises the real actuator without
inventing a new quantum; the 575 minimum-timeslice HAL returns zero, so zero is
not a valid negative test here.

The three reviewer-facing groups are: native/no-request controls (first two
rows), accepted typed-BPF behavior (legal and duplicate), and invalid/conflict
behavior (invalid and conflict), with the two independent-field rows as the
causal control. All eight rows test the same commit-path hypothesis and are one
experiment.

## Smallest implementation

### 575 driver source

1. Add `kernel-open/common/inc/nv-gpu-sched-init-diagnostic.h`: one fixed,
   address-free context and phase/field enums. It carries ABI size/version,
   `hClient`, `hResource`, GPU/subdevice, `grpID`, `runlistId`, `engineType`,
   the already allocated `tsgUniqueId` as constructor epoch, immutable defaults,
   minimum timeslice, both request attempted/conflict/value triples, both
   validation results/effective values, the observed native-call status and
   post-call field, final constructor status/fields, and final-snapshot-valid.
   It exports no RM pointer and adds no policy input.
2. Add one `void nv_gpu_sched_init_diagnostic(const ... *ctx)` declaration to
   both scheduler-hook headers and one `noinline` barrier-only implementation
   beside `nv_gpu_sched_gsp_control_complete` in
   `kernel-open/nvidia/nv-gpu-sched-hooks.c`. It must never dispatch struct_ops,
   mutate RM state, replace a status, or relax a `notrace` function.
3. Instrument only `kchangrpapiConstruct_IMPL` in
   `src/nvidia/src/kernel/gpu/fifo/kernel_channel_group_api.c`. Emit:
   `VALIDATED` immediately after the production validator;
   `NATIVE_RETURN/TIMESLICE` and `NATIVE_RETURN/INTERLEAVE` immediately after
   the existing setter calls and before their existing error branches; and one
   `CONSTRUCTOR_RETURN` for every observation activated at task-init. Snapshot
   final CPU fields while the object is valid and under its existing lock,
   before failure cleanup can destroy it. Preserve the setters, their order,
   the timeslice `NV_TRUE` argument, error status, and all original jumps.
   Explicitly cover the post-`ctxBufPoolReserve` lock-reacquire early return;
   never dereference the group after cleanup. Constructors that exit before
   task-init remain outside this targeted matrix.
4. Add a CPU-only ABI/event-construction test and Makefile under
   `kernel-open/tests/sched-init-diagnostic/`. It checks exact field offsets and
   sizes shared with the observer, all phases/fields, `tsgUniqueId` use, status
   preservation, and that diagnostic emission has no return channel. Extend
   existing transition-validator tests only if a production helper is factored;
   do not add a shadow validator.

No UVM source change is needed. The existing GSP completion hook in
`rpcRmApiControl_GSP` already fires after the real wait for commands
`0xa06c0103` (timeslice) and `0xa06c0107` (interleave), carrying handles,
input, transport status, and GSP status.

### Experiment repository

1. Add `extension/revision_init_trace.{h,bpf.c,c}`. One observer attaches only
   to the new diagnostic hook and the existing GSP-completion hook, filters the
   exact gated target TGID, emits bounded ring records, and reports observed,
   emitted, read-error, and drop counters per source. It filters for both
   scheduling commands **before** ring output rather than retaining the
   timeslice-only filter in the current GPreempt smoke observer; unrelated
   controls never consume ring capacity.
2. Add `extension/revision_init_loader.c`, a generic libbpf-object loader for
   the seven already compiled `revision_init_*.bpf.o` fixtures. Before load it
   sets the fixture's `target_tgid` rodata to the already spawned target PID,
   attaches only that object's scheduler struct_ops map, prints a ready record,
   then on bounded shutdown dumps every request-map entry and all four stats.
   It creates no persistent global pin; if an owned pin is needed for process
   coordination, it is fresh, private, and removed by the same process.
3. Add `extension/revision-init/run_live.py` plus pure-CPU tests. Reuse the
   existing owned-process-group cleanup, two shared leases, safety snapshots,
   telemetry and exec-gate patterns. The runner owns the observer, optional
   fixture loader, target, logs and fresh outputs. It performs the matrix and
   joins policy, native-diagnostic and GSP records; it does not implement a new
   transition model.
4. Extend only the explicit opt-in Makefile targets for the observer, generic
   loader and offline tests. Do not add this live matrix to the default build.

The target needs a pre-CUDA pause so the observer and fixture can be configured
for its stable TGID before CUDA constructs a channel group. Prefer a small
optional stdin gate in a private copy/wrapper of the existing vector target,
while preserving its kernel, launch count and full numerical oracle. Do not use
process-name matching, attach after CUDA initialization, or treat a successful
launch as evidence that the setter ran.

## Per-cell evidence gates

All gates are mandatory; failure is retained and the cell is not retried under
a changed oracle.

1. Target: exact PID/PGID ownership, observer and optional fixture ready before
   release, exit zero, exactly 8 launches, 32,768 values, zero mismatches.
2. Policy: native has no struct_ops link. Every BPF cell has exactly one owned
   scheduler struct_ops link. `INIT_SEEN == INIT_RECORDED == request-map
   entries > 0`, `INIT_RECORD_ERROR=0`, every record is complete and belongs
   to the target, and recorder returns exactly match the frozen row.
3. Diagnostic framing: for every target constructor epoch, exactly one
   VALIDATED and one CONSTRUCTOR_RETURN occur in order. Native-return phases
   exactly match the expected accepted fields. No duplicate, missing,
   unreadable, out-of-order or unmatched event is allowed. Observer read errors
   and ring drops are zero.
4. Validation: defaults in the policy record equal diagnostic defaults;
   attempted/conflict/value and validation status/effective values match the
   row. Every native-return status is `NV_OK`, and its post-call CPU value is
   the validated value. CONSTRUCTOR_RETURN is `NV_OK`, snapshot-valid, and has
   the expected final fields.
5. GSP: within the same target PID/TID and hClient/hResource constructor
   interval, each expected native field has exactly one matching command with
   valid input, expected value, zero transport status, valid GSP status and
   zero GSP status. Rejected/default fields have none. GSP events outside a
   matched target interval or later CUDA controls do not count.
6. Cleanup: stop and verify the target's whole owned process group before
   stopping its loader/observer; then verify every recorded owned link/map/pin
   is absent. Continuous kernel/GPU monitors remain alive through cleanup, and
   their bounded queries show no foreign compute PID.
7. Block: both fresh repetitions of all eight rows pass with no kernel warning,
   BUG/Oops/panic/Xid, GPU reset, service failure, leaked module reference or
   residual BPF object. A failed cell stops the campaign after recovery.

## Full-core lifecycle and recovery boundary

This experiment requires a separately reviewed core-module build and lifecycle
coordinator; adapting the UVM-only coordinator without these extra checks is
unsafe. Root is the only executor.

Before mutation, hold the existing GPU and struct_ops lock **inodes** without
creating or replacing them. Require the pinned kernel/driver/GPU, no compute
process, no local interactive user session, no existing struct_ops policy, and
capture the active state of GDM and nvidia-persistenced. Capture the complete
loaded NVIDIA module **subset**, dependency order, versions, parameters, BTF
ABI, device-node set and ordinary file paths/sizes. Require a fresh candidate
stage and the exact known-good 849ea75d restore stage to contain an artifact
for every module in that captured subset. Do not require, load or restore a
module that was absent initially, and do not overwrite either stage.

Stop only services that were active. Recheck holders and compute clients, then
remove exactly the captured subset in reverse dependency order (for this
currently observed host: UVM, DRM, modeset, core). Never force removal. Load
exactly that subset in forward dependency order by explicit candidate paths
with the captured parameters. Validate exact module-set equality plus every
module's version/BTF/parameters, the expected device-node set, 400 W power
limit, GPU health and idle state before restarting originally active services
and before any cell. The candidate BTF must expose both diagnostic hooks with
exact address-free layouts and the existing typed kfunc/struct_ops ABI. Run a
native numerical preflight before the matrix.

Recovery is physical and unconditional even if result recording fails or a
signal arrives: stop all owned groups, detach only owned links, stop the two
services if active, verify zero holders, remove exactly the candidate subset in
reverse dependency order, then insert exactly the captured baseline subset in
forward dependency order from the exact old staged paths with captured
parameters. Validate that the restored module set and device-node set equal the
pre-mutation snapshots, then validate every known-good module, GPU health, idle
state and 400 W limit **before** restarting only the services originally
active. Withhold services and report a hard recovery failure if that exact old
runtime cannot be proven restored; do not fall back to `modprobe`, `depmod`,
force unload, reboot, broad kills, or a different module. Mask/queue
SIGINT/SIGTERM during physical recovery and publish `complete: true` only after
the leases are closed and a second post-restore snapshot passes.

Required pre-live gates are: driver build and CPU tests; semantic patch review
of the actual constructor instrumentation (the CPU ABI test alone cannot cover
its placement); fresh-stage inspection; lifecycle failure-injection tests for
every unload, insert, validation, service and record-write failure; a root
read-only preflight; and a real native numerical preflight. None is a paper
result.

## Supported claim if all 16 cells pass

> On one RTX 5090 running Linux 6.15.11 and NVIDIA 575.57.08, gpubpf's typed
> scheduler-init callback reproduced the tested APPLY, DEFAULT, REPEAT,
> CONFLICT, and RANGE request semantics in fresh CUDA constructors: accepted
> fields reached one successful native setter and post-wait GSP control, while
> absent, invalid, and conflicting fields issued neither; mixed valid/invalid
> fields committed independently, and all 16 bounded controls completed with
> correct CUDA output.

This supports the concrete scheduler-init transition mechanism only. It does
not establish performance, physical scheduling-quantum behavior, verifier
rejection (the invalid values are verifier-accepted but transition-rejected),
universal no-op semantics, all constructors, all GPUs/drivers, or recovery from
a machine crash.
