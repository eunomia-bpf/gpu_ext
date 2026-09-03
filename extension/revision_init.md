# Scheduler-init transition candidate (575)

2026-09-03: the seven request fixtures **compile**, and the CPU test of the
production-shared 575 recorder/validator **passes 28 cases / 705 assertions**.
[Original logs and scope](../docs/experiment/revision-safety/init-matrix-cpu-575-01/execution.md)
retain both the initial macro-name build failure and its corrected build.
**No BPF object was loaded or attached; no GPU workload ran for this test.**
No driver or existing fixture/tracer/runner was modified. There is deliberately no runnable live
matrix yet: the current module lacks the native-init observation points below.
These files do not change the completed **7/4/3/7 load-only** or **12-case,
145-assertion CPU** results in
[the readiness record](../docs/experiment/revision-safety/driver-test-readiness.md).

## Request matrix and exact expectations

`revision_init_requests.h` supplies the same request sequences to the seven
`revision_init_*.bpf.c` wrappers and `revision_init_cpu_test.c`. The CPU
test calls the **actual 575 shared recorder and validator**, not a replacement
model. Its wrappers do not invoke or mock native scheduling setters. It covers
seven cases at four synthetic defaults (`0`, `1`, `1024`, `UINT64_MAX`); the
successful CPU run is not native transition evidence.

Let `D` be the callback's actual default timeslice and `I` its default
interleave, expected to be MEDIUM=1 in this constructor. `D^1` differs from `D`
without overflow. Status codes from `nv-gpu-transition-validator.h` are
APPLY=0, DEFAULT=1, REPEAT=2, CONFLICT=4, RANGE=6. A kfunc's APPLY means
**recorded**, not that native validation or a setter has executed.

| Fixture suffix | Timeslice / interleave requests | Recorder returns (TS / IL) | Validation (TS / IL) | Required post-init native calls (TS / IL) | Final CPU values |
| --- | --- | --- | --- | --- | --- |
| `no_request` | none / none | none / none | 1 / 1 | 0 / 0 | D / I |
| `legal` | D / 0 | 0 / 0 | 0 / 0 | 1 / 1 | D / 0 |
| `invalid_interleave` | none / 3 | none / 0 | 1 / 6 | 0 / 0 | D / I |
| `duplicate` | D,D / 0,0 | 0,2 / 0,2 | 0 / 0 | 1 / 1 | D / 0 |
| `conflict` | D,D^1,D / 0,2,0 | 0,4,4 / 0,4,4 | 4 / 4 | 0 / 0 | D / I |
| `independent_interleave` | D / 3 | 0 / 0 | 0 / 6 | 1 / 0 | D / I |
| `independent_timeslice` | D,D^1 / 0 | 0,4 / 0 | 4 / 0 | 0 / 1 | D / 0 |

All seven normal constructors should return NV_OK=0; policy range/conflict
rejection must not fail an otherwise valid constructor. Native setters that
are called must separately return NV_OK. Current generated 575
[`g_kernel_fifo_nvoc.h:900`](../../gpu_ext-kernel-575/src/nvidia/generated/g_kernel_fifo_nvoc.h)
implements the minimum-timeslice HAL as zero: **timeslice=0 is not a valid
live negative case**. Requesting D deliberately proves the accepted actuator
path without imposing a new scheduling quantum; LOW=0 proves an actual
interleave-field change from MEDIUM.

The BPF objects default to `target_tgid=0`, which affects nobody. A future
loader must configure the real gated workload PID before loading. Observations
are bounded to 64 entries keyed by full PID/TID, TSG, and runlist. They record
only callback input and recorder results. Reservation failure makes no request;
duplicate identities are not overwritten. Any `INIT_RECORD_ERROR`, incomplete
record, zero engagement, or mismatch between `INIT_SEEN`, `INIT_RECORDED`, and
map-entry counts must fail the future test. These records contain **no native
commit, constructor, or firmware-success claim**.

After POD completed, root authorized and this agent executed only the CPU
expectation test and the BPF compile-only target:

```bash
taskset -c 17 make -C extension test_revision_init_fixtures
taskset -c 17 make -C extension revision_init_fixtures
```

Neither target loads or attaches BPF. The explicit default header root is
`../../gpu_ext-kernel-575` relative to `extension`; do not substitute 610.
Both commands exited zero after the compile-only macro-name correction;
neither is an admission, attachment, or native-init execution test.

## Minimal native diagnostic-hook patch proposal — not implemented

Current 575 source revision is `849ea75d`, kernel `6.15.11-061511-generic`.
The RM constructor and setters are `notrace`, not listed as attachable filter
functions, and their RM structure types are absent from live module BTF.
Kallsyms names alone do not authorize or establish probe attachment. Do not
hardcode RM offsets or bypass notrace; the existing
[driver record](../docs/experiment/driver-575-linux-6.15-runtime.md) documents
this restriction. The currently attachable `nv_gpu_sched_task_init` and
`nv_gpu_sched_gsp_control_complete` do not expose the missing full boundary.

A later separately reviewed **core nvidia module** patch/rebuild/reload is
needed; no UVM changes or new ioctl are required. The smallest source scope is:

1. Add a read-only diagnostic context header beside
   `kernel-open/common/inc/nv-gpu-rpc-diagnostic.h`, declare one diagnostic hook
   in `kernel-open/nvidia/nv-gpu-sched-hooks.h`, and define it next to
   `nv_gpu_sched_gsp_control_complete` in `nv-gpu-sched-hooks.c:403` with the
   same `noinline` plus barrier convention. It must not dispatch struct_ops,
   write driver state, replace status, or relax notrace. Use phase/field tags
   within this hook, not new policy interfaces. C safely copies values from RM;
   BPF observes only the diagnostic context and current PID/TID.
2. In `src/nvidia/src/kernel/gpu/fifo/kernel_channel_group_api.c:329–359`, retain
   a constructor-local observation, initially inactive, when defaults are
   ready. Copy `grpID`, `runlistId`, `engineType`, `pGpu->gpuInstance`, actual
   `gpumgrGetSubDeviceInstanceFromGpu(pGpu)`, `pParams->{hClient,hResource}`,
   the immutable defaults, expected/observed identity and phase, both recorded
   request values/attempted/conflict flags, the minimum passed to validation,
   and both validation outcomes. Emit VALIDATED immediately after the existing
   validator. Interleave uses `pInterleaveLevel[subdevInst]`, **not** gpuInstance
   as an array index. Never export RM pointers.
3. At the two policy-only calls at `:362–387`, expand each
   `NV_ASSERT_OK_OR_GOTO` into the same call/status/error branch plus one
   NATIVE_RETURN observation **before** its failure jump. Preserve the
   timeslice `NV_TRUE` argument. Record field, requested value, real `NvU32`
   status, and post-call CPU value read while the group is valid. Do not count
   the native default timeslice call in `kernel_channel_group.c:186–191` or
   the default MEDIUM call at `kernel_channel_group_api.c:320–327`; both precede
   task_init. Zero policy calls for rejected fields must be observable, not
   inferred from a recorder return or a successful CUDA launch.
4. Snapshot the group's final CPU fields while it is valid under the GPU lock,
   before the `failed:` cleanup at `:645` can destroy/free it. Emit exactly one
   CONSTRUCTOR_RETURN for each active observation at the final return near
   `:703`, carrying the actual final status and whether that snapshot still
   represents a successfully constructed object. `done:` may subsequently
   fail `ctxBufPoolReserve`, retry `failed:`, or take its lock-reacquire
   `NV_ASSERT_OK_OR_RETURN`: cover that post-init early return explicitly too.
   Do not dereference a freed group at `done:`. Copy constructors and exits
   before task_init are outside this matrix and must not be mispaired.

The finite target is the existing
[`workloads/bpftime-device-smoke/vector.cu`](../workloads/bpftime-device-smoke/vector.cu):
8 launches and **32768 checked values**. No new CUDA harness is needed. A later
runner must obtain the existing GPU/struct_ops leases, pass native correctness
first, spawn an owned exec-gated target (stable PID), install only its own
tracer/policy, release that target, and require the complete per-constructor
matrix above. Associate diagnostics with target PID/TID plus GPU/runlist/TSG
and constructor epoch; associate existing RPC observations with the captured
`hClient/hResource` and only the VALIDATED-to-CONSTRUCTOR_RETURN interval.
Capture events before later CUDA control calls overwrite init values. Any
overflow, missing phase, unreadable context, duplicate or unmatched event,
unexpected setter call, nonzero native/constructor status, or failed numerical
check is failure, not a skipped cell. Stop/verify the target's owned process
group before stopping its probe, then verify every owned group and BPF link is
gone, including survivors after a leader exit. Reuse the existing owned-PGID
cleanup helpers; do not globally detach policies.

For the current GSP client, existing
`src/nvidia/src/kernel/vgpu/rpc.c:10398–10414,10576–10593` already observes both
SET_TIMESLICE (`0xa06c0103`) and SET_INTERLEAVE_LEVEL (`0xa06c0107`) **after the
real RPC wait**, with input/transport/GSP status. Require matching successful
RPCs for accepted fields separately from native-setter success. This can prove
the validated host-to-GSP control request completed; it cannot measure the
physical scheduling quantum or guarantee later CUDA calls did not replace it.
Absent new native diagnostic events, leave the full native-init requirement
open rather than relabeling callback, RPC-only, load-only, or CPU evidence.
