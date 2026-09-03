# R5 Phase B production transition audit

Date: 2026-08-31

Status update, 2026-09-03: the source diagnosis below describes the pre-repair
interface and is preserved as a historical audit. Later production paths use
a shared validator; the [575 execution](sched-load-575-02/execution.json)
passed 12 CPU cases/145 assertions and seven kernel load-only fixtures
(4 admissions, 3 rejections). No policy was attached. The subsequent
[replacement prefetch run](prefetch-invalid-575-02/result-review.md) executes
native, legal BYPASS and invalid-action-99 controls and observes native fallback
for every invalid request. Native scheduler-init commits remain open; other
prefetch invalidity classes remain outside that narrow result. See
[current boundaries](driver-test-readiness.md).

Disposition: `GAP`. The current production paths do not expose a shared
transition-validation seam that can satisfy the frozen numeric, stale, and
conflict tests. Per the approved plan, no duplicated model is substituted and
no transition-safety claim is made.

This was a read-only source audit of the test-sched driver and the 610 port. It
loaded no BPF program, changed no driver state, and consumed zero real
preflights. No file/content hashes, checksums, fingerprints, or digests were
generated or used.

## Path trace

### Scheduler initialization outputs

1. `bpf_nv_gpu_set_timeslice` assigns its `u64` argument directly to
   `ctx->timeslice` in
   `kernel-open/nvidia/nv-gpu-sched-hooks.c:87-92`.
2. `bpf_nv_gpu_set_interleave` assigns its `u32` argument directly to
   `ctx->interleave_level` in the same file at lines 97-102.
3. `nv_gpu_sched_task_init` invokes the attached struct-ops callback under an
   RCU read-side critical section, but performs no output validation itself
   (`nv-gpu-sched-hooks.c:377-393`).
4. `kchangrpInit_IMPL` creates a stack context, invokes the hook, then copies
   every nonzero timeslice and interleave output directly into the live channel
   group before calling the native timeslice setter
   (`src/nvidia/src/kernel/gpu/fifo/kernel_channel_group.c:186-212`).

The native `kfifoChannelGroupSetTimeslice_IMPL` rejects values below the native
minimum before its own assignment
(`src/nvidia/src/kernel/gpu/fifo/kernel_fifo.c:1665-1696`). However, the hook
path has already copied that value into `pKernelChannelGroup->timesliceUs`
before calling it. Thus a rejected value is not a no-op on the driver-owned
field; the surrounding creation path fails afterward.

Likewise, `kchangrpSetInterleaveLevel_IMPL` accepts only LOW, MEDIUM, and HIGH
(`kernel_channel_group.c:695-723`), but the hook output is copied into the
channel group before this setter is reached when a channel is later added
(`kernel_channel_group.c:604-610`). These native setters are real validation
code, but the gpubpf path does not route its output through them before the
first mutation. They therefore cannot establish the frozen invalid-input,
prior-state-preserved behavior.

### Bind admission output

`nv_gpu_sched_bind` resets `ctx->allow` to one, calls the policy, and returns
the modified scalar (`nv-gpu-sched-hooks.c:413-422`). The caller interprets any
zero value as `NV_ERR_BUSY_RETRY` and every nonzero value as allow
(`kernel_channel_group_api.c:1095-1111`). This is deterministic admission
semantics, but it is not a versioned state transition: it records neither the
source state observed by the policy nor whether a competing/repeated request
already committed. It therefore cannot establish the planned stale or
conflicting-transition behavior.

### UVM action and prefetch-region outputs

1. `bpf_gpu_set_prefetch_region` directly stores `first` and `outer` after only
   a null-pointer check (`kernel-open/nvidia-uvm/uvm_bpf_struct_ops.c:140-148`).
2. The page-prefetch wrappers return the callback's integer after an enum cast,
   without a shared range validator (`uvm_bpf_struct_ops.c:368-408`).
3. `compute_prefetch_region` treats `BYPASS` and `ENTER_LOOP` specially and all
   other action values as the default branch
   (`kernel-open/nvidia-uvm/uvm_perf_prefetch.c:112-145`). This limits the
   effect of an unknown action at that call site, but it neither reports a
   rejected transition nor supplies a common validation seam.
4. The later region arithmetic clamps some endpoints to the maximum region
   (`uvm_perf_prefetch.c:147-165`), after BPF-controlled values have already
   entered arithmetic. The setter itself checks neither ordering nor numeric
   bounds, so the planned invalid-input/no-state-change assertion is not
   available.

### List mutation and deferred migration

The test-sched source checks only null pointers and `list_empty` before calling
`list_move` or `list_move_tail` (`uvm_bpf_struct_ops.c:150-175`). That does not
prove that the chunk belongs to the supplied source list. The 610 port's same
kfuncs retain only the null checks and call the list primitives directly
(`gpu_ext-kernel-610/kernel-open/nvidia-uvm/uvm_bpf_struct_ops.c:155-173`).
Neither version represents an expected source state or conflict token.

`bpf_gpu_migrate_range` converts a policy-supplied integer handle back to a
`uvm_va_space_t *` and checks only pointer nonzero and length nonzero
(`uvm_bpf_struct_ops.c:180-191`). Its production callee states that the caller
must ensure the VA space remains alive, then acquires the pointed-to object's
lock (`kernel-open/nvidia-uvm/uvm_migrate.c:735-774`). There is no retained
reference or generation check between deferred policy work and dereference.
Consequently a stale-object no-op test cannot be implemented safely through
the current interface.

## Frozen matrix result

| Transition class | Production observation | Executable matched test | Result |
| --- | --- | --- | --- |
| Numeric timeslice/interleave | direct context assignment and pre-validation copy; native setters can reject later | no pre-mutation shared validator and no prior-state-preserving result | GAP |
| Numeric action/region | enum cast/default fall-through and post-hoc endpoint clamp | no common rejection/no-op seam | GAP |
| Stale object | raw deferred VA-space handle; caller owns lifetime guarantee | unsafe to synthesize stale pointer | GAP |
| Conflicting/repeated request | one-shot stack contexts; no source version or commit token | state conflict is not represented | GAP |
| List source-state | mutation under PMM list lock, but no membership/source-list validator | no production-shared predicate | GAP |

## Required next implementation plan

Before Phase B can become executable evidence, a separately reviewed plan must
define a small driver-owned validator used by the production call sites and by
CPU-only tests. At minimum it must:

- return explicit `APPLY`, `NOOP_STALE`, `NOOP_CONFLICT`, and `REJECT_RANGE`
  outcomes;
- validate timeslice, interleave, action, and region ordering/bounds before
  mutation;
- couple each state-changing request to object identity plus expected source
  state or generation;
- retain or safely resolve deferred VA-space references before dereference;
- validate list membership against the intended source list while the PMM list
  lock is held; and
- preserve native defaults on every rejection/no-op path.

That implementation is not authorized by the current experiment plan and is
not attempted here. The aggregate R5 result remains at most `PARTIAL` even
though Phase A passed.
