# Q2 safety and design: source-backed text and remaining evidence

Updated 2026-09-03. This document supplies concise material for the promised
transition-validation pseudocode, SIMT algorithm, rejection examples, failure
taxonomy, and trusted computing base (TCB). It does not change the paper or
claim that unfinished runtime validation has passed.

The audit was read-only except for this document: no build, program load,
GPU execution, driver replacement, or new safety experiment was performed.
Previously retained test results are distinguished from current source checks.

## 1. Implemented components and deployment boundaries

Source locations below use these explicit roots:

- **D610**: `kernel-module/nvidia-module`, checked-out commit `c4fd5655`.
- **D575**: sibling worktree `gpu_ext-kernel-575`, commit `849ea75d`.
- **V**: sibling worktree `bpftime-r5`, commit `36610ee`, branch
  `revision/r5-safety-evidence`.
- **R**: sibling worktree `bpftime`, commit `d6316fa`, with existing local
  changes preserved; the experiment runtime build is `build-cuda-pr503`.

These are not interchangeable artifacts. In particular, R's current
`build-cuda-pr503/CMakeCache.txt:587` sets `ENABLE_EBPF_VERIFIER=OFF`. V contains
the GPU verifier and its conditional attach integration; this does not prove
that current device-policy performance runs use the verifier.

| Item | Current evidence | Remaining boundary |
| --- | --- | --- |
| Scheduler initialization requests | D610/D575 use immutable inputs, typed recording setters, independent per-field validation, and native setters; [575 load-only evidence](experiment/revision-safety/sched-load-575-02/execution.json) admits three scheduler controls and rejects two direct writes, within the seven-fixture scheduler/PMM suite | An explicit native initialization commit-path safety test is still missing; load admission/rejection does not execute the constructor or native setters |
| UVM prefetch requests | Production code validates actions, original-width endpoints, checked translation, and narrowing | Retained live invalid-region/action fallback testing is missing |
| PMM reorder requests | Production code uses callback-local requests, root membership/generation and lock-held validation | One retained 610 kernel-native ioctl and positive/negative BPF-load pair passed; this is not every driver/hardware combination |
| GPU SIMT verification | V implements PREVAIL, uniformity dataflow and SIMT checks; retained CPU rejection/control tests passed | Current R build has verification disabled; strict verifier-to-device deployment is not established by the CPU tests |
| Historical safety taxonomy | 50 retained rows; existing source reconciliation supports 46 fully and 4 partially | Original session corpus is unavailable; several manuscript descriptions overstate this evidence |

The earlier [transition GAP audit](experiment/revision-safety/phase-b-transition-audit.md)
describes the pre-repair interface. Later production integrations supersede
that implementation diagnosis; they do not retroactively turn absent live
tests into completed tests.

## 2. Transition validation: text and pseudocode

### Insertion-ready English text

Policy programs request resource-management changes through typed setters;
they do not directly update protected decision state. A setter records the
first request, including its original numeric width. Repeating the same
request is idempotent within that callback, whereas a different second request
latches a conflict. After the callback, driver code validates the request
against its current resource state before invoking the native actuator.
Validation and fallback are resource-specific: rejecting a PMM reorder
preserves the entry list state, while an invalid initial prefetch selection
uses native prefetch behavior. These checks protect the covered transitions,
not the performance quality of an otherwise valid policy.

For PMM reordering, the driver records the owning PMM, root identity, source
list and generation. It checks these fields while holding the existing PMM
list lock, permits only the used/unused allocation lists, and performs a
validated head/tail move through one driver-owned helper. Cross-list changes
advance the generation; same-list reordering does not. This is a
callback-local contract, not a general asynchronous transaction system.

### Illustrative pseudocode matching the production PMM path

```text
record(request, destination, position):
    if not request.attempted:
        request = (attempted=true, conflict=false, destination, position)
        return RECORDED                     // not a resource commit
    if request.conflict: return CONFLICT
    if request.pair == (destination, position): return REPEAT
    request.conflict = true
    return CONFLICT

finish_access(pmm, root, decision, raw_action):
    assert pmm.list_lock is held
    now = (pmm, root, root.generation, root.source_list)
    result = validate_identity_state_and_request(decision, now)
    if raw_action is neither DEFAULT nor BYPASS: return PRESERVE
    if result == APPLY:
        revalidate under the same lock
        driver_move(root, checked_used_or_unused_list, checked_head_or_tail)
        return COMMITTED
    if result == NO_REQUEST and raw_action == DEFAULT: return NATIVE_TAIL_MOVE
    return PRESERVE
```

The source calls the first-record result `NV_GPU_TRANSITION_APPLY`; the
pseudocode spells it `RECORDED` to avoid confusing setter acceptance with a
completed list operation. There is only one post-callback commit in the
access path. A fresh callback may legally request the opposite position.

Source anchors:

- D610 `kernel-open/common/inc/nv-gpu-transition-validator.h:139` records
  scalar requests; `:450` records PMM requests; `:475` validates identity,
  generation/source, conflict and request range; `:503` defines access
  action routing.
- D610 `kernel-open/nvidia-uvm/uvm_bpf_struct_ops.c:144` and `:157` expose
  recording setters; `:201` rejects direct BTF-structure writes with
  `-EACCES`.
- D610 `kernel-open/nvidia-uvm/uvm_pmm_gpu.c:383` updates lists and root
  generation under the existing lock; `:435` validates current membership;
  `:487` commits; `:510` applies access routing.
- D610 `src/nvidia/src/kernel/gpu/fifo/kernel_channel_group_api.c:325`
  initializes immutable observations, invokes the scheduler hook, validates
  both fields, and calls native setters. Its phase is a synchronous
  constructor marker, not a deferred-object generation token.
- D610 `kernel-open/nvidia-uvm/uvm_perf_prefetch.c:117` validates initial
  requests; `:149` checks iterator translation before narrowing;
  `nv-gpu-transition-validator.h:336` and `:374` check ranges and overflow.

Do not write that every invalid request universally becomes a silent no-op.
The newer D575 persistent-timeslice path returns `NV_ERR_INVALID_ARGUMENT`
for an invalid attempted request before changing the RM control payload:
`src/nvidia/src/kernel/gpu/fifo/kernel_channel_group_api.c:1482` and
`kernel-open/common/inc/nv-gpu-timeslice-control.h:42`. It preserves the
original authorized RM control and only updates local bookkeeping after
successful RPC completion. The older unsafe integer-VA-space migration
interface was removed from the validated interface, rather than made safe by
a new lifetime protocol; see the [prefetch integration record](experiment/revision-transition-validator/phase-b-prefetch-results.md).

## 3. SIMT verifier algorithm and enforcement scope

### Insertion-ready English text

The GPU-verifier prototype first applies PREVAIL with GPU-specific helper and
map descriptions to check base eBPF properties, including memory bounds and
termination. It then computes forward uniformity dataflow over reachable
instructions, tracking register values, stack bytes, pointer provenance and
map identity. The final SIMT pass requires uniform branch operands, map keys,
shared-map update values and atomic target addresses, and rejects prohibited
helpers. Unsupported or insufficiently uniform values at these checked uses
are rejected. These are executable checks over a restricted instruction and
helper model, not a proof of the compiler, GPU hardware, or entire runtime.

Host kernel `struct_ops` programs are admitted by Linux's verifier. The tested
GPU path instead uses the userspace PREVAIL-plus-SIMT implementation; the two
must not be described as the same unmodified Linux verifier. Runtime
enforcement additionally requires a verifier-enabled build and strict mode.
Warning-only and verification-disabled modes do not reject unsafe programs.

```text
verify_gpu(program, map_descriptions):
    reject malformed input or unsupported GPU map descriptions
    reject if PREVAIL(program, GPU helper/map model, termination=true) fails
    state[entry] = initial uniformity/pointer/stack state
    worklist = [entry]
    while worklist is not empty:
        pc = pop(worklist)
        output = transfer(program[pc], state[pc], helper/map semantics)
        for successor in control_flow_successors(pc):
            if join_into(state[successor], output) changes it:
                enqueue(successor)
    for reachable instruction and its input state:
        require uniform operands for conditional branches
        require uniform addresses for atomics
        require uniform keys/flags for map operations
        require uniform values for shared-map stores/updates/helper outputs
        reject prohibited helpers and nonuniform checked host-bridge payloads
    accept only if all checks pass
```

Source anchors within V:

- `bpftime-verifier/src/gpu/gpu_verifier.cpp:267`: PREVAIL precedes uniformity
  analysis and SIMT checking; `:309` is the public tested entry point.
- `bpftime-verifier/src/gpu/uniformity_analysis.hpp:18`: tags `UNKNOWN`,
  `UNIFORM`, `VARYING`; `:41`: registers, 512 stack bytes, pointer provenance
  and map descriptors. `uniformity_analysis.cpp:691` defines the implemented
  join, and `:760` the worklist algorithm. The pseudocode does not assert a
  separately proved soundness theorem for this abstract domain.
- `bpftime-verifier/src/gpu/simt_safety_check.cpp:174`: the concrete checks.
- `attach/nv_attach_impl/nv_attach_impl.cpp:228`: conditional verification;
  only strict mode returns `GPU_VERIFIER_REJECTED`. Its
  `CMakeLists.txt:58` enables this code only with `ENABLE_EBPF_VERIFIER`.
- `runtime/include/bpftime_config.hpp:40`: warning, strict and disabled
  modes; warning is the default at `:65`.

## 4. Concrete rejection examples

These are retained test outcomes, not tests newly executed by this audit.
All seven V pairs call `verify_gpu_program`, rather than invoking an isolated
SIMT predicate. The [Phase A record](experiment/revision-safety/phase-a-results.md)
reports 28 assertions in five targeted cases and 137 assertions in the
23-case complete verifier suite.

| Rejected program | Closely matched accepted control | V test source |
| --- | --- | --- |
| Eight-byte stack write at `fp-520` | Same write at `fp-8` | `gpu_revision_safety_test.cpp:146` |
| Helper-dependent backward loop with no proven bound | One-iteration constant-bounded loop | `:170` |
| Conditional branch on lane ID | Branch on warp ID | `:197` |
| Map key derived from lane ID | Key derived from warp ID | `:225` |
| Shared-map value derived from lane ID | Value derived from warp ID | `:268` |
| Atomic target from a per-thread map | Same instruction sequence with a shared-map address | `:308` |
| Helper 506, `bpf_gpu_membar` | Helper 510, `bpf_get_warp_id` | `:355` |

The full test path is `V/bpftime-verifier/test/gpu_revision_safety_test.cpp`.
Some old test labels and the Phase A prose call helper 510 a block ID; the
actual helper table at `V/bpftime-verifier/src/gpu/gpu_platform.cpp:167`
identifies it as **warp ID**. The table above follows the code.

A separate real Linux-verifier example is a direct store into the PMM
decision's hidden request field, versus a typed reorder-setter call:
[`revision_pmm_fixture.bpf.h`](../extension/revision_pmm_fixture.bpf.h), lines
34 and 37. The [retained 610 console](experiment/revision-transition-validator/phase-b-pmm-live-1-logs/console.txt)
records one admission, one rejection, and the successful kernel-native PMM
test ioctl. The [fixture loader](../extension/revision_sched_verifier.c)
loads but does not attach these programs; it requires positive controls
before accepting `-EACCES` as a negative result.

## 5. Failure taxonomy and historical-count correction

### Insertion-ready English text

We distinguish program rejection, invalid resource transitions, valid but
ineffective policies, and failures outside the policy-verification boundary.
Program checks reject unsafe accesses or unsupported SIMT behavior before
execution. Transition checks reject or route invalid requests before the
covered native mutation. Neither layer guarantees throughput, fairness, or
freedom from workload OOM, tool failures, driver bugs or GPU faults. Runtime
correctness and engagement measurements are therefore reported separately
from verifier acceptance.

| Failure class | Detection and response | Evidence scope |
| --- | --- | --- |
| Base-program bounds/termination violation | PREVAIL rejects the GPU prototype program; Linux verifier covers kernel-loaded host policies | Seven GPU test pairs and separate kernel-load fixtures above |
| Unsupported SIMT control flow or side effect | Uniformity/SIMT pass rejects at checked uses | CPU verifier execution; requires enabled strict deployment |
| Illegal range/action, stale PMM state, conflicting request | Driver validates before mutation; preserves state or follows the resource-specific native/error route | Shared production header tests; one live PMM ioctl |
| Valid policy makes poor decisions or uses stale statistics | Detect through outcome/engagement metrics; revise or detach the policy | Not a verifier guarantee; historical negative results retained |
| Driver, workload, loader, resource-exhaustion or GPU fault | Diagnose and recover at the relevant layer; no universal instant-recovery guarantee | Historical source records include Xid and system OOM |

A fresh text-row recount of the
[historical 50-row table](eval/agent/q2_safety_taxonomy.md) gives:
24 `LOGIC_BUG`, 18 `PERF_REGRESSION`, 2 `VERIFIER_REJECT`, 2 `XID_FAULT`,
2 `SYSTEM_HANG`, 1 `BUILD_FAIL`, and 1 `DRIVER_BUG`.
This is not the manuscript's `24 + 18 + 2 verifier + 2 GPU-verifier overflow
+ 4 other` classification. In particular, the two recorded Xid faults must
not be relabeled as verifier-caught GPU overflows. The direct sources include
`docs/experiment/plans/msched.md:125,127` and the system-OOM record at
`docs/experiment/plans/xcoord.md:247,328`.

The existing [source reconciliation](experiment/revision-safety/phase-c-event-sources.md)
classifies 46 events as supported and four as partial (1, 5, 8, 49). Original
sessions were unavailable to that audit. It supports a repository-record
taxonomy, not independently replayable session extraction. Absence of an OS
kernel-panic record is not evidence of no GPU fault, no system failure, or
recovery without restarts.

## 6. Trusted computing base and non-guarantees

### Insertion-ready English text

The threat model trusts administrators who select hooks and deploy policies.
For host policies, the TCB includes Linux's verifier/JIT, the driver hook and
validation code, native resource actuators and their locking/lifetime rules.
For verified device policies it additionally includes the userspace base and
SIMT verifiers, helper/map semantics, eBPF-to-PTX and GPU compilation,
instrumentation/trampoline code, and the map/loader runtime. NVIDIA driver,
firmware and GPU hardware remain trusted. Verification of policy bytecode
does not establish correctness of these components or of arbitrary
application code, nor does it provide per-tenant confidentiality, fair
scheduling, side-channel resistance or a performance guarantee.

The new host-uBPF algorithm ports are a distinct execution domain. For
example, [`finemoe_policy_bridge.cpp`](../workloads/finemoe/finemoe_policy_bridge.cpp)
loads and JIT-compiles with uBPF at lines 14 and 32; its numerical and decision
checks do not establish Linux kernel-verifier admission or device-SIMT
verification. Results must retain their actual execution-domain labels.

## 7. What is still genuinely unfinished

1. Integrate this accurately scoped material into the paper; the current
   document alone does not satisfy a promise to revise design/implementation.
2. Establish enabled, strict GPU verification on the actual deployed
   device-policy runtime, retaining rejection-before-launch and positive
   execution controls. R's disabled build cannot supply that evidence.
3. Retain covered native scheduler-init commit tests on the matching custom
   driver. The [575 run](experiment/revision-safety/sched-load-575-02/execution.json)
   now records all seven scheduler/PMM load-only outcomes (4 admissions,
   3 rejections) and the shared-header CPU validator's 12 cases/145 assertions.
   Neither result executes native initialization commits or live prefetch
   fallback. The actual custom BTF types were checked; no module reload or
   policy attachment was needed for the load-only suite.
4. Retain live prefetch invalid-output/fallback evidence rather than treating
   offline build success as that test.
5. Correct the historical taxonomy wording and narrow the four partial rows;
   recover/release the original session corpus before claiming transcript
   replayability. No unavailable transcript has been reconstructed here.

Existing implemented validators and CPU tests are substantive progress.
They do not justify universal stale-request safety for removed deferred
interfaces, a full-stack formal-verification claim, or an assertion that
every existing BPF performance campaign ran with verification enabled.

## 8. Paper-integration outline, 2026-09-03

Under the user's continuous-revision authorization, integrate Q2 directly in
the active `tex/` sources without a separate approval pause. Preserve existing
labels and packages. Move the safety subsection into `tex/revision_safety.tex`:
first distinguish program checks from transition checks, then give the PMM
record/validate/commit pseudocode and resource-specific fallback, the tested
PREVAIL/SIMT algorithm and rejection examples, and finally failure classes and
TCB limits. Keep warp execution and cross-layer maps in `design.tex` with
scoped claims. Update `implementation.tex` to identify the actual typed host
interfaces, verifier-enabled strict-mode prototype, and the current
performance runtime's verification-disabled boundary; remove blanket
no-atomics/no-cross-SM assertions. Retain precise source/evidence TODOs for
unrun safety tests. The coordinating author owns evaluation counts and all
other paper sections.
