# R5 production transition-validator implementation plan

Status: revision 3 for independent review; do not execute until approved.

## Revision question

Can gpubpf route policy-controlled scheduler values, UVM prefetch results, and
PMM list moves through driver-owned validation seams that reject invalid input
and convert stale, repeated, or conflicting requests into explicit no-ops before
mutating driver state?

This is the implementation follow-up required by the `GAP` in
`docs/experiment/revision-safety/phase-b-transition-audit.md`. It makes no paper
changes and does not treat a successful module build as safety evidence.

## Hypothesis and declared ceiling

A production-shared validator can make these behaviors executable and
reproducible without loading unsafe policy inputs:

1. scheduler outputs preserve native defaults on absent, invalid, stale, or
   conflicting requests, including the legal `LOW=0` interleave value;
2. prefetch actions and absolute half-open regions are validated before
   coordinate arithmetic or use;
3. repeated identical scheduler decisions are idempotent, while incompatible
   decisions against one snapshot do not commit;
4. PMM reorder requests require driver-recorded source-list membership under
   `pmm->list_lock`; and
5. the existing raw integer-to-VA-space migration kfunc is no longer exposed.

The production tree has no native retain/release API for `uvm_va_space_t`.
`uvm_va_space_mm_retain()` retains the associated `mm_struct`, not the VA-space
object, and `uvm_va_space_destroy()` explicitly relies on teardown-specific
queue synchronization rather than a VA-space reference count. Therefore this
experiment does **not** invent an opaque token or ad-hoc reference count. It
removes `bpf_gpu_migrate_range(u64 va_space_handle, ...)` from the registered
kfunc surface and records deferred migration as `PARTIAL`. The aggregate result
cannot exceed `PARTIAL` until a separately reviewed native lifetime design is
available.

The implementation must use the same validation definitions in production and
tests. A copied model, documentation-only state machine, or test-only parser
does not count.

## Artifact and no-hash boundary

- Primary driver branch: `kernel-module/nvidia-module` branch `test-sched`.
- Current-driver port: sibling `gpu_ext-kernel-610` branch
  `port/nvidia-610.43.02`, updated only after the primary implementation and
  CPU tests pass.
- Host: Linux 7.1.12 with the official 610.43.02 core stack. No module is
  replaced during the CPU/build phases.
- Use ordinary Git revisions for source-control bookkeeping. Never generate,
  refresh, compare, or record file/content hashes, checksums, fingerprints, or
  digests as implementation or experiment evidence.

## Production-shared definitions

Add `kernel-open/common/inc/nv-gpu-transition-validator.h`. It contains the
exact side-effect-free scheduler, action, and region validator definitions used
by both driver call sites and host tests. Kernel-only PMM mutation remains in
`kernel-open/nvidia-uvm/uvm_pmm_gpu.c`, because list locking and ownership cannot
be represented faithfully in a host-only model.

Use this internal result vocabulary:

```text
APPLY
NOOP_DEFAULT
NOOP_REPEAT
NOOP_STALE
NOOP_CONFLICT
REJECT_ACTION
REJECT_RANGE
REJECT_IDENTITY
```

Every consumer switches over all results. Rejection and no-op paths leave the
pre-call live state unchanged. This enum is an implementation return value, not
a research-workflow schema.

## Frozen scheduler contract

### Context and field access

Replace the ambiguous output values with hidden decision state:

```text
has_timeslice, requested_timeslice, timeslice_conflict
has_interleave, requested_interleave, interleave_conflict
```

Absence is represented only by `has_* == false`; numeric zero is never a
sentinel. `bpf_nv_gpu_set_interleave(ctx, 0)` therefore explicitly requests the
native `LOW` level. `bpf_nv_gpu_set_timeslice(ctx, 0)` is an explicit invalid
request when zero is below the native minimum.

`nv_gpu_sched_ops_is_valid_access()` permits BPF reads only from the immutable
input prefix (`tsg_id`, `engine_type`, `default_timeslice`,
`default_interleave`, and `runlist_id`) and rejects every direct context write.
Only the two setter kfuncs may update hidden decision state. A BPF program
cannot bypass presence, repeat, or conflict tracking by writing a context
output or metadata field directly.

Each setter applies these rules within the callback-local context:

- first call records presence and the exact value;
- an identical later call returns `NOOP_REPEAT` and keeps the first value;
- a different later call marks that field conflicted, returns
  `NOOP_CONFLICT`, and makes the post-callback validator retain the native
  default for that field.

### Identity and phase comparison

The existing callback inside `kchangrpInit_IMPL` runs too early: the enclosing
`kchangrpapiConstruct_IMPL` subsequently applies native `MEDIUM`, overwriting
any policy interleave selection. Relocate the policy invocation to immediately
after that native default-interleave setter. At this point both native defaults
are established and the TSG is not yet returned to the caller.

Immediately before the relocated callback, `kchangrpapiConstruct_IMPL` records
an immutable `expected` snapshot containing the live `grpID`,
`pKernelChannelGroup->runlistId`, and `TASK_CONSTRUCT_DEFAULTS_READY` phase.
Immediately after the callback and immediately before either commit, it
constructs a separate `current` snapshot by re-reading those two live object
fields and the call site's current phase. The validator compares `expected`
with `current`; it never compares a BPF-writable copy with itself.

A TSG/runlist mismatch returns `REJECT_IDENTITY`; a phase mismatch returns
`NOOP_STALE`. Both fields retain native defaults. The synchronous hook normally
keeps the snapshots equal; focused tests mutate the independent `current`
snapshot to exercise stale handling.

### Validation and commit

- no setter call returns `NOOP_DEFAULT` and preserves the corresponding native
  default;
- a requested timeslice below
  `kfifoRunlistGetMinTimeSlice_HAL(pKernelFifo)` returns `REJECT_RANGE`;
- an interleave request is legal only when it equals native `LOW=0`,
  `MEDIUM=1`, or `HIGH=2`;
- a conflict retains the default even if the first value was legal; and
- accepted values are passed to `kfifoChannelGroupSetTimeslice()` and
  `kchangrpSetInterleaveLevel()` without any prior live-field copy. Their
  failure uses the enclosing construct routine's existing `failed` path.

The post-callback decision contains independent results for timeslice and
interleave. One rejected field does not suppress an unrelated accepted field.
Native setter failure follows the existing `kchangrpInit_IMPL` error path.

## Frozen UVM action and coordinate contract

### Coordinates

All BPF-visible prefetch regions are absolute VA-block page indices with
half-open semantics `[first, outer)`. `max_prefetch_region`, initial callback
output, iterator `current_region`, and accepted iterator output all use that
same coordinate system. A non-empty output is legal only when:

```text
max.first <= first < outer <= max.outer
```

`(0, 0)` is the sole empty-region encoding and is legal regardless of
`max.first`. Other zero-length regions are rejected.

Replace the narrow callback output pointer with a callback-local
`uvm_bpf_prefetch_decision_t`. Its requested endpoints and the corresponding
`bpf_gpu_set_prefetch_region()` parameters are `u64`; presence/conflict fields
are hidden from direct BPF writes. Thus the driver retains the BPF program's
original values rather than truncating them at the kfunc boundary.

The shared validator checks those original `u64` endpoints against the empty,
ordering, maximum-region, VA-block, and `uvm_page_index_t` constraints before
any narrowing. Every intermediate is `NvU64`; addition is checked before
execution, subtraction requires `lhs >= rhs`, and narrowing occurs last.
Clamping is not validation.

The native bitmap-tree traversal remains internally relative. Before an
iterator callback, its native `subregion` is converted to absolute coordinates
by the shared checked translation:

```text
widened = max.first + relative_endpoint
require widened >= bitmap_tree->offset
absolute_endpoint = widened - bitmap_tree->offset
require max.first <= absolute_endpoint <= max.outer
```

The native DEFAULT path uses the same checked translation before returning its
result. A translation failure yields the empty region and records
`REJECT_RANGE`; it never reaches the old unchecked post-hoc arithmetic.

### Initial callback actions

The driver gives the callback a fresh wide decision candidate and validates the
raw integer action before interpreting it:

- `DEFAULT`: discard the candidate and execute the native traversal;
- `BYPASS`: validate the candidate as absolute, then accept it and skip native
  traversal;
- `ENTER_LOOP`: discard the candidate and execute the iterator-callback
  traversal; and
- every other integer: return `REJECT_ACTION`, restore DEFAULT plus an empty
  candidate, and execute the native traversal.

An invalid candidate paired with `BYPASS` returns `REJECT_RANGE` and executes
the native DEFAULT path. Thus invalid policy output cannot suppress the safe
native fallback.

### Iterator callback actions

Each traversal step passes an absolute, validated narrow `current_region` input
and a new wide decision candidate. The callback return is no longer ignored:

- `DEFAULT`: ignore its candidate and continue;
- `BYPASS`: validate and commit its absolute candidate, then continue; the last
  legal `BYPASS` selection wins, matching the existing in-tree selection
  pattern;
- `ENTER_LOOP` or any other value: reject that step, ignore its candidate, and
  continue without changing the last accepted selection.

In-tree iterator policies already return `BYPASS` when they select a region and
`DEFAULT` otherwise; they are updated so any region supplied to the setter is
absolute. Policies that never use the iterator remain behaviorally unchanged.

## Frozen PMM membership contract

Add a driver-owned `uvm_pmm_root_list_state_t` field to each
`uvm_gpu_root_chunk_t`:

```text
NONE, FREE, VA_BLOCK_USED, VA_BLOCK_UNUSED, EVICTION, LAZY_FREE
```

The same root metadata records a membership generation. Every native ownership
or cross-list transition advances it; same-list head/tail reorder does not.

The state is protected by `pmm->list_lock` and is updated in the same critical
section as every native root-list operation. The required audited transition
sites are:

- `chunk_update_lists_locked()`: `NONE`, `FREE`, or `VA_BLOCK_USED`;
- `root_chunk_update_eviction_list()`: `VA_BLOCK_USED` or
  `VA_BLOCK_UNUSED`;
- `chunk_start_eviction()`: `EVICTION` after removal from an eviction list;
- eviction recovery through `chunk_update_lists_locked()`;
- referenced-page lazy-free enqueue: `LAZY_FREE`;
- `process_lazy_free_entry()`: `NONE` before native free processing; and
- root-chunk initialization/deinitialization assertions.

The audit is source-complete only when every list mutation in
`uvm_pmm_gpu.c` that can alias a root chunk uses one unified helper. The audit
must classify every `chunk`, `result`, `root_chunk`, and `subchunk` alias, not
just textual `root_chunk->chunk.list` matches. It explicitly includes
`free_next_available_root_chunk()` (`result` is a root),
`pin_free_chunks_func()`, `find_free_chunk_locked()`, lazy-free enqueue/drain,
and root initialization in addition to the named paths above. The helper
updates state and generation in the same lock section for root targets and
leaves root membership unchanged for proven subchunk-only targets.
`list_empty()` is not accepted as membership evidence.

Replace the raw-list kfuncs with a callback-local typed request:

```text
bpf_gpu_request_reorder(decision_ctx, destination, position)
```

Immediately before each `gpu_block_activate` or `gpu_block_access` callback,
the driver creates a hidden `uvm_bpf_pmm_decision_ctx` containing the owning
PMM, root chunk, observed source state, observed generation, and an empty
request record. `destination` is only `VA_BLOCK_USED` or `VA_BLOCK_UNUSED`;
`position` is `HEAD` or `TAIL`. The kfunc records a request but never mutates a
list. Within this one callback invocation an identical second setter returns
`NOOP_REPEAT`; a different second setter marks the local decision conflicted
and returns `NOOP_CONFLICT`.

After the callback, while still under the same `pmm->list_lock`, the production
commit helper verifies:

1. `pmm` is the chunk's owning PMM;
2. the root chunk's current state and generation still equal the callback's
   observed source and generation;
3. the destination enum resolves to the corresponding list head in that same
   PMM; and
4. the root is not `NONE`, `FREE`, `EVICTION`, or `LAZY_FREE`.

If the local record is conflicted, nothing commits. Otherwise the helper
performs at most one `list_move()`/`list_move_tail()`. A cross-list commit
updates source state and advances membership generation in that same critical
section; a same-list reorder preserves membership generation. A wrong source,
foreign PMM, or post-eviction/lazy request is
`NOOP_STALE`/`REJECT_IDENTITY`.

No policy request record persists across callbacks. A later hook therefore
re-evaluates real-time source membership and may legally request HEAD after an
earlier TAIL (or vice versa); movement of other nodes cannot turn the later
request into a false repeat. In-tree policies are mechanically ported from raw
list pointers to the invocation-local request kfunc. The request kfunc never
traverses a list or accepts a destination pointer.

## Deferred VA-space lifetime disposition

Delete `bpf_gpu_migrate_range(u64 va_space_handle, ...)` from
`uvm_bpf_kfunc_ids_set` and from the public BPF header. Any in-tree experimental
policy that references it is excluded from the validated build list and marked
unavailable with the documented native-lifetime blocker. No integer is cast to
`uvm_va_space_t *` in the registered BPF interface.

Restoring deferred migration requires a separate reviewed design that names a
native owner, non-reusable token identity/generation, lookup synchronization,
teardown invalidation, and an acquisition that actually keeps
`uvm_va_space_t` alive. `uvm_va_space_mm_retain()` alone is explicitly
insufficient.

## Frozen executable case matrix

| Case | Invalid/stale/conflicting input | Matched control | Required outcome |
| --- | --- | --- | --- |
| Timeslice presence/minimum | no call; zero; minimum minus one | native minimum | absent retains default; invalid rejects; minimum applies |
| Interleave presence/range | no call; 3; maximum integer | explicit LOW=0, MEDIUM=1, HIGH=2 | absence retains default; invalid rejects; all named levels apply |
| Identity/source phase | wrong TSG/runlist; non-init current phase | independently reconstructed matching current snapshot | stale/identity no-op versus apply |
| Repeated decision | same setter twice; then different value | one setter | identical repeat no-op; conflict retains default |
| Initial action | negative and above ENTER_LOOP with empty candidate | DEFAULT, BYPASS, ENTER_LOOP | invalid falls back to native DEFAULT; legal routes exactly |
| Initial region | reversed, noncanonical empty, below/above max | empty, one-page, full-max absolute regions | invalid falls back to native; controls accepted exactly |
| Iterator action | ENTER_LOOP and out-of-range value | DEFAULT and BYPASS | invalid candidate ignored; last legal BYPASS wins |
| Raw endpoint width | one above `uvm_page_index_t`, one above block outer, and maximum `u64` | largest legal narrow endpoint | original wide invalid values reject before narrowing; boundary control accepted |
| Region translation | subtraction underflow and widened-add overflow | largest legal translated endpoint | reject before arithmetic; boundary control accepted |
| PMM source state | wrong expected list, foreign PMM, eviction/lazy state | member of expected used/unused list | invalid/stale no-op; valid move exactly once |
| PMM repeat/conflict | same setter twice and different setter in one callback | one request; later callback reverses HEAD/TAIL | local repeat/conflict no-op; cross-callback control is re-evaluated and applies |
| Raw VA-space handle | any nonzero integer handle | none until native lifetime design exists | kfunc is absent from registered BTF surface |

## Execution protocol

### Phase A: production-header host tests

1. Add `kernel-open/tests/transition-validator/transition_validator_test.c`.
   It includes the exact production
   `kernel-open/common/inc/nv-gpu-transition-validator.h`; it does not copy the
   rules.
2. Give every scalar/snapshot/action/coordinate matrix row exact before/after
   state and result-enum assertions.
3. Run the focused target, then its containing host-test target once.
4. Inspect test registration so every frozen case runs exactly once.

### Phase B: driver integration, kernel-native PMM tests, and builds

1. Remove the early policy invocation from `kchangrpInit_IMPL`; wire the shared
   header and relocated post-default invocation into
   `kchangrpapiConstruct_IMPL`, and wire it into `compute_prefetch_region()`.
   Wire the kernel-only membership helper into `uvm_pmm_gpu.c` and the typed
   kfunc in `uvm_bpf_struct_ops.c`.
2. Add `uvm_test_pmm_bpf_transition()` to the existing
   `kernel-open/nvidia-uvm/uvm_pmm_test.c` test-ioctl path. It exercises the
   exact production PMM helper with real `list_lock`, list heads, root metadata,
   every root-list alias helper, wrong-source, foreign-PMM, callback-local
   repeat/conflict, cross-callback reversal, eviction, and lazy-state controls.
   A host-only mock cannot satisfy the PMM row.
3. Add five actual BPF load fixtures and a loader that calls
   `bpf_object__load()` without attaching struct_ops:
   `sched-input-write` (load-fail), `sched-hidden-write` (load-fail),
   `sched-immutable-read` (load-pass), `sched-timeslice-setter` (load-pass), and
   `sched-interleave-low-setter` (load-pass). The focused selector must report
   exactly five attempted fixtures, two expected verifier rejections, and three
   admissions. A compiled C selftest or direct call to
   `nv_gpu_sched_ops_is_valid_access()` does not satisfy this row.
4. Build all five open modules for the primary supported configuration, then
   port the same semantics to 610.43.02 and build all five modules for Linux
   7.1.12. Inspect diagnostics, touched BTF types/kfuncs, and the absence of the
   raw migration kfunc from BTF registration. Build success alone is not a
   runtime claim.
5. The kernel-native PMM test and real BPF verifier-load fixtures require a
   safely loadable test stack and together count as one live preflight. If the
   display-owned running stack cannot be left untouched, do not run them and
   report the affected rows `PARTIAL`.

### Phase C: optional live admission preflights

Only after independent review and successful CPU/build phases may at most three
live preflights be used, including the kernel-native test from Phase B:

1. kernel-native PMM/verifier integration test on an idle, safely replaceable
   stack;
2. accepted scheduler decision; and
3. rejected UVM prefetch action using an out-of-range action integer paired
   only with the legal empty `(0, 0)` region, proving native fallback remains
   active.

No stale pointer, foreign list node, conflicting list mutation, invalid region,
range underflow, or overflow is sent to a live driver. Those cases remain
CPU/kernel-selftest-only.

## Acceptance and falsification

- `PASS` for the implemented subset: every frozen scheduler/action/region/list
  pair executes through production-shared code; rejection/no-op preserves the
  exact prior state; the kernel-native PMM/verifier tests run; both driver lines
  build; and no containing test regresses.
- Aggregate `PARTIAL`: mandatory even if the implemented subset passes, because
  deferred migration remains unavailable pending a native VA-space lifetime
  design. It is also `PARTIAL` if a safely loadable kernel-native test stack is
  unavailable.
- `FAIL`: invalid input mutates live state, numeric zero is still overloaded as
  interleave absence, direct context writes remain admitted, overflow reaches
  arithmetic, a raw VA-space pointer remains registered, PMM membership relies
  on `list_empty()`, a conflict commits, a containing suite regresses, or a test
  selector runs zero cases.

Runtime preflights are reported separately and cannot repair failed CPU or
kernel-native cases.

## Safety and stopping rules

- Do not change the running driver during CPU/build work.
- Do not stop GDM, unload display-owned core NVIDIA modules, install modules,
  switch packages, or reboot without explicit authorization.
- Stop on an unexpected GPU client, module-state change, Xid, hang, leaked
  struct-ops link, or any proposal to synthesize a stale kernel pointer.
- Preserve unrelated worktree changes and use isolated branches/worktrees.
- Commit and push every coherent validated batch in both repositories.

## Review gate

An independent reviewer must approve numeric presence semantics, context field
mutability, the two-snapshot identity comparison, the absolute half-open
coordinate contract, iterator action semantics, PMM state coverage, the
explicit deferred-lifetime `PARTIAL`, kernel-native tests, and live-preflight
boundary before implementation starts. Any semantic change after approval
requires another plan review.
