# R5 production transition-validator implementation plan

Status: draft for independent review; do not execute until approved.

## Revision question

Can gpubpf route every policy-controlled scheduler and UVM state change through
a small driver-owned validation seam that rejects invalid values and converts
stale, repeated, or conflicting requests into explicit no-ops before mutating
driver state?

This is the implementation follow-up required by the `GAP` in
`docs/experiment/revision-safety/phase-b-transition-audit.md`. It makes no paper
changes and does not treat a successful module build as safety evidence.

## Hypothesis

A production-shared validator can make the following behaviors executable and
reproducible without loading unsafe policy code:

1. scheduler outputs preserve the native default on an invalid timeslice or
   interleave request;
2. prefetch actions and regions are validated before arithmetic or use;
3. repeated identical decisions are idempotent, while incompatible decisions
   against one snapshot return a conflict no-op;
4. list transitions require membership in the expected source list while the
   PMM list lock is held; and
5. deferred migration cannot dereference an unretained raw VA-space pointer.

The implementation must use the same validation functions in production and
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

## Required production changes

### 1. Common result vocabulary

Define a kernel-internal result enum used by the scheduler and UVM validators:

```text
APPLY
NOOP_DEFAULT
NOOP_REPEAT
NOOP_STALE
NOOP_CONFLICT
REJECT_RANGE
REJECT_IDENTITY
```

The enum is an implementation return value, not a research-workflow schema.
Every consumer must switch explicitly over all results; rejection/no-op paths
must leave the pre-call state unchanged.

### 2. Scheduler task-init transition

Replace the current pre-validation copies in `kchangrpInit_IMPL` with a shared
validator that receives:

- immutable observed identity: TSG ID, runlist ID, and initialization phase;
- native defaults and native minimum timeslice;
- the policy-requested timeslice and interleave; and
- a per-context decision mask recording whether each output was set once,
  repeated identically, or set incompatibly.

Rules:

- zero retains the native default;
- a nonzero timeslice below the native minimum is `REJECT_RANGE`;
- interleave must be LOW, MEDIUM, or HIGH when nonzero;
- identity/source-phase mismatch is `NOOP_STALE` or `REJECT_IDENTITY` before
  any live-field write;
- a repeated identical setter is `NOOP_REPEAT`;
- a second incompatible setter for the same field is `NOOP_CONFLICT`; and
- accepted values are committed through the native setters, not copied before
  those setters validate them.

The scheduler struct-ops verifier must expose identity/input fields read-only
and only documented decision fields writable. A policy must not be able to
rewrite `tsg_id`, `runlist_id`, defaults, or internal decision metadata.

### 3. UVM action and region transition

Add a shared validator after each page-prefetch callback and before
`compute_prefetch_region` consumes the result. It must:

- accept only DEFAULT, BYPASS, and ENTER_LOOP actions;
- reject `first > outer`;
- reject endpoints outside the callback's `max_prefetch_region` coordinate
  contract;
- check every addition/subtraction used to translate coordinates before doing
  it; and
- restore the default empty region and DEFAULT action on rejection.

The implementation must document whether callback outputs are relative or
absolute and update in-tree policies consistently. A test that merely clamps an
invalid value after unchecked arithmetic does not count.

### 4. PMM list transition

Make head/tail movement a driver-owned transition performed under
`pmm->list_lock`. The validator must confirm that the chunk is linked in the
expected source list and that the requested destination is one of the two
driver-provided eviction lists for the same PMM instance. Empty, foreign,
already-moved, or conflicting requests are explicit no-ops.

Do not traverse an unbounded list from BPF context. The production implementation
must use driver-owned metadata or a bounded/native membership predicate.

### 5. Deferred VA-space lifetime

Remove the integer-to-pointer `va_space_handle` contract from the sleepable
migration kfunc. Replace it with an opaque driver-owned token whose lookup
retains a live VA-space reference before dereference and whose invalidation is
synchronized with VA-space teardown. Migration must also reject zero length and
address-plus-length overflow before taking the VA-space lock.

A token lookup after invalidation must return `NOOP_STALE`/an error without
dereferencing freed memory. If the existing UVM lifetime APIs cannot support
this contract, stop and record a design blocker rather than adding an ad-hoc
reference count.

## Frozen executable case matrix

| Case | Invalid/stale/conflicting input | Matched control | Required outcome |
| --- | --- | --- | --- |
| Timeslice minimum | minimum minus one | native minimum and default zero | reject with live state unchanged; controls apply/default |
| Interleave range | zero-adjacent invalid value and maximum integer | LOW, MEDIUM, HIGH, default zero | reject invalid; accept named levels; default unchanged |
| Identity/source phase | wrong TSG/runlist or non-init phase | matching identity in init phase | stale/identity result versus one apply |
| Repeated decision | same setter twice; then different value | one setter or identical repeat | identical repeat no-op; different repeat conflict no-op |
| Action range | negative and value above ENTER_LOOP | all three legal actions | invalid becomes DEFAULT without region use |
| Region ordering/bounds | first greater than outer; endpoint outside max | empty, one-page, full-max regions | reject with empty output; controls accepted exactly |
| Region arithmetic | values that would overflow coordinate translation | largest legal non-overflowing region | reject before arithmetic; boundary control accepted |
| List source state | empty, foreign-list, already-moved chunk | member of expected source list | invalid/stale no-op; valid move exactly once |
| Deferred token | unknown and invalidated token | retained live token | stale returns without dereference; live migration admission proceeds |
| Migration arithmetic | zero length and address-plus-length overflow | one-page and maximum legal range | reject invalid; controls reach native range lookup |

## Execution protocol

### Phase A: shared pure-validator tests

1. Add a host-buildable C test target that includes the exact production
   validator definitions. Do not copy the rules into the test.
2. Give every matrix row exact before/after state assertions and exact result
   enum assertions.
3. Run the focused target, then the containing driver/unit-test suite once.
4. Inspect test registration so every frozen case runs exactly once.

### Phase B: driver integration and builds

1. Wire the shared validators into the production scheduler and UVM paths.
2. Re-run the focused and containing CPU suites.
3. Build all five open modules for the primary supported kernel configuration.
4. Port the same semantic change to 610.43.02, rerun the CPU tests there, and
   build all five modules for Linux 7.1.12.
5. Inspect compiler diagnostics and BTF availability for the touched public
   structs/kfuncs. Build success does not establish runtime behavior.

### Phase C: optional live admission preflights

Only after independent review of Phases A/B may up to three live preflights be
used:

1. accepted scheduler decision on an idle/display-maintenance-approved stack;
2. rejected scheduler decision proving the native default remains active; and
3. UVM accepted/rejected action-region pairs with attach/hit evidence.

No stale raw pointer, foreign list node, conflicting list mutation, or invalid
region is ever sent to a live driver. Those cases remain CPU-only.

## Acceptance and falsification

- `PASS`: all frozen CPU pairs execute through production-shared validators;
  rejection/no-op preserves exact prior state; both driver lines build; and no
  existing containing test regresses.
- `PARTIAL`: scalar/action/region cases pass but safe object-lifetime or list
  membership integration has a documented native-API blocker.
- `FAIL`: an invalid request mutates state, an identity field remains writable
  by policy, overflow reaches arithmetic, a stale token dereferences an object,
  a conflict commits, any containing suite regresses, or a test selector runs
  zero cases.

Runtime preflights are not required for a CPU/build `PASS`; if run, their result
is reported separately as admission evidence and cannot repair a failed CPU
case.

## Safety and stopping rules

- Do not change the running driver during Phases A/B.
- Do not stop GDM, unload display-owned core NVIDIA modules, install modules,
  switch packages, or reboot without explicit authorization.
- Stop on an unexpected GPU client, module-state change, Xid, hang, leaked
  struct-ops link, or any proposal to synthesize a stale kernel pointer.
- Preserve unrelated worktree changes and use isolated branches/worktrees.
- Commit and push every coherent validated batch in both repositories.

## Review gate

An independent reviewer must approve the result vocabulary, field mutability,
coordinate contract, native lifetime API, case matrix, and live-preflight
boundary before implementation starts. Any change to these semantics after
approval requires another plan review.
