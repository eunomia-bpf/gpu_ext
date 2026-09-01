# R5 verifier and transition-safety evidence plan

Status: revision 2 after a blocked review; do not execute until approved.

## Revision question

The Author response promises transition-validation pseudocode, the SIMT
verifier algorithm, rejected-policy examples, a failure-mode taxonomy, and an
explicit TCB.  This experiment asks a narrower, falsifiable question before
paper editing: which rejection and no-op behaviors are implemented and
reproducible in the current artifacts?

## Hypothesis and scope

The combined safety path should demonstrate all of the following without
loading unsafe policy code on the GPU:

1. the base verifier rejects invalid memory access and unbounded control flow;
2. the SIMT pass rejects a lane-varying branch and lane-varying external side
   effects while accepting the matched warp-uniform controls;
3. driver-owned transition validation rejects out-of-range values and turns
   stale or conflicting requests into explicit no-ops; and
4. each of the 50 historical agent-study safety events has a concrete source
   location, or is marked unsupported rather than inferred from another
   derived report.

This is a safety-evidence experiment, not a formal-verification claim, a GPU
performance experiment, or evidence that every driver transition is covered.
It makes no paper changes.

## Artifact identities

- Main repository: the current `gpu_ext` branch and its normal Git revision.
- SIMT verifier: `bpftime` branch `codex/gpu-simt-verifier-clean` at Git commit
  `44c3511`, tested in a separate clean worktree so the existing dirty bpftime
  checkout is untouched.
- Driver transition code: the `kernel-module/nvidia-module` submodule at Git
  commit `24d3e5b7` and the 610 port at Git commit `74a036fe` for source
  comparison.  Runtime driver replacement is outside this plan.
- Historical event inventory: `docs/eval/agent/q2_safety_taxonomy.md`, checked
  against repository files and Git history.  `q5_safety_events_from_sessions`
  is a separate 27-event transcript analysis and must not be substituted for
  the 50-event artifact inventory.

Git revisions are bookkeeping identities.  No file/content hash, checksum,
fingerprint, or digest may be generated, refreshed, compared, or recorded.

## Case matrix and controls

| Case | Unsafe input | Matched control | Required observation |
|---|---|---|---|
| Base bounds | out-of-bounds stack/map access | in-bounds access | unsafe rejected by PREVAIL/base verifier; control accepted |
| Base loop | data-dependent backward loop without a proven bound | compile-time bounded loop | unsafe rejected; control accepted |
| SIMT branch | branch predicate derived from `thread_idx`/lane ID | predicate derived from block ID or a constant | unsafe rejected with the warp-uniform branch diagnostic; control accepted |
| SIMT map key/value | lane-derived key or shared-map payload | warp-uniform key and payload | unsafe rejected with the relevant SIMT diagnostic; control accepted |
| SIMT atomic/helper | lane-varying atomic address or prohibited helper | uniform target and allowed helper | unsafe rejected; control accepted |
| Numeric transition | overflowing or out-of-range timeslice/interleave/action/region | minimum, maximum, and default legal values | invalid request leaves the driver-owned state at its prior/default value |
| Stale transition | transition for an object no longer in the required source state | same transition from its required state | stale request is an explicit no-op; valid control changes state once |
| Conflicting transition | two incompatible requests against one state snapshot | two compatible/idempotent requests | at most one incompatible transition commits; repeat/stale request is a no-op |

The first five cases execute the real verifier library.  The last three must
exercise driver-owned validation code shared with the production call path;
a documentation-only model or duplicated Python/C implementation does not
count.  If no such callable validation seam exists, the result is `GAP` and a
separate implementation-and-test plan is required before claiming those
behaviors.

## Protocol

### Phase A: isolated verifier build

1. Create a sibling bpftime worktree from `codex/gpu-simt-verifier-clean`.
2. Add the workspace no-hash rule to that worktree if it is not inherited from
   the parent directory.
3. Add one test source,
   `bpftime-verifier/test/gpu_revision_safety_test.cpp`, to the existing
   `bpftime_verifier_tests` target.  It must contain these exact Catch2 test
   names, all tagged `[gpu][revision-safety]`:
   - `revision base verifier bounds pair`
   - `revision base verifier loop pair`
   - `revision SIMT branch pair`
   - `revision SIMT map side-effect pairs`
   - `revision SIMT atomic and helper pairs`
4. Every unsafe section and every control section must call the public
   `verify_gpu_program` entry point.  Direct calls to `analyze_uniformity` or
   `check_simt_safety` may support lower-level unit tests but do not satisfy
   this matrix.  Each pair asserts:
   - unsafe: a non-empty rejection plus the expected base-verifier or named
     SIMT diagnostic;
   - control: an empty optional (accepted); and
   - unsafe/control differ in only the property being tested.
5. The exact required sections are:
   - bounds: out-of-bounds stack access rejected; same-width in-bounds stack
     access accepted;
   - loop: data-dependent backward loop with no proven bound rejected;
     constant-bounded backward loop accepted;
   - branch: `thread_idx` predicate rejected with
     `Warp-Uniform Branch Conditions`; block-ID predicate accepted;
   - map side effects: lane-derived key rejected with
     `Map Helper Key Uniformity`, lane-derived shared-map value rejected with
     `Shared Map Value Uniformity`, and a warp-uniform key/value control for
     each is accepted;
   - atomic/helper: lane-varying atomic target rejected with
     `Atomic Operations on Uniform Addresses`, the same operation on a
     uniform target accepted, prohibited helper 506 rejected with
     `Prohibited Helpers`, and an otherwise equivalent allowed-helper control
     accepted.
6. Configure and build only `bpftime_verifier_tests`, disabling linker build
   IDs for newly linked experiment binaries.
7. Before running results, invoke Catch2 listing with the exact selector
   `[gpu][revision-safety]` and require the five names above exactly once.  Run
   the targeted group with the same selector and verbose assertions, then run
   the full verifier binary once.  Preserve stdout/stderr and exit status as
   plain text.  A missing test name or zero-test invocation is a failure.

No GPU or driver mutation is needed for Phase A.

### Phase B: transition-validation seam

1. Trace each policy output from kfunc/struct-ops callback to the driver state
   mutation while holding the relevant lock/reference.
2. Record whether the production path checks object identity, source state,
   numeric range, overflow, and conflicting/repeated requests.
3. Compile and run tests against the production validation function(s).  Use
   exact before/after state assertions and return/result categories; do not
   infer safety from a successful module build.
4. Do not unload, replace, or reload NVIDIA modules in this plan.  A build-only
   kernel result is clearly labeled as such.

### Phase C: 50-event source reconciliation

1. Parse the 50 numbered rows in `q2_safety_taxonomy.md` and require exactly
   one unique number from 1 through 50.
2. For each row, record at least one concrete repository path plus a line,
   section, or normal Git commit that directly supports the incident and its
   recovery.  A citation to Q1/Q3/Q5/Q6 alone is insufficient because those
   are derived reports.
3. Classify each row as `SUPPORTED`, `PARTIAL`, or `UNSUPPORTED`.  Do not fill
   missing evidence from memory or reconstructed prompts.
4. Report the separate transcript-corpus blocker: the original sessions named
   in `docs/eval/agent/README.md` remain absent locally unless directly found.

## Acceptance and falsification rules

- `PASS` requires every executable unsafe/control pair to produce the planned
  opposite outcomes through `verify_gpu_program`, the five exact targeted test
  names appearing once, the full verifier suite passing, and
  production-shared transition validation for numeric, stale, and conflicting
  requests.
- `PARTIAL` is required if verifier cases pass but any transition class lacks a
  production-shared validation seam, or any historical event lacks direct
  source evidence.
- `FAIL` is required for an unsafe verifier case that is accepted, a valid
  control that is rejected, a transition test that mutates state after an
  invalid request, a crash/hang, or a test invocation selecting zero cases.
- Existing CUDA danger demonstrations may motivate a case, but do not count as
  verifier rejection evidence.
- Derived tables and prose are not raw transcript evidence.

## Safety and stopping rules

- At most three real preflights may be used.  Phase A CPU-only unit tests and
  compile checks are not GPU preflights; any command that loads a BPF program
  or touches live driver state is a preflight.
- Stop immediately on a GPU client appearing unexpectedly, a struct-ops leak,
  a driver/module-state change, an Xid, a hang, or an unsafe case reaching a
  live transition path.
- Do not stop GDM, unload core NVIDIA modules, switch packages, reboot, or run
  the deadlock CUDA demonstration.
- Preserve unrelated dirty files and the existing bpftime checkout.

## Evidence and review

Commit and push each coherent, verified batch.  Store the plan review, command
logs, exact test counts, case outcomes, transition audit, event-source matrix,
and result review under `docs/experiment/revision-safety/`.  No aggregate claim
is promoted until a fresh reviewer checks raw outputs, case selection, matched
controls, and scope language.
