# Independent review of proposal 3 revision 1

Reviewed repository state: commit `4036764` plus the repository-wide no-hash
instruction subsequently committed as `6c44d7e`.

## Verdict

APPROVE WITH REPAIRS

## Mandatory repairs

1. Add a repair-specific numerical correctness gate before timing. The current
   32 tests prove patch application, source shape, and chunk boundaries, but do
   not execute `MoEMLP` or compare values. Test 1/256-row repaired fast-path
   outputs against the upstream behavior and 257/353-row outputs against
   concatenated ≤256-row reference evaluations using the same expert
   parameters, with explicit synchronization and a declared tolerance. A
   deterministic end-to-end completion alone could preserve a stable but
   incorrect chunk implementation.

2. Bind the passed preflight to the exact unreplaced build used for timing
   without checksums. Record the resolved extension/runtime filenames and
   ordinary file metadata—at minimum sizes and stable file identity/timestamps—
   in the preflight result, then require equality before the timed schedule and
   each MoE launch. Reject any rebuild/replacement. The current runner accepts
   any `status="passed"` preflight JSON, while admission verifies source/patch
   state but only requires runtime binaries to exist. Also replace or clearly
   supersede `build-smoke.md`, which records Python 3.10/Torch 2.8 and therefore
   conflicts with the active Python 3.12/Torch 2.13 build declaration. Record
   the actual repaired rebuild's filenames, sizes, toolchain, flags, and
   architecture inspection.

3. Record and enforce the three-attempt repaired-preflight budget. Arbitrary
   `--output` paths currently permit unlimited preflight launches. Maintain an
   append-only attempt index or inspect the fixed repaired-protocol directory
   namespace before launch; preserve every failed directory and refuse attempt
   four. Deterministic repeated failures must still stop immediately as
   specified.

## Optional notes

- The source repair is scientifically fair if labeled exactly as required: it
  is a public commit plus a disclosed compatibility/correctness repair, not the
  unmodified public artifact and not a gpubpf contribution. The report correctly
  discloses that chunking affects measured prefill and avoids claiming zero
  overhead.
- Preserving proposal 2's failed 353-row attempt while granting up to three
  attempts to the materially revised source protocol is legitimate. The old
  attempt is evidence about the unmodified artifact, not a sample from the
  repaired protocol.
- Token accounting, O_DIRECT preflight, cache activity, completed-eviction
  evidence, all-six-hook engagement, ownership, cleanup, timeout, and valid-
  block gates are otherwise strong. Failed/partial blocks are retained but
  excluded from analysis; fewer than five valid complete blocks yields only
  “inconclusive.”
- Upstream commit, exact changed-file set, both reverse-applicable patches,
  module versions/BTF, build flags, file inventories/sizes, tests, and enforced
  no-rebuild continuity are adequate identity controls without checksum work
  once repair 2 is implemented.
- The work remains within the user's experiments-only instruction: the repair
  plan keeps WRITE closed and authorizes no paper edits.

GPU execution remains unauthorized until all three mandatory repairs are
implemented and independently re-reviewed.

## Revision 2 repair disposition

All three mandatory repairs have now been implemented for re-review:

1. the repaired `_store` exposes a standalone numerical comparison that runs
   `MoEMLP::forward()` for 1, 256, 257, and 353 BF16 rows, explicitly
   synchronizes the actual and reference paths, and applies `rtol=1e-2`,
   `atol=1e-2`;
2. admission records resolved runtime filenames plus size, device/inode, and
   modification/change timestamps, and the timed runner requires an exact
   match before the schedule and every MoE launch; the active repaired build is
   recorded in `build-smoke.md`; and
3. preflight uses only the fixed `raw/repaired-preflight/attempt-01` through
   `attempt-03` namespace, preserves results, refuses overwrite/nonsequential
   attempts, and refuses unchanged retry after a deterministic failure.

The standalone numerical comparison passed all four sizes on the RTX 5090,
all 36 offline tests passed, both source patches reverse-apply to the admitted
worktree, `_store` contains sm_120 device code, and read-only admission passed.
This disposition records implementation evidence; it does not change the
independent verdict. The 120B preflight remains unauthorized pending re-review.

## Follow-up review

The first follow-up review blocked timing because `run_full_schedule()` checked
only the preflight directory's parent. A directory with an arbitrary basename
could therefore sit under `raw/repaired-preflight` without consuming one of the
three fixed attempt names.

The repair now accepts only the resolved `attempt-01`, `attempt-02`, or
`attempt-03` path, requires the result's attempt number to match the directory,
and requires the same directory's admitted runtime inventory to match the
result. A new offline test covers the accepted path, an arbitrary basename,
and a mismatched attempt number. All 36 offline tests and read-only admission
then passed.

Final follow-up verdict: **APPROVE**.

The repaired-protocol 120B preflight is authorized. Timing remains unauthorized
until a complete repaired preflight passes every correctness and engagement
gate.

## Revision 3 launcher-repair review

Revision-2 attempt 1 completed the exact 512+64-token warm-up and then failed
only because the outer `strace` wrapper did not inherit the server's frozen CPU
set. Revision 3 moves the unchanged `taskset -c 0-7` prefix outside `strace`,
records the executed command, and permits attempt 2 only when the deterministic
predecessor is the specifically reviewed revision 2. Current, missing, or any
other protocol identifier remains fail-closed.

A fresh read-only review checked the raw attempt, launcher, attempt accounting,
38 offline tests, read-only admission, diff scope, and unchanged model,
requests, metrics, source repair, commands, and schedule.

Verdict: **APPROVE**.

Fixed-namespace attempt 2 is authorized. Timing remains unauthorized until a
complete repaired preflight passes every correctness and engagement gate.

## Revision 4 deterministic-repair review

Attempt 2 completed the warm-up and both eight-prompt smoke passes, but six of
eight exact-output pairs differed. Revision 4 initially removed concurrent
in-place writes to the shared accumulator and imposed expert-index reduction
order while retaining four expert compute streams.

A fresh read-only review returned **BLOCK** before attempt 3. It found three
result-invalidating defects:

1. each worker constructed its `token_mask` and gathered input before installing
   the guard for its non-blocking external CUDA stream, leaving an unchecked
   producer/consumer stream handoff before `MoEMLP::forward()`;
2. the output-publication `cudaStreamSynchronize()` return value was ignored;
   and
3. a caught expert-forward exception decremented `pending_` and allowed
   `WaitHiddenStates()` to return a partial result.

The implementation now moves the external-stream guard before all worker
PyTorch mask/input operations, uses the existing checked CUDA-call wrapper at
the publication barrier, records the first worker failure, clears partial
pending output, and raises that failure from `WaitHiddenStates()`. These are
implementation repairs only; the model, requests, exact-output oracle, four
expert compute threads, schedule, metrics, and fixed three-attempt budget are
unchanged. The revised implementation must be rebuilt, rerun through both GPU
gates, and receive a follow-up review before the final attempt is authorized.

### Follow-up round 1

The repaired implementation was rebuilt for sm_120. The row gate and all four
deterministic-reduction arrival orders passed exactly; 40 offline tests, all
three reverse-application checks, diff checks, dependency checks, and fresh
read-only admission also passed.

The same reviewer verified that the external-stream guard now covers the
worker's mask/input construction, dequantization, forward pass, and output
publication; the publication synchronization is checked; and the first worker
failure is carried to `WaitHiddenStates()`, where partial pending output is
cleared and the error is raised before any result is returned. The reviewer
found no new lock cycle or stale-error path. The exact-output oracle, four
expert compute threads, model, request schedule, and fixed attempts 1--3 remain
unchanged.

Follow-up verdict: **APPROVE**.

The fixed-namespace attempt 3 is authorized. Timing remains unauthorized until
that complete real preflight passes every correctness and engagement gate.
