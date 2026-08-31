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
