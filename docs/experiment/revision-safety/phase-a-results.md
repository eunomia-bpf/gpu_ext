# R5 Phase A verifier results

Date: 2026-08-31

Disposition: `PASS` for the isolated verifier phase. This is not yet an R5
aggregate result: transition validation and the 50-event source reconciliation
remain open.

## Artifact and execution boundary

- Source: `bpftime` branch `revision/r5-safety-evidence`, Git commit `36610ee`.
- Entry point under test: public `verify_gpu_program` for every unsafe case and
  every matched control.
- Build target: `bpftime_verifier_tests` only.
- Runtime boundary: CPU-only verifier execution. No BPF program was loaded, no
  GPU context was created, and no live driver state changed. This consumed zero
  real preflights.
- Evidence contains no file/content hashes, checksums, fingerprints, or
  digests. Git revision names are source-control identities only.

## Frozen case outcomes

| Pair | Unsafe observation | Matched control | Outcome |
| --- | --- | --- | --- |
| Base bounds | 8-byte write at stack offset -520 rejected with `Lower bound must be at least 0` | same-width write at -8 accepted | PASS |
| Base loop | helper-derived backward loop rejected with `Could not prove termination` | one-iteration constant-bounded backward loop accepted | PASS |
| SIMT branch | lane-ID-derived predicate rejected with `Warp-Uniform Branch Conditions` | block-ID-derived predicate accepted | PASS |
| SIMT map key | lane-derived key rejected with `Map Helper Key Uniformity` | block-uniform key accepted | PASS |
| SIMT map value | lane-derived shared-map value rejected with `Shared Map Value Uniformity` | block-uniform value accepted | PASS |
| SIMT atomic | per-thread-map address rejected with `Atomic Operations on Uniform Addresses` | shared-map address accepted | PASS |
| Helper admission | prohibited helper 506 rejected with `Prohibited Helpers` | allowed helper 510 accepted | PASS |

The atomic pair uses the same lookup and atomic instruction sequence. Only the
map address-space type differs: the unsafe program receives a per-thread map
value pointer, while the accepted control receives a shared-map value pointer.

## Command record

Configuration used the repository's vendored Catch2 source and disabled linker
build IDs for the newly linked test executable. The incremental build completed
successfully:

```text
[4/5] Building CXX object bpftime-verifier/CMakeFiles/bpftime_verifier_tests.dir/test/gpu_revision_safety_test.cpp.o
[5/5] Linking CXX executable bpftime-verifier/bpftime_verifier_tests
```

Exact selector listing:

```text
$ build-r5-v2/bpftime-verifier/bpftime_verifier_tests --list-tests '[gpu][revision-safety]'
Matching test cases:
  revision base verifier bounds pair
      [gpu][revision-safety]
  revision base verifier loop pair
      [gpu][revision-safety]
  revision SIMT branch pair
      [gpu][revision-safety]
  revision SIMT map side-effect pairs
      [gpu][revision-safety]
  revision SIMT atomic and helper pairs
      [gpu][revision-safety]
5 matching test cases
```

Targeted execution with verbose assertions ended with:

```text
$ build-r5-v2/bpftime-verifier/bpftime_verifier_tests '[gpu][revision-safety]' -s
===============================================================================
All tests passed (28 assertions in 5 test cases)
```

Full verifier regression execution ended with:

```text
$ build-r5-v2/bpftime-verifier/bpftime_verifier_tests
===============================================================================
All tests passed (137 assertions in 23 test cases)
```

All three commands exited with status 0. The selector listed each required test
name exactly once; neither result depended on a zero-test invocation.

## Interpretation

This phase demonstrates executable rejection examples across the base verifier
and the SIMT-specific rules while showing that closely matched safe programs
remain expressible. It does not demonstrate driver transition validation,
formal verification of the full stack, live GPU enforcement, or the provenance
of the historical 50 safety events. Those claims remain gated by Phases B and
C and by fresh result review.
