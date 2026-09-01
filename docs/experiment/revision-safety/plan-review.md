# R5 safety-evidence plan reviews

## Round 1 — BLOCK

The independent reviewer found that the fixed
`bpftime_verifier_tests` suite did not cover every promised unsafe/control pair.
Some existing cases called internal `check_simt_safety` directly instead of the
public `verify_gpu_program` pipeline, and the plan named neither a new test
source nor exact selectors.  In particular, base bounds, a data-dependent
backward loop, and matched controls for map key/value, atomic, and prohibited
helper cases were incomplete.  Therefore a green full suite would not prove
the planned matrix.

The reviewer separately confirmed that the plan's transition `GAP/PARTIAL`
rule is necessary: current timeslice, interleave, and prefetch-region helpers
write output fields directly and do not by themselves demonstrate a shared
numeric/stale/conflict validator.

## Revision 2 response

Phase A now requires a named `gpu_revision_safety_test.cpp` source with five
exact Catch2 tests selected by `[gpu][revision-safety]`.  Every unsafe and
matched control path must call `verify_gpu_program`, assert the expected
accept/reject result and diagnostic, and differ only in the tested property.
The test listing must contain all five names exactly once before either the
targeted group or full suite can count.
