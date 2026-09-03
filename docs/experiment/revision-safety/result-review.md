# R5 safety-evidence result review

Date: 2026-08-31

Status update, 2026-09-03: this review's original verdict and observations
remain below. Its pre-repair Phase B diagnosis is supplemented by the later
production-shared validator and [575 execution](sched-load-575-02/execution.json):
12 CPU cases/145 assertions and seven load-only fixtures (4 admissions,
3 rejections), with no policy attachment. A still later
[three-cell invalid-prefetch run](prefetch-invalid-575-02/result-review.md)
supplies live action-99 fallback evidence and old-runtime restoration. Native
scheduler-init commit remains open. See [current boundaries](driver-test-readiness.md).

Reviewer verdict: `PASS` for the reported scoped result. Aggregate R5 remains
`PARTIAL` by experiment outcome, not because the review failed.

The fresh reviewer did not participate in the R5 plan review and performed a
read-only check of the frozen plan, bpftime test implementation, production
driver paths, result documents, and event-source matrix.

## Checks accepted

- The exact `[gpu][revision-safety]` selector lists all five required test
  names exactly once.
- All seven unsafe/control pairs call the public `verify_gpu_program` entry
  point. Targeted execution passes 28 assertions in 5 test cases; the complete
  verifier binary passes 137 assertions in 23 test cases.
- The transition result is correctly labeled `GAP`: current production paths
  include pre-validation field writes, lack a unified validator and
  generation/conflict token, and do not establish VA-space lifetime or list
  membership safety. No duplicated test model is presented as evidence.
- The Q2 inventory contains exactly one row for every number from 1 through
  50. The reconciliation contains 46 `SUPPORTED`, 4 `PARTIAL`, and 0
  `UNSUPPORTED` rows; each `PARTIAL` row explicitly narrows the missing
  recovery, causal, or exact-counter evidence.
- The expected historical session corpus is absent, so the result correctly
  declines to claim transcript replayability.
- Branch/revision identities, source changes, and result documents are
  mutually consistent. No paper file was edited for R5.
- No file/content hash, checksum, fingerprint, or digest gate appears in the
  experiment or review.

## Approved interpretation

Phase A is executable verifier evidence. Phase B is a source-backed production
gap. Phase C is a repository-source reconciliation, not a raw-session replay.
Together they justify a reviewed `PARTIAL` R5 result and a separate transition
validator implementation experiment; they do not justify full-stack formal
verification or a replayable 50-event transcript claim.
