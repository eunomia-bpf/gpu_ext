# Independent observer review

Verdict: **READY for the live campaign under the documented bounded
sampled-exclusivity assumption.** This review was read-only and ran no GPU
work.

The review initially rejected the runner for five concrete reasons: sparse PID
coverage, a release timestamp recorded before pipe close, a tail sample taken
before loader detachment, link cleanup that later failures could skip, and
raising signal handlers that could interrupt child registration or cleanup. A
second pass also found that completion timestamps could admit a query started
before a lifecycle gate, and that separately bounding query duration and idle
time allowed a nearly two-second sample cadence.

The admitted implementation now:

- records query start and finish on the shared monotonic clock;
- admits pause, ready, post-release, and post-detach samples only when the query
  starts strictly after the corresponding saved gate;
- separately bounds query duration, idle gaps, start gaps, and finish gaps to
  one second and rejects every sampled foreign PID;
- verifies owned tracing and policy link disappearance independently of later
  monitor, stream, and safety checks; and
- queues SIGINT/SIGTERM without throwing, propagating cancellation only after
  all owned cleanup attempts and the incomplete execution record.

The final CPU-only suite passes 18/18. The remaining disclosed limitation is
intrinsic to sampling: a foreign process whose complete lifetime falls between
two queries is not observable. Therefore the evidence may claim only bounded
sampled exclusivity, never continuous or target/VA-space attribution.
