# OpenCode launchlate RM review outcome

Verdict: **PASS; no confirmed blocker.**

The review confirmed the conservative RAW-to-PTIMER interval arithmetic,
32-ns expansion at each endpoint, affine classification without midpoint or
negative-latency repair, both 200/200 control gates, exact gpubpf and NVBit
220-launch identities, the 10% uncertainty gate, the exact three-arm schedule,
and strict raw-log replay.  It also confirmed that method-string and campaign
gates prevent the old cross-clock, CUPTI, public-midpoint, calibration-only, or
failed records from becoming performance results.

Its seven follow-ups were nonblockers:

1. Full mode is fixed to pp=512 and ten blocks in `validate_plan()` and
   `analyze()`; the complete analyzer also emits all ten raw triples, geometric
   means, the median effect, and the fixed-seed interval.
2. The separately reviewed bpftime helper implementation calls
   `CLOCK_MONOTONIC_RAW`; a fresh verifier-enabled runtime build and the helper
   execution test passed while standard helper 5 remained unchanged.
3. The NVBit source does read `%globaltimer`.  The copied-source schema was
   strengthened after review to require that instruction explicitly.
4. `one_valid()` rejecting two independently valid attempts is stricter than
   selecting a replacement: normal resume stops after the first valid attempt,
   while duplicate valid evidence fails closed.
5. The analyzer's apparent constant NVBit block counters follow exact
   `one_match()` checks, so duplicate result or calibration records are rejected.
6. The complete analyzer reopens llama-bench JSON and recomputes throughput,
   pairing, degradation, raw triples, median, and interval rather than trusting
   runner summaries.
7. The complete runner writes controls before correctness or timing and binds
   their raw records to the same boot, driver, cleanup, and safety evidence.

This is a code-and-plan audit, not a measurement.  It does not convert the CPU
self-tests or clock controls into Table 1 results.  The fresh GPU controls,
correctness gate, preflight, and ten-block campaign are still required.
