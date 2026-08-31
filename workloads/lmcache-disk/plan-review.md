# LMCache local-NVMe plan reviews

## Round 1 — independent review

Decision: **APPROVE WITH REQUIRED REPAIRS**

Required repairs recorded from the independent reviewer:

1. Make the strongest current stable LMCache-compatible triplet primary;
   retain the historical submitted revision only as a labeled bridge.
2. Freeze exact commands, environment, imported paths/hashes, schedule,
   resume semantics, atomic completion, invalid preservation, and manifest
   mismatch behavior.
3. Prove store completion and exact engagement with token IDs, cold zero-hit
   evidence, fully persisted expected chunks, exact warm hit/retrieval counts,
   request correlation, no fallback/eviction, and capacity calculations.
4. Trace the exact disk workload and reject every `.pt` open lacking
   `O_DIRECT`; retain raw traces and remove unsupported disk-I/O-byte claims.
5. Time throughput over warm requests only; freeze deterministic outputs;
   use paired relative-rate inference and performance-blind retry/attempt caps.
6. Prefer P95/maximum over an eight-sample P99 and scope uncertainty to the
   fixed model, prefix set, and SSD.

Repair status: implemented in `plan.md`, `schedule.json`, the schema-2 prompt
artifact generator, and `run_lmcache_disk.py`; round-2 review is pending.

## Round 2 — independent review

Decision: **APPROVE WITH REQUIRED REPAIRS**

The reviewer accepted the current stable triplet, request-ID mapping, exact
prompt arrays and aligned hits, persistence barrier and footprint derivation,
warm-only metrics, paired analysis, capped blind retries, atomic markers, and
completed-attempt validation.  Six remaining blockers were recorded:

1. add the two vLLM connector/factory modules to the frozen import manifest;
2. admit the actual requested output/cache filesystem, not the harness path;
3. bind gates/resume to runner and canonical launch configuration and prevent
   inherited runtime-environment contamination;
4. reject any partial or contradictory warm hit/retrieval and cold-store log;
5. require successful, contained, identical 48-path O_DIRECT read/write sets;
6. require complete authenticated gate evidence and compare model artifacts.

Repair status: all six items are implemented.  CPU structural tests pass
(6/6), and full admission now fails only on the declared external conditions:
driver 610.43.02 instead of 575.57.08, two unrelated SGLang GPU processes,
and 32,137 MiB residual GPU memory.  The files are frozen for round 3.

## Round 3 — final independent review

The reviewer verified that all six round-2 blockers are closed: frozen vLLM
connector imports; actual output/cache mount admission; runner, plan, command,
environment, and artifact binding; exact request evidence; successful and
contained 48-object O_DIRECT coverage; and complete authenticated gate/model
evidence.  The randomized complete-block protocol, fixed stopping rule,
performance-blind invalidation, warm-only metrics, paired bootstrap, and exact
resume semantics were also accepted.  No new paper-facing correctness blocker
was found.

Approval does not waive runtime admission.  Driver 610.43.02, the unrelated
SGLang processes, and residual GPU memory remain explicit launch blockers.

FINAL DECISION: APPROVE
