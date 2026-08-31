# LMCache local-NVMe revision experiment

This directory implements the revision-plan extension from the existing
LMCache CPU comparison to LMCache's local-NVMe backend.  The primary experiment
uses current stable LMCache `v0.5.4` with official vLLM `0.27.1+cu129`; the
historical submitted LMCache build is retained only as provenance.

Tracked reproducibility artifacts:

- `plan.md`: predeclared question, gates, estimands, stopping, and blockers;
- `plan-review.md`: independent review decisions and required repairs;
- `run_lmcache_disk.py`: fail-closed preflight/smoke/run/analyze harness;
- `test_runner.py`: CPU-only structural gate tests;
- `prompts.json`: public exact token arrays and expected aligned hits;
- `schedule.json`: all 15 precomputed attempts for ten valid blocks;
- `artifacts-current.json`: wheel, source, build, and imported-module hashes;
- `current-requirements.txt`: full primary Python environment freeze;
- `build-smoke.md`: current and historical build evidence.

The runner never signals a process it did not start. It refuses a mismatched
driver, foreign GPU process, residual GPU memory, or missing final plan
approval. Admission later passed on an idle RTX 5090, but all three allowed
real preflight attempts failed before serving a request. The preserved failures
and exact causes are recorded in `plan.md` and `build-smoke.md`; none is a
performance result, and this protocol must not be relaunched under a new output
name.

The pre-run driver deviation is recorded in `plan.md`: all three cells use
the same 610.43.02 stack, with the reviewed workload and analysis unchanged.
No custom module replacement is needed for this storage-only comparison.
