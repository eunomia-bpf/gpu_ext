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

The runner never signals a process it did not start.  GPU execution remains
disabled while the host driver differs from `575.57.08`, foreign SGLang
processes occupy the RTX 5090, or final independent plan approval is absent.
The current admission check therefore exits with an explicit blocker report;
offline validation continues without touching the GPU.
