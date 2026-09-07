# LMCache local-NVMe revision experiment

This directory implements the revision-plan extension from the existing
LMCache CPU comparison to LMCache's local-NVMe backend.  The primary experiment
uses current stable LMCache `v0.5.4` with official vLLM `0.27.1+cu129`; the
historical submitted LMCache build is retained only as provenance.

## Current results (2026-09-07)

Performance work is no longer paused. Completed results on driver 575.57.08:

- [Five-arm serving comparison](results-575-lmcache-gds-five-arm-20260906.md):
  25 cells; recompute, CPU, disk FIFO, native and BPF. These disk arms use
  all-submit inputs and measure the mechanism floor, not adaptive-policy gain.
- [Write-budget comparison](results-575-gds-write-budget-20260907.md):
  25 cells; native and BPF share the same live-event-driven admission policy.
- [Request-buffer reuse](results-575-gds-ioctl-reuse-20260907.md):
  18,000 isolated decisions; paired median BPF call cost falls 24.200%.
  This is not an end-to-end storage speedup.
- [Shared I/O concurrency comparison](results-575-gds-write-workers-20260907.md):
  30 cells; BPF four/default workers has paired median read p99 -29.541%
  and write throughput +6.359%, with both improving in three of five blocks.
  Native's stronger default-worker result and all adverse pairs are retained.

Each report links its raw results and reproduction commands. Storage uses real
cuFile compatibility-mode I/O; these results do not demonstrate hardware
NVMe-to-GPU P2P. The concurrency study measures storage-request latency, not
vLLM TTFT. Do not repeat completed cells when updating analysis scripts.

## Historical setup and failed attempts

The text below records the earlier 610-era preparation and failed attempts.
Its launch restrictions, review decisions and stopping rules are historical,
not current instructions or prerequisites for performance measurement. Keep
the failure records; do not interpret them as a pause on the completed 575 work.

Tracked historical reproducibility artifacts:

- `plan-v2.md`: historical predeclared question, gates, estimands, and stopping;
- `plan-review-v2.md`: independent revision-2 review and launch decision;
- `plan.md` and `plan-review.md`: closed revision-1 failure provenance;
- `run_lmcache_disk.py`: thin one-cell vLLM adapter and recomputable analysis;
- `lmcache_primitives.py`: low-level launch/request/validation helpers;
- `test_runner.py`: CPU-only structural gate tests;
- `prompts.json`: public exact token arrays and expected aligned hits;
- `schedule.json`: all 15 precomputed attempts for ten valid blocks;
- `artifacts-current.json`: wheel, source, build, and exact import paths;
- `current-requirements.txt`: full primary Python environment freeze;
- `build-smoke.md`: current and historical build evidence.

The runner never signals a process it did not start. It refuses a mismatched
driver, foreign GPU process, or residual GPU memory. It deliberately has no
approval parser or promotion marker. Inspection later passed on an idle RTX
5090, but all three allowed
real preflight attempts failed before serving a request. The preserved failures
and exact causes are recorded in `plan.md` and `build-smoke.md`; none is a
performance result, and this protocol must not be relaunched under a new output
name. Revision 2 uses the proposed 0.98 startup budget and semantic evidence;
it does not generate or compare content fingerprints. Final independent review
passed the offline repair and blocked another launch because the three-attempt
cap is already exhausted.

After the user separately requested a fast code-first check, one bounded,
single-prefix `lmcache_disk` dependency smoke completed under the unchanged
0.98 memory budget. It proves server startup, one exact cold store, six durable
local-disk chunks, and one exact warm retrieval. It had no trace and supplies
no syscall-level O_DIRECT or performance evidence. This smoke does not reset
the closed experiment or authorize the three-cell comparison; see
`build-smoke.md`.

The pre-run driver deviation is recorded in `plan.md`: all three cells use
the same 610.43.02 stack, with the reviewed workload and analysis unchanged.
No custom module replacement is needed for this storage-only comparison.
