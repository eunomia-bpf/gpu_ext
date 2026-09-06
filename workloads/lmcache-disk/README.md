# LMCache local-NVMe revision experiment

This directory implements the revision-plan extension from the existing
LMCache CPU comparison to LMCache's local-NVMe backend.  The primary experiment
uses current stable LMCache `v0.5.4` with official vLLM `0.27.1+cu129`; the
historical submitted LMCache build is retained only as provenance.

Tracked reproducibility artifacts:

- `plan-v2.md`: active predeclared question, gates, estimands, and stopping;
- `plan-review-v2.md`: independent revision-2 review and launch decision;
- `plan.md` and `plan-review.md`: closed revision-1 failure provenance;
- `run_lmcache_disk.py`: thin one-cell vLLM adapter and recomputable analysis;
- `lmcache_primitives.py`: active low-level launch/request/validation helpers;
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

## UVM-KV performance runner pressure options

`run_uvm_kv_perf.py` (kind `lmcache_uvm_kv_perf`, CPU-only tests in
`test_uvm_kv_perf.py`) accepts two optional pressure knobs for the
recoverability-arbitration pressure cells; both default to off and leave the
existing CLI and behavior unchanged:

- `--kv-cache-memory-bytes N` (positive integer): explicit vLLM KV pool size
  in bytes, appended to the server argv of every cell as
  `--kv-cache-memory-bytes N`.
- `--pressure-gib N` (default 0 disables) with optional `--pressure-passes P`
  (default 1), `--pressure-pause-ms M` (default 0), and `--pressure-binary
  PATH` (default `workloads/uvm-policy-mechanism/uvm_fault_stream`): when N >
  0, each cell starts the owned UVM fault-stream tenant before the BPF loader
  (debt arm) and before the vLLM server, so its CUDA context and managed
  allocation exist before the model fills VRAM; stdout/stderr is captured in
  the cell `pressure.log` and the runner waits (bounded 30 s) for the exact
  `READY pid=` and `MONITOR_PID:` lines. The tenant holds its monitor wait
  through the cold requests and their store barriers; the runner writes a
  newline to its stdin after the loader warm signal and immediately before
  the warm requests, and stops the tenant process group with a bounded
  SIGINT/SIGTERM/SIGKILL during cleanup (before the server stop). The tenant
  argv, readiness, release outcome, return code,
  `pressure-result.json` (when the tenant writes it), and errors are recorded
  in the cell `result.json`; a launch or readiness failure is recorded and
  the cell continues. No retries, gates, or filtering are added.
