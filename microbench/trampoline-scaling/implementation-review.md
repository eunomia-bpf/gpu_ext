# Implementation review

OpenCode session `ses_f95d50784ffeTgbkexUZR295da` reviewed the frozen plan,
CUDA target, BPF handlers, loader, build rules, runner, tests, and README in
read-only mode. Snapshots, editing, shell tools, task delegation, and network
access were disabled.

The review focused only on blocking correctness, safety, and reproducibility
issues: matched-arm fairness, target-specific PTX engagement, the complete map
oracle, process/shared-memory cleanup, read-only leases, resume behavior,
fail-closed parsing, and claim scope.

## Initial verdict

`READY`.

The first model response consumed its output budget without emitting a visible
verdict. A continuation in the same session requested only the compact verdict
and returned `READY`; no blocker was omitted from a longer visible response.

## Post-review hardening and re-review

Four conservative changes were made after the initial verdict:

- baseline and build commands now run in owned process groups with bounded
  cleanup;
- resume checks ordinary source metadata and all compiled/runtime audit gates;
- each resume attempt gets a new telemetry directory; and
- shared memory is retained rather than unlinked if any owned process survives.

The updated runner and tests were attached to the same read-only OpenCode
session for a focused re-review. Final verdict: `READY`.

A final delta check covered the rule that an arm becomes resumable only after
post-run GPU safety settles, plus propagation of campaign/summary failures and
its injected-failure unit test. Verdict: `READY`.

No GPU, `sudo`, or experiment result was involved in either review.
