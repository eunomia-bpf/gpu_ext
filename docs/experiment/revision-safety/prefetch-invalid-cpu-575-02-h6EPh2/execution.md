# Q2 prefetch fixture: CPU cleanup regression checks

2026-09-03. This step changed only the Q2 runner, its synthetic tests and
runbook. No GPU, BPF attachment, driver/module, EB/Fine worker, public GPreempt
helper, or Git operation was performed by this agent.

The necessary local fixes are:

- Require **leader poll has completed and owned PGID is empty** before returning
  cleanup success. This follows EB's repaired check without importing its build
  script or modifying the shared GPreempt helper. The final implementation keeps
  Q2's existing SIGINT 8 s, SIGTERM 5 s, SIGKILL 5 s grace periods; it does not
  take the old empty-PGID `wait(timeout=1)` shortcut.
- Attempt every monitor even if an earlier one cannot be stopped. Record each
  attempted PID, return code and error, close logs, and fail the cell rather
  than skipping later monitors or marking incomplete cleanup valid.

Both CPU-17 invocations of this command exited **0**:

```sh
taskset -c 17 python3 -B extension/revision-prefetch/test_offline.py -v
```

- [tests.log](tests.log): all **5 tests passed** for the initial local adaptation.
  It still used EB's TERM/KILL grace periods.
- [tests-02.log](tests-02.log): all **5 tests passed** after retaining Q2's prior
  INT/TERM/KILL grace periods. This is the final-source test run.

The two added tests use synthetic process objects, not real PIDs: an empty
group with delayed leader reaping must not call the one-second wait or claim
success while the leader remains alive; a failed first monitor must not skip
the second, and the retained record must show both outcomes with `complete=false`.
The original three record-gate tests also continue to pass. Neither test run
launched a child workload, attached BPF, or exercised a GPU.

C/BPF source and build inputs were unchanged, so the successful
[first build](../prefetch-invalid-cpu-575-01-EXpHQx/execution.md) was not repeated.
Real attachment, three functional controls, and actual signal/owned-cleanup
behavior remain unverified. The observed compute-mask remains **pre-filter**,
not the final hint or DMA mask.
