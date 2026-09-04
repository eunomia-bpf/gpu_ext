# OpenCode formal-result review

Date: 2026-09-03 (America/Vancouver)

- Model: `opencode/ling-3.0-flash-fin-free`
- Session: `ses_f95392401ffe5mMLWd33V7gBzg`
- Configuration: snapshots and sharing disabled; deny-all permissions; write,
  edit, shell, fetch, and task tools disabled; no tool call executed.
- Verdict: `READY`.

The reviewer confirmed that the offline analyzer re-parses every native,
instrumented, and probe log in all 15 cells; recomputes the exact 34,560
positive tuples and 2,560 negative drops; checks per-cell lifecycle plus
campaign and per-cell safety state; excludes the failed `full-575-01` attempt;
and limits the result to bounded raw-record expressibility and exact readback.
It found no concrete blocker.
