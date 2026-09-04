# OpenCode read-only review attempts

Date: 2026-09-04

OpenCode was invoked as a read-only subagent with inline configuration
containing `"snapshot": false`; edit, shell, web-fetch, and task permissions
were denied. No OpenCode process edited files or ran a command.

Three bounded attempts were made with the Spark gateway models. Each emitted a
session `step_start` event but no review text. The first was interrupted after
the formal CPU preflight had independently exposed and repaired two harness
confounders; the second timed out after 180 seconds; the final reduced-context
attempt was stopped on the orchestrator's instruction to close the stage. Thus
OpenCode produced **no verdict**, and this file must not be cited as an
independent approval.

The executable evidence remains the compiler warnings-as-errors gates, the
paired outcomes, and the two retained regression runs in `execution/raw/`.
An OpenCode verdict can be retried later without rerunning or changing the
frozen matrix.
