# Section VI full attempt 01: interrupted and excluded

The 2026-09-03 `575-section-vi-full-01` campaign is **not a formal performance
result**. Root stopped it after observing external CPU contention. Preserve
the entire attempt, including nine completed cells (three blocks) and the
interrupted tenth cell, `block-03/fifo`. Do not select the completed prefix,
replace individual slow cells, or pool this attempt with a later campaign.

## Interference observation and stop

The interactive workspace OpenCode process, PID 445972 / PGID 445972, began
at 13:48:17 UTC and had access to CPUs 0–23. This was not either of the bounded
OpenCode review processes recorded in [the review report](opencode-review.md).
Root's live `pidstat -p 445972 1 3` observations were:

| UTC interval | Three CPU samples | Mean CPU |
| --- | --- | ---: |
| 16:24:05–16:24:07, before admission | 3%, 2%, 2% | 2.33% |
| 16:37:18–16:37:20, during the campaign | 254%, 196%, 146% | 198.67% |

These are transcribed observations from the root tool session, not a saved
continuous process trace. The later window demonstrates material CPU activity
but does not quantify its effect on each cell or establish a policy difference.
No foreign GPU compute application was observed. Root sent SIGTERM only to
its owned EB coordinator, PID 636094; it did not kill or stop the interactive
OpenCode process. No performance-dependent parameter change was made.

## Interrupted-cell cleanup failure

The [interrupted result](raw/575-section-vi-full-01/block-03/fifo/result.json)
records `InterruptedError` and a **cleanup failure**: the owned-process helper
found an empty non-zombie process group, then its separate one-second leader
wait expired. This cell's cleanup gate did not pass. The saved post-safety
snapshot nevertheless has no GPU compute applications, struct-ops links/maps,
Xids, or kernel abnormalities, and UVM reference count is zero. Root also
checked afterwards that coordinator 636094 and target 674333 were gone and
that GDM and persistenced were active. Later clean state does not erase the
recorded cleanup failure.

The repair requires both a reaped leader and an empty owned process group
within the existing bounded TERM/KILL phases. Because that helper is part of
the recorded runtime inventory, future admission requires a **fresh complete
three-arm numerical/shadow preflight**, then five new randomized full blocks.
The earlier [preflight 01](correctness-results-575-01.md) remains valid evidence
for the implementation it actually tested, not the subsequently repaired helper.

## Retained evidence

The complete [campaign](raw/575-section-vi-full-01/campaign.json), failure record,
ten launch/result/telemetry/log sets and nine worker results contain **51 files,
17,521,126 bytes**, before this report. All raw content is retained unchanged.
The [coordinator log](../../../docs/experiment/revision-safety/eb-section-vi-full-01/coordinator.log)
records the interruption and restoration of both services. Only this new
non-cache raw directory's ownership was reassigned for publication.

Before the replacement campaign, temporarily isolate the exact interfering
process on CPU 17, outside the executor's worker CPUs 1–5 and main/Torch CPUs
8–11. Record and verify process/thread identities and original affinity, monitor
the restriction, and restore it on exit without overwriting independent changes.
This reduces direct CPU-core contention; it does not establish complete memory
bandwidth or machine-wide isolation. Keep all policy parameters and the same
five-block randomized plan unchanged. The replacement is not complete yet.
