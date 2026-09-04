# OpenCode raw-map failure review request

Date: 2026-09-03 (America/Vancouver)

Model: `opencode/ling-3.0-flash-fin-free`

The reviewer was given the failed cell record, its diagnostic stream, the C
probe, and the Python runner. It was asked to rank the possible causes of the
eighth-cell BPF-object open failure, identify missing diagnostics, review
fail-closed shared-memory cleanup, and check whether a retry could weaken the
experiment. A second review received the resulting patch plus its CPU test and
checked process races, inode replacement, diagnostics, and evidence gates.

Both sessions used `snapshot=false`, `share=disabled`, deny-all permissions,
and disabled write, edit, shell, fetch, and task tools. No tool call executed.

