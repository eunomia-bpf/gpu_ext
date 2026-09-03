# RTX 5090 invalid-prefetch transition controls

Date: 2026-09-03. Result: complete. This is a three-cell functional safety
experiment on Linux 6.15.11, NVIDIA 575.57.08, and an RTX 5090. It is not a
performance comparison; each cell ran once and the policy intentionally changes
prefetch traversal, so the saved kernel times and callback totals must not be
compared as throughput results.

## What the three controls establish

| Cell | Decisions | Policy calls | Native effects | Bypass effects | Output |
|---|---:|---:|---:|---:|---|
| native action 0 | 42,053 | 0 | 42,053 | 0 | 42,053 non-empty |
| legal bypass action 1 | 131,072 | 131,072 | 0 | 131,072 | 131,072 empty |
| invalid action 99 | 41,882 | 41,882 | 41,882 | 0 | 41,882 non-empty |

Every decision had exactly one matched wrapper entry/exit, SELECTED event,
FINISHED event, and completed frame. The diagnostic hook ran exactly twice per
decision. All observer, ordering, nesting, state, action, request, traversal,
output, and map error counters were zero, and all BPF program recursion-miss
counters were zero.

The native control never called a policy and completed ten native iterations
per decision. The legal action called the typed setter once per decision and
returned an empty region without native traversal. The out-of-range action 99
also called the typed setter once per decision, but every request fell back to
the native traversal, completed ten iterations, and produced a non-empty
region. Thus the live result demonstrates that an invalid policy action becomes
the intended native fallback rather than an unchecked state mutation.

All three 8 GiB targets completed with zero data mismatches. Their compute
monitors saw no foreign compute PID, had no query errors, and kept all measured
query/start/finish/idle gaps below the one-second bound. Every owned tracing and
policy link disappeared, all three safety monitors remained alive until owned
cleanup, and no cell recorded a kernel/GPU abnormality.

## Restoration

The coordinator removed the diagnostic candidate and inserted the exact
admitted old UVM stage with all 53 captured parameters. It revalidated the old
UVM ABI, NVIDIA core identity, two UVM device nodes, idle state, and empty
struct-ops state before restarting nvidia-persistenced and GDM. Both services
returned to active/running with `Result=success`; the live diagnostic interface
is absent, UVM reference count is zero, power limit is 400 W, GPU memory is back
to 15 MiB, and current-boot dmesg/journal abnormality lists remain empty.

`lifecycle.json` and `summary.json` are authoritative. `postflight.json` is an
independent revalidation of the cell equations and restored live state.
