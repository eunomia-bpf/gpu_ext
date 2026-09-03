# Real CPU-affinity canary: passed

At 16:58:42–16:58:43 UTC on 2026-09-03, the repaired guard ran against the
previously identified interactive OpenCode PID 445972, process start ticks
1704347, workspace cwd `/home/yunwei37/workspace/gpu`. It recorded all 34
initial thread identities and CPUs 0–23, pinned them to CPU 17, checked the
restriction twice, and restored all 34 to their original masks. The
[guard record](guard.json) has `complete=true`, no errors, child exit 0 and an
empty owned child group. Final recorded masks are uniformly CPUs 0–23.

The child was a CPU-only Python command; its [output](child.log) confirms the
separate coordinator affinity CPUs 8–17. It imported no CUDA runtime or model
and launched no GPU work. No signal was sent to OpenCode, nor were its session
or configuration changed. This canary overlapped the separately identified
**untimed** EB numerical preflight, not a formal performance cell.

This validates the real ordinary pin/restore path for the current process,
not arbitrary interruption/PID-race safety or GPU cleanup. The latter remain
covered only to the extent of their separately recorded tests and controller
gates. The formal campaign must obtain its own fresh complete guard record.
