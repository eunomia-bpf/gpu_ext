# XSched complete bounded host-submission trace

2026-09-08. Diagnostic follow-up to the failed host-mutex candidate
`bbeb6599`; not a new policy, successful repair or performance sample.
The first-32 diagnostic limit prevented deciding whether later commands
were submitted. Root changed only both host log limits from 32 to 512 in
the isolated tool HAL and its saved `xsched-level2-sm120.patch`.

The HAL build/install exited zero. The same failed `native_port` cell was
run once: 2 LC + 4 BE processes, 4 streams each, 50 commands per stream,
340 blocks, 256 threads and 9511106 recurrence repetitions. Full paths
and arguments are in `cells/protocol.json`. Both revision locks were held;
the driver and tool binary from the preceding candidate were unchanged.

The runner exited 1 after `be1` failed before its service result. Per-process
JSON retains stdout, stderr, arguments, environment and cleanup exit codes.

| Worker | Exit | HAL Launch lines | Publication lines | Tool nonzero-launch counter | First mismatch |
|---|---:|---:|---:|---:|---:|
| be1 | 2 | 201 | 201 | 200 | 87040 |
| be2 | 2 | 201 | 201 | 200 | 87040 |
| be3 | 2 | 201 | 201 | 200 | 87040 |
| be4 | 2 | 201 | 201 | 200 | 87040 |
| lc1 | 0 | 200 | 200 | 200 | none reported |
| lc2 | 0 | 200 | 200 | 200 | none reported |

Every one of the 16 BE queue managers has all distinct command indices
1 through 50 in its host Launch records. Each BE process additionally has
one type-2 resume, for 201 total Launch records. The 512 cap is not reached.
For example, `be1` manager `0x631de35f2860` records command 2 at
20:21:56.813831 and command 50 at 20:21:56.820458, after its type-2 command-1
record at 20:21:56.476376. Thus the earlier conjecture that the affected
queue worker never submitted commands 2--50 is contradicted by this trace.

All four mismatches report zero instead of `0x1.45ef1cp+4`. The first
mismatch does not enumerate all affected outputs. The discrepancy between
201 HAL/publication records and 200 tool nonzero-launch counts is a new
diagnostic lead, not proof of which call is absent or why. HAL logging is
before `LaunchWrapper` and is not proof of successful CUDA submission or
GPU execution; likewise the tool counter is not an execution trace.
Possible launch failure, callback suppression and zero-value consumption
remain to be distinguished. The original abort/replay rule is unchanged.

Root returned these results to the same OpenCode tool-repair session,
which had been resumed after an actual `finish=length`, not interrupted
for silence. No BPF or completed-baseline cell was repeated. Cleanup
returned the GPU to 0% utilization, 1 MiB used and P8 without reset or
module reload. The expanded diagnostic logging can affect timing; no
service/throughput comparison is claimed from this run.
