# Cached-entry continuation: guardian executes, BE output fails

After the delayed-preload and cached-index fixes, the existing two-arm
continuation again runs only native_port then bpf_port, with one block,
50 tasks per stream, 9511106 repetitions, 340 blocks, 256 threads, two LC
and four BE processes. No completed baseline is repeated. The command's
arguments and paths are retained in `cells/protocol.json` and `run.log`.

The native BE process now reaches `running` and logs an instrumented entry
and 200 guarded launches. The previous publication assertion and logging
SIGSEGV are gone. It then exits 2 because the workload's existing per-lane
recurrence output comparison fails, before emitting a service result. LC1
exits zero with 200 launches. The runner exits 1; BPF is unstarted. This is
not evidence of a completed policy performance comparison or a speedup.

The built tool is the cached-index rebuild recorded in
`../level2-launch-stack-20260908.7gAYEJ/rebuild.log`. Source `40ceb34d` also
captured a concurrently edited callback-mutex change; the local source
owner subsequently removes it because the actual stack shows NVBit's own
callback serialization. Future runs must use the consolidated source and
a new build, not infer an exact source/binary pairing from that commit alone.
The current failure record is retained as an integration diagnostic.

All owned workers and xserver have ended. Unlike earlier crash cleanup,
NVML continues to report 100% GPU utilization, P0 and 1 MiB despite no
listed compute applications. The recent kernel-log tail shows no matching
NVRM/Xid message. No driver reset or reload was performed, and the GPU is
not claimed idle from this observation. The shared leases are released.

Local Qwen now owns the remaining actual guardian/abort/replay defect;
do not disable the output comparison to turn dropped work into a claimed
performance gain. Keep the original baseline and all failed attempts.
