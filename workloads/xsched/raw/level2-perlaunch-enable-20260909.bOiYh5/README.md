# Per-launch enable repairs XSched tool-context delivery

The only executable change from the preceding callback-placement candidate
is nvbit_enable_instrumented(ctx, f, true, false) immediately after the
per-launch nvbit_set_at_launch call. The installed official mem_trace example
also enables instrumentation after setting its value on every launch, whereas
the preceding candidate enabled only during first insertion.

The host-tool build exits zero. The same native_port workload now finishes
all six workers (four BE, two LC), all 1200 requested kernels, all exit zero,
with no sample_diagnostics. BE1/2/3/4 retained XG5 marks advance through slots
50–53 and kernel indices 47–50, instead of repeatedly reading early slot 1.
The run retains real replay launches: BE1/2/3/4 report 207/201/207/200 launches
for 200 submitted kernels each. This supports the per-launch enable ordering
repair on this NVBit tool path, not a fix to the separate native cuXtra route.

The unchanged XG5 probe and host diagnostics are enabled. This successful
native sample is retained for the subsequent matched baseline/native/BPF
comparison rather than repeated. Its LC service p99 is 760289.344 us and
BE throughput is 9.894779 kernels/s; neither alone establishes policy benefit.
Service time is not full arrival-to-completion latency.

The prior tool is restored by run.sh and compares equal to its pre-run copy.
GPU is idle afterward (0%, 1 MiB, P8). No module change or persistence restart
is required. Build/run commands, source snapshot and worker outputs are
retained here; large objects/libraries remain local. No manuscript is edited.
