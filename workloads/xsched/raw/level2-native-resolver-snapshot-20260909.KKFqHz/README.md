# Native relay resolver is present at the first launch

2026-09-09 05:30:09 PDT, installed native candidate efb36b2c.
This targeted GDB observation reuses the previous workload, environment and
GO input, but stops at the first AdoptWindowRelayExtra entry before launching
the instrumented kernel. It does not repeat a throughput cell or the old
exit-handler backtrace. Both shared leases cover execution; no UVM module
or target parameter/entrypoint is changed. GDB deliberately exits after the
snapshot; xserver is cleaned up by the existing shell trap.

## Observation

The resolver's probed byte is 1. Its three stored function pointers exactly
equal the corresponding exported symbols in the loaded shim:

| Function | Stored pointer and exported-symbol address |
| --- | --- |
| XgGetRelayOriginalParams | 0x7ffff73cf6f0 |
| XgFindRelayOriginalParams | 0x7ffff73d0950 |
| XgHasRelayLayouts | 0x7ffff73cfc70 |

The loaded-library list contains both the isolated shim libcuda.so.1 and
the real system libcuda.so. The resolver therefore has not selected absent
or different functions at this first relay.

The raw registry memory is also retained. Its libstdc++ hashtable header
shows bucket_count 13 and element_count 2 (field order from the local
/usr/include/c++/13/bits/hashtable.h:387–390). The registry is not empty.
This does not identify which entry the subsequent lookup selects or prove
that its returned parameter layout is correct.

## Consequence

This contradicts the root's preceding hypothesis that wrong dynamic-library
resolution explains the first failure. Do not add another library-name
fallback as a claimed fix based on that hypothesis. The earlier yNZVZG
trace still shows first launch error 701 before the later parameter-window
overlap message and exit-handler SIGSEGV. A later CUDA failure could affect
metadata queries; that causal direction remains unproven.

The next repair must address the first failing launch using its actual
parameter layout and driver launch behavior. These are diagnostic
observations, not performance numbers or a successful native Level-2 run.

run-backtrace.sh and load-backtrace.gdb retain the exact commands.
backtrace.log, lifecycle.log and xserver-debug.log retain the outputs.
The prior debug-go.txt input is referenced by its preserved absolute path.
No manuscript or previous measurement is changed.

## Follow-up: the first compute constructor recovers the original layout

A second targeted snapshot at 05:33:44 PDT stops on the first five-parameter
GetXgOriginalLayout invocation, captures its input/output pointers in GDB
convenience variables, and steps to its return. It does not change target
memory or call new inferior functions.

| Layout | Count | Offsets | Sizes |
| --- | ---: | --- | --- |
| Loaded input | 5 | 0, 8, 0x10, 0x14, 0x18 | 8, 8, 4, 4, 0x1508 |
| Returned original | 5 | 0, 8, 0x10, 0x14, 0x18 | 8, 8, 4, 4, 8 |

The boolean return is 1. Hence the initial constructor lookup succeeds and
recovers the original 0x20-byte extent. A missing initial original-layout
lookup is not an explanation of the first launch 701. The later overlap
message must not be assumed to precede that first failure. This does not yet
prove the later relay upload is correct or explain the driver's rejection
of the grown launch buffer.

load-layout.gdb, run-layout.sh, layout.log, layout-lifecycle.log and
xserver-layout.log retain the exact second observation. It uses the same
frozen binary, workload/environment, shared leases and deliberate post-snapshot
exit. No throughput measurement or completed native Level-2 run is claimed.

## Matched original-entry control: first launch still returns 701

At 06:07:54 PDT, the same installed efb36b2c binary, META_EXTEND=1 and
META_KPARAM=1 were used with the existing ORIGINAL_ENTRY_CONTROL=1 option.
The first CudaCommand::LaunchWrapper returned 0x2bd (CUDA error 701).
GDB did not modify the launch buffer or target memory; it exited after this
first return. This is not a completed workload or a throughput measurement.

Unlike the older HX4CV3 shortened-buffer control, this observation retains
META_KPARAM=1. Switching to the original entry does not by itself eliminate
the current first-launch rejection. The common extended metadata and launch
parameter path remain under investigation; this result does not establish
which field causes the rejection.

run-original-launch.sh, load-original-launch.gdb, original-launch.log,
original-launch-lifecycle.log and xserver-original-launch.log preserve this
observation. Both shared leases and the existing xserver cleanup trap were
used; no driver reload was required.
