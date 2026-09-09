# Native lookup candidate: first failure precedes the exit-handler crash

Root reused the existing read-only GDB backtrace template against the current
native candidate efb36b2c on 2026-09-09 04:45:38–04:45:39 PDT. The target is
BE process 2, which exited with SIGSEGV in the six-process lxBvNj attempt.
Only output paths and the BE process ID differ from the old diagnostic;
no kernel parameter, entrypoint or driver state was patched by GDB.
Both shared GPU/struct-ops leases cover this run. No source was changed.

The trace records this order:

1. Metadata grows the loaded parameter extent to 0x1520, and instrumentation
   reaches the launch path.
2. The first launch returns CUDA error 701 (too many resources requested).
   Later launches return error 4; the log still reports original parameter
   extent 0x1520 beyond the blob-relative window base 0x1500.
3. SIGSEGV occurs in glibc __run_exit_handlers(status=1), reached through
   InstrumentManager::Launch. Other launch threads are also in exit handling.

This localizes the observed SIGSEGV after launch failure, not in the earlier
constructor initialization branch. It does not prove the first launch failed
solely because of the reported extent, or establish why the registry lookup
still misses. Those remain the implementation issue. Fixing only shutdown
would not make native Level-2 run. No performance sample comes from GDB.

run-backtrace.sh starts the original xserver and cleans it up on exit;
load-backtrace.gdb runs the existing workload then prints all-thread stacks
and instructions at the stopped PC. debug-go.txt is the reused GO input.
backtrace.log, xserver-debug.log and lifecycle.log preserve the observation.
GDB/runner exit zero means the diagnostic completed, not workload success.
The saved original UVM and current GDS/KV loaders are untouched. No manuscript
or old result was edited.

