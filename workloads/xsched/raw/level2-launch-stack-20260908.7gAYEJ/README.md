# Actual first-launch crash stack and bounded repair

After launcher repair `42049461`, the tool loads in the real workload and
the old missing-publication-function error disappears. The full-shape native
cell still exits -11 before the BE running event; its records remain in
`../level2-delayed-preload-20260908.pdvxNk/`. BPF has not run.

`debug_one.py` runs one native BE process with four streams under GDB and
the existing global scheduler, solely to obtain the actual crash stack.
This is not the six-process performance comparison or a timing sample.
`stack.log` shows SIGSEGV in `Instr::getIdx`, called by the tool CUDA event
callback after insertion/enabling. Other launch threads wait on NVBit's
internal callback mutex; this trace does not establish simultaneous entry
into our callback or justify adding a new synchronization scheme.

The implicated source logs `entry->getIdx()` after
`nvbit_enable_instrumented`. Root caches that diagnostic index before
insertion/enabling and logs the saved scalar instead, avoiding access to
the earlier instruction object at that point. Native/BPF decisions,
published context, trampoline, HAL and workload are unchanged.

`make -j2 guard-tool` rebuilds successfully; `rebuild.log` records the
compiler/linker commands. Both diagnostic and build use the shared leases.
GDB exits zero after capturing SIGSEGV; that zero exit is not a successful
workload run. All owned processes have ended and GPU returns to idle.
The next action is to retry the failed/unstarted policy cells, retaining
the previously completed baseline and all failure records.
