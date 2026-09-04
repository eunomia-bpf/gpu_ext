# Cross-layer raw-record harness: strict read-only review

Act as an independent systems/code reviewer.  The attached files are the
entire review surface.  Do not use tools, edit files, run commands, browse, or
delegate.  No GPU experiment has run; this is source and CPU/compile readiness
only.

Return a leading verdict, exactly `READY` or `REQUIRED FIXES`, followed by a
concise rationale.  Treat a defect as blocking when it can invalidate or
misrepresent a future real run.  Cite the relevant file/function for each
blocker.

Audit these properties:

1. The device BPF handler writes a distinct raw tuple containing per-thread
   launch sequence and block/thread xyz, while a separate per-thread array
   maintains independently checkable aggregates.
2. CUDA creates the ground-truth tuple from its own built-ins; native and
   instrumented target processes are independent; the host joins every tuple
   exactly rather than comparing only counts.
3. Positive cells cover 256 and 2,048 threads with exact zero-drop accounting.
   The six-launch/capacity-four negative deterministically overflows because
   draining begins only after target exit, accounts for every omission, and is
   explicitly rejected as incomplete evidence.
4. Preflight is one complete randomized three-cell block.  Formal execution is
   five complete randomized blocks, requires a passed compatible preflight,
   uses fresh target/probe processes and private segments, and never overwrites
   output.
5. Malformed, duplicate, missing, out-of-range, bad-size, dirty, unaccounted,
   or silently dropped records fail closed.  Aggregate shards are checked
   individually, and ring totals reconcile with callback totals.
6. Exact process-group and shared-segment ownership cleanup is safe.  The
   shared leases and existing GPU/driver/UVM/service/kernel-log safety gates
   remain in force.  Dry-run performs no artifact inspection, output write,
   lease acquisition, process launch, or GPU operation.
7. Documentation preserves the honest boundary: current fixed-size ABI only;
   no performance, strict-verifier, on-chip shard, automatic placement, or
   arbitrary/unbounded data-structure claim.

