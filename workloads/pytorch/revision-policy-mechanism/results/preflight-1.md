# Preflight 1: stopped at UVM-fd admission

- Cell reached: native no-prefetch only.
- Driver/module: NVIDIA 610.43.02 custom UVM module on Linux
  7.1.12-070112-generic, with `uvm_perf_prefetch_enable=0`.
- Workload state: CUDA and the managed allocator initialized; the benchmark was
  paused before its first benchmark allocation. No epoch or timing ran.
- Observed target fds: both fd 9 and fd 10 resolved to `/dev/nvidia-uvm`, while
  the reviewed plan required exactly one owned fd.
- Diagnostic: the privileged monitor successfully registered the V2 event
  tracker on fd 9 and emitted `ready`; before workload release it reported zero
  migrations, zero prefetch migrations/bytes, and zero dropped migration events.
- Decision: admission failed because the multiplicity contradicted the plan.
  The monitor and paused benchmark were terminated without releasing the main
  workload. This attempt supplies no correctness or performance sample.
- Recovery: no compute client remained; the custom 610 UVM module was reloaded
  with `uvm_perf_prefetch_enable=1`, refcount zero, and no attached struct_ops
  policy.

Repair for preflight 2: enumerate every owned UVM fd and select the unique fd
that successfully emits the monitor `ready` record. An ineligible auxiliary fd
may reject tracker initialization; zero or multiple ready candidates remain a
hard stop.

No file/content hashes, checksums, or digests were generated, refreshed,
compared, or recorded for this attempt.
