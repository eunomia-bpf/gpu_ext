# Preflight 2: stopped at the declared row timeout

- Cell reached: native no-prefetch only.
- Admission: the paused CUDA process owned two `/dev/nvidia-uvm` fds. The V2
  tracker emitted `ready` only for fd 9; fd 10 rejected tracker initialization
  with NV status 22. This satisfied the repaired unique-ready rule.
- Workload: 8,000,000-node random GCN, ten edges per node, 128 features, hidden
  size 256, chunked propagation, one warm-up and one requested measured epoch.
- Managed-allocation peak observed before termination: approximately 36.09 GB
  on the 32 GB RTX 5090.
- Completed warm-up: loss 2.3035, synchronized time 450.921 seconds.
- Timeout: the process exceeded the declared 15-minute per-row limit during the
  measured epoch and was terminated. It did not emit a final benchmark JSON, so
  this attempt supplies no correctness or performance sample.
- Final UVM event record: 452,148 migrations, 6,723,469,312 migrated bytes, zero
  prefetch migrations, zero prefetch bytes, and zero dropped migration events.
  The nonzero total proves the zero-prefetch observation was not an all-zero
  monitor failure.
- BPF cell: not launched after the native timeout.
- Recovery: the workload and monitor exited; no compute client remained; the
  custom 610 UVM module was reloaded with prefetch enabled, refcount zero, and
  no attached struct_ops policy.

Decision: close this plan without using the third preflight allowance. The
required event-instrumented full-size semantic preflight cannot complete its
warm-up plus measured epoch inside the declared 15-minute row bound. Because
event tracking itself may add substantial cost, this attempt does not establish
the runtime of an uninstrumented 8M GCN cell and must not be used to estimate or
report that runtime. A new plan must use a shorter real page-fault workload while
preserving the same native-versus-gpubpf policy comparison.

No file/content hashes, checksums, or digests were generated, refreshed,
compared, or recorded for this attempt.
