# Packed-argument BPF correctness canary

Passed 2026-09-03 01:31:13–01:32:30 UTC on the declared `849ea75d`
NVIDIA-575/Linux-6.15 driver stage, fixed at 400 W. The coordinated runner
session exited zero and released both GPU/struct_ops leases.

This verifies the BPF argument-packing optimization introduced in `d649b15`,
after the separately retained three-mode preflight. `paper_policy_buffers.py`
bulk-copies the same 24-byte candidate ABI, exact float64 bits, identity and
input ordinal. It does not filter, sort or choose candidates. Native selectors,
policy mathematics, workload and CUDA execution are unchanged. The helper is
explicitly included in this run's admission inventory.

All four expert row sizes and four accumulation arrival orders had zero
maximum absolute/relative error. Two complete nonstream 512-input/64-output
requests and one complete SSE parity request all matched the same-frontend
goldens exactly. The stream retained 64 token frames plus DONE, with independent
engine and serving metrics both increasing by exactly 64 generated tokens.

Actual host uBPF JIT engagement:

- 6,912 prefetch-rank calls, 4,608 EAMC-match calls and 12,645 scored-eviction
  calls; all three same-snapshot native shadow checks had zero mismatches.
- All three JIT shutdown summaries report zero errors. The request controller
  completed three requests, retained six phase EAMs, and had no active or
  aborted request trace at the final drain.
- 3,797 completed prefetches transferred 50,296,823,808 bytes; outcomes conserve
  as 2,295 first-use hits + 1,497 unused-prefetch evictions + 5 unused residents.

Server exit code was zero. The final safety record shows GPU 2 MiB, utilization
0%, no compute clients, UVM refcount zero, empty struct_ops, no new Xid and no
new RM unhandled-interrupt warning.

Shadow verification was enabled, so request latencies are diagnostic only.
This is neither a repeated performance comparison nor a claim that the policy
ran inside kernel UVM hooks. The three-mode formal comparison must freeze this
optimized runtime and disable the redundant shadow oracle in every timed arm.
