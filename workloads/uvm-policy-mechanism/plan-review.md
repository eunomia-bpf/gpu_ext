# Plan Review: short UVM same-policy mechanism comparison

One independent reviewer completed three bounded, read-only rounds. No GPU
workload ran during review.

## Round 1: BLOCK

- Enabled prefetch has a preferred-location first-touch branch before the BPF
  callback, so an unrestricted claim of policy equivalence was too broad.
- The plan did not explicitly require loader/link detach before the next module
  reload.

Resolution: limit the claim to the CPU-resident, non-first-touch path created by
CPU initialization before timing; require hook coverage for every expected
fault region; and require owned loader termination, detach, no remaining memory
struct_ops, and UVM refcount zero after every BPF cell.

## Round 2: BLOCK

- Exactly 131,072 callback calls was too strict because speculative hardware
  faults and VA-block retries may add calls beyond the unique demand addresses.

Resolution: start external counters only after the workload pause and monitor
readiness; require wrapper and helper counts to match and both to be at least
131,072; record additional calls without invalidating them. Zero actual prefetch
migrations/drops and full numerical validation remain mandatory.

## Round 3: APPROVE

The reviewer approved the repaired plan for execution.

The review and experiment workflow does not generate, refresh, compare, or
record file/content hashes, checksums, or digests.
