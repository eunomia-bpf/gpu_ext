# Plan Review: native no-prefetch versus gpubpf no-prefetch

The same independent reviewer evaluated the plan for three bounded rounds. No
GPU workload was launched during review.

## Round 1: BLOCK

- The proposed LRU access hook was dead on the target GCN workload, so a
  near-zero result would not measure mechanism cost.
- The activation hook could not replace it because the native LRU move happens
  before the non-bypassable callback.
- No executable policy artifact existed.

Resolution: replace LRU with the driver's native no-prefetch switch versus the
same no-prefetch outcome expressed through the live page-prefetch hook.

## Round 2: BLOCK

- Callback counts alone did not cover the preferred-location first-touch path,
  which may emit prefetch migrations before the hook.
- The candidate policy printed on every fault and lacked an ownership-safe
  readiness protocol.

Resolution: add an untimed target-process UVM V2 migration-event monitor; require
zero prefetch migrations, bytes, and dropped migration events in both cells;
add a minimal no-map/no-print policy and a loader that destroys only its own
link. External kprobes separately prove live callback and helper execution in
the BPF cell.

## Round 3: BLOCK, minimal execution repair applied

- With host Yama `ptrace_scope=1`, an unprivileged sibling monitor cannot use
  `pidfd_getfd()` on the paused benchmark.
- The reviewer confirmed the UVM event ABI and counters, policy semantics,
  absence of timed maps/prints, and loader ownership behavior were otherwise
  sound.

Resolution: the preflight launches the monitor with `sudo -n` and requires its
`ready` record before releasing the benchmark. Any fd-duplication or readiness
failure stops the preflight. This is an execution-privilege repair only; it does
not alter either timed cell. The three-round review cap is exhausted, so no
fourth review is requested.

Throughout review and execution, the workflow does not generate, refresh,
compare, or record file/content hashes, checksums, or digests.
