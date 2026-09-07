# LMCache: ioctl allocation-reuse implementation (dev-only)

Implements the adapter change from `ioctl-reuse-plan-20260907.md`. Root
reviews, runs the real-driver measurement, and publishes results; no runner,
timing, executor, policy, or driver change is included here.

## What changed (only `lmcache_gds_policy_adapter.py`)

- `PolicyRequest.pack_into(buffer, request_id, object_id)`: writes all input
  fields and re-zeros all output fields (action, outputPriority, deferNs,
  batchTarget, callerTgid, rmStatus) plus the pads into a caller-owned
  136-byte buffer. `pack()` is untouched.
- `BpfDecider` checks `LMCACHE_GDS_IOCTL_REUSE=1` at construction. When set,
  the instance keeps one `bytearray(136)`, one ctypes view bound to it, and
  (only when no custom `ioctl_func` was injected) the bound `_libc().ioctl`
  callable. Each decision in the reuse path: `pack_into` every field, one
  command-82 ioctl on the reused buffer/view, then parse outputs while still
  holding `BpfDecider._lock` so the shared buffer cannot be repacked by
  another decision between the ioctl and the parse.
- Default (env var unset): `decide()` keeps the original legacy body
  byte-for-byte, plus one initial `if self._shared is not None` guard.
  Fresh `request.pack()` bytearray, fresh `from_buffer` pointer through
  `_uvm_ioctl`, output parse after lock release. No extra helper calls on
  the legacy hot path.

## What is preserved

- One real command-82 ioctl per admission; no native substitution, no
  skipped reads; identical 136-byte ABI on the wire (tests assert legacy and
  reuse hand byte-identical input buffers to the mocked driver).
- GIL behavior: the reuse path calls the same bound libc ioctl, so GIL
  release/retention still follows `LMCACHE_GDS_IOCTL_KEEP_GIL` as before.
- Locks, error handling (negative ioctl result -> `OSError` with errno;
  nonzero `rmStatus` or unknown action -> `GdsDecisionError`), and the custom
  ioctl injection contract. With reuse on, an injected function receives the
  shared buffer; it is overwritten on the next decision.
- Nothing else in the workload changed: shared executor, policy files,
  native implementation, timing definitions, and existing results are
  untouched.

## Tests (development only; mocked ioctl, no GPU, no real driver)

`test_lmcache_gds_policy_adapter.py` (standard-library unittest):

- repeated requests overwrite every input and re-zero every output, so the
  driver sees exactly what a fresh legacy pack would have held;
- legacy and reuse return identical `Decision` sequences and present
  byte-identical input buffers through the same mocked native-policy driver;
- reuse is opt-in: default still allocates a fresh buffer per decision;
- `rmStatus`, unknown-action, and `OSError` handling match on both paths;
- the bound-callable branch reports errno correctly on a real non-driver fd;
- concurrent decisions on the shared buffer stay consistent.

Run:

```
cd workloads/lmcache-disk/gds-control
python3 -m unittest test_lmcache_gds_policy_adapter -v
python3 -m unittest test_lmcache_gds_backend_adapter -v
```

## Next (root)

Per the plan: six rotated 1,000-decision blocks of {native, legacy BPF,
`LMCACHE_GDS_IOCTL_REUSE=1` BPF} against the live driver, 100 warmups,
per-call nanoseconds, actions, and errors into
`raw/gds-ioctl-reuse-575-20260907/`; compare paired block medians.
