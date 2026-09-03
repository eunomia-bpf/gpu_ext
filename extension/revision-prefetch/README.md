# Q2 live invalid-prefetch fallback fixture

Status: **driver and observer CPU gates pass; live module admission and all
three functional controls remain open.** The closed
[`attempt01`](../../docs/experiment/revision-safety/prefetch-invalid-575-01/results.md)
has zero completed controls because its structure-return fentry target was
rejected before the target was released. Never resume or overwrite it.

The hypothesis is that a valid callback returning unsupported action `99`, with
a legal empty `(0,0)` request, is routed by the production initial-prefetch
validator to native traversal and leaves a real managed-memory target
numerically correct. This fixture is functional instrumentation, never a timing
benchmark.

## Address-free observation chain

Kernel revision `0c109956` adds one read-only diagnostic hook around the
existing decision path. Its 88-byte context contains only copied action,
request, bounds, validation/effect, output-region, phase, and native-traversal
scalars. It contains no pointer or address-derived token and does not change
policy dispatch, validation, branching, actuation, or return values. The
reviewed source/build record is
[`prefetch-driver-build-575-01`](../../docs/experiment/revision-safety/prefetch-driver-build-575-01/execution.md).

The BPF fixture uses exactly three tracing programs and one optional struct_ops
program:

- wrapper entry creates one frame keyed by the complete `pid_tgid`;
- wrapper exit copies the actual return, request and maximum region;
- diagnostic SELECTED verifies validation/effect and stores stable scalars;
- diagnostic FINISHED compares those scalars, verifies actual output bounds and
  traversal completion, then deletes the frame;
- the fixed policy requests `(0,0)` and returns only 1 or 99.

It does not attach to `get_range`, `compute_prefetch_mask`, or the iterator
wrapper, and it persists no tree, mask, stack, output, or context address. This
removes the failed structure-return attachment and the earlier address exposure.
Full `pid_tgid` ordering plus one outstanding frame per task replaces pointer
correlation. It still cannot attribute callbacks to a target VA space, so an
otherwise exclusive GPU window and a fail-closed compute-process sampler are
required. The sampler proves an empty pre-target sample, target-only samples at
pause, attachment-ready, and post-release points, plus an empty sample after
target exit and loader detachment. Consecutive samples across that lifecycle
may be at most one second apart; each query duration and the idle, start, and
finish gaps between adjacent queries are bounded separately. A lifecycle point
accepts only a query that started after that point, so a query spanning release
or detachment is stale. Cross-process ordering uses the system monotonic clock;
wall time is informational only. This is bounded sampled
exclusivity, not proof against a client whose entire lifetime falls between two
samples.

## Fixed controls and gates

Each mode runs a fresh 8 GiB / 64 KiB `uvm_fault_stream` process and verifies
all 131,072 values:

| Mode | Selected evidence | Finished evidence |
| --- | --- | --- |
| Native | action 0, no request, `NOOP_DEFAULT`, `NATIVE` | completed=1 and iterations>0 |
| BYPASS | action 1, legal `(0,0)`, `APPLY`, `BYPASS` | completed=0, iterations=0, output `(0,0)` |
| Invalid99 | action 99, legal `(0,0)`, `APPLY`, `NATIVE` | completed=1, iterations>0, output empty or within copied bounds |

`APPLY` describes region validation, not acceptance of action 99. The iteration
value counts native loop-body entries, not every helper call in traversal
macros. The output is `compute_prefetch_region`'s actual return, not the final
filtered prefetch mask, completed DMA, or PCIe traffic.

Every decision must follow wrapper-enter → wrapper-exit → SELECTED → FINISHED.
Counts, program run counts and output classifications must reconcile exactly;
all frames must be deleted; recursion misses and all observer error counters
must be zero. Native has no policy calls, while both BPF modes require one
policy call and successful region setter per wrapper. A missing/duplicate phase,
unexpected effect, illegal output, absent traversal, attachment failure, foreign
compute PID, numerical error, Xid, timeout, surviving owned process/link, or
module/service restoration failure rejects the cell and stops the campaign.

## Current executable preparation

The current CPU commands pass:

```sh
taskset -c 17 python3 -B extension/revision-prefetch/test_offline.py
taskset -c 17 make -C extension/revision-prefetch -j1
```

Evidence is retained in
[`prefetch-observer-cpu-575-03`](../../docs/experiment/revision-safety/prefetch-observer-cpu-575-03/execution.md).
`run_safety.py` itself never reloads a module. It requires Linux 6.15.11,
driver 575.57.08, the loaded diagnostic BTF/prototype, both existing lease
inodes, 400 W, prefetch enabled, no preexisting compute client/UVM reference or
struct_ops link, CPUs 8–17, and a fresh output directory. Before release it
requires all three monitors alive, a GPU-telemetry sample, and a fresh
target-only compute sample; after exit it requires the bounded lifecycle and
tail sample above. Any monitor error, foreign sampled PID, or sampling gap
rejects the cell.

SIGINT and SIGTERM are queued by a non-throwing handler. Body checkpoints turn
the first queued signal into cancellation, but owned target, loader, link,
monitor, stream, and safety cleanup remains non-interruptible; the signal is
propagated only after the incomplete execution record is written.

Do not invoke the runner against the currently loaded pre-diagnostic module.
The next step is an owned lifecycle wrapper that stages the built diagnostic
module, proves refcount-zero replacement and live fentry admission, runs
native/BYPASS/invalid99 once each, and restores the prior module and services
before writing a campaign-level completion record.
