# Stale cross-layer state / thrashing harness

This directory is the CPU/source boundary for Reviewer D's stale-state
sensitivity experiment.  It contains no GPU result.  No live cell has been
run.

## Frozen question and matrix

The workload allocates 40 GiB with `cudaMallocManaged` and alternates six
two-second measured phases over that same allocation:

1. dense, sparse;
2. dense, sparse;
3. dense, sparse.

Dense launches walk successive 256 MiB logical spans (one checked word per
64 KiB region). Sparse launches repeatedly select one region in every 32.
Every returned word is compared with its exact expected value after every
launch. A 1.2-second unmeasured sparse bootstrap precedes the first dense
phase so that even the 1-second delayed arm starts with a real published
snapshot rather than a fabricated default.

The formal matrix is 21 cells: one driver-default UVM control and the Cartesian
product of `{native,bpf}` by `{fresh,100 ms,1 s}` in each of three paired
blocks. The order is fixed by seed `20260903`.  This matrix answers two separate
questions:

- mechanism cost: native versus BPF at the same delay in the same block;
- information cost: fresh versus 100 ms or 1 s within one implementation and
  block.

The default-UVM row is a context control, not a substitute member of either
causal pair. A delayed row with no extra wrong-phase decisions, migrations, or
slowdown is retained as a valid negative result.

## Current hard boundary

Live execution is intentionally refused by `run_study.py live`. The current
575 `gpu_mem_ops.gpu_page_prefetch` ABI supplies a fault-local page index,
bitmap tree, maximum region, and decision object. It supplies no atomically
published cross-layer snapshot or source timestamp. Existing BPF policies can
own private maps, but the driver has no native same-algorithm consumer of such
a map. A userspace `cudaMemPrefetchAsync` approximation would change both the
decision point and actuation mechanism, so it is not used as the native arm.

The minimum missing interface is one driver-owned, atomically readable record
containing `(sequence, phase, source_mono_ns, published_mono_ns)`, exposed
read-only to both:

- a native in-driver implementation of `stale_state_policy_model.h`; and
- a uniquely named BPF policy implementing the same model.

The driver must also expose matched decision diagnostics for both arms:
snapshot sequence and phase, decision monotonic time, requested/output region,
and final effect. Those diagnostics are needed to join decisions to the
host-truth timeline and count decision age and wrong-phase decisions without
host-generated proxy counters. A private BPF map alone does not close this
boundary. No existing policy source is modified here.

## Real records required from every future cell

The analyzer accepts ordinary raw records only. It never synthesizes missing
counts.

- `phase-truth.jsonl`: workload-emitted start/end events for bootstrap and all
  six measured phases, including scheduled offsets and monotonic timestamps.
- `workload-result.json`: phase/kernel/end-to-end time, checked values, exact
  mismatch count, and first mismatch.
- `uvm-events.jsonl`: actual UVM Tools GPU-fault, migration, migrated-byte,
  prefetch-migration/byte, thrashing, eviction, and dropped-event counts.
- policy rows only: `snapshot-publications.jsonl`, `policy-decisions.jsonl`,
  and `policy-final.json`. Every decision is joined to a real host-truth
  interval. Publications identify the shared driver publisher and consuming
  implementation. Decisions retain the fault page, legal maximum region,
  output region, and a matched driver-diagnostic effect. Dense-snapshot
  prefetch must equal the full legal maximum and sparse-snapshot discard must
  equal the empty region; both must engage. Missing snapshots, record loss,
  and request errors invalidate a row.
  Callback, snapshot-read/helper, effect-request, and retained-record totals
  must close exactly; any helper or record error invalidates the row.
- `execution.json`, `safety-before.json`, `safety-after.json`,
  `gpu-telemetry.csv`, `compute-apps.jsonl`, and `kernel-monitor.log` for
  ownership, exclusivity, cleanup, telemetry, and kernel-safety evidence.

`discarded_prefetch_decisions` means an observed sparse snapshot caused the
policy to request the real empty-region bypass at the driver hook. It is not
inferred from a difference between request counts and migration counts.

## CPU-only commands

```bash
make test-offline
python3 -B run_study.py dry-run full \
  --output /absolute/future/raw/stale-state-575-full-01 \
  --preflight /absolute/future/raw/stale-state-575-preflight-01
python3 -B run_study.py cpu-preflight
python3 -B run_study.py analyze \
  --input /absolute/future/raw/stale-state-575-full-01
```

`dry-run` does not inspect artifacts, create output, acquire leases, start a
process, or query a GPU. `cpu-preflight` uses the monotonic clock to confirm
that fresh, 100 ms, and 1 s publications are distinguishable; it is dependency
evidence only. Building the CUDA workload is separate:

```bash
make build-sources
```

The future live coordinator must reuse the existing read-only exclusive lease
files `/tmp/gpubpf-revision-gpu0.lock` and
`/tmp/gpubpf-revision-struct-ops.lock`, the continuous compute/GPU/kernel
monitors, and the pre/post safety checks. As in the existing lease/telemetry
runner, that coordinator—not the monitor sibling—must duplicate the owned
workload's UVM fd and pass it to `uvm_event_monitor` as an inherited fd. It must
never run a formal campaign
until the shared driver interface above exists and an excluded complete
preflight passes. A formal `campaign.json` points to the absolute preflight
root; analysis revalidates all seven preflight cells before accepting the 21
formal cells.
