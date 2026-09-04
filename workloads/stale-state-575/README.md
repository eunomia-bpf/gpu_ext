# Stale cross-layer state / thrashing harness

This directory is the CPU/source boundary for Reviewer D's stale-state
sensitivity experiment.  It contains no GPU result.  No live cell has been
run. The native decision model now also has a real host-uBPF JIT consumer and
a deterministic native/JIT differential test. That is dependency evidence;
it does not close the live-interface boundary below. The exact CPU execution
and review are retained in
[`jit-preparation-20260904.md`](jit-preparation-20260904.md).

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

## Source/build boundary now closed

Live execution is intentionally refused by `run_study.py live`. The current
installed 575 module still has only the legacy
`gpu_mem_ops.gpu_page_prefetch` ABI. The source artifact
[`driver-bridge-v1.patch`](driver-bridge-v1.patch) now adds the missing
driver-owned snapshot, matched native consumer, append-only BPF callback and
setter, common diagnostics, and lifecycle counters against driver revision
`6a5b3bb5`. [`driver-bridge-v1/`](driver-bridge-v1/) contains the matching
read-only BPF policy and CPU ABI test. The exact source/build evidence and the
remaining live gate are recorded in
[`driver-bridge-v1-readiness.md`](driver-bridge-v1-readiness.md).

The patch implements one driver-owned, atomically readable record containing
`(sequence, phase, source_mono_ns, published_mono_ns)`, exposed read-only to
both:

- a native in-driver implementation of `stale_state_policy_model.h`; and
- a uniquely named BPF policy implementing the same model.

It also exposes matched decision diagnostics for both arms:
snapshot sequence and phase, decision monotonic time, requested/output region,
and final effect. Those diagnostics are needed to join decisions to the
host-truth timeline and count decision age and wrong-phase decisions without
host-generated proxy counters. A private BPF map alone would not close this
boundary.

This is not yet a live result: the patch has not been installed, the new
module has not been loaded, the BPF policy has not passed the live verifier or
attached, and no GPU cell has run. Those operations require a controlled
module-load window followed by an excluded seven-cell preflight.

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
  Callback, snapshot-read, decision-request, effect-request, diagnostic, and
  retained-record totals must close exactly; any request, effect, or record
  error invalidates the row.
- `execution.json`, `safety-before.json`, `safety-after.json`,
  `gpu-telemetry.csv`, `compute-apps.jsonl`, and `kernel-monitor.log` for
  ownership, exclusivity, cleanup, telemetry, and kernel-safety evidence.

`discarded_prefetch_decisions` means an observed sparse snapshot caused the
policy to request the real empty-region bypass at the driver hook. It is not
inferred from a difference between request counts and migration counts.

## CPU-only commands

```bash
make test-offline
make test-driver-bridge
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
evidence only. `make test-offline` also compiles the bounded policy to BPF
bytecode, executes it through bpftime's uBPF JIT, and differentially checks it
against the native model over malformed inputs and 306,000 seeded decisions.
The test requires an existing bpftime uBPF build and has no native fallback.
Building the CUDA workload is separate:

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
