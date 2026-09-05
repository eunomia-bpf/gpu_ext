# Stale cross-layer state / thrashing harness

This directory implements Reviewer D's stale-state sensitivity experiment. It
contains live setup and partial excluded-preflight traces, but no complete
preflight or formal GPU result. The native decision model also has a real
host-uBPF JIT consumer and a deterministic native/JIT differential test. That
is dependency evidence; it does not turn an interrupted preflight into a
result. The exact CPU execution and review are retained in
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

Full live execution is intentionally refused by `run_study.py live`. The
current installed 575 module still has only the legacy
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

`coordinator.py::TruthFDCoordinator` now consumes the workload-owned truth
pipe after the workload emits `workload_ready`. It releases the workload while
the bridge remains off, then configures the selected consumer immediately
before publishing the first truth-derived snapshot after the frozen delay.
This preserves the 1.2-second bootstrap without counting its initial
no-snapshot interval as a policy error. It publishes only exact `phase_start`
timestamps, validates the driver-captured publication timestamp and
status-observation window, reconciles final common counters, and disables
its owned generation on success or failure. The default-UVM path never
configures, publishes, or disables policy state. Its return schema explicitly
marks `truth_source=workload_phase_fd`, `synthetic_source=false`, and
`experiment_evidence=false`: this closes a coordinator implementation gap but
is not a complete cell.
The exact CPU validation record is in
[`truth-fd-coordinator-readiness.md`](truth-fd-coordinator-readiness.md).

The diagnostic observer and preflight orchestration source are now present.
`driver-bridge-v1/live_loader` owns one fentry observer link; the native arm
disables creation of the struct_ops program and map, while the BPF arm also
owns exactly one struct_ops link. It retains a real verifier log, emits only
completed driver diagnostics, and rejects owner, ABI, action/effect, ordering,
or ring-buffer-loss errors. `observer_protocol.py` independently validates
that stream and reconciles it against the driver's common counters. Ownership
comes from the VA-space creator TGID carried by the driver diagnostic, not the
possibly unrelated UVM worker in `current`. The raw-cell validator replays the
observer and verifier evidence and requires exact raw/normalized decisions.
`live_runner.py` supplies the seven-cell workload, truth-FD, UVM Tools,
compute-client, GPU telemetry, kernel-monitor, safety, and cleanup lifecycle.
`run_module_lifecycle.py` is the outer UVM-only stage/load/restore hook built
on the previously exercised revision-prefetch lifecycle primitives.

This remains implementation and live-engagement readiness, not a result.
Controlled attempts have loaded the stale-state module, passed live verifier
and attach checks, and executed real 40-GiB cells, but every seven-cell
excluded preflight is incomplete or rejected. No partial cell is carried into
another attempt, and formal execution remains disabled. The retained attempt
history and current restore-ABI blocker are in
[`live-preflight-readiness.md`](live-preflight-readiness.md); the owner08
interruption boundary is recorded separately in
[`owner08-interruption-20260905.md`](owner08-interruption-20260905.md).

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
python3 -B live_runner.py dry-run \
  --output "$PWD/raw/stale-state-575-preflight-01" \
  --inherited-lease-fds 11 12
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

The implemented live runner accepts the two already locked read-only lease
descriptors from the outer lifecycle. The runner—not a monitor sibling—uses
`pidfd_getfd` to duplicate exactly the owned CUDA 12.9 workload's two
`/dev/nvidia-uvm` FDs. The monitor uses `UVM_TOOLS_INIT_EVENT_TRACKER_V2` to
require exactly one driver-validated VA-space FD and one secondary MM FD; it
does not choose by numeric order or accept an arbitrary same-device FD. The
baseline does not start the BPF
loader and refuses all policy files; native starts only the observer; BPF
starts the observer plus its owned struct_ops policy. It must never run a formal campaign
until the shared driver interface above exists and an excluded complete
preflight passes. A formal `campaign.json` points to the absolute preflight
root; analysis revalidates all seven preflight cells before accepting the 21
formal cells.

The offline suite replays the exact 15-record workload stream through real OS
pipes for the default arm and all six native/BPF delay conditions. It rejects
wrong identity, duplicate/extra/trailing records, late delivery, dirty
counters, and incomplete cleanup. The bridge remains an in-memory test double
in those tests; they do not claim that the proc endpoint ran.
