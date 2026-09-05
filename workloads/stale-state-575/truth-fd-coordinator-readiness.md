# Stale-state truth-FD coordinator readiness

Date: 2026-09-04
Scope: source and CPU validation only; no module load, BPF attach, or GPU cell

## Outcome

The workload-owned phase-truth pipe now has a fail-closed coordinator in
`coordinator.py::TruthFDCoordinator`. After validating the exact
`workload_ready` identity, it configures the selected native or BPF generation
and only then invokes the caller-owned release gate. Each of the seven exact
`phase_start` records is published to the driver-owned proc endpoint at the
configured 0, 100, or 1000 ms delay. The coordinator retains source,
eligibility, write-window, driver-publication, and status-observation
timestamps.

The coordinator rejects malformed, duplicate, extra, reordered, late, or
incomplete truth records. It also rejects early/late publication observations,
wrong consumer activity, error counters, non-closing callback/effect counters,
and dirty cleanup. A configured generation is disabled on both success and
failure. The default-UVM path never configures, publishes, or disables policy
state and returns no publication rows.

The returned schema says `truth_source=workload_phase_fd`,
`synthetic_source=false`, and
`evidence_scope=coordinator_only_not_complete_cell`. It also says
`experiment_evidence=false`, because this component does not collect the
driver diagnostics, UVM events, workload result, safety snapshots, leases, or
continuous monitors required for a valid cell.

## Validation completed

- `python3 -B workloads/stale-state-575/test_offline.py`: 39 tests passed.
- All six native/BPF delay conditions consumed an exact 15-record byte stream
  through a real OS pipe and closed publication, observation, counter, and
  cleanup checks in the in-memory bridge model.
- The default arm consumed the same pipe while a guard bridge rejected any
  attempted policy mutation.
- Negative cases passed for wrong ready identity/type, duplicate and extra
  fields, trailing records, late delivery, partial-line pipe timeout, late
  status observation, dirty live counters, and post-configuration parse
  failure with cleanup. A policy decision after driver publication but before
  the later host status observation remains valid; the latter is not mislabeled
  as a consumer barrier.
- `make test-offline`: passed the same 39 Python tests, the 15-check driver ABI
  test, the native model assertions, and the 306,012-call native/host-uBPF JIT
  differential with zero contract errors.
- Python bytecode compilation and `git diff --check` passed.

## Remaining live gate

This is not evidence that the proc endpoint or GPU path ran. The installed
driver still lacks the staged bridge. The next controlled step is to install
and load the prepared module, admit/attach the BPF policy and diagnostic
observer, integrate leases plus UVM/compute/GPU/kernel monitors around this
coordinator, and run one excluded seven-cell preflight. Only a passing
preflight can enable the 21-cell formal campaign.
