# Stale-state live preflight implementation readiness

Date: 2026-09-04
Scope: implementation and offline validation only; no module load, BPF attach,
or GPU execution

## Implemented path

The excluded seven-cell path is now represented end to end in source:

- `live.bpf.c` observes the versioned driver diagnostic through fentry and
  places only completed records in a 64 MiB ring buffer. The same object
  contains the policy, but the native loader disables both policy autoload and
  struct_ops-map creation before load.
- The driver records the VA-space creator TGID at creation and carries that
  stable owner in every diagnostic. This avoids attributing UVM worker-thread
  callbacks through `current`; the observer filters on the driver-owned
  identity instead.
- `live_loader.c` owns every link, records the actual verifier output, filters
  by the stable workload owner, validates the 176-byte diagnostic semantics, and
  fails if selected/finished/retained counts differ or any event is lost.
- `observer_protocol.py` re-parses the retained JSON with duplicate-key and
  exact-schema rejection, verifies ownership, and creates `policy-final.json`
  only when every observer and driver counter closes.
- The raw-cell validator independently replays `policy-observer.jsonl`, checks
  the verifier transcript and observer stderr, and requires the normalized
  decision stream to match the raw observer decisions exactly. A baseline row
  rejects every observer, verifier, publication, decision, and final-policy
  artifact.
- `TruthFDCoordinator.before_release` lets the runner attach UVM Tools and the
  observer while the workload is blocked. The bridge remains off until the
  first bootstrap snapshot becomes eligible, avoiding expected GPU work in an
  initial no-snapshot interval from being mislabeled as a policy failure.
- `live_runner.py` owns the fresh output tree and all child processes. It
  captures an empty compute boundary before launch, starts continuous GPU,
  compute, and kernel monitoring, duplicates only the owned workload UVM FD,
  runs the truth-FD coordinator, stops/reaps only its children, checks the
  post-cell idle state, and invokes the frozen raw-cell validator. The
  baseline never loads the observer/policy object and refuses policy files.
- `run_module_lifecycle.py` accepts an explicit candidate `nvidia-uvm.ko`,
  stages it under a fresh stale-state namespace, stops the same services and
  uses the same UVM-only remove/insert checks as the previously exercised
  revision-prefetch lifecycle, passes both lease descriptors to the preflight
  child, and restores the admitted 575 UVM module and original service state
  in `finally`. Before removal it proves that the live module's BTF interface,
  version, parameters, and non-diagnostic role match the admitted restore.
  Linux does not expose the loaded module's source pathname, so this is an
  explicit compatibility admission rather than a claim that its on-disk
  origin can be reconstructed.

## Offline checks

- Driver bridge patch check, 30-assertion pure-model test, and a fresh complete
  `nvidia-uvm.ko` build pass after adding stable owner identity. The resulting
  BTF has a 176-byte diagnostic whose final member is `owner_tgid`.
- Workload bridge build/test: the 15-check ABI test, observer fentry section,
  policy struct_ops section, generated skeleton, and `-Werror` userspace
  loader build pass.
- Python suite: 46 tests pass, including real-pipe truth replay, observer
  native/BPF ownership, duplicate JSON, event-loss, counter-drift,
  raw/normalized mismatch, missing verifier evidence, baseline artifact
  rejection, before-release failure, and delayed-bootstrap configuration
  ordering.
- The live runner dry-run emits the exact seven-cell order, declares no module
  or GPU action, and preserves a null baseline policy command.
- Python bytecode compilation and whitespace checks pass.

These checks are not live verifier, module-lifecycle, GPU-engagement, UVM
event, numerical-correctness, or performance evidence.

## Remaining gate

Independent source review passed before execution. The controlled operator
must now provide a fresh candidate module path and fresh stage, preflight, and
lifecycle-record paths. The first live attempt is the excluded seven-cell
preflight only. Any module restoration error, verifier/attach
failure, missing/foreign client, dropped diagnostic or UVM event, policy
counter mismatch, numerical error, monitor gap, surviving child, or nonempty
post-state invalidates the attempt. Formal 21-cell execution remains disabled
until that complete preflight passes.
