# POD setup-versus-steady follow-up

This is an implementation-ready, unrun follow-up to the completed POD-Attention
operator study.  It reuses the real operator path through
[`../run_phase_study.py`](../run_phase_study.py); it does not replace or modify
the prior campaign or any prior raw result.

## Frozen matrix

The formal campaign is exactly 15 fresh processes: `pod_inline`, `pod_cuda`,
and `pod_bpf` in a seeded randomized order inside each of five paired blocks.
Every process runs only the previously valid Llama-3-8B / decode-batch-32
shape, with 10 warmups and 100 retained steady samples.  A separate excluded
preflight is one randomized three-arm block with three samples per arm.

The monotonic timeline records coordinator-side cell start, loader spawn and
ready (BPF only), client spawn and exit, and cleanup completion.  The child
records process-main entry, standard-library import completion, runtime-import
start and completion, immediately before the first diagnostic launch, after
its synchronization, warmup completion, and steady-sample completion.  The
first successful adapter launch is recorded once per exact CUfunction name;
the frozen shape is expected to launch one of the six registered alternatives,
while the other five stay explicitly unobserved in that cell.

The runner inherits the existing hard FP16 comparison and full FP32
characterization, CTA/atomic exactly-once audit, device engine 2, bridge launch
and shared-memory checks, exact loader inventory, driver and telemetry checks,
runtime inventory, owned private-segment cleanup, post-safety checks, and both
exclusive leases.  A missing marker, changed runtime, failed gate, incomplete
preflight, dirty cleanup, or output collision fails the campaign.

## CPU-only inspection

This command prints the exact formal matrix without reading build artifacts,
creating the output directory, acquiring a lease, or starting any process:

```bash
python3 ../run_phase_study.py full \
  --dry-run \
  --output ../raw/phase-full-UNRUN \
  --preflight ../raw/phase-preflight-UNRUN
```

The dry-run is planning evidence only.  Offline tests can be run with:

```bash
(cd .. && python3 -m unittest test_phase_study)
```

## Claim boundary

These markers can decompose bounded deployment, first-launch, warmup, and
recurring costs for this POD adapter on this one shape.  Loader-ready time and
first-launch/synchronization are deliberately reported as bounded phases, not
as a general binary-attachment cost.  Even a successful real run cannot claim
arbitrary attachment cost, a constant total trampoline cost, strict-verifier
admission, or full POD serving-system performance.
