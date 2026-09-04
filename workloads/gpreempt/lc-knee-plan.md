# GPreempt LC knee sweep — frozen supporting experiment

Prepared 2026-09-03 America/Vancouver (2026-09-04 UTC). This experiment is
supporting evidence for RQ3. It
locates the foreground load knee under a fixed closed-loop background supply;
it is not a new headline or decisive experiment and is not an equivalence
test. Its scope is fixed before execution and must not be expanded after seeing
results.

## Frozen matrix

- Foreground VGG19 periodic-FIFO rates: exactly 500, 625, and 800 requests/s.
- Background ResNet152 load: closed-loop continuous in every cell.
- Arms: native CUDA stream priorities, original-C GPreempt, and BPF/JIT
  GPreempt, using the existing host-mapped compatibility transport.
- Preflight: one 10-second three-arm block at 800 requests/s. It exercises the
  most demanding prespecified point and is excluded from estimates.
- Full run: three paired blocks × three rates × three arms × 60 seconds = 27
  cells. Rate and arm positions are independently seeded Latin rotations.
- No rate, block, arm, workload, metric, or repetition may be appended after
  execution. A different scope is a separate prospective experiment, never an
  extension of this campaign.

The models, batch size, CUDA graphs, preprocessing, timing window, seed,
numerical checks, policy inputs, kernel repetition, 400 W host safety policy,
driver requirement, leases, telemetry, engagement checks, cleanup checks, raw
request audit, and paired-ratio estimator remain those of
[`load-study-plan.md`](load-study-plan.md). Periodic foreground accounting keeps
all offered requests and backlog visible; continuous background has no invented
offered-request denominator.

## Commands

CPU-only plan inspection and tests:

```bash
python3 -B run_load_study.py preflight --study lc-knee --plan
python3 -B run_load_study.py full --study lc-knee --plan
python3 -B test_load_study.py
python3 -B test_analyze_load_study.py
```

Only after separate GPU coordination and a valid preflight:

```bash
sudo -n python3 -B run_load_study.py preflight --study lc-knee \
  --output raw/lc-knee-preflight-01
sudo -n python3 -B run_load_study.py full --study lc-knee \
  --preflight raw/lc-knee-preflight-01 \
  --output raw/lc-knee-full-01
python3 -B analyze_load_study.py raw/lc-knee-full-01 \
  --output raw/lc-knee-full-01/independent-audit.json
```

The full runner fails closed before creating its output directory unless
`--preflight` names a separate campaign whose frozen plan is LC800/10s/three
arms, whose summary is completed and complete, and whose three raw cells pass
the independent analyzer's correctness, inventory, safety, cleanup, telemetry,
and engagement checks. The preflight path is recorded in the full plan; cells
are not copied or pooled. Plan-only output reports the requirement without
reading that path.

Use a new output directory for every attempt. Preflight success does not
authorize changing this matrix, and no partial prefix is a completed result.
