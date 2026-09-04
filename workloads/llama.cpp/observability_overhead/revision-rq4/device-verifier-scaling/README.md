# Device-verifier admission scaling

This CPU-only harness measures the one-time latency of bpftime's real
`verify_gpu_program` API for the frozen experiment in [`plan.md`](plan.md).
It never loads a policy, opens a GPU device, or uses either GPU experiment
lease. It also never builds in or links against `build-table1-575-strict`.

## Isolated build

The build uses the current bpftime verifier sources, system libbpf only as the
verifier's link dependency, the repository's local Catch2 source, and a
separate `Release` directory. It builds only the probe and its verifier-library
dependencies with at most two build jobs.

```bash
taskset -c 16-23 nice -n 10 ./build_isolated.sh
```

The probe embeds the bpftime Git revision and build type. Its `--describe`
mode constructs and checks the program but does not invoke the verifier or read
a clock.

## Offline checks

These tests do not call the verifier. When `VERIFIER_SCALING_PROBE` is set,
they additionally exercise all ten `--describe` arms, which still do not call
the verifier.

```bash
VERIFIER_SCALING_PROBE=/home/yunwei37/workspace/gpu/bpftime-device-verifier-scaling-build/verifier_scaling_probe \
  python3 -m unittest -v test_verifier_scaling.py
python3 run_verifier_scaling.py --dry-run
```

## Real preflight and full run

Do not treat preflight as a result. Run it only after the GPU campaign and
concurrent CPU builds have released their assigned cores, even though this
harness itself is CPU-only:

```bash
env CUDA_VISIBLE_DEVICES= taskset -c 23 \
  python3 run_verifier_scaling.py --preflight \
    --probe /home/yunwei37/workspace/gpu/bpftime-device-verifier-scaling-build/verifier_scaling_probe \
    --bpftime-root /home/yunwei37/workspace/gpu/bpftime-table1-575 \
    --output-dir raw/preflight-01

env CUDA_VISIBLE_DEVICES= taskset -c 23 \
  python3 run_verifier_scaling.py \
    --probe /home/yunwei37/workspace/gpu/bpftime-device-verifier-scaling-build/verifier_scaling_probe \
    --bpftime-root /home/yunwei37/workspace/gpu/bpftime-table1-575 \
    --output-dir raw/scaling-575-01

python3 analyze_verifier_scaling.py raw/scaling-575-01
```

The analyzer reopens every raw stdout, stderr, and execution record; rebuilds
the schedule independently; and computes medians, paired CFG ratios, and
Theil--Sen exponents with 20,000 fixed-seed block bootstraps. More than 10% of
cells with wall time above 1.25 times process CPU time, or any major page fault
during a timed API call, makes the hypothesis verdict inconclusive without
deleting rows. Every raw row remains part of the run.

The runner records source revision/status and executable path/size/mtime at
both boundaries. The analyzer invalidates a run if the verifier source or
probe changes while cells are collected; it does not use content hashes.
