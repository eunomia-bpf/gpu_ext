# RTX 5090 device-trampoline scaling

This directory contains a standalone supporting experiment for Reviewer D's
question about device-side trampoline scaling. It measures the current
bpftime-backed gpubpf hook; it does not assume that the handler runs once per
warp. The audited runtime source currently replaces the explicit hook stub
with an ordinary PTX `call`/`call.uni` and uses the default minimal save mode
without a register guard.

The two frozen series are:

- fixed active work: 2,048 active warps while launched blocks increase from
  256 to 4,096; and
- fixed launch geometry: 4,096 blocks x 256 threads while active warps
  increase from 2,048 to 32,768.

Every cell runs the same CUDA binary in three arms: native, an attached
return-only BPF handler, and an attached per-thread map-counter handler. The
counter arm is an engagement/losslessness control, not a claim that all useful
policies have the same cost.

## Offline validation

The following commands do not launch CUDA or use `sudo`:

```sh
make -j4 \
  BPFTIME_ROOT=/home/yunwei37/workspace/gpu/bpftime-table1-575 \
  BPFTIME_BUILD=/home/yunwei37/workspace/gpu/bpftime-table1-575/build-table1-575
python3 -m unittest -v test_offline.py
python3 -m py_compile run_scaling.py test_offline.py
```

The tests cover the matrix, compiled PTX hook-site schema, application output
oracle, complete BPF-map segment oracle, target-specific attach evidence,
runtime feature gate, deterministic pairing, ambient-injection rejection, and
read-only coordination leases. Missing lock files are never created.

Each scheduled arm owns a separate telemetry process and raw CSV. The runner
stops that process, closes its stream, and validates its samples before the
post-arm safety check, so telemetry cannot hold the UVM reference being
checked. It then allows the ordinary fixed 60-second window for NVIDIA UVM's
asynchronous reference release, while continuing to reject a nonzero final
count, active compute process, new kernel anomaly, or nonempty struct_ops
state. The validated telemetry path and summary are attached to the arm, and
the arm is checkpointed only after every gate passes.

## Real execution

Run the three-arm dependency preflight first, using a new output directory:

```sh
python3 run_scaling.py \
  --phase preflight \
  --output raw/preflight-575-01
```

Only after that result is `complete`, run the ten paired blocks:

```sh
python3 run_scaling.py \
  --phase full \
  --output raw/full-575-01
```

Use `--resume` with the exact same arguments after an interruption. Resume
accepts only the frozen parameter/schedule record and skips only already-valid
arms. It never selects retries based on performance.

The runner requires the pre-created GPU and structural-operation lock files,
an idle RTX 5090 on driver 575.57.08, the 400 W power service, an anomaly-free
kernel log, no extant struct-ops objects, and the selected CUDA/LLVM runtime.
It owns a unique shared-memory segment and process group for each attached
arm, validates exact identity before removal, and records raw application,
loader, agent, telemetry, and safety evidence. A measurement is valid only if:

1. all 1,048,576 output slots match the active/inactive oracle;
2. the separate marker reports exactly 32 callbacks;
3. the agent records both exact kernel targets, the marker fallback transform,
   the target stub transform, module load, and successful attach;
4. the counter arm's complete map contents match the independent per-thread
   oracle; and
5. both BPF links detach, owned processes exit, the private segment is gone,
   telemetry passes, and the GPU returns to its admitted safety state.

The selected Table 1 runtime has device verification disabled. Therefore this
experiment provides performance, correctness, and engagement evidence only;
it is not verifier-enforcement evidence. No GPU result exists in this
directory until a real run produces a `raw/` record.

See [plan.md](plan.md) for the frozen design and [plan-review.md](plan-review.md)
for the OpenCode pre-execution review trail.
