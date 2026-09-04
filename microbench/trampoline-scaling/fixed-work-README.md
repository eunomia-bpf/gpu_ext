# Fixed-work block-organization follow-up

This follow-up reuses the validated RTX 5090 trampoline-scaling harness while
holding total launched work and dynamic warps fixed. It tests a bounded
block-organization claim; it does not assume once-per-warp dispatch or claim
that block count is independently manipulable from threads per block.

CPU-only validation, which does not launch CUDA, is:

```sh
python3 -m unittest -v test_offline.py test_fixed_work.py
python3 -m py_compile run_scaling.py run_fixed_work.py \
  analyze_fixed_work.py test_offline.py test_fixed_work.py
```

Compiling the harness also does not execute a GPU workload:

```sh
make -j4 \
  BPFTIME_ROOT=/home/yunwei37/workspace/gpu/bpftime-table1-575 \
  BPFTIME_BUILD=/home/yunwei37/workspace/gpu/bpftime-table1-575/build-table1-575
```

After acquiring the existing GPU and struct-ops leases, execute a new
dependency preflight and then the full campaign:

```sh
python3 run_fixed_work.py \
  --phase preflight \
  --output raw/fixed-work-preflight-575-01
python3 run_fixed_work.py \
  --phase full \
  --output raw/fixed-work-full-575-01
python3 analyze_fixed_work.py \
  --result raw/fixed-work-full-575-01/result.json
```

The full run has ten paired blocks, three seed-randomized and position-balanced
arms, five cells per arm, and 150 timed measurements. Cell order is randomized
once per block and shared by all arms. The primary metric is the paired endpoint
difference-in-differences between 128x1,024 and 4,096x32, normalized by mean
endpoint native time. A predeclared all-five guard also compares each of the
other four organizations with 128x1,024 using Bonferroni-adjusted 98.75%
paired-bootstrap intervals. The bounded statement is supported only when the
endpoint 95% interval and all four guard intervals lie inside +/-1%.

The analyzer treats `result.json` only as the frozen schedule and a set of raw
directory locators. It independently reopens each arm's application,
loader/map, agent-bootstrap, telemetry, paired-safety, and lifecycle evidence;
derived validity or summary dictionaries in `result.json` are not gates. Each
arm and telemetry source must be distinct. Consequently, earlier campaigns
that do not contain these raw safety/lifecycle files are intentionally rejected
by this analyzer rather than upgraded into publication evidence.

See [fixed-work-plan.md](fixed-work-plan.md) and
[fixed-work-plan-review.md](fixed-work-plan-review.md).
