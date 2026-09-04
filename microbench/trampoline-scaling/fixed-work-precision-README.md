# Fixed-work precision follow-up

This prospective follow-up attempts to resolve the valid but inconclusive
ten-block fixed-work result without changing the tested kernel, handler, hook
site, per-kernel work, or +/-1% decision margin. It aggregates 512 identical
launches in each CUDA-event interval and uses exactly 48 fresh randomized
blocks. The old result selected this fixed budget but is never pooled into the
new analysis.

CPU-only validation is:

```sh
python3 -m unittest -v test_offline.py test_fixed_work.py \
  test_fixed_work_precision.py
python3 -m py_compile run_scaling.py run_fixed_work.py \
  analyze_fixed_work.py run_fixed_work_precision.py \
  analyze_fixed_work_precision.py test_fixed_work.py \
  test_fixed_work_precision.py
```

The existing fixed-work binaries are reused because the precision change is
only a runner parameter change. After acquiring the existing read-only GPU and
struct-ops leases, run:

```sh
python3 run_fixed_work_precision.py \
  --phase preflight \
  --output raw/fixed-work-precision-preflight-575-01
python3 run_fixed_work_precision.py \
  --phase full \
  --output raw/fixed-work-precision-full-575-01
python3 analyze_fixed_work_precision.py \
  --result raw/fixed-work-precision-full-575-01/result.json
```

The full design is fixed at 48 blocks, 144 fresh arm processes, and 720 timed
cells. Each process uses 16 warmups, 512 timed launches, and 16 hook repetitions
per thread. All six arm permutations appear eight times; every arm occupies
every position 16 times. Each block has one independently randomized cell order
shared by its three arms.

The analyzer reopens every raw application, loader/map, bootstrap, telemetry,
safety, and lifecycle file. `result.json` contributes only the frozen schedule
and raw-directory locators. The endpoint median 95% interval and all four
Bonferroni-adjusted 98.75% guard intervals must fit wholly inside +/-1% to
support the bounded claim. The run is never extended or stopped based on
precision. See [fixed-work-precision-plan.md](fixed-work-precision-plan.md) and
[fixed-work-precision-plan-review.md](fixed-work-precision-plan-review.md).
