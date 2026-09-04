# RQ4 matched observability experiment

- Phase: `preflight`
- Driver: `575.57.08`
- Target: `_Z9rope_normILb1ELb0Ef6__halfEvPKT1_PT2_iiiiiPKifff14rope_corr_dimsfPKfPKli`
- Blocks requested: `1`

| Config | Valid blocks | Attempts | Prefill tok/s geomean |
|---|---:|---:|---:|
| baseline | 1 | 1 | 7099.88 |
| gpubpf_kernelretsnoop | 1 | 1 | 132.33 |
| nvbit_kernelretsnoop | 1 | 1 | 134.91 |
| gpubpf_threadhist | 1 | 1 | 5331.84 |
| nvbit_threadhist | 1 | 1 | 6662.24 |

## Paired effects

- kernelretsnoop: 1 paired blocks; mean -0.04 pp, 95% CI [-0.04, -0.04].
- threadhist: 1 paired blocks; mean -18.74 pp, 95% CI [-18.74, -18.74].
