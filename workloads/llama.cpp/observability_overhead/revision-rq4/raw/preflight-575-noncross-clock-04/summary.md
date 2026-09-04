# RQ4 matched observability experiment

- Phase: `preflight`
- Driver: `575.57.08`
- Target: `_Z9rope_normILb1ELb0Ef6__halfEvPKT1_PT2_iiiiiPKifff14rope_corr_dimsfPKfPKli`
- Blocks requested: `1`

| Config | Valid blocks | Attempts | Prefill tok/s geomean |
|---|---:|---:|---:|
| baseline | 1 | 1 | 7074.94 |
| gpubpf_kernelretsnoop | 1 | 1 | 131.23 |
| nvbit_kernelretsnoop | 1 | 1 | 132.66 |
| gpubpf_threadhist | 1 | 1 | 5464.88 |
| nvbit_threadhist | 1 | 1 | 6663.09 |

## Paired effects

- kernelretsnoop: 1 paired blocks; mean -0.02 pp, 95% CI [-0.02, -0.02].
- threadhist: 1 paired blocks; mean -16.94 pp, 95% CI [-16.94, -16.94].
