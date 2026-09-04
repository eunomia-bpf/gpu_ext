# RQ4 matched observability experiment

- Phase: `preflight`
- Driver: `575.57.08`
- Target: `_Z9rope_normILb1ELb0Ef6__halfEvPKT1_PT2_iiiiiPKifff14rope_corr_dimsfPKfPKli`
- Blocks requested: `1`

| Config | Valid blocks | Attempts | Prefill tok/s geomean |
|---|---:|---:|---:|
| baseline | 1 | 1 | 7052.77 |
| gpubpf_kernelretsnoop | 1 | 1 | 131.60 |
| nvbit_kernelretsnoop | 1 | 1 | 135.63 |
| gpubpf_threadhist | 1 | 1 | 4773.51 |
| nvbit_threadhist | 1 | 1 | 6677.75 |

## Paired effects

- kernelretsnoop: 1 paired blocks; mean -0.06 pp, 95% CI [-0.06, -0.06].
- threadhist: 1 paired blocks; mean -27.00 pp, 95% CI [-27.00, -27.00].
