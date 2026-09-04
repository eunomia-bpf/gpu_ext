# RQ4 matched observability experiment

- Phase: `preflight`
- Driver: `575.57.08`
- Target: `_Z9rope_normILb1ELb0Ef6__halfEvPKT1_PT2_iiiiiPKifff14rope_corr_dimsfPKfPKli`
- Blocks requested: `1`

| Config | Valid blocks | Attempts | Prefill tok/s geomean |
|---|---:|---:|---:|
| baseline | 1 | 1 | 7034.56 |
| gpubpf_kernelretsnoop | 1 | 1 | 133.21 |
| nvbit_kernelretsnoop | 1 | 1 | 135.56 |
| gpubpf_threadhist | 1 | 1 | 5296.53 |
| nvbit_threadhist | 1 | 1 | 6671.40 |

## Paired effects

- kernelretsnoop: 1 paired blocks; mean -0.03 pp, 95% CI [-0.03, -0.03].
- threadhist: 1 paired blocks; mean -19.54 pp, 95% CI [-19.54, -19.54].
