# RQ4 matched observability experiment

- Phase: `preflight`
- Driver: `575.57.08`
- Target: `_Z9rope_normILb1ELb0Ef6__halfEvPKT1_PT2_iiiiiPKifff14rope_corr_dimsfPKfPKli`
- Blocks requested: `1`

| Config | Valid blocks | Attempts | Prefill tok/s geomean |
|---|---:|---:|---:|
| baseline | 1 | 1 | 7059.59 |
| gpubpf_kernelretsnoop | 1 | 1 | 132.79 |
| nvbit_kernelretsnoop | 1 | 1 | 136.31 |
| gpubpf_threadhist | 1 | 1 | 5323.57 |
| nvbit_threadhist | 1 | 1 | 6669.64 |

## Paired effects

- kernelretsnoop: 1 paired blocks; mean -0.05 pp, 95% CI [-0.05, -0.05].
- threadhist: 1 paired blocks; mean -19.07 pp, 95% CI [-19.07, -19.07].
