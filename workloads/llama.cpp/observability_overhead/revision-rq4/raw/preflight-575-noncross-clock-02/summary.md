# RQ4 matched observability experiment

- Phase: `preflight`
- Driver: `575.57.08`
- Target: `_Z9rope_normILb1ELb0Ef6__halfEvPKT1_PT2_iiiiiPKifff14rope_corr_dimsfPKfPKli`
- Blocks requested: `1`

| Config | Valid blocks | Attempts | Prefill tok/s geomean |
|---|---:|---:|---:|
| baseline | 1 | 1 | 7042.82 |
| gpubpf_kernelretsnoop | 0 | 1 | n/a |
| nvbit_kernelretsnoop | 1 | 1 | 87.18 |
| gpubpf_threadhist | 1 | 1 | 4696.38 |
| nvbit_threadhist | 1 | 1 | 6680.84 |

## Paired effects

- kernelretsnoop: 0 paired blocks; incomplete.
- threadhist: 1 paired blocks; mean -28.18 pp, 95% CI [-28.18, -28.18].
