# RQ4 matched observability experiment

- Phase: `preflight`
- Driver: `575.57.08`
- Target: `_Z9rope_normILb1ELb0Ef6__halfEvPKT1_PT2_iiiiiPKifff14rope_corr_dimsfPKfPKli`
- Blocks requested: `1`

| Config | Valid blocks | Attempts | Prefill tok/s geomean |
|---|---:|---:|---:|
| baseline | 1 | 1 | 7032.67 |
| gpubpf_kernelretsnoop | 0 | 1 | n/a |
| nvbit_kernelretsnoop | 1 | 1 | 87.11 |
| gpubpf_threadhist | 1 | 1 | 4790.49 |
| nvbit_threadhist | 1 | 1 | 6674.33 |

## Paired effects

- kernelretsnoop: 0 paired blocks; incomplete.
- threadhist: 1 paired blocks; mean -26.79 pp, 95% CI [-26.79, -26.79].
