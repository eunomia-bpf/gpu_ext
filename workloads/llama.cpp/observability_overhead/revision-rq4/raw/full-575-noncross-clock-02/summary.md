# RQ4 matched observability experiment

- Phase: `full`
- Driver: `575.57.08`
- Target: `_Z9rope_normILb1ELb0Ef6__halfEvPKT1_PT2_iiiiiPKifff14rope_corr_dimsfPKfPKli`
- Blocks requested: `10`

| Config | Valid blocks | Attempts | Prefill tok/s geomean |
|---|---:|---:|---:|
| baseline | 10 | 10 | 38056.93 |
| gpubpf_kernelretsnoop | 10 | 10 | 128.37 |
| nvbit_kernelretsnoop | 10 | 10 | 144.30 |
| gpubpf_threadhist | 10 | 10 | 36531.77 |
| nvbit_threadhist | 10 | 10 | 34136.77 |

## Paired effects

- kernelretsnoop: 10 paired blocks; mean -0.04 pp, 95% CI [-0.04, -0.04].
- threadhist: 10 paired blocks; mean 6.29 pp, 95% CI [6.13, 6.47].
