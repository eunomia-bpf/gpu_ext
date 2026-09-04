# RQ4 matched observability experiment

- Phase: `full`
- Driver: `575.57.08`
- Target: `_Z9rope_normILb1ELb0Ef6__halfEvPKT1_PT2_iiiiiPKifff14rope_corr_dimsfPKfPKli`
- Blocks requested: `10`

| Config | Valid blocks | Attempts | Prefill tok/s geomean |
|---|---:|---:|---:|
| baseline | 10 | 10 | 38045.27 |
| gpubpf_kernelretsnoop | 0 | 10 | n/a |
| nvbit_kernelretsnoop | 10 | 10 | 144.30 |
| gpubpf_threadhist | 0 | 10 | n/a |
| nvbit_threadhist | 10 | 10 | 34141.82 |

## Paired effects

- kernelretsnoop: 0 paired blocks; incomplete.
- threadhist: 0 paired blocks; incomplete.
