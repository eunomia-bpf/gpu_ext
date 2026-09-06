# RQ4 Table 1 launchlate performance-only campaign

- Kind: `launchlate_perf_only`
- Timestamp: `20260905_203821`
- Driver: `575.57.08`
- Target: `_Z9rope_normILb1ELb0Ef6__halfEvPKT1_PT2_iiiiiPKifff14rope_corr_dimsfPKfPKli`
- Blocks requested: `10`
- pp: `512`

Gates bypassed: RM/PTIMER calibration, 1.5 us bracket precision check, source-schema checks, and engagement/correctness/verifier gates. No cell is rejected or retried on hook accounting.

| Config | Valid blocks | Attempts | Prefill tok/s geomean | Mean overhead vs baseline |
|---|---:|---:|---:|---:|
| baseline | 10 | 10 | 37762.86 | - |
| gpubpf_launchlate | 10 | 10 | 37177.99 | 1.55% |
| nvbit_launchlate | 10 | 10 | 33858.20 | 10.33% |

## Blocks

| Block | baseline tok/s | gpubpf tok/s | gpubpf overhead | NVBit tok/s | NVBit overhead |
|---:|---:|---:|---:|---:|---:|
| 1 | 37588.32 | 37255.27 | 0.89% | 33548.52 | 10.75% |
| 2 | 37738.46 | 37253.48 | 1.29% | 34150.14 | 9.51% |
| 3 | 37470.49 | 37310.27 | 0.43% | 34628.30 | 7.59% |
| 4 | 37988.22 | 37184.49 | 2.12% | 34790.16 | 8.42% |
| 5 | 37629.66 | 37106.29 | 1.39% | 33567.45 | 10.80% |
| 6 | 37826.58 | 36999.81 | 2.19% | 33472.09 | 11.51% |
| 7 | 37781.91 | 37312.53 | 1.24% | 33871.85 | 10.35% |
| 8 | 37831.83 | 37089.83 | 1.96% | 33834.73 | 10.57% |
| 9 | 37779.48 | 37072.15 | 1.87% | 33369.11 | 11.67% |
| 10 | 37996.97 | 37197.19 | 2.10% | 33383.94 | 12.14% |

Positive overhead means token/s degradation relative to the same-block no-probe baseline.
Probe counters are recorded per cell for audit only and never gate a cell.
