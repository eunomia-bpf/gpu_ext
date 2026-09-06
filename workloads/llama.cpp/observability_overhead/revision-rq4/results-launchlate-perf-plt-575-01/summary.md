# RQ4 Table 1 launchlate performance-only campaign

- Kind: `launchlate_perf_only`
- Timestamp: `20260905_203555`
- Driver: `575.57.08`
- Target: `_Z9rope_normILb1ELb0Ef6__halfEvPKT1_PT2_iiiiiPKifff14rope_corr_dimsfPKfPKli`
- Blocks requested: `1`
- pp: `512`

Gates bypassed: RM/PTIMER calibration, 1.5 us bracket precision check, source-schema checks, and engagement/correctness/verifier gates. No cell is rejected or retried on hook accounting.

| Config | Valid blocks | Attempts | Prefill tok/s geomean | Mean overhead vs baseline |
|---|---:|---:|---:|---:|
| baseline | 1 | 1 | 37842.42 | - |
| gpubpf_launchlate | 1 | 1 | 37088.27 | 1.99% |
| nvbit_launchlate | 1 | 1 | 33376.35 | 11.80% |

## Blocks

| Block | baseline tok/s | gpubpf tok/s | gpubpf overhead | NVBit tok/s | NVBit overhead |
|---:|---:|---:|---:|---:|---:|
| 1 | 37842.42 | 37088.27 | 1.99% | 33376.35 | 11.80% |

Positive overhead means token/s degradation relative to the same-block no-probe baseline.
Probe counters are recorded per cell for audit only and never gate a cell.
