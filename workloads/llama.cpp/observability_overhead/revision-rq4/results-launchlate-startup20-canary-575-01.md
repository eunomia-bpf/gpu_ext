# RQ4 launchlate startup-20 canary, perf-only

Raw run: `raw/table1-launchlate-startup20-canary-575-01` (`launchlate_perf_only`, timestamp `20260905_162745`). Fresh RTX 5090, driver 575.57.08, tinyllama-1.1b-chat Q4_K_M on one pp512 block, target symbol `rope_norm`, `probe_startup_s=20`, one attempt per cell. This canary bypassed the RM/PTIMER calibration, 1.5 us bracket precision check, source-schema checks, and the engagement/correctness/verifier gates, so no clock precision criterion applies to it and no cell is rejected or retried on hook accounting.

| Config | pp tok/s | Overhead vs baseline | Probe counters |
|---|---:|---:|---|
| baseline | 37914.35 | - | - |
| gpubpf_launchlate | 37389.31 | 1.38% | host_launches=0, device_entries=0 |
| nvbit_launchlate | 35196.21 | 7.17% | selected_launches=44, device_entries=44 |

The gpubpf cell preloaded the bpftime agent and ran to completion, but its probe recorded zero host launches and zero device entries while the NVBit cell on the same block recorded 44 selected launches and 44 device entries. The target launches therefore happened; the bpftime agent simply never engaged them. The gpubpf 37389.31 token/s number is an inert-preload diagnostic only: it measures the cost of the preloaded-but-inactive agent and is not a valid launchlate Table 1 result.

The zero-engagement outcome after the 20 s startup window disproves the fixed-startup-delay hypothesis for this attach failure; waiting longer at startup does not make the agent's handlers engage. The next step is agent handler refresh/attach repair so the launchlate probe actually pairs host launches with device entries before any Table 1 cell is claimed.

Primary machine evidence: `raw/table1-launchlate-startup20-canary-575-01/result.json` and `raw/table1-launchlate-startup20-canary-575-01/summary.md`.
