# RTX 5090 STRICT uniform-map analysis

- Run status: valid
- Tested hypothesis: inconclusive
- Raw arm processes replayed: 72
- Attached cells with exact STRICT acceptance: 60

## Arm latency

| Arm | median us/launch |
|---|---:|
| native | 2.014250 |
| noop | 3.928500 |
| device_update | 3.860250 |
| host_update | 3.882750 |
| device_lookup | 3.841500 |
| host_lookup | 4.148250 |

## Co-primary paired effects

| Comparison | pairs | ratio (97.5% interval) | delta us (97.5% interval) |
|---|---:|---:|---:|
| host_vs_device_update | 12 | 1.0008 [0.9891, 1.0143] | 0.003250 [-0.042500, 0.055000] |
| host_vs_device_lookup | 12 | 1.0778 [1.0644, 1.0833] | 0.298500 [0.251000, 0.319000] |

## Descriptive controls

| Comparison | pairs | ratio (95% interval) | delta us (95% interval) |
|---|---:|---:|---:|
| noop_vs_native | 12 | 1.9779 [1.9035, 2.3796] | 1.973250 [1.836750, 2.263250] |
| device_update_vs_noop | 12 | 0.9818 [0.9654, 1.0042] | -0.071750 [-0.138000, 0.016000] |
| device_lookup_vs_noop | 12 | 0.9910 [0.9610, 1.0193] | -0.035000 [-0.155750, 0.073500] |

The two placement intervals are Bonferroni-adjusted co-primary comparisons.
The controls are descriptive and do not isolate a causal component cost.
Map-effect readback is idempotent: it proves the final nonzero effect, not callback invocation cardinality or verifier soundness.

Scope: one 32-thread block; constant key/value; scalar per-thread callbacks; STRICT verifier; RTX 5090; lookup and update only
