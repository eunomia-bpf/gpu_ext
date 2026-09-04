# RTX 5090 device-map placement analysis

- Run status: valid
- Tested hypothesis: supported
- Raw arm processes replayed: 128

## Arm latency

| Arm | median us/launch |
|---|---:|
| native | 1.985500 |
| noop | 3.917750 |
| device_update | 3.845500 |
| host_update | 36.185499 |
| rpc_update | 33843.156875 |
| device_lookup | 3.822750 |
| host_lookup | 4.174750 |
| rpc_lookup | 33830.915469 |

## Paired effects

| Comparison | pairs | ratio (interval) | delta us (interval) | confidence |
|---|---:|---:|---:|---:|
| host_vs_device_update | 16 | 9.4307 [9.3789, 9.4896] | 32.346999 [31.908751, 32.834001] | 97.5% |
| host_vs_device_lookup | 16 | 1.0904 [1.0797, 1.1113] | 0.345750 [0.309500, 0.418750] | 97.5% |
| rpc_vs_device_update | 16 | 8843.7589 [8762.4048, 8888.6052] | 33839.265625 [33799.904500, 34050.692828] | 95.0% |
| rpc_vs_device_lookup | 16 | 8990.6410 [8793.7436, 9091.8033] | 33827.080719 [33793.565656, 34951.764469] | 95.0% |
| noop_vs_native | 16 | 1.9912 [1.9148, 2.4731] | 1.972750 [1.847000, 2.290500] | 95.0% |

The two host-mapped/device-resident intervals are the Bonferroni-adjusted co-primary comparisons. RPC and no-op comparisons are descriptive.

Scope: one 32-thread block; current verification-disabled scalar per-thread runtime; RTX 5090; lookup and update only
