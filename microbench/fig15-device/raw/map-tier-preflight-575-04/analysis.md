# RTX 5090 device-map placement analysis

- Run status: valid_preflight
- Tested hypothesis: not_tested
- Raw arm processes replayed: 8

## Arm latency

| Arm | median us/launch |
|---|---:|
| native | 2.512000 |
| noop | 5.808000 |
| device_update | 8.864000 |
| host_update | 27.311999 |
| rpc_update | 35383.697500 |
| device_lookup | 9.280000 |
| host_lookup | 10.128000 |
| rpc_lookup | 33770.992300 |

Preflight establishes execution only; it is not a paper result.
