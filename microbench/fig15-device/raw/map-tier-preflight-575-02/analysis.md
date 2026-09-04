# RTX 5090 device-map placement analysis

- Run status: valid_preflight
- Tested hypothesis: not_tested
- Raw arm processes replayed: 8

## Arm latency

| Arm | median us/launch |
|---|---:|
| native | 2.608000 |
| noop | 7.808000 |
| device_update | 9.360000 |
| host_update | 27.120000 |
| rpc_update | 33704.753900 |
| device_lookup | 9.184000 |
| host_lookup | 10.608000 |
| rpc_lookup | 33706.558250 |

Preflight establishes execution only; it is not a paper result.
