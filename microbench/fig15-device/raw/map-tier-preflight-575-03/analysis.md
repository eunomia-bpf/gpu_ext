# RTX 5090 device-map placement analysis

- Run status: valid_preflight
- Tested hypothesis: not_tested
- Raw arm processes replayed: 8

## Arm latency

| Arm | median us/launch |
|---|---:|
| native | 2.480000 |
| noop | 5.776000 |
| device_update | 9.040000 |
| host_update | 27.104000 |
| rpc_update | 33760.208150 |
| device_lookup | 9.312000 |
| host_lookup | 10.480000 |
| rpc_lookup | 33771.663650 |

Preflight establishes execution only; it is not a paper result.
