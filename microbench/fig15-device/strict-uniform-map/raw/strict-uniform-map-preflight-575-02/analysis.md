# RTX 5090 STRICT uniform-map analysis

- Run status: valid_preflight
- Tested hypothesis: not_tested
- Raw arm processes replayed: 6
- Attached cells with exact STRICT acceptance: 5

## Arm latency

| Arm | median us/launch |
|---|---:|
| native | 2.416000 |
| noop | 5.712000 |
| device_update | 8.976000 |
| host_update | 8.864000 |
| device_lookup | 9.360000 |
| host_lookup | 10.416000 |

Preflight establishes execution only; it is not a paper result.
