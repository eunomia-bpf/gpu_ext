# RTX 5090 fixed-work trampoline block organization

- Phase: `preflight`
- Status: `complete`

| Cell | Blocks | Threads/block | Active warps | Arm | Pairs | Median delta (ms) | Median overhead | 95% paired-bootstrap interval |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 2 | 1024 | 128 | 4096 | noop | 1 | 0.004032 | 68.108% | [68.108%, 68.108%] |
| 2 | 1024 | 128 | 4096 | counter | 1 | 0.021248 | 358.919% | [358.919%, 358.919%] |

Positive overhead means attached execution was slower than its paired native run.
This experiment measures the current runtime; it does not assume once-per-warp dispatch.
