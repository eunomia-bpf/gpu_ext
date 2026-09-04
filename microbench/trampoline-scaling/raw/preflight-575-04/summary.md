# RTX 5090 device-trampoline scaling

- Phase: `preflight`
- Status: `complete`

| Cell | Blocks | Active warps | Arm | Pairs | Median delta (ms) | Median overhead | 95% paired-bootstrap interval |
|---:|---:|---:|---|---:|---:|---:|---:|
| 0 | 256 | 2048 | noop | 1 | 0.004000 | 61.576% | [61.576%, 61.576%] |
| 0 | 256 | 2048 | counter | 1 | 0.007808 | 120.197% | [120.197%, 120.197%] |

Positive overhead means attached execution was slower than its paired native run.
This experiment measures the current runtime; it does not assume once-per-warp dispatch.
