# RTX 5090 fixed-work trampoline block organization

- Phase: `full`
- Status: `complete`

| Cell | Blocks | Threads/block | Active warps | Arm | Pairs | Median delta (ms) | Median overhead | 95% paired-bootstrap interval |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 0 | 128 | 1024 | 4096 | noop | 10 | 0.001840 | 5.637% | [2.756%, 7.133%] |
| 0 | 128 | 1024 | 4096 | counter | 10 | 0.558816 | 1713.101% | [1693.694%, 1727.022%] |
| 1 | 256 | 512 | 4096 | noop | 10 | 0.001424 | 4.251% | [0.976%, 9.108%] |
| 1 | 256 | 512 | 4096 | counter | 10 | 0.582112 | 1768.684% | [1745.426%, 1791.892%] |
| 2 | 1024 | 128 | 4096 | noop | 10 | 0.001344 | 4.131% | [0.972%, 10.288%] |
| 2 | 1024 | 128 | 4096 | counter | 10 | 0.576448 | 1753.848% | [1709.227%, 1773.265%] |
| 3 | 2048 | 64 | 4096 | noop | 10 | 0.000272 | 0.809% | [-1.571%, 6.016%] |
| 3 | 2048 | 64 | 4096 | counter | 10 | 0.587664 | 1774.135% | [1712.285%, 1813.955%] |
| 4 | 4096 | 32 | 4096 | noop | 10 | 0.000976 | 2.664% | [-0.156%, 8.616%] |
| 4 | 4096 | 32 | 4096 | counter | 10 | 0.581920 | 1594.296% | [1537.902%, 1648.515%] |

Positive overhead means attached execution was slower than its paired native run.
This experiment measures the current runtime; it does not assume once-per-warp dispatch.
