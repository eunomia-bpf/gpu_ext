# Fixed-work trampoline analysis

- Run status: **valid**
- Tested hypothesis: **inconclusive**
- Endpoint decision: **inconclusive**
- Endpoint effect: **-2.7049%** (95% paired-bootstrap interval [-5.4735%, 3.1331%])
- All-five organization guard: **inconclusive** (four Bonferroni-adjusted 98.75% intervals)
- Predeclared materiality interval: **[-1.0%, +1.0%]**
- Raw evidence replayed: **150 cells in 30 distinct arms**

| Blocks | Threads/block | Contrast vs. 128x1,024 | 98.75% interval | Decision |
|---:|---:|---:|---:|---|
| 256 | 512 | -1.4066% | [-4.6880%, 7.4910%] | inconclusive |
| 1024 | 128 | 0.8768% | [-6.2737%, 8.0730%] | inconclusive |
| 2048 | 64 | -2.7464% | [-9.6558%, 1.3123%] | inconclusive |
| 4096 | 32 | -2.7049% | [-5.6351%, 5.7089%] | inconclusive |

| Blocks | Threads/block | No-op delta (us) | Counter delta (us) |
|---:|---:|---:|---:|
| 128 | 1024 | 1.8400 | 558.8160 |
| 256 | 512 | 1.4240 | 582.1120 |
| 1024 | 128 | 1.3440 | 576.4480 |
| 2048 | 64 | 0.2720 | 587.6640 |
| 4096 | 32 | 0.9760 | 581.9200 |

Claim boundary: fixed total work and dynamic warps on this kernel and RTX 5090; not universal block-count independence or warp-leader execution.
