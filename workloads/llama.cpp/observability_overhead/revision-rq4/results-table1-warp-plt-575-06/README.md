# results-table1-warp-plt-575-06
Hardware: NVIDIA RTX 5090, driver 575.57.08. Workload: llama.cpp pp512.
Coverage: 10 rotated blocks, 70 numeric rc0 cells.
Baseline mean throughput is 37586.322536 tok/s. Mean tool throughput and
same-block mean overhead are:

| tool | gpubpf mean tok/s (overhead) | NVBit mean tok/s (overhead) |
| --- | --- | --- |
| kernelretsnoop | 3493.6654284 (90.7050859%) | 142.4363463 (99.6210304%) |
| threadhist | 36471.0324999 (2.9653081%) | 33694.9158263 (10.3501103%) |
| launchlate | 37502.5450369 (0.2208152%) | 34279.7492721 (8.7959164%) |
Interpretation: gpubpf lower overhead in all three configurations; kernelretsnoop remains expensive.
Note: only the 10 initially missing gpubpf launchlate cells were filled after correcting the uprobe binary path; other completed cells were not rerun.
