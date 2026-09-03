FineMoE dynamic-set policy versus demand-only and all-positive speculation on
Qwen1.5-MoE-A2.7B-Chat BF16, RTX 5090, NVIDIA 575.57.08. Each of five randomized
paired blocks contains all four arms; each cell generates 128 verified tokens
from eight held-out inputs after one excluded warmup. (a) Application-window
throughput: all five cells are shown, with a short line for the arm median.
(b) Each narrow stacked bar is one cell (left to right: blocks 00–04), showing
logical completed speculative payload in decimal GB through final drain,
partitioned by first demand use, eviction without use, or unused residency.
Resident-unused is right-censored, not waste; the small top segments retain
their true scale. Demand-only has no speculative copies, shown as crosses at
zero. These are payload accounting quantities, not measured PCIe traffic or
time saved; primary throughput excludes final drain. Native C and real host
uBPF JIT use the same dynamic-set policy and execution path. Both reduce unused
speculation and improve throughput over the all-positive ablation, but neither
beats demand-only throughput on this workload. This is a component policy port,
not a full FineMoE reproduction or evidence of a new BPF policy. Paired-effect
confidence intervals are reported separately, not drawn around these medians.

The vector PDF has a 7-inch canvas, matching the paper's letter-page width minus
its 0.75-inch side margins. Fonts are 7.5–8 points at that width; do not shrink
the two-panel figure into one column. The adjacent CSV retains all twenty
points in bytes and token/s, reconstructed from original worker records and
checked against the existing independent analysis by `../plot_results.py`.
