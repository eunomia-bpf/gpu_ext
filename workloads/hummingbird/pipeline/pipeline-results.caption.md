# Hummingbird pipeline comparison — completed 40-cell campaign

Hummingbird completion-fence ablation on the common VGG foreground and
ResNet152 background frontend. Rows use periodic and BurstGPT-derived
foreground arrivals, respectively; background demand is continuous. Columns
show foreground scheduled-arrival-to-numerically-verified-output p99 in
milliseconds (lower is better) and background verified completions inside the
60-second measurement window per second (higher is better). Each bar is a
descriptive median across five randomized paired blocks; all five individual
cell values are plotted, with within-bar horizontal position identifying
blocks 00–04. Native C (square markers, solid bars) and actual host-uBPF JIT
(diamond markers, hatched bars) use the same policy and executor at a fixed
host outstanding-event bound of one (d1) or two (d2). Both arrival modes share
the same y-axis scale for each metric, starting at zero.

Foreground p99 includes every started request, including any final verified
completion after the window. All offered foreground requests were verified
inside their windows in this campaign, so no incomplete-coverage crosses are
needed. This coverage is distinct from meeting the latency SLO. Background
goodput excludes post-window completions. The original profile and
1,811,879 ns foreground SLO are fixed.
The accompanying report supplies all-offered SLO attainment and the four
within-arrival paired comparisons: C d2/d1, BPF d2/d1, BPF/C at d1, and BPF/C
at d2. Their geometric-mean ratios and five-block bootstrap intervals are
not confidence intervals around the descriptive median bars drawn here.

Bound 2 increases background goodput by approximately 15% for both native C
and BPF on both arrival modes, but the joint foreground-protection criterion
is not established. BurstGPT SLO attainment decreases in both implementations.
Seven of the eight BPF/C p99 and goodput ratio intervals include one; the
exception is BurstGPT d1 background goodput, where BPF is 0.24% lower
(paired 95% interval: −0.35% to −0.13%). These results do not establish
statistical equivalence or a protection-preserving overall win.
This figure excludes the old fixed-policy baseline and old 50-cell campaign;
it is a completion-fence ablation, not a full Hummingbird reproduction.
Outstanding host event records are not hardware queue occupancy, and the BPF
policy here executes on the host, not inside a device kernel.

## Reproduction and figure QA

`plot_results.py` reads the closed, complete, exercised formal analysis JSON
and never runs the analyzer, experiment or GPU. Its three synthetic projection
tests passed; the figure contains all 40 audited cells from
`raw/full-575-01/analysis.json`. The script has no CSV exporter; no extra export
framework was added. To reproduce, select a fresh output prefix:

```bash
taskset -c 17 python3 -B workloads/hummingbird/pipeline/test_plot_results.py
taskset -c 17 python3 -B workloads/hummingbird/pipeline/plot_results.py \
  --analysis workloads/hummingbird/pipeline/raw/full-575-01/analysis.json \
  --output-prefix workloads/hummingbird/pipeline/figures/pipeline-comparison-recheck
```

The generated PDF is 504 × 334.8 points (7 × 4.65 inches) with embedded
TrueType text; PNG uses 300 dpi, with shared 7.5–8 point paper typography.
Actual color and 100-dpi grayscale PDF rendering were inspected: all four
panels, axes, five-block markers and hatch/shape distinctions remain legible.
The first layout's footer overlapped the lower x-axis labels; only the bottom
margin was increased, and that first layout is retained as `*-layout-01.*`.
The final `figures/pipeline-comparison.pdf` and `.png` have no such overlap;
`pipeline-comparison-grayscale.png` records the grayscale inspection.
Include at the paper's two-column text width with `figure*`, not one column.
Paper integration and final compiled-page inspection remain root's work.
Missing/nonfinite p99 values reject plotting;
they are never replaced with zero or silently removed. Zero goodput and
incomplete LC coverage remain visible.
