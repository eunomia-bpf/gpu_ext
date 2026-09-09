# Completed four-ratio sweep

All 300 cells across 3 workloads, 4 ratios, 5 policies and 5 repetitions
completed with zero tenant/tool exit errors. The existing analyzer verified
all saved cells and generated both absolute completion-time and paired
high-priority speedup plots. All 12 plotted points use five complete pairs.
Ratio 1.8 contains only the user cancellation marker, not measurements.

K-Means at requested ratio 1.5: median high-priority completion is
26.546135 s with scheduling and 26.183911 s with memory plus scheduling.
Paired median speedup over Default is 7.127865x and 7.252826x respectively.
The paired median high-priority improvement is 1.3780%. Low-priority
completion medians are 48.424704 s and 48.889332 s respectively.
The raw values for all five policies are in final-completion-summary.json.

The timing evidence does not establish the page-migration mechanism causing
the speedup. The explicit CPU build overlap inventory identifies six GEMM
ratio 1.2 cells; all observations are retained unchanged. The premature
completion notification and failed first restoration are documented without
rewriting their original logs. The manual retry restored driver, services
and storage loaders; GPU ownership was explicitly handed to the experiment
Codex after lock release. No new GPU experiment, commit or push was performed.

Only independent previews were refreshed in this completion turn. The paper's
frozen observation CSV and figure were not replaced without approval of the
new K-Means endpoint. The previously approved prose range still contains its
1.3780% improvement.
