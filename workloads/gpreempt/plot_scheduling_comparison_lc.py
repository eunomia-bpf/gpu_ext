#!/usr/bin/env python3
"""Render the matched scheduling comparison as a single-column LC-latency figure.

Two stacked panels from the published per-point data of the 2x2 figure
(figures/scheduling-comparison-2x2.points.json): (a) XSched LC queue-entry
p99 (three implementations), (b) GPreempt LC response p99 (three
implementations across three BE loads). Bars are medians. Background-side
throughput/goodput is reported in the paper text, not in this figure.
No GPU execution and no paper edits.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

import plot_load_study as gp_plot
from plot_scheduling_comparison_bars import XS_ARMS, XS_TICKS, GP_TICKS, \
    SCENARIO_TICKS, COLORS, STYLE, DEFAULT_POINTS, load_points


def _labels(panel, bars, values, fontsize=5.5):
    for bar, value in zip(bars, values):
        panel.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                   f"{value:.3g}", ha="center", va="bottom", rotation=90,
                   fontsize=fontsize)


def _draw(data: dict, paths: list[Path]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    xs, gp = data["xsched"], data["gpreempt"]
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(2, 1, figsize=(3.4, 2.6))
        panel = axes[0]
        values = [statistics.median(p["queue_p99_s"] for p in xs if p["arm"] == arm)
                  for arm in XS_ARMS]
        _labels(panel, panel.bar(range(3), values, color=COLORS, width=.62),
                values, fontsize=6.5)
        panel.set_ylim(0, max(values) * 1.3)
        panel.set_xticks(range(3), XS_TICKS)
        panel.set_ylabel("LC queue-entry\np99 (s)")
        panel.set_title("(a) XSched workload", loc="left", fontsize=7.5, pad=7)
        panel = axes[1]
        tops = []
        for arm_index, (arm, color) in enumerate(zip(gp_plot.ARMS, COLORS)):
            values = [statistics.median(p["response_p99_ms"] for p in gp
                                        if p["scenario"] == scenario and p["arm"] == arm)
                      for scenario in gp_plot.SCENARIOS]
            offset = (arm_index - 1) * .26
            _labels(panel, panel.bar([i + offset for i in range(3)], values,
                                     color=color, width=.24), values)
            tops.append(max(values))
        panel.set_ylim(0, max(tops) * 1.3)
        panel.set_xticks(range(3), SCENARIO_TICKS)
        panel.set_xlabel("BE supply (req/s)")
        panel.set_ylabel("LC response\np99 (ms)")
        panel.set_title("(b) GPreempt workload", loc="left", fontsize=7.5, pad=7)
        for panel in axes:
            panel.yaxis.set_major_locator(MaxNLocator(nbins=4))
            panel.ticklabel_format(axis="y", style="plain", useOffset=False)
            panel.grid(axis="y", alpha=.25, linewidth=.6)
        handles = [Line2D([0], [0], color=color, marker="s", markersize=5,
                          linewidth=0, label=label)
                   for color, label in zip(COLORS, ("Native", "Original", "BPF port"))]
        figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.0),
                      ncol=3, frameon=False, handlelength=1.1,
                      handletextpad=.4, columnspacing=.7)
        figure.tight_layout(rect=(0, 0, 1, .92), h_pad=1.2)
        try:
            for path in paths:
                figure.savefig(path, dpi=300)
        finally:
            plt.close(figure)


def render(points_path: Path, prefix: Path) -> list[Path]:
    data = load_points(points_path)
    paths = [prefix.with_suffix(suffix) for suffix in (".pdf", ".png")]
    if any(path.exists() for path in paths):
        raise FileExistsError("output exists; choose a new explicit figure prefix")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    _draw(data, paths)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, default=DEFAULT_POINTS,
                        help="published per-point data of the 2x2 figure")
    parser.add_argument("--output-prefix", type=Path,
                        default=Path(__file__).resolve().parent / "figures"
                        / "scheduling-comparison-lc-bars")
    args = parser.parse_args()
    outputs = render(args.points, args.output_prefix)
    print("\n".join(str(path) for path in outputs))
