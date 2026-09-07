#!/usr/bin/env python3
"""Render the matched scheduling comparison as a compact 1x4 bar figure.

Reads the published per-point data of the 2x2 figure
(figures/scheduling-comparison-2x2.points.json) and draws within-workload
median bars: (a) XSched LC queue-entry p99, (b) XSched BE throughput,
(c) GPreempt LC response p99, (d) GPreempt BE goodput. No GPU execution,
no paper edits, and the 2x2 outputs are never touched.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

import plot_load_study as gp_plot

XS_ARMS = ("native", "xsched", "bpftime_hpf")
XS_TICKS = ("Native", "Orig.\nXSched", "BPF-HPF")
GP_TICKS = ("Native", "Original C", "BPF-GPreempt")
SCENARIO_TICKS = ("100", "200", "cont.")
COLORS = gp_plot.COLORS
STYLE = {"font.family": "DejaVu Sans", "font.size": 8,
         "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
         "legend.fontsize": 7, "axes.spines.top": False,
         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42}
DEFAULT_POINTS = (Path(__file__).resolve().parent / "figures"
                  / "scheduling-comparison-2x2.points.json")


def load_points(points_path: Path) -> dict:
    data = json.loads(Path(points_path).read_text())
    if data.get("schema") != "scheduling_comparison_2x2_v1":
        raise ValueError("points file is not the published scheduling comparison data")
    xs, gp = data["xsched"], data["gpreempt"]
    if sorted(p["arm"] for p in xs) != sorted(arm for arm in XS_ARMS for _ in range(10)):
        raise ValueError("XSched points must cover three arms x ten blocks")
    expected_gp = sorted((scenario, arm) for scenario in gp_plot.SCENARIOS
                         for arm in gp_plot.ARMS for _ in range(5))
    if sorted((p["scenario"], p["arm"]) for p in gp) != expected_gp:
        raise ValueError("GPreempt points must cover three scenarios x three arms x five blocks")
    for point in (*xs, *gp):
        for key, value in point.items():
            if key in ("arm", "scenario", "source_cell"):
                continue
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"non-finite measurement in point {point}")
    return data


def medians(points: list[dict], metric: str, key: str, values) -> list[float]:
    return [statistics.median(p[metric] for p in points if p[key] == value)
            for value in values]


def _bar_labels(panel, bars, values):
    for bar, value in zip(bars, values):
        panel.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                   f"{value:.3g}", ha="center", va="bottom", rotation=90,
                   fontsize=6)


def _draw(data: dict, paths: list[Path]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    xs, gp = data["xsched"], data["gpreempt"]
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(1, 4, figsize=(7.2, 2.0))
        panels = (
            (axes[0], "queue_p99_s", "LC queue-entry\np99 (s)",
             medians(xs, "queue_p99_s", "arm", XS_ARMS), XS_TICKS, None),
            (axes[1], "background_kernels_s", "BE throughput\n(kernels/s)",
             medians(xs, "background_kernels_s", "arm", XS_ARMS), XS_TICKS, None),
            (axes[2], "response_p99_ms", "LC response\np99 (ms)", None,
             SCENARIO_TICKS, "BE supply (req/s)"),
            (axes[3], "background_goodput_rps", "BE goodput\n(req/s)", None,
             SCENARIO_TICKS, "BE supply (req/s)"),
        )
        for column, (panel, metric, ylabel, xs_values, ticks, xlabel) in enumerate(panels):
            if column < 2:
                bars = panel.bar(range(3), xs_values, color=COLORS, width=.62)
                _bar_labels(panel, bars, xs_values)
                top = max(xs_values)
            else:
                source, key = (gp, "scenario")
                tops = []
                for arm_index, (arm, color) in enumerate(zip(gp_plot.ARMS, COLORS)):
                    values = [statistics.median(p[metric] for p in source
                                                if p[key] == scenario and p["arm"] == arm)
                              for scenario in gp_plot.SCENARIOS]
                    offset = (arm_index - 1) * .26
                    bars = panel.bar([i + offset for i in range(3)], values,
                                     color=color, width=.24)
                    _bar_labels(panel, bars, values)
                    tops.append(max(values))
                top = max(tops)
            panel.set_ylim(0, top * 1.28)
            panel.set_xticks(range(3), ticks)
            if xlabel:
                panel.set_xlabel(xlabel)
            panel.set_ylabel(ylabel)
            panel.yaxis.set_major_locator(MaxNLocator(nbins=4))
            panel.ticklabel_format(axis="y", style="plain", useOffset=False)
            panel.grid(axis="y", alpha=.25, linewidth=.6)
            panel.text(.97, .97, f"({'abcd'[column]})", transform=panel.transAxes,
                       fontsize=8, va="top", ha="right")
        handles = [Line2D([0], [0], color=color, marker="s", markersize=5,
                          linewidth=0, label=label)
                   for color, label in zip(COLORS, GP_TICKS)]
        figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(.72, 1.04),
                      ncol=3, frameon=False, handlelength=1.1,
                      handletextpad=.4, columnspacing=.9)
        figure.tight_layout(rect=(0, 0, 1, .9), w_pad=1.6)
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
                        / "scheduling-comparison-1x4-bars")
    args = parser.parse_args()
    outputs = render(args.points, args.output_prefix)
    print("\n".join(str(path) for path in outputs))
