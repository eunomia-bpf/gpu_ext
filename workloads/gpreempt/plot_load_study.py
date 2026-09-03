#!/usr/bin/env python3
"""Plot all cells of a completed, independently audited GPreempt load study.

Points are per-cell statistics; horizontal bars are five-block medians, not
confidence intervals. No CUDA execution or paper-source mutation occurs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

ARMS = ("native", "original_gpreempt", "bpf_gpreempt")
SCENARIOS = ("be100", "be200", "be_continuous")
LABELS = ("Native baseline", "Original-C GPreempt", "BPF GPreempt")
COLORS = ("#666666", "#D97706", "#0072B2")


def plot_points(audit: dict) -> list[dict]:
    """Reject incomplete or nonfinite inputs instead of silently dropping them."""
    if not audit.get("complete") or not audit.get("formal_eligible"):
        raise ValueError("a complete formal audit is required; preflight is not a result")
    if audit.get("rejected_cells") or audit.get("incomplete_cells"):
        raise ValueError("rejected or missing cells must be resolved and reported first")
    if set(audit["scenarios"]) != set(SCENARIOS):
        raise ValueError("all three prespecified scenarios are required")
    points = []
    for scenario in SCENARIOS:
        data = audit["scenarios"][scenario]
        if not data["complete"] or data["valid_paired_blocks"] != 5:
            raise ValueError("each scenario needs five complete paired blocks")
        seen = set()
        for cell in data["per_cell_points"]:
            key = (cell["block"], cell["arm"])
            if key in seen or key[0] not in range(5) or key[1] not in ARMS:
                raise ValueError("duplicate or unexpected cell")
            seen.add(key)
            foreground = cell["metrics"]["vgg_rt"]
            background = cell["metrics"]["resnet152_be"]
            latency = foreground["p99_response_us"] / 1000
            goodput = background["goodput_rps"]
            coverage = foreground["completion_coverage"]
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError("foreground response p99 must be positive and finite")
            if not math.isfinite(goodput) or goodput < 0:
                raise ValueError("background goodput must be finite and nonnegative")
            if coverage is None or not math.isfinite(coverage) or not 0 <= coverage <= 1:
                raise ValueError("periodic foreground requires explicit completion coverage")
            conditional = foreground["p99_is_conditional"]
            if type(conditional) is not bool or conditional != (coverage < 1):
                raise ValueError("foreground censoring label disagrees with completion coverage")
            points.append({"scenario": scenario, "block": key[0], "arm": key[1],
                           "response_p99_ms": latency, "background_goodput_rps": goodput,
                           "conditional": conditional, "completion_coverage": coverage})
        if seen != {(block, arm) for block in range(5) for arm in ARMS}:
            raise ValueError("missing paired cell")
    return points


def render(audit: dict, prefix: Path) -> list[Path]:
    points = plot_points(audit)
    paths = [prefix.with_suffix(suffix) for suffix in (".pdf", ".png")]
    if any(path.exists() for path in paths):
        raise FileExistsError("figure output exists; use a new explicit output prefix")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    definitions = (("response_p99_ms", "LC response p99 (ms, lower is better)"),
                   ("background_goodput_rps", "BE goodput (req/s, higher is better)"))
    for panel, (metric, label) in zip(axes, definitions):
        for scenario_index, scenario in enumerate(SCENARIOS):
            for arm_index, (arm, color) in enumerate(zip(ARMS, COLORS)):
                cells = sorted((p for p in points if p["scenario"] == scenario and p["arm"] == arm),
                               key=lambda p: p["block"])
                center = scenario_index + (arm_index - 1) * 0.23
                for block_index, cell in enumerate(cells):
                    conditional = metric == "response_p99_ms" and cell["conditional"]
                    panel.scatter(center + (block_index - 2) * 0.025, cell[metric],
                                  marker="^" if conditional else "o", s=22,
                                  facecolors="none" if conditional else color,
                                  edgecolors=color, linewidths=0.8, alpha=0.65, zorder=3)
                median = statistics.median(cell[metric] for cell in cells)
                panel.plot([center - 0.078, center + 0.078], [median, median],
                           color=color, linewidth=2.3, zorder=4)
        panel.set_xticks(range(3), ["100 req/s", "200 req/s", "Continuous"])
        panel.set_xlabel("Background supply (LC fixed at 100 req/s)")
        panel.set_ylabel(label)
        panel.set_xlim(-0.5, 2.5)
        panel.set_ylim(bottom=0)
        panel.grid(axis="y", color="#dddddd", linewidth=0.5, zorder=0)

    handles = [Line2D([0], [0], color=color, marker="o", markersize=4, linewidth=2,
                      label=label) for color, label in zip(COLORS, LABELS)]
    figure.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
                  bbox_to_anchor=(0.5, 1.01))
    conditional = [point for point in points if point["conditional"]]
    note = "Each point: one 60 s cell. Bars: five-block medians. Host-mapped flag transport."
    if conditional:
        coverage = min(point["completion_coverage"] for point in conditional)
        note += f"\nOpen triangles: completed-only p99; minimum LC coverage {coverage:.1%}."
    else:
        note += "\nAll offered LC requests verified; BE goodput excludes completions after the window."
    figure.text(0.5, 0.015, note, ha="center", va="bottom", fontsize=8)
    figure.tight_layout(rect=(0, 0.14, 1, 0.9))
    prefix.parent.mkdir(parents=True, exist_ok=True)
    try:
        for path in paths:
            figure.savefig(path, dpi=200, bbox_inches="tight")
    finally:
        plt.close(figure)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    arguments = parser.parse_args()
    result = render(json.loads(arguments.audit.read_text()), arguments.output_prefix)
    print("\n".join(str(path) for path in result))
