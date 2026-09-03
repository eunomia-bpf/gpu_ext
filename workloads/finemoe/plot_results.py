#!/usr/bin/env python3
"""Plot every completed FineMoE cell; no workload execution or new statistics."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

import analyze_results as analysis

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gpreempt"))
from plot_scheduling_comparison import COLORS, STYLE

ARMS = analysis.ARMS
LABELS = ("Demand-only", "All-positive\nablation", "FineMoE\nnative C", "FineMoE\nBPF")
CATEGORIES = ("first_use", "evicted_unused", "resident_unused")
METRICS = ("tokens_per_second", "drained.prefetch_copy_bytes",
           *(f"drained.prefetch_{kind}_bytes" for kind in CATEGORIES))


def load_points(campaign: Path) -> list[dict]:
    """Reuse the published audit and reconstruct only plotted raw quantities.

    This does not rerun the 36-array numerical preflight. Its saved audit remains
    the correctness evidence; every plotted value is checked against that audit.
    """
    audit = json.loads((campaign / "independent-analysis.json").read_text())
    if (audit.get("complete") is not True or audit.get("valid_blocks") != 5
            or audit.get("valid_cells") != 20):
        raise ValueError("requires the complete five-block campaign")
    cells = audit["cells"]
    expected = {(block, arm) for block in range(5) for arm in ARMS}
    if len(cells) != 20 or {(cell["block"], cell["arm"]) for cell in cells} != expected:
        raise ValueError("missing, duplicate, or unexpected block/arm")
    points = []
    for cell in sorted(cells, key=lambda c: (c["block"], ARMS.index(c["arm"]))):
        relative = Path(f"block-{cell['block']:02d}") / cell["arm"] / "worker-result.json"
        worker = json.loads((campaign / relative).read_text())
        if worker["arm"] != cell["arm"]:
            raise ValueError(f"wrong raw arm: {relative}")
        metrics, _ = analysis.reconstruct(worker)
        for key in METRICS:
            if not math.isclose(metrics[key], cell["metrics"][key], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"raw/audit metric mismatch: {relative}: {key}")
        point = {"block": cell["block"], "arm": cell["arm"],
                 "tokens_per_second": metrics["tokens_per_second"],
                 "prefetch_bytes": metrics["drained.prefetch_copy_bytes"],
                 **{f"{kind}_bytes": metrics[f"drained.prefetch_{kind}_bytes"]
                    for kind in CATEGORIES}, "source": str(relative)}
        points.append(point)
    return points


def draw(points: list[dict], prefix: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    # Letter paper minus 0.75-inch side margins in docs/paper/main.tex: 7 inches.
    # Preserve the canvas width; tight-bbox resizing could shrink printed fonts.
    with plt.rc_context(STYLE):
        figure, (throughput, payload) = plt.subplots(1, 2, figsize=(7, 2.85))
        arm_colors = (COLORS[0], "#8064A2", COLORS[1], COLORS[2])
        part_colors = (COLORS[2], "#AAAAAA", COLORS[1])
        hatches = ("", "///", "...")
        for index, (arm, color, marker) in enumerate(zip(ARMS, arm_colors, ("o", "^", "s", "D"))):
            cells = [point for point in points if point["arm"] == arm]
            values = [point["tokens_per_second"] for point in cells]
            for offset, cell in enumerate(cells):
                x = index + (offset - 2) * .115
                throughput.scatter(x, cell["tokens_per_second"], s=20, marker=marker,
                                   facecolors="none", edgecolors=color, linewidths=1, zorder=3)
                bottom = 0
                for kind, part_color, hatch in zip(CATEGORIES, part_colors, hatches):
                    height = cell[f"{kind}_bytes"] / 1e9
                    payload.bar(x, height, bottom=bottom, width=.095, color=part_color,
                                edgecolor="#444444", linewidth=.4, hatch=hatch, zorder=3)
                    bottom += height
                if not bottom:
                    payload.plot(x, 0, marker="x", markersize=4, color=COLORS[0],
                                 markeredgewidth=1, clip_on=False, zorder=4)
            throughput.plot([index - .3, index + .3], [statistics.median(values)] * 2,
                            color=color, linewidth=1.4, zorder=4)
        for panel, letter in ((throughput, "(a)"), (payload, "(b)")):
            panel.set_xticks(range(4), LABELS)
            panel.set_xlim(-.5, 3.5)
            panel.set_ylim(bottom=0)
            panel.yaxis.set_major_locator(MaxNLocator(nbins=5))
            panel.grid(axis="y", alpha=.25, linewidth=.6, zorder=0)
            panel.text(.015, .975, letter, transform=panel.transAxes, va="top", fontsize=8)
        throughput.set_ylim(0, max(p["tokens_per_second"] for p in points) * 1.15)
        payload.set_ylim(0, max(p["prefetch_bytes"] for p in points) / 1e9 * 1.15)
        throughput.set_ylabel("Throughput (token/s)")
        payload.set_ylabel("Completed speculative payload (GB)")
        legend = [Patch(facecolor=color, edgecolor="#444444", hatch=hatch, label=label)
                  for color, hatch, label in zip(part_colors, hatches,
                      ("First demand use", "Evicted unused", "Resident unused (censored)"))]
        figure.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .99),
                      ncol=3, frameon=False, columnspacing=1.5, handlelength=1.8)
        figure.text(.5, .025,
                    "Five paired blocks: each marker/bar is one cell. Throughput: short line = median.",
                    ha="center", fontsize=7.5)
        figure.subplots_adjust(left=.075, right=.985, bottom=.245, top=.825, wspace=.33)
        try:
            for suffix in (".pdf", ".png"):
                figure.savefig(prefix.with_suffix(suffix), dpi=300)
        finally:
            plt.close(figure)


def render(campaign: Path, prefix: Path) -> None:
    points = load_points(campaign)
    paths = [prefix.with_suffix(ext) for ext in (".pdf", ".png", ".csv")]
    if any(path.exists() for path in paths):
        raise FileExistsError("choose a fresh output prefix; existing artifacts are retained")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    draw(points, prefix)
    with paths[2].open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(points[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=HERE / "raw/full-v1")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    render(args.campaign, args.output_prefix)
