#!/usr/bin/env python3
"""Render the five matched policy ports as a 1x5 panel figure.

Each panel is one independent replication: bars for baseline (no policy),
the original implementation, and the gpubpf port of the same policy, from
port-panels.json (transcribed published values with per-panel sources).
XSched and GPREEMPT are in the separate scheduling figure; mechanism cost
is reported in text. No GPU execution and no paper edits.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

COLORS = {"baseline": "#666666", "original": "#D97706", "port": "#0072B2"}
LEGEND = (("baseline", "Baseline"), ("original", "Native policy"), ("port", "BPF port"))
STYLE = {"font.family": "DejaVu Sans", "font.size": 7,
         "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
         "legend.fontsize": 7, "axes.spines.top": False,
         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42}
HERE = Path(__file__).resolve().parent


def load_panels(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text())
    if data.get("schema") != "matched_port_panels_v1":
        raise ValueError("panel file is not the matched-port panel data")
    panels = data["panels"]
    if len(panels) != 5 or len({p["id"] for p in panels}) != 5:
        raise ValueError("expected five distinct panels")
    for panel in panels:
        for group in panel["groups"]:
            for key, value in group.items():
                if key == "label":
                    continue
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"bad value in panel {panel['id']}: {group}")
    return panels


def _draw(panels: list[dict], paths: list[Path]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(1, 5, figsize=(7.2, 1.6))
        for panel, axis in zip(panels, axes.flat):
            width = .8 / max(len(LEGEND) - 1, 1)
            positions = {}
            for group_index, group in enumerate(panel["groups"]):
                arms = [key for key, _ in LEGEND if key in group]
                for slot, arm in enumerate(arms):
                    center = group_index + (slot - (len(arms) - 1) / 2) * width * 1.1
                    positions.setdefault(arm, []).append(
                        axis.bar(center, group[arm], color=COLORS[arm],
                                 width=width))
            top = max(g[k] for g in panel["groups"] for k in g if k != "label")
            axis.set_ylim(0, top * 1.32)
            axis.set_xticks(range(len(panel["groups"])),
                            [g["label"] for g in panel["groups"]])
            if len(panel["groups"]) > 1:
                axis.tick_params(axis="x", labelrotation=20)
            axis.set_ylabel(panel["metric"])
            axis.set_title(panel["title"], fontsize=7, pad=2)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)
            axis.grid(axis="y", alpha=.25, linewidth=.6)
        handles = [Line2D([0], [0], color=COLORS[key], marker="s", markersize=5,
                          linewidth=0, label=label) for key, label in LEGEND]
        figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.02),
                      ncol=3, frameon=False, handlelength=1.1,
                      handletextpad=.4, columnspacing=1.0)
        figure.tight_layout(rect=(0, 0, 1, .94), h_pad=1.0, w_pad=1.5)
        try:
            for path in paths:
                figure.savefig(path, dpi=300)
        finally:
            plt.close(figure)


def render(panels_path: Path, prefix: Path) -> list[Path]:
    panels = load_panels(panels_path)
    paths = [prefix.with_suffix(suffix) for suffix in (".pdf", ".png")]
    if any(path.exists() for path in paths):
        raise FileExistsError("output exists; choose a new explicit figure prefix")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    _draw(panels, paths)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=Path, default=HERE / "port-panels.json")
    parser.add_argument("--output-prefix", type=Path,
                        default=HERE / "figures" / "matched-port-panels")
    args = parser.parse_args()
    outputs = render(args.panels, args.output_prefix)
    print("\n".join(str(path) for path in outputs))
