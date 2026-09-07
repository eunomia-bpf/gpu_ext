#!/usr/bin/env python3
"""Render the device-side observability overhead comparison as a 1x2 bar figure.

Left panel: submitted-paper P40 setup (values transcribed in
p40-submitted-table1.json with provenance). Right panel: RTX 5090 campaign
summary (revision-rq4/results-table1-warp-plt-575-06/summary.json, ten
paired runs per arm). Log scale because overheads span three orders of
magnitude. No GPU execution and no paper edits.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

TOOLS = ("kernelretsnoop", "threadhist", "launchlate")
TOOL_TICKS = ("kernelret-\nsnoop", "threadhist", "launchlate")
SYSTEMS = ("gpubpf", "nvbit")
SYSTEM_TICKS = ("gpubpf", "NVBit")
COLORS = ("#0072B2", "#D97706")
STYLE = {"font.family": "DejaVu Sans", "font.size": 8,
         "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
         "legend.fontsize": 7, "axes.spines.top": False,
         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42}
HERE = Path(__file__).resolve().parent
DEFAULT_P40 = HERE / "p40-submitted-table1.json"
DEFAULT_5090 = (HERE / "revision-rq4" / "results-table1-warp-plt-575-06"
                / "summary.json")


def p40_overheads(path: Path) -> dict[str, list[float]]:
    data = json.loads(Path(path).read_text())
    if sorted(data["tools"]) != sorted(TOOLS):
        raise ValueError("P40 transcription must cover exactly the three tools")
    values = {}
    for system in SYSTEMS:
        values[system] = []
        for tool in TOOLS:
            value = data["tools"][tool][system]
            if type(value) not in (int, float) or not 0 < value < 100:
                raise ValueError(f"P40 {tool}/{system} overhead out of range: {value}")
            values[system].append(float(value))
    return values


def rtx5090_overheads(path: Path) -> dict[str, list[float]]:
    summary = json.loads(Path(path).read_text())
    arms = {entry["arm"]: entry for entry in summary["arms"]}
    expected = {f"{system}_{tool}" for system in SYSTEMS for tool in TOOLS} | {"baseline"}
    if set(arms) != expected:
        raise ValueError("RTX 5090 summary arms differ from the six tool arms plus baseline")
    values = {}
    for system in SYSTEMS:
        values[system] = []
        for tool in TOOLS:
            entry = arms[f"{system}_{tool}"]
            if entry.get("cells") != 10:
                raise ValueError(f"RTX 5090 {system}_{tool} does not have ten cells")
            value = entry["mean_overhead_pct"]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 100:
                raise ValueError(f"RTX 5090 {system}_{tool} overhead out of range: {value}")
            values[system].append(float(value))
    return values


def _draw(p40: dict[str, list[float]], rtx: dict[str, list[float]],
          paths: list[Path]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(3.4, 1.6))
        for column, (panel, values, title) in enumerate(
                zip(axes, (p40, rtx), ("(a) P40, Llama 1B prefill", "(b) RTX 5090, TinyLlama-1.1B prefill"))):
            for system_index, (system, color) in enumerate(zip(SYSTEMS, COLORS)):
                offset = (system_index - .5) * .4
                bars = panel.bar([i + offset for i in range(3)], values[system],
                                 color=color, width=.38)
                for bar, value in zip(bars, values[system]):
                    panel.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.12,
                               f"{value:.3g}", ha="center", va="bottom", fontsize=6.5)
            panel.set_yscale("log")
            panel.set_ylim(.1, 250)
            panel.set_yticks((.1, 1, 10, 100), ("0.1", "1", "10", "100"))
            panel.set_xticks(range(3), TOOL_TICKS)
            panel.set_title(title, fontsize=7.5, pad=3)
            panel.grid(axis="y", alpha=.25, linewidth=.6, which="major")
            panel.minorticks_off()
        axes[0].set_ylabel("overhead (%), log")
        handles = [Line2D([0], [0], color=color, marker="s", markersize=5,
                          linewidth=0, label=label)
                   for color, label in zip(COLORS, SYSTEM_TICKS)]
        figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.02),
                      ncol=2, frameon=False, handlelength=1.1,
                      handletextpad=.4, columnspacing=1.2)
        figure.tight_layout(rect=(0, 0, 1, .84), w_pad=1.6)
        try:
            for path in paths:
                figure.savefig(path, dpi=300)
        finally:
            plt.close(figure)


def render(p40_path: Path, rtx_path: Path, prefix: Path) -> list[Path]:
    p40, rtx = p40_overheads(p40_path), rtx5090_overheads(rtx_path)
    paths = [prefix.with_suffix(suffix) for suffix in (".pdf", ".png")]
    if any(path.exists() for path in paths):
        raise FileExistsError("output exists; choose a new explicit figure prefix")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    _draw(p40, rtx, paths)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p40", type=Path, default=DEFAULT_P40,
                        help="transcribed submitted-paper P40 values with provenance")
    parser.add_argument("--rtx5090", type=Path, default=DEFAULT_5090,
                        help="RTX 5090 campaign summary.json")
    parser.add_argument("--output-prefix", type=Path,
                        default=HERE / "figures" / "obs-overhead-bars")
    args = parser.parse_args()
    outputs = render(args.p40, args.rtx5090, args.output_prefix)
    print("\n".join(str(path) for path in outputs))
