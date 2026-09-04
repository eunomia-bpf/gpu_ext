#!/usr/bin/env python3
"""Plot the paired RTX 5090 trampoline-scaling result from raw records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ARMS = ("baseline", "noop", "counter")
BLOCKS = tuple(range(10))


def load_pairs(path: Path) -> dict[tuple[int, int, str], float]:
    result = json.loads(path.read_text())
    if result.get("status") != "complete" or result.get("failures") != []:
        raise RuntimeError("full campaign is not complete and failure-free")

    records = result.get("records", [])
    if len(records) != len(BLOCKS) * len(ARMS):
        raise RuntimeError("full campaign does not contain exactly 30 arms")

    pairs: dict[tuple[int, int, str], float] = {}
    for record in records:
        block = record.get("block")
        arm = record.get("arm")
        measurements = record.get("measurements", [])
        if block not in BLOCKS or arm not in ARMS or not record.get("valid"):
            raise RuntimeError("invalid block, arm, or arm validity")
        if len(measurements) != 9:
            raise RuntimeError("each full arm must contain all nine cells")
        if arm != "baseline":
            engagement = record.get("engagement", {})
            gate = record.get("agent_gate", {})
            if (engagement.get("marker_callbacks") != 32 or
                    not engagement.get("clean_detach") or
                    not gate.get("routing_order_valid")):
                raise RuntimeError("attached-arm engagement gate failed")
        for measurement in measurements:
            cell = measurement.get("cell")
            key = (block, cell, arm)
            if (cell not in range(9) or key in pairs or
                    measurement.get("mismatches") != 0 or
                    measurement.get("checked_values") != 1_048_576):
                raise RuntimeError("measurement completeness/correctness gate failed")
            pairs[key] = float(measurement["elapsed_ms"])

    expected = {
        (block, cell, arm)
        for block in BLOCKS for cell in range(9) for arm in ARMS
    }
    if set(pairs) != expected:
        raise RuntimeError("paired matrix is incomplete")
    return pairs


def paired_us(
    pairs: dict[tuple[int, int, str], float], cell: int, arm: str
) -> np.ndarray:
    return np.asarray([
        (pairs[block, cell, arm] - pairs[block, cell, "baseline"]) * 1_000
        for block in BLOCKS
    ])


def median_interval(values: np.ndarray, seed: int = 1797) -> tuple[float, float, float]:
    if values.shape != (10,) or not np.all(np.isfinite(values)):
        raise RuntimeError("paired bootstrap requires ten finite observations")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(10_000, len(values)), replace=True)
    boot = np.median(samples, axis=1)
    median = float(np.median(values))
    low, high = np.percentile(boot, [2.5, 97.5])
    return median, float(low), float(high)


def series(
    pairs: dict[tuple[int, int, str], float], cells: tuple[int, ...], arm: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    estimates = [median_interval(paired_us(pairs, cell, arm)) for cell in cells]
    medians = np.asarray([item[0] for item in estimates])
    lows = medians - np.asarray([item[1] for item in estimates])
    highs = np.asarray([item[2] for item in estimates]) - medians
    return medians, lows, highs


def plot(input_path: Path, output_path: Path) -> None:
    pairs = load_pairs(input_path)
    plt.rcParams.update({
        "figure.figsize": (6.9, 2.45),
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "font.family": "serif",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
    })

    figure, axes = plt.subplots(1, 2)
    panels = (
        (axes[0], (0, 1, 2, 3, 4), np.asarray([256, 512, 1024, 2048, 4096]),
         "Launched blocks", "(a)"),
        (axes[1], (4, 5, 6, 7, 8), np.asarray([2048, 4096, 8192, 16384, 32768]),
         "Active warps", "(b)"),
    )
    styles = {
        "noop": ("#0072B2", "o", "-", "Return-only handler"),
        "counter": ("#E69F00", "s", "--", "Per-thread counter"),
    }
    handles = []
    labels = []
    for axis, cells, x_values, x_label, panel_label in panels:
        axis.axhline(0, color="#777777", linewidth=1.0, linestyle=":")
        for arm in ("noop", "counter"):
            color, marker, linestyle, label = styles[arm]
            medians, low, high = series(pairs, cells, arm)
            handle = axis.errorbar(
                x_values, medians, yerr=np.vstack([low, high]),
                color=color, marker=marker, markersize=4.2,
                linestyle=linestyle, capsize=2, label=label,
            )
            if axis is axes[0]:
                handles.append(handle)
                labels.append(label)
        axis.set_xscale("log", base=2)
        axis.set_xticks(x_values, [str(value) for value in x_values])
        axis.set_xlabel(x_label)
        axis.set_ylabel("Paired latency increment (µs)")
        axis.set_ylim(bottom=0)
        axis.text(-0.14, 1.02, panel_label, transform=axis.transAxes,
                  fontsize=8, fontweight="bold")

    figure.legend(handles, labels, loc="upper center", ncol=2,
                  frameon=False, bbox_to_anchor=(0.5, 1.03))
    figure.subplots_adjust(left=0.085, right=0.99, bottom=0.22,
                           top=0.82, wspace=0.30)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=HERE / "raw/full-575-01/result.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "figures/trampoline-scaling.pdf",
    )
    args = parser.parse_args()
    plot(args.input.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
