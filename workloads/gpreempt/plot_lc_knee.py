#!/usr/bin/env python3
"""Plot a completed, independently audited GPreempt LC-knee study.

Small points are paired 60-second cells and large markers/lines are three-block
medians.  LC p99 is explicitly marked as conditional whenever completion
coverage is below 100%.  This script reads only the completed audit JSON; it
does not inspect campaign directories or run CUDA workloads.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

ARMS = ("native", "original_gpreempt", "bpf_gpreempt")
SCENARIOS = ("lc500", "lc625", "lc800")
RATES = (500, 625, 800)
LABELS = ("Native baseline", "Original-C GPreempt", "BPF GPreempt")
COLORS = ("#666666", "#D97706", "#0072B2")
MARKERS = ("o", "s", "^")
AUDIT_SCHEMA = "gpreempt_lc_knee_audit_v1"
P99_POPULATION = "all_started_and_verified_including_after_window"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_number(value, message: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(value), message)
    return float(value)


def _validate_audit_header(audit: dict) -> None:
    expected = {
        "schema": AUDIT_SCHEMA,
        "study": "lc-knee",
        "evidence_role": "supporting",
        "mode": "full",
        "complete": True,
        "formal_eligible": True,
        "formal_complete": True,
        "valid_cells": 27,
        "required_cells": 27,
        "prespecified_lc_rates_rps": list(RATES),
        "post_hoc_rate_additions_allowed": False,
    }
    for key, expected_value in expected.items():
        actual = audit.get(key)
        matches = actual is expected_value if type(expected_value) is bool else actual == expected_value
        _require(matches, f"audit.{key} must be {expected_value!r}")
    for key in ("rejected_cells", "incomplete_cells", "unexpected_cells"):
        _require(audit.get(key) == [], f"audit.{key} must be empty")
    _require(set(audit.get("scenarios", {})) == set(SCENARIOS),
             "the three prespecified LC-knee scenarios are required exactly")

    gate = audit.get("preflight_gate")
    _require(isinstance(gate, dict), "a completed independent preflight gate is required")
    gate_expected = {
        "schema": AUDIT_SCHEMA,
        "study": "lc-knee",
        "mode": "preflight",
        "scenario": "lc800",
        "valid_cells": 3,
        "complete": True,
        "formal_complete": False,
    }
    for key, expected_value in gate_expected.items():
        actual = gate.get(key)
        matches = actual is expected_value if type(expected_value) is bool else actual == expected_value
        _require(matches,
                 f"audit.preflight_gate.{key} must be {expected_value!r}")
    campaign = gate.get("campaign")
    _require(isinstance(campaign, str) and Path(campaign).is_absolute(),
             "audit.preflight_gate.campaign must be an absolute path")


def plot_data(audit: dict) -> dict:
    """Validate the formal LC-knee audit and return exact paired plot data."""
    _require(isinstance(audit, dict), "audit must be a JSON object")
    _validate_audit_header(audit)
    points = []
    for scenario, rate in zip(SCENARIOS, RATES):
        summary = audit["scenarios"][scenario]
        _require(isinstance(summary, dict), f"scenario {scenario} must be an object")
        _require(summary.get("complete") is True, f"scenario {scenario} is incomplete")
        _require(summary.get("valid_paired_blocks") == 3,
                 f"scenario {scenario} needs three valid paired blocks")
        _require(summary.get("required_blocks") == 3,
                 f"scenario {scenario} must require three blocks")
        cells = summary.get("per_cell_points")
        _require(isinstance(cells, list) and len(cells) == 9,
                 f"scenario {scenario} needs exactly nine cells")
        seen = set()
        for cell in cells:
            _require(isinstance(cell, dict), "each cell must be an object")
            block = cell.get("block")
            arm = cell.get("arm")
            _require(type(block) is int and block in range(3), "unexpected block")
            _require(arm in ARMS, "unexpected policy arm")
            _require(cell.get("scenario") == scenario, "cell scenario does not match its group")
            key = (block, arm)
            _require(key not in seen, "duplicate paired cell")
            seen.add(key)

            begin_ns = cell.get("begin_ns")
            end_ns = cell.get("end_ns")
            _require(type(begin_ns) is int and type(end_ns) is int and begin_ns >= 0,
                     "cell timestamps must be nonnegative integer nanoseconds")
            _require(end_ns - begin_ns == 60_000_000_000,
                     "LC-knee plot accepts only the planned 60-second cells")
            metrics = cell.get("metrics")
            _require(isinstance(metrics, dict), "cell metrics must be an object")
            foreground = metrics.get("vgg_rt")
            background = metrics.get("resnet152_be")
            _require(isinstance(foreground, dict) and isinstance(background, dict),
                     "both planned task metrics are required")

            offered = foreground.get("offered_requests")
            _require(type(offered) is int and offered == rate * 60,
                     "offered LC requests must match the prespecified rate and duration")
            p99_us = _finite_number(foreground.get("p99_response_us"),
                                    "LC response p99 must be finite")
            _require(p99_us > 0, "LC response p99 must be positive")
            coverage = _finite_number(foreground.get("completion_coverage"),
                                      "LC completion coverage must be explicit and finite")
            _require(0 < coverage <= 1, "LC completion coverage must be in (0, 1]")
            conditional = foreground.get("p99_is_conditional")
            _require(type(conditional) is bool and conditional == (coverage < 1),
                     "conditional-p99 label must agree with completion coverage")
            _require(foreground.get("response_p99_population") == P99_POPULATION,
                     "LC response-p99 population is not the audited population")
            sample_count = foreground.get("p99_sample_count")
            _require(type(sample_count) is int and 0 < sample_count <= offered,
                     "LC p99 sample count must be positive and no larger than offered")
            _require(math.isclose(sample_count / offered, coverage,
                                  rel_tol=0, abs_tol=1e-12),
                     "LC completion coverage must equal p99 samples / offered requests")
            goodput = _finite_number(background.get("goodput_rps"),
                                     "BE goodput must be finite")
            _require(goodput >= 0, "BE goodput must be nonnegative")

            points.append({
                "scenario": scenario,
                "rate_rps": rate,
                "block": block,
                "arm": arm,
                "lc_response_p99_ms": p99_us / 1000.0,
                "be_goodput_rps": goodput,
                "completion_coverage": coverage,
                "conditional_p99": conditional,
            })
        _require(seen == {(block, arm) for block in range(3) for arm in ARMS},
                 f"scenario {scenario} does not contain the exact paired-cell matrix")

    groups = []
    for scenario, rate in zip(SCENARIOS, RATES):
        for arm in ARMS:
            cells = sorted((point for point in points
                            if point["scenario"] == scenario and point["arm"] == arm),
                           key=lambda point: point["block"])
            _require([cell["block"] for cell in cells] == [0, 1, 2],
                     "each rate/arm group must have the same three paired blocks")
            groups.append({
                "scenario": scenario,
                "rate_rps": rate,
                "arm": arm,
                "blocks": [cell["block"] for cell in cells],
                "cell_count": len(cells),
                "median_lc_response_p99_ms": statistics.median(
                    cell["lc_response_p99_ms"] for cell in cells),
                "median_be_goodput_rps": statistics.median(
                    cell["be_goodput_rps"] for cell in cells),
                "median_completion_coverage": statistics.median(
                    cell["completion_coverage"] for cell in cells),
                "minimum_completion_coverage": min(
                    cell["completion_coverage"] for cell in cells),
                "any_conditional_p99": any(cell["conditional_p99"] for cell in cells),
            })
    return {"points": points, "groups": groups}


def render(audit: dict, prefix: Path) -> list[Path]:
    """Write publication-sized vector and raster versions to a new prefix."""
    data = plot_data(audit)
    prefix = Path(prefix)
    paths = [prefix.with_suffix(suffix) for suffix in (".pdf", ".png")]
    if any(path.exists() for path in paths):
        raise FileExistsError("figure output exists; use a new explicit output prefix")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    definitions = (
        ("lc_response_p99_ms", "median_lc_response_p99_ms", "LC response p99 (ms)"),
        ("be_goodput_rps", "median_be_goodput_rps", "BE goodput (req/s)"),
    )
    arm_offsets = (-0.19, 0.0, 0.19)
    block_offsets = (-0.035, 0.0, 0.035)

    for panel_index, (panel, (metric, median_metric, ylabel)) in enumerate(
            zip(axes, definitions)):
        for rate_index, rate in enumerate(RATES):
            # Thin gray lines expose the within-block three-arm pairing.
            for block in range(3):
                paired = [next(point for point in data["points"]
                               if point["rate_rps"] == rate
                               and point["block"] == block and point["arm"] == arm)
                          for arm in ARMS]
                xs = [rate_index + offset + block_offsets[block]
                      for offset in arm_offsets]
                panel.plot(xs, [point[metric] for point in paired],
                           color="#c7c7c7", linewidth=0.6, zorder=1)

            for arm_index, (arm, color, marker) in enumerate(
                    zip(ARMS, COLORS, MARKERS)):
                cells = sorted((point for point in data["points"]
                                if point["rate_rps"] == rate and point["arm"] == arm),
                               key=lambda point: point["block"])
                center = rate_index + arm_offsets[arm_index]
                for cell in cells:
                    x_value = center + block_offsets[cell["block"]]
                    conditional = panel_index == 0 and cell["conditional_p99"]
                    panel.scatter(x_value, cell[metric], marker=marker, s=20,
                                  facecolors="none" if conditional else color,
                                  edgecolors=color, linewidths=0.8, alpha=0.75, zorder=3)
                    if conditional:
                        panel.annotate(f'{cell["completion_coverage"]:.0%}',
                                       (x_value, cell[metric]), xytext=(0, 5),
                                       textcoords="offset points", ha="center",
                                       va="bottom", fontsize=7, color=color)
                group = next(group for group in data["groups"]
                             if group["rate_rps"] == rate and group["arm"] == arm)
                panel.scatter(center, group[median_metric], marker=marker, s=50,
                              facecolors=color, edgecolors="white", linewidths=0.7,
                              zorder=5)

        # Colored lines join each policy's three-block median across offered rates.
        for arm_index, (arm, color) in enumerate(zip(ARMS, COLORS)):
            groups = [next(group for group in data["groups"]
                           if group["rate_rps"] == rate and group["arm"] == arm)
                      for rate in RATES]
            panel.plot([index + arm_offsets[arm_index] for index in range(3)],
                       [group[median_metric] for group in groups],
                       color=color, linewidth=1.5, zorder=4)
        panel.set_xticks(range(3), [str(rate) for rate in RATES])
        panel.set_xlabel("Offered LC rate (req/s)")
        panel.set_ylabel(ylabel)
        panel.set_xlim(-0.45, 2.45)
        panel.set_ylim(bottom=0)
        panel.grid(axis="y", color="#dddddd", linewidth=0.5, zorder=0)
        panel.text(-0.14, 1.02, f"({chr(ord('a') + panel_index)})",
                   transform=panel.transAxes, ha="left", va="bottom",
                   fontsize=8, fontweight="bold")

    arm_handles = [Line2D([0], [0], color=color, marker=marker, markersize=5,
                          linewidth=1.5, label=label)
                   for color, marker, label in zip(COLORS, MARKERS, LABELS)]
    coverage_handles = [
        Line2D([0], [0], color="#333333", marker="o", markerfacecolor="#333333",
               markersize=4, linewidth=0, label="LC p99: 100% complete"),
        Line2D([0], [0], color="#333333", marker="o", markerfacecolor="none",
               markersize=4, linewidth=0, label="LC p99: conditional (coverage labelled)"),
    ]
    figure.legend(handles=arm_handles + coverage_handles, loc="upper center", ncol=5,
                  frameon=False, bbox_to_anchor=(0.5, 1.015), columnspacing=1.0,
                  handlelength=1.5)
    note = ("Small points connected in gray: paired 60 s cells; large markers/colored lines: "
            "three-block medians. LC p99 uses all started+verified requests, including after-window completions.\n"
            "Open LC points are conditional and labelled with completion coverage; "
            "filled LC points have 100% coverage.")
    figure.text(0.5, 0.012, note, ha="center", va="bottom", fontsize=7)
    figure.tight_layout(rect=(0, 0.17, 1, 0.88))
    prefix.parent.mkdir(parents=True, exist_ok=True)
    try:
        for path in paths:
            figure.savefig(path, dpi=200, bbox_inches="tight")
    finally:
        plt.close(figure)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path,
                        help="completed lc-knee audit JSON from analyze_load_study.py")
    parser.add_argument("--output-prefix", type=Path, required=True)
    arguments = parser.parse_args()
    outputs = render(json.loads(arguments.audit.read_text()), arguments.output_prefix)
    print("\n".join(str(path) for path in outputs))
