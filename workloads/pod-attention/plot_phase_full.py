#!/usr/bin/env python3
"""Render the phase-full-575-01 POD result from its closed analysis only.

The three panels deliberately use different scales because they describe
different measurement boundaries: timed operator calls, the complete audited
100-sample loop, and cold elapsed time before Python main.  This script never
reads per-cell raw records and never runs CUDA or the workload.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "raw" / "phase-full-575-01" / "analysis.json"
DEFAULT_PREFIX = HERE / "figures" / "phase-full-575-01"

ARMS = ("pod_inline", "pod_cuda", "pod_bpf")
PLOTTED_ARMS = ("pod_cuda", "pod_bpf")
ARM_LABELS = ("CUDA\nadapter", "BPF\nadapter")
ARM_COLORS = ("#666666", "#0072B2")
ARM_MARKERS = ("o", "D")
OPERATOR_METRICS = (("cuda_ms", "CUDA event"), ("host_wall_ms", "Host wall"))
BLOCKS = 5

# Match the current full-width paper plots in workloads/gpreempt and
# workloads/finemoe: final-size sans serif text, open top/right spines, and the
# same gray/blue baseline-system palette.  The 7-inch canvas is not tight-cropped
# so LaTeX need not shrink the type below its intended size.
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def positive(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite measurement")
    return float(value)


def positive_series(value: object, name: str) -> list[float]:
    require(isinstance(value, list) and len(value) == BLOCKS,
            f"{name} must contain exactly five paired-block values")
    return [positive(item, f"{name}[{index}]") for index, item in enumerate(value)]


def checked_arm_series(container: object, name: str) -> dict[str, list[float]]:
    require(isinstance(container, dict) and set(container) == set(ARMS),
            f"{name} must contain the complete three-arm campaign")
    return {arm: positive_series(container[arm], f"{name}.{arm}") for arm in ARMS}


def checked_medians(container: object, series: dict[str, list[float]], name: str) -> dict[str, float]:
    require(isinstance(container, dict) and set(container) == set(ARMS),
            f"{name} must contain all three arms")
    medians = {arm: positive(container[arm], f"{name}.{arm}") for arm in ARMS}
    for arm in ARMS:
        require(math.isclose(medians[arm], statistics.median(series[arm]),
                             rel_tol=1e-12, abs_tol=1e-12),
                f"{name}.{arm} disagrees with the five block values")
    return medians


def checked_ratio(record: object, numerator: list[float], denominator: list[float],
                  name: str) -> dict[str, object]:
    require(isinstance(record, dict), f"missing paired ratio: {name}")
    require(record.get("blocks") == BLOCKS and record.get("lower_is_better") is True,
            f"{name} must be a lower-is-better five-block ratio")
    blocks = positive_series(record.get("block_ratios"), f"{name}.block_ratios")
    expected = [left / right for left, right in zip(numerator, denominator)]
    require(all(math.isclose(saved, actual, rel_tol=1e-12, abs_tol=1e-12)
                for saved, actual in zip(blocks, expected)),
            f"{name} block ratios disagree with the plotted measurements")
    estimate = positive(record.get("geometric_mean_ratio"), f"{name}.geometric_mean_ratio")
    recomputed = math.exp(statistics.fmean(math.log(value) for value in expected))
    require(math.isclose(estimate, recomputed, rel_tol=1e-12, abs_tol=1e-12),
            f"{name} geometric mean disagrees with its five ratios")
    interval = record.get("confidence_interval_95")
    require(isinstance(interval, list) and len(interval) == 2,
            f"{name} needs one saved 95% interval")
    low, high = (positive(value, f"{name}.confidence_interval_95") for value in interval)
    require(low <= estimate <= high, f"{name} 95% interval does not contain its estimate")
    return {"ratio": estimate, "block_ratios": blocks, "ci95": [low, high]}


def project_analysis(analysis: object) -> dict[str, object]:
    """Validate the closed analysis contract and project only plotted values."""
    require(isinstance(analysis, dict), "analysis root must be an object")
    for key in ("complete", "formal_complete"):
        require(analysis.get(key) is True, f"requires the completed formal campaign: {key}")
    require(analysis.get("protocol") == "pod-device-setup-phases-v1"
            and analysis.get("numeric_protocol") == "pod-fp16-upstream-match-v2",
            "analysis uses another experiment or numerical protocol")
    require(analysis.get("fresh_process_cells") == 15
            and analysis.get("blocks") == BLOCKS
            and analysis.get("measured_operator_samples_per_cell") == 100
            and analysis.get("phase_observations_per_arm") == BLOCKS,
            "analysis is not the complete 15-cell, five-block, 100-sample campaign")
    require(analysis.get("phase_estimator") == "median of five fresh-process block durations"
            and analysis.get("ratio_estimator")
            == "geometric mean of five paired block ratios; lower is better",
            "saved estimators differ from the figure semantics")
    uncertainty = analysis.get("uncertainty")
    require(isinstance(uncertainty, dict)
            and uncertainty.get("method") == "whole-block percentile bootstrap with shared resamples"
            and uncertainty.get("draws") == 10000
            and uncertainty.get("seed") == 20260907
            and uncertainty.get("confidence") == 0.95
            and uncertainty.get("scope")
            == "pointwise intervals; no equivalence test or multiple-comparison adjustment",
            "figure requires the saved 10,000-draw whole-block 95% intervals")
    boundary = analysis.get("claim_boundary")
    require(isinstance(boundary, str)
            and "RTX 5090" in boundary
            and "not a generic attach-latency estimate" in boundary
            and "not operator latency or an end-to-end serving workload" in boundary,
            "analysis is missing the phase-boundary interpretation safeguards")

    operator = analysis.get("operator_latency")
    require(isinstance(operator, dict)
            and operator.get("cell_estimator")
            == "arithmetic mean of all 100 unfiltered synchronized samples",
            "operator values need the saved all-100-sample cell estimator")
    operator_blocks = operator.get("block_means_ms")
    operator_medians = operator.get("median_of_five_cell_means_ms")
    operator_ratios = operator.get("paired_ratios", {}).get("device_bpf_vs_cuda_adapter")
    require(isinstance(operator_blocks, dict)
            and set(operator_blocks) == {name for name, _ in OPERATOR_METRICS}
            and isinstance(operator_medians, dict)
            and set(operator_medians) == {name for name, _ in OPERATOR_METRICS}
            and isinstance(operator_ratios, dict),
            "operator analysis lacks CUDA-event or synchronized host-wall results")
    projected_operator = []
    for metric, label in OPERATOR_METRICS:
        series = checked_arm_series(operator_blocks[metric], f"operator.{metric}.blocks")
        medians = checked_medians(operator_medians[metric], series,
                                  f"operator.{metric}.medians")
        ratio = checked_ratio(operator_ratios.get(metric), series["pod_bpf"],
                              series["pod_cuda"], f"operator.{metric}.bpf_vs_cuda")
        projected_operator.append({
            "metric": metric,
            "label": label,
            "baseline_median_ms": medians["pod_cuda"],
            "bpf_median_ms": medians["pod_bpf"],
            "overhead_percent": (ratio["ratio"] - 1) * 100,
            "block_overhead_percent": [(value - 1) * 100 for value in ratio["block_ratios"]],
            "ci95_percent": [(value - 1) * 100 for value in ratio["ci95"]],
        })

    phases = analysis.get("block_phase_ms")
    phase_medians = analysis.get("median_phase_ms")
    paired = analysis.get("paired_ratios", {}).get("device_bpf_vs_cuda_adapter")
    require(isinstance(phases, dict) and isinstance(phase_medians, dict)
            and isinstance(paired, dict), "analysis lacks saved phase values or paired ratios")
    projected_phases = {}
    for key, output_key in (("steady_samples_ns", "steady"),
                            ("pre_python_main_ns", "cold")):
        require(key in phases and key in phase_medians and key in paired,
                f"analysis lacks required phase: {key}")
        series_ms = checked_arm_series(phases[key], f"phase.{key}.blocks_ms")
        medians_ms = checked_medians(phase_medians[key], series_ms,
                                     f"phase.{key}.medians_ms")
        ratio = checked_ratio(paired[key], series_ms["pod_bpf"], series_ms["pod_cuda"],
                              f"phase.{key}.bpf_vs_cuda")
        projected_phases[output_key] = {
            "blocks_s": {arm: [value / 1000 for value in series_ms[arm]]
                         for arm in PLOTTED_ARMS},
            "medians_s": {arm: medians_ms[arm] / 1000 for arm in PLOTTED_ARMS},
            **ratio,
        }

    return {
        "operator": projected_operator,
        "steady": projected_phases["steady"],
        "cold": projected_phases["cold"],
        "blocks": BLOCKS,
        "samples_per_cell": 100,
    }


def load_plot_data() -> dict[str, object]:
    """Read the one authorized result source."""
    return project_analysis(json.loads(SOURCE.read_text()))


def _draw_arm_points(panel, phase: dict[str, object]) -> None:
    for arm_index, (arm, color, marker) in enumerate(
            zip(PLOTTED_ARMS, ARM_COLORS, ARM_MARKERS)):
        values = phase["blocks_s"][arm]
        for block, value in enumerate(values):
            panel.scatter(arm_index + (block - 2) * .055, value, marker=marker, s=22,
                          facecolors="white", edgecolors=color, linewidths=1, zorder=3)
        median = phase["medians_s"][arm]
        panel.plot([arm_index - .24, arm_index + .24], [median, median],
                   color=color, linewidth=2.1, zorder=4)
    panel.set_xticks(range(2), ARM_LABELS)
    panel.set_xlim(-.48, 1.48)
    panel.grid(axis="y", alpha=.25, linewidth=.6, zorder=0)


def draw(data: dict[str, object], pdf_path: Path, png_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter

    with plt.rc_context(STYLE):
        figure, (operator, steady, cold) = plt.subplots(
            1, 3, figsize=(7, 2.65), gridspec_kw={"width_ratios": (1.12, 1, 1.05)})

        for index, metric in enumerate(data["operator"]):
            points = metric["block_overhead_percent"]
            for block, value in enumerate(points):
                operator.scatter(index + (block - 2) * .04, value, marker="o", s=17,
                                 facecolors="white", edgecolors="#0072B2", linewidths=.9,
                                 alpha=.85, zorder=3)
            estimate = metric["overhead_percent"]
            low, high = metric["ci95_percent"]
            operator.errorbar(index, estimate,
                              yerr=[[estimate - low], [high - estimate]], fmt="D",
                              markersize=4.5, color="#0072B2", markerfacecolor="#0072B2",
                              capsize=3, elinewidth=1.2, capthick=1.2, zorder=4)
            label_y = max(high, *points) + .075
            operator.text(index, label_y, f"+{estimate:.2f}%", ha="center", va="bottom",
                          fontsize=7.5, color="#005A8D")
        operator.axhline(0, color="#666666", linewidth=.8, linestyle="--", zorder=1)
        operator.set_xticks(range(2), [item["label"] for item in data["operator"]])
        operator.set_xlim(-.48, 1.48)
        operator.set_ylim(0, 2.35)
        operator.set_ylabel("BPF overhead vs. CUDA adapter (%)")
        operator.grid(axis="y", alpha=.25, linewidth=.6, zorder=0)

        _draw_arm_points(steady, data["steady"])
        steady.set_ylim(0, 1.95)
        steady.set_ylabel("100-sample audited loop (s)")
        for index, arm in enumerate(PLOTTED_ARMS):
            value = data["steady"]["medians_s"][arm]
            offset = .07 if arm == "pod_cuda" else -.07
            alignment = "bottom" if arm == "pod_cuda" else "top"
            steady.text(index, value + offset, f"{value:.3f} s", ha="center", va=alignment,
                        fontsize=7.2, color=ARM_COLORS[index])
        steady.text(.97, .97, f"paired GM: {data['steady']['ratio']:.3f}×",
                    transform=steady.transAxes, ha="right", va="top", fontsize=7.5)

        _draw_arm_points(cold, data["cold"])
        cold.set_yscale("log")
        cold.set_ylim(.01, 1000)
        cold.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
        cold.yaxis.set_minor_formatter(NullFormatter())
        cold.set_ylabel("Cold pre-Python time (s, log)")
        cold.text(0, data["cold"]["medians_s"]["pod_cuda"] * 1.55,
                  f"{data['cold']['medians_s']['pod_cuda'] * 1000:.0f} ms",
                  ha="center", va="bottom", fontsize=7.2, color=ARM_COLORS[0])
        cold.text(1, data["cold"]["medians_s"]["pod_bpf"] / 2.3,
                  f"{data['cold']['medians_s']['pod_bpf']:.0f} s",
                  ha="center", va="top", fontsize=7.2, color=ARM_COLORS[1])
        cold.text(.97, .97, f"paired GM: {data['cold']['ratio']:,.0f}×",
                  transform=cold.transAxes, ha="right", va="top", fontsize=7.5)

        for panel, letter in zip((operator, steady, cold), ("(a)", "(b)", "(c)")):
            panel.text(.015, .985, letter, transform=panel.transAxes,
                       va="top", ha="left", fontsize=8)

        figure.text(.5, .075,
                    "Each point: one of five paired blocks. (a) diamond/whisker: paired GM/95% block-bootstrap CI; (b,c) short lines: medians.",
                    ha="center", va="bottom", fontsize=7)
        figure.text(.5, .025,
                    "(b) includes correctness + decision audits; (c) is this injection path, not generic attach latency.",
                    ha="center", va="bottom", fontsize=7)
        figure.subplots_adjust(left=.078, right=.992, bottom=.30, top=.96, wspace=.48)
        try:
            figure.savefig(pdf_path, dpi=300)
            figure.savefig(png_path, dpi=300)
        finally:
            plt.close(figure)


def caption(data: dict[str, object]) -> str:
    cuda = data["operator"][0]
    host = data["operator"][1]
    return (
        "Device-policy costs for the frozen POD operator shape and runtime on an RTX 5090 "
        f"({data['blocks']} fresh-process paired blocks; {data['samples_per_cell']} synchronized "
        "operator samples per cell). (a) BPF-adapter overhead relative to the same CUDA adapter "
        "for CUDA-event and synchronized host-wall operator latency. Open circles are block ratios; "
        "diamonds and whiskers are paired geometric means and pointwise 95% whole-block bootstrap "
        f"intervals. Median cell means are {cuda['baseline_median_ms']:.3f} vs. "
        f"{cuda['bpf_median_ms']:.3f} ms (CUDA event) and {host['baseline_median_ms']:.3f} vs. "
        f"{host['bpf_median_ms']:.3f} ms (host wall). (b) Complete 100-sample loop duration; "
        "it includes correctness checks and a full decision audit after every timed operator and "
        "is neither operator latency nor end-to-end serving latency. (c) Elapsed time before the "
        "first Python module statement on the current injection path, shown on a logarithmic axis; "
        "it is not generic attach latency. Short bars in (b,c) are medians and open markers are all "
        f"five blocks. BPF adds {cuda['overhead_percent']:.2f}% CUDA-event and "
        f"{host['overhead_percent']:.2f}% host-wall operator latency, whereas the audited loop is "
        f"{data['steady']['ratio']:.3f}× and cold pre-Python time is "
        f"{data['cold']['medians_s']['pod_bpf']:.0f} s "
        f"({data['cold']['ratio']:,.0f}× vs. the CUDA adapter). Lower is better; intervals are "
        "pointwise and are not equivalence tests.\n"
    )


def output_paths(prefix: Path) -> tuple[Path, Path, Path]:
    resolved = prefix.resolve()
    try:
        resolved.relative_to(HERE)
    except ValueError as error:
        raise ValueError("figure outputs must remain under workloads/pod-attention") from error
    return (resolved.with_suffix(".pdf"), resolved.with_suffix(".png"),
            resolved.with_suffix(".caption.md"))


def render(prefix: Path = DEFAULT_PREFIX, *, replace_derived: bool = False) -> tuple[Path, Path, Path]:
    data = load_plot_data()
    paths = output_paths(prefix)
    if not replace_derived and any(path.exists() for path in paths):
        raise FileExistsError("derived figure exists; choose a fresh explicit output prefix")
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    draw(data, paths[0], paths[1])
    paths[2].write_text(caption(data))
    return paths


def summary(data: dict[str, object]) -> dict[str, object]:
    return {
        "source": str(SOURCE),
        "blocks": data["blocks"],
        "samples_per_cell": data["samples_per_cell"],
        "operator_overhead_percent": {
            item["metric"]: item["overhead_percent"] for item in data["operator"]
        },
        "steady_loop_ratio": data["steady"]["ratio"],
        "bpf_cold_pre_python_seconds": data["cold"]["medians_s"]["pod_bpf"],
        "cold_pre_python_ratio": data["cold"]["ratio"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_PREFIX,
                        help="fresh output prefix under workloads/pod-attention")
    parser.add_argument("--check-only", action="store_true",
                        help="validate the fixed analysis and print plotted quantities")
    parser.add_argument("--replace-derived", action="store_true",
                        help="replace only the three derived files at the selected prefix")
    arguments = parser.parse_args()
    projected = load_plot_data()
    if arguments.check_only:
        print(json.dumps(summary(projected), indent=2))
    else:
        print("\n".join(str(path) for path in render(
            arguments.output_prefix, replace_derived=arguments.replace_derived)))
