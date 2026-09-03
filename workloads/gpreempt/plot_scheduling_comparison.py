#!/usr/bin/env python3
"""Prepare a 2x2 within-workload scheduling comparison from final audited data.

XSched: ten paired blocks, original Level-1 executor, queue-entry p99 in seconds.
GPreempt: five paired blocks per load, arrival-to-verified p99 in milliseconds.
No GPU execution, compilation, paper edits, or confidence bars on medians.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

import plot_load_study as gp_plot

XS_ARMS = ("native", "xsched", "bpftime_hpf")
XS_ALL_ARMS = (*XS_ARMS, "gpubpf")
XS_LABELS = ("Native", "Original XSched", "BPF-HPF")
GP_LABELS = ("Native", "Original C", "BPF-GPreempt")
MARKERS = ("o", "s", "D")
COLORS = gp_plot.COLORS
STYLE = {"font.family": "DejaVu Sans", "font.size": 8,
         "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
         "legend.fontsize": 7.5, "axes.spines.top": False,
         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42}
DEFAULT_XS_AUDIT = (Path(__file__).resolve().parents[1] / "xsched/raw/"
                    "full-persistent-575-20260903/independent-raw-audit.json")


def positive(value, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite measurement")
    return float(value)


def xsched_points(audit_path: Path) -> list[dict]:
    """Read only cells listed by the final raw audit; do not rerun its workload."""
    audit_path = Path(audit_path)
    campaign = audit_path.parent
    audit = json.loads(audit_path.read_text())
    required = {"status": "passed", "audited_cells": 46, "complete_blocks": 10,
                "mixed_cells": 40, "isolated_controls": 6}
    if any(audit.get(key) != value for key, value in required.items()):
        raise ValueError("XSched needs the completed 46-cell raw audit, not a pilot")
    for key in ("actual_order_verified", "aggregate_recomputation_equal",
                "per_cell_policy_engagement_and_safety_verified",
                "per_worker_argv_environment_clock_numerics_verified"):
        if audit.get(key) is not True:
            raise ValueError(f"missing XSched raw-audit guarantee: {key}")
    protocol = json.loads((campaign / "protocol.json").read_text())
    required_protocol = {"phase": "full", "repetitions": 10, "tasks_per_stream": 50,
                         "streams_per_process": 4, "lc_processes": 2, "be_processes": 4,
                         "xsched_level": 1}
    if (any(protocol.get(key) != value for key, value in required_protocol.items())
            or protocol.get("full_50_kernel_10_block_protocol") is not True
            or protocol.get("short_budget_difference") is not None
            or len(protocol.get("configs", [])) != 4
            or set(protocol["configs"]) != set(XS_ALL_ARMS)):
        raise ValueError("XSched full Level-1 workload configuration differs from the audited experiment")
    schedule = protocol.get("schedule", [])
    if len(schedule) != 10 or any(len(order) != 4 or set(order) != set(XS_ALL_ARMS) for order in schedule):
        raise ValueError("XSched schedule must contain all four original arms in each of ten blocks")
    cells = [(f"block-{block:02d}-{index}-{arm}", block, index, arm)
             for block, order in enumerate(schedule) for index, arm in enumerate(order)]
    controls = {f"control-{role}-{index}" for role in ("lc", "be") for index in range(3)}
    expected = controls | {name for name, *_ in cells}
    verified = audit.get("verified_directories", [])
    if len(verified) != len(expected) or set(verified) != expected:
        raise ValueError("XSched audit does not cover exactly the recorded campaign")
    actual = {path.parent.name for pattern in ("block-*/result.json", "control-*/result.json")
              for path in campaign.glob(pattern)}
    if actual != expected or any(campaign.glob("*/failure.json")):
        raise ValueError("XSched campaign has missing, extra, or failed cells")
    summary = json.loads((campaign / "summary.json").read_text())
    if summary.get("protocol") != protocol or summary.get("complete_blocks") != 10:
        raise ValueError("XSched summary belongs to a different protocol or incomplete campaign")
    samples = protocol["tasks_per_stream"] * protocol["streams_per_process"]
    output_values = samples * protocol["blocks"] * protocol["threads"]
    points = []
    for directory, block, index, arm in cells:
        row = json.loads((campaign / directory / "result.json").read_text())
        if (row.get("config") != arm or type(row.get("block")) is not int or row["block"] != block
                or row.get("order_index") != index or row.get("error") or row.get("cleanup_errors")
                or any(row.get(key) != protocol[key] for key in ("reps", "tasks_per_stream", "blocks", "threads"))
                or row.get("lc_samples") != protocol["lc_processes"] * samples
                or row.get("be_completed") != protocol["be_processes"] * samples
                or row.get("outputs_validated_per_process") != output_values):
            raise ValueError(f"XSched cell metadata/counts differ from audited protocol: {directory}")
        latency = positive(row.get("lc_p99_us"), "XSched queue-entry p99")
        throughput = positive(row.get("be_throughput_kernels_s"), "XSched background throughput")
        if arm in XS_ARMS:
            points.append({"block": block, "arm": arm, "queue_p99_s": latency / 1_000_000,
                           "background_kernels_s": throughput, "source_cell": directory})
    for arm in XS_ARMS:
        selected = [point for point in points if point["arm"] == arm]
        if len(selected) != 10:
            raise ValueError("XSched plotted arm is missing a block")
        recorded = summary["configs"][arm]
        for metric, key, scale in (("queue_p99_s", "lc_p99_median_us", 1_000_000),
                                   ("background_kernels_s", "be_throughput_median", 1)):
            if not math.isclose(statistics.median(p[metric] for p in selected),
                                positive(recorded[key], key) / scale, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("XSched per-cell values disagree with its audited aggregate")
    return sorted(points, key=lambda point: (point["block"], XS_ARMS.index(point["arm"])))


def comparison_data(xs_audit_path: Path, gp_audit: dict) -> dict:
    # Keep the already reviewed complete-45-cell contract as the sole GP entry.
    gp_points = gp_plot.plot_points(gp_audit)
    return {"schema": "scheduling_comparison_2x2_v1", "xsched": xsched_points(xs_audit_path),
            "gpreempt": gp_points,
            "scope": {"xsched": "same original XSched frontend and Level-1 executor; userspace BPF-HPF decisions",
                      "gpreempt": "same original-C/BPF GPreempt policy; explicit host-mapped flag compatibility",
                      "excluded_arm": "XSched driver-only gpubpf uses a different policy and is not plotted",
                      "cross_workload_latency_comparison": False,
                      "error_bars": "none; all per-cell points and medians are shown, not paired CIs"}}


def caption(data: dict) -> str:
    conditional = [point for point in data["gpreempt"] if point["conditional"]]
    coverage_note = (f"Hollow markers in (b) identify conditional completed-request p99: "
                     f"{len(conditional)} cells have unstarted foreground backlog; minimum final foreground "
                     f"completion coverage is {min(p['completion_coverage'] for p in conditional):.1%}. "
                     "These points do not establish an all-offered latency improvement."
                     if conditional else "All offered foreground requests were eventually verified in every GPreempt cell.")
    return (
        "Scheduling-policy implementations, compared within each workload. "
        "Panels (a,c), left: the XSched burst workload, with two foreground (LC) and four background (BE) "
        "processes, four streams per process, and 50 kernels per stream; each of the three displayed arms "
        "has ten complete paired-block cells. (a) is foreground host submission to first CTA entry p99 "
        "in seconds, including queueing; it is not kernel completion latency or hardware preemption time. "
        "(c) is aggregate background kernels per second, from the common BE release to its last stream "
        "completion. Native CUDA, original XSched HPF, and userspace JIT BPF-HPF are compared; BPF-HPF "
        "still uses the original XSched frontend and Level-1 launch executor. The separately measured "
        "driver-only gpubpf arm uses a different policy and is intentionally excluded. "
        "Panels (b,d), right: VGG19 foreground at 100 requests/s and ResNet152 background at 100 requests/s, "
        "200 requests/s, or continuous closed-loop supply; batch one, CUDA graphs, and 200 microseconds "
        "preprocessing are fixed. Each arm/load has five paired 60-second cells. (b) is scheduled arrival "
        "to GPU-synchronized, numerically verified output p99 in milliseconds, including FIFO waiting; "
        "its population includes the final already-started request if it finishes after the window. "
        "(d) counts only verified background completions inside the half-open measurement window, divided "
        "by 60 seconds. Native CUDA stream priorities, original-C GPreempt, and BPF-GPreempt are compared; "
        "both GPreempt policy arms use the same explicit host-mapped flag transport, not original GDRCopy. "
        "Markers show every cell and short horizontal bars show within-arm medians; marker shape and "
        "position identify the arm. No confidence intervals are drawn on those medians, and paired-effect "
        "intervals must be read from the separate raw audit. " + coverage_note + " "
        "The panels show original-policy implementation costs and latency/throughput tradeoffs, not a "
        "cross-workload ranking: left and right have different workloads, latency definitions, time units, "
        "and throughput units. Neither latency axis measures a hardware preemption quantum; this figure "
        "does not establish statistical equivalence or reproduce the papers' original hardware speedups.\n")


def _draw(data: dict, paths: list[Path], width: float) -> None:
    """Render only when explicitly invoked after all GPU measurement has ended."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(2, 2, figsize=(width, 4.65), sharex="col")
        definitions = ((0, 0, "queue_p99_s", "LC queue-entry p99 (s)", "(a)"),
                       (1, 0, "background_kernels_s", "BE throughput (kernels/s)", "(c)"),
                       (0, 1, "response_p99_ms", "LC response p99 (ms)", "(b)"),
                       (1, 1, "background_goodput_rps", "BE goodput (req/s)", "(d)"))
        for row, column, metric, label, letter in definitions:
            panel = axes[row, column]
            groups = (None,) if column == 0 else gp_plot.SCENARIOS
            arms = XS_ARMS if column == 0 else gp_plot.ARMS
            source = data["xsched" if column == 0 else "gpreempt"]
            for group_index, group in enumerate(groups):
                for arm_index, (arm, color, marker) in enumerate(zip(arms, COLORS, MARKERS)):
                    cells = sorted((p for p in source if p["arm"] == arm
                                    and (group is None or p["scenario"] == group)), key=lambda p: p["block"])
                    center = arm_index if column == 0 else group_index + (arm_index - 1) * .24
                    jitter = .025 if column == 0 else .024
                    for index, cell in enumerate(cells):
                        conditional = column == 1 and row == 0 and cell["conditional"]
                        panel.scatter(center + (index - (len(cells) - 1) / 2) * jitter, cell[metric],
                                      marker=marker, s=20, facecolors="none" if conditional else color,
                                      edgecolors=color, linewidths=1, alpha=.7, zorder=3)
                    median = statistics.median(cell[metric] for cell in cells)
                    half_width = .15 if column == 0 else .08
                    panel.plot([center - half_width, center + half_width], [median, median],
                               color=color, linewidth=2, zorder=4)
            panel.set_ylabel(label)
            # Zero-based headroom keeps markers visible even when cells cluster
            # tightly around a nonzero value, as in the XSched BE panel.
            panel.set_ylim(0, max(point[metric] for point in source) * 1.08)
            panel.set_xlim(-.5, 2.5)
            panel.yaxis.set_major_locator(MaxNLocator(nbins=5))
            panel.ticklabel_format(axis="y", style="plain", useOffset=False)
            panel.grid(axis="y", alpha=.25, linewidth=.6)
            panel.text(.012, .985, letter, transform=panel.transAxes, fontsize=8, va="top")
        axes[1, 0].set_xticks(range(3), ["Native", "Original\nXSched", "BPF-HPF"])
        axes[1, 0].set_xlabel("XSched burst workload")
        axes[1, 1].set_xticks(range(3), ["100 req/s", "200 req/s", "Continuous"])
        axes[1, 1].set_xlabel("GPreempt: BE supply (LC 100 req/s)")
        for column, labels in enumerate((XS_LABELS, GP_LABELS)):
            handles = [Line2D([0], [0], color=color, marker=marker, markersize=4.5,
                              linewidth=1.2, label=label)
                       for color, marker, label in zip(COLORS, MARKERS, labels)]
            axes[0, column].legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, 1.06),
                                   ncol=3, frameon=False, handlelength=1.2,
                                   handletextpad=.35, columnspacing=.8)
        note = "Each marker: one cell. Bars: medians (XS: 10 blocks; GP: 5 per load)."
        if any(point["conditional"] for point in data["gpreempt"]):
            note += "\nHollow GP latency markers: conditional p99; see caption for completion coverage."
        else:
            note += "\nDifferent workload/metric units across columns; not hardware preemption latency."
        figure.text(.5, .015, note, ha="center", va="bottom", fontsize=7.5)
        figure.tight_layout(rect=(0, .11, 1, .98), h_pad=1.5, w_pad=2)
        try:
            for path in paths:
                # Keep the chosen final-width canvas, avoiding tight-bbox rescaling.
                figure.savefig(path, dpi=300)
        finally:
            plt.close(figure)


def render(xs_audit_path: Path, gp_audit_path: Path, prefix: Path, width: float = 7.2) -> list[Path]:
    if not math.isfinite(width) or not 6.8 <= width <= 8.0:
        raise ValueError("2x2 requires a 6.8–8 inch two-column canvas; do not shrink below readable size")
    data = comparison_data(xs_audit_path, json.loads(Path(gp_audit_path).read_text()))
    paths = [prefix.with_suffix(suffix) for suffix in (".pdf", ".png", ".caption.md", ".points.json")]
    if any(path.exists() for path in paths):
        raise FileExistsError("output exists; choose a new explicit figure prefix")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    _draw(data, paths[:2], width)
    paths[2].write_text(caption(data))
    data["sources"] = {"xsched_audit": str(Path(xs_audit_path).resolve()),
                       "gpreempt_audit": str(Path(gp_audit_path).resolve())}
    paths[3].write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gpreempt_audit", type=Path, help="final independently audited 45-cell study")
    parser.add_argument("--xsched-audit", type=Path, default=DEFAULT_XS_AUDIT)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--width-inches", type=float, default=7.2,
                        help="intended full-width printed size; verify after paper integration")
    args = parser.parse_args()
    outputs = render(args.xsched_audit, args.gpreempt_audit, args.output_prefix, args.width_inches)
    print("\n".join(str(path) for path in outputs))
