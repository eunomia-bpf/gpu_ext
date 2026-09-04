#!/usr/bin/env python3
"""Independently audit the aggregate Markdown inputs behind the current Fig. 15."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import NamedTuple


HERE = Path(__file__).resolve().parent
GPU_EXT = HERE.parents[1]
DEFAULT_OLD = (
    GPU_EXT / "docs/paper/img/results-raw/runtime/old/micro_vec_add_result.md"
)
DEFAULT_NEW = GPU_EXT / "docs/paper/img/results-raw/runtime/micro_vec_add_result.md"

PANEL_A_OPERATIONS = (
    "Empty probe",
    "Entry probe",
    "Exit probe",
    "Entry+Exit",
    "GPU Ringbuf",
    "Global timer",
    "Per-GPU-thread array",
    "Memtrace",
    "GPU Array map update",
    "GPU Array map lookup",
)


class Comparison(NamedTuple):
    operation: str
    old_time_us: float
    new_time_us: float
    old_baseline_us: float
    new_baseline_us: float
    old_overhead_us: float
    new_overhead_us: float
    corrected_reduction_pct: float
    plotted_old_overhead_us: float
    plotted_reduction_pct: float


def parse_markdown(path: Path) -> dict[tuple[str, str], float]:
    """Parse only successful aggregate rows; reject duplicates."""
    results: dict[tuple[str, str], float] = {}
    row = re.compile(
        r"^\|\s*(?P<name>[^|]+?)\s*\|\s*(?P<workload>[^|]+?)\s*\|"
        r"\s*(?P<time>[0-9]+(?:\.[0-9]+)?)\s*\|"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        match = row.match(line)
        if not match:
            continue
        key = (match.group("name"), match.group("workload"))
        if key in results:
            raise ValueError(f"duplicate result row in {path}: {key}")
        results[key] = float(match.group("time"))
    if not results:
        raise ValueError(f"no benchmark rows found in {path}")
    return results


def compare_panel_a(
    old: dict[tuple[str, str], float],
    new: dict[tuple[str, str], float],
) -> list[Comparison]:
    """Recompute old/new overhead using each summary's own tiny baseline."""
    old_baseline = old[("Baseline (tiny)", "tiny")]
    new_baseline = new[("Baseline (tiny)", "tiny")]
    comparisons = []
    for operation in PANEL_A_OPERATIONS:
        key = (f"{operation} (tiny)", "tiny")
        old_time = old[key]
        new_time = new[key]
        old_overhead = old_time - old_baseline
        new_overhead = new_time - new_baseline
        if old_overhead <= 0:
            raise ValueError(f"nonpositive old overhead for {operation}")
        plotted_old = old_time - new_baseline
        comparisons.append(
            Comparison(
                operation=operation,
                old_time_us=old_time,
                new_time_us=new_time,
                old_baseline_us=old_baseline,
                new_baseline_us=new_baseline,
                old_overhead_us=old_overhead,
                new_overhead_us=new_overhead,
                corrected_reduction_pct=(old_overhead - new_overhead)
                / old_overhead
                * 100.0,
                plotted_old_overhead_us=plotted_old,
                plotted_reduction_pct=(plotted_old - new_overhead)
                / plotted_old
                * 100.0,
            )
        )
    return comparisons


def map_absolute_ratios(
    new: dict[tuple[str, str], float],
) -> dict[str, float]:
    """Reproduce the only same-label absolute ratios available in the summary."""
    return {
        "update": new[("CPU Array map update (minimal)", "minimal")]
        / new[("GPU Array map update (tiny)", "tiny")],
        "lookup": new[("CPU Array map lookup (minimal)", "minimal")]
        / new[("GPU Array map lookup (tiny)", "tiny")],
    }


def render(old_path: Path, new_path: Path) -> str:
    old = parse_markdown(old_path)
    new = parse_markdown(new_path)
    comparisons = compare_panel_a(old, new)
    ratios = map_absolute_ratios(new)
    lines = [
        "# Independent replay of the legacy Fig. 15 aggregates",
        "",
        f"- Old aggregate: `{old_path}`",
        f"- New aggregate: `{new_path}`",
        "- This is an arithmetic audit, not a valid performance reproduction: the inputs "
        "contain one aggregate per arm and no retained repetitions, raw timing samples, "
        "correctness record, or mechanism-engagement record.",
        "",
        "## Panel (a): baseline subtraction",
        "",
        "The plotting script subtracts the new 5.15-us baseline from both series. The "
        "correct replay subtracts 5.23 us from the old series and 5.15 us from the new "
        "series. Arithmetic repair cannot identify either series as eGPU or warp aggregation.",
        "",
        "| Operation | old overhead (us) | new overhead (us) | corrected reduction | plotted reduction |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        lines.append(
            f"| {item.operation} | {item.old_overhead_us:.2f} | "
            f"{item.new_overhead_us:.2f} | {item.corrected_reduction_pct:.1f}% | "
            f"{item.plotted_reduction_pct:.1f}% |"
        )
    outside = [
        item.operation
        for item in comparisons
        if item.corrected_reduction_pct < 60.0 - 1e-9
        or item.corrected_reduction_pct > 80.0 + 1e-9
    ]
    lines.extend(
        [
            "",
            "The corrected aggregate reductions are not uniformly 60--80%; operations "
            f"outside that interval are: {', '.join(outside)}.",
            "",
            "## Panel (b): available absolute ratios",
            "",
            f"- Standard CPU array update / device GPU-array update: {ratios['update']:.1f}x.",
            f"- Standard CPU array lookup / device GPU-array lookup: {ratios['lookup']:.1f}x.",
            "",
            "These ratios are descriptive only. The CPU rows use the three-iteration "
            "`minimal` run and a standard BPF array reached through the serialized host-helper "
            "bridge. The GPU rows use the 10,000-iteration `tiny` run and a device-resident "
            "GPU array. Their BPF bodies are not operation-matched: CPU update performs one "
            "update, GPU update performs lookup plus update, CPU lookup mutates the returned "
            "pointer, and GPU lookup has no retained result. The figure also plots CPU absolute "
            "latency beside GPU baseline-subtracted overhead.",
            "",
            "## Verdict",
            "",
            "The Markdown summaries can reproduce the plotting arithmetic, but cannot support "
            "the causal labels `eGPU`, `warp aggregation`, or a general 6000x CPU-versus-GPU "
            "map claim. A prospective, operation-matched run is required.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--new", type=Path, default=DEFAULT_NEW)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = render(args.old, args.new)
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
