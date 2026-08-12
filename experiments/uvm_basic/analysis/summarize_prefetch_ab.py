#!/usr/bin/env python3
"""Summarize independent-process UVM demand versus explicit-prefetch runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


PHASES = {
    "cpu_prefetch_to_cpu_ms": "cpu_prefetch_to_cpu",
    "cpu_retouch_ms": "cpu_retouch",
    "gpu_prefetch_ms": "gpu_prefetch_after_retouch",
    "kernel_after_retouch_ms": None,
}


def percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def stats(values: list[float]) -> dict[str, float | int | str]:
    if not values:
        return {key: "UNAVAILABLE" for key in ("count", "mean", "median", "min", "max", "stdev", "p95")}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p95": percentile95(values),
    }


def read_runs(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    runs: list[dict[str, Any]] = []
    errors: list[str] = []
    manifests = sorted(root.glob("run_*.tsv"))
    paths: list[Path]
    if manifests:
        with manifests[-1].open(newline="") as source:
            paths = [Path(row["result"]) for row in csv.DictReader(source, delimiter="\t")
                     if row.get("result") not in (None, "UNAVAILABLE")]
    else:
        paths = sorted(root.glob("*/*/*.jsonl"))
    for path in paths:
        rows = []
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                errors.append(f"{path}:{line_no}: {error}")
                rows = []
                break
        if not rows:
            continue
        phases = {row["phase"]: row for row in rows if not row.get("skipped")}
        case = str(rows[0].get("case", "UNAVAILABLE"))
        kernel = phases.get(f"kernel_after_retouch_{case}")
        run = {
            "source_file": str(path.relative_to(root)),
            "run_id": rows[0].get("run_id"),
            "case": case,
            "bytes_per_array": rows[0].get("bytes_per_array"),
            "correct": bool(kernel and kernel.get("correct")) and all(
                row.get("correct", False) for row in rows if not row.get("skipped")
            ),
        }
        for field, phase in PHASES.items():
            selected = phases.get(phase) if phase else kernel
            run[field] = selected.get("elapsed_ms") if selected else "UNAVAILABLE"
        runs.append(run)
    return runs, errors


def read_csv_first(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="") as source:
        return next(csv.DictReader(source), {})


def latest_profile(root: Path, case: str) -> dict[str, Any]:
    reports = sorted(root.glob(f"{case}_*.nsys-rep"))
    if not reports:
        return {"case": case, "status": "UNAVAILABLE"}
    prefix = reports[-1].with_suffix("")
    totals = read_csv_first(Path(str(prefix) + "_stats_um_total_sum.csv"))
    phase_name = f"kernel_after_retouch_{case}"
    phase_files = sorted(root.glob(f"{prefix.name}_phase_{phase_name}_um_total_sum_nvtx=*.csv"))
    kernel = read_csv_first(phase_files[-1]) if phase_files else {}
    prefetch_files = sorted(root.glob(f"{prefix.name}_phase_gpu_prefetch_after_retouch_um_total_sum_nvtx=*.csv"))
    prefetch = read_csv_first(prefetch_files[-1]) if prefetch_files else {}
    def phase_metric(row: dict[str, str], key: str) -> str:
        if not row:
            return "UNAVAILABLE"
        return row.get(key) or "0"
    return {
        "case": case,
        "status": "AVAILABLE",
        "report": reports[-1].name,
        "h2d_mb_total": totals.get("Total HtoD Migration Size (MB)", "UNAVAILABLE"),
        "d2h_mb_total": totals.get("Total DtoH Migration Size (MB)", "UNAVAILABLE"),
        "gpu_faults_total": totals.get("Total GPU PageFaults", "UNAVAILABLE"),
        "cpu_faults_run_total": totals.get("Total CPU Page Faults", "UNAVAILABLE"),
        "kernel_h2d_mb": phase_metric(kernel, "Total HtoD Migration Size (MB)"),
        "kernel_gpu_faults": phase_metric(kernel, "Total GPU PageFaults"),
        "prefetch_h2d_mb": phase_metric(prefetch, "Total HtoD Migration Size (MB)"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    experiment = parser.parse_args().experiment_dir.resolve()
    root = experiment / "results" / "prefetch_ab"
    root.mkdir(parents=True, exist_ok=True)
    runs, parse_errors = read_runs(root)
    (experiment / "results" / "prefetch_ab_parse_errors.txt").write_text(
        "\n".join(parse_errors) + ("\n" if parse_errors else "")
    )
    fields = ["source_file", "run_id", "case", "bytes_per_array", *PHASES, "correct"]
    with (experiment / "results" / "prefetch_ab_runs.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(runs)

    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(int(run["bytes_per_array"]), str(run["case"]))].append(run)
    summary_fields = ["bytes_per_array", "case", "metric", "evidence_class",
                      "count", "mean", "median", "min", "max", "stdev", "p95"]
    summary_rows: list[dict[str, Any]] = []
    for (size, case), group in sorted(grouped.items()):
        for metric in PHASES:
            values = [float(row[metric]) for row in group if row[metric] != "UNAVAILABLE"]
            summary_rows.append({"bytes_per_array": size, "case": case, "metric": metric,
                                 "evidence_class": "PROGRAM_TIMING", **stats(values)})
    with (experiment / "results" / "prefetch_ab_summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, summary_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)

    profiles = [latest_profile(root, case) for case in ("demand", "prefetch")]
    lines = [
        "# Explicit GPU Prefetch A/B Results", "",
        "Evidence classes: program timings are `PROGRAM_TIMING`; migration and fault counts are `NSIGHT_UVM`.", "",
        "Demand and prefetch cases run in separate processes. Both start from a new managed allocation, run CPU first-touch and two kernels, prefetch A/B/C to the CPU, and retouch A/B. Only the explicit-prefetch case migrates A/B/C to the GPU before its measured post-retouch kernel.", "",
        "## Timing", "",
        "| Bytes/array | Case | Metric | Count | Mean ms | Median ms | p95 ms | Correct |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        correct = all(item["correct"] for item in grouped[(int(row["bytes_per_array"]), str(row["case"]))])
        lines.append(f"| {row['bytes_per_array']} | {row['case']} | {row['metric']} | {row['count']} | "
                     f"{row['mean']} | {row['median']} | {row['p95']} | {correct} |")
    lines += ["", "## Nsight", "",
              "| Case | HtoD total MB | DtoH total MB | GPU faults total | CPU faults run total | Post-retouch kernel HtoD MB | Post-retouch kernel GPU faults | Explicit-prefetch HtoD MB |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for profile in profiles:
        lines.append(f"| {profile['case']} | {profile.get('h2d_mb_total', 'UNAVAILABLE')} | "
                     f"{profile.get('d2h_mb_total', 'UNAVAILABLE')} | {profile.get('gpu_faults_total', 'UNAVAILABLE')} | "
                     f"{profile.get('cpu_faults_run_total', 'UNAVAILABLE')} | {profile.get('kernel_h2d_mb', 'UNAVAILABLE')} | "
                     f"{profile.get('kernel_gpu_faults', 'UNAVAILABLE')} | {profile.get('prefetch_h2d_mb', 'UNAVAILABLE')} |")
    lines += ["", "CPU fault counts are run-wide totals because this Nsight export does not safely attribute them to NVTX phases. Timing alone is not residency evidence. The hypothesis that explicit prefetch shifts HtoD migration out of the kernel and reduces GPU faults is accepted only if the Nsight phase data above supports it.", ""]
    (experiment / "docs" / "PREFETCH_AB_RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
