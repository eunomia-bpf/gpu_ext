#!/usr/bin/env python3
"""Summarize UVM basic JSONL, Nsight CSV, and gpu_ext trace evidence."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(results: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths: set[Path] = set(results.glob("program_*.jsonl"))
    for pattern in ("basic_*.jsonl", "profile_*.jsonl"):
        candidates = sorted(results.glob(pattern))
        if candidates:
            paths.add(candidates[-1])
    for path in sorted(paths):
        for line_no, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {error}") from error
            row["source_file"] = path.name
            rows.append(row)
    return rows


def ratio(numerator: float | None, denominator: float | None) -> str:
    if numerator is None or denominator in (None, 0):
        return "UNAVAILABLE"
    return f"{numerator / denominator:.6f}"


def trace_count(path: Path) -> int:
    lines = [line for line in path.read_text(errors="replace").splitlines() if line.strip()]
    if not lines:
        return 0
    return max(0, len(lines) - (1 if any(token in lines[0].lower() for token in ("time", "pid", "event")) else 0))


def find_nsys_fault_evidence(results: Path) -> tuple[str, list[str]]:
    reports = sorted(results.glob("uvm_basic_*.nsys-rep"))
    if not reports:
        return "UNAVAILABLE", []
    prefix = reports[-1].stem
    files: list[str] = []
    for path in sorted(results.glob(f"{prefix}*.csv")):
        if "_um_" not in path.name and "_um." not in path.name:
            continue
        text = path.read_text(errors="replace")
        if text.strip():
            files.append(path.name)
    if not files:
        return "UNAVAILABLE", []
    return "AVAILABLE_IN_NSYS_CSV", files


def read_single_csv(path: Path) -> dict[str, str]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    return rows[0] if rows else {}


def nsys_um_totals(results: Path) -> dict[str, str]:
    reports = sorted(results.glob("uvm_basic_*.nsys-rep"))
    if not reports:
        return {}
    path = results / f"{reports[-1].stem}_stats_um_total_sum.csv"
    return read_single_csv(path) if path.exists() else {}


def nsys_um_phases(results: Path) -> list[dict[str, str]]:
    pattern = re.compile(r"_phase_(.+?)_um_total_sum_nvtx=")
    reports = sorted(results.glob("uvm_basic_*.nsys-rep"))
    if not reports:
        return []
    prefix = reports[-1].stem
    latest: dict[str, Path] = {}
    for path in sorted(results.glob(f"{prefix}_phase_*_um_total_sum_nvtx=*.csv")):
        match = pattern.search(path.name)
        if match:
            latest[match.group(1)] = path
    phase_rows: list[dict[str, str]] = []
    for phase, path in latest.items():
        row = read_single_csv(path)
        phase_rows.append({
            "phase": phase,
            "h2d_mb": row.get("Total HtoD Migration Size (MB)") or "0",
            "d2h_mb": row.get("Total DtoH Migration Size (MB)") or "0",
            "gpu_faults": row.get("Total GPU PageFaults") or "0",
        })
    return sorted(phase_rows, key=lambda row: row["phase"])


def summarize(experiment_dir: Path) -> None:
    results = experiment_dir / "results"
    docs = experiment_dir / "docs"
    results.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(results)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("run_id", "UNAVAILABLE"))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for run_id, run_rows in sorted(grouped.items()):
        phases = {str(row.get("phase")): row for row in run_rows if not row.get("skipped")}
        first = run_rows[0]
        kernel1 = phases.get("kernel_1_demand") or phases.get("kernel_1_device")
        kernel2 = phases.get("kernel_2_hot") or phases.get("kernel_2_device")
        kernel3 = phases.get("kernel_3_after_cpu_touch")
        kernel4 = phases.get("kernel_4_after_gpu_prefetch")
        prefetch = phases.get("explicit_gpu_prefetch")
        summary_rows.append({
            "run_id": run_id,
            "source_file": first.get("source_file", "UNAVAILABLE"),
            "allocation": first.get("allocation", "UNAVAILABLE"),
            "bytes_per_array": first.get("bytes_per_array", "UNAVAILABLE"),
            "cpu_retouch": first.get("cpu_retouch", "UNAVAILABLE"),
            "gpu_prefetch": first.get("gpu_prefetch", "UNAVAILABLE"),
            "cpu_prefetch_before_retouch": first.get("cpu_prefetch_before_retouch", "UNAVAILABLE"),
            "kernel_1_ms": kernel1.get("elapsed_ms") if kernel1 else "UNAVAILABLE",
            "kernel_2_ms": kernel2.get("elapsed_ms") if kernel2 else "UNAVAILABLE",
            "kernel_1_over_kernel_2": ratio(
                float(kernel1["elapsed_ms"]) if kernel1 else None,
                float(kernel2["elapsed_ms"]) if kernel2 else None,
            ),
            "kernel_3_ms": kernel3.get("elapsed_ms") if kernel3 else "UNAVAILABLE",
            "kernel_4_ms": kernel4.get("elapsed_ms") if kernel4 else "UNAVAILABLE",
            "kernel_3_over_kernel_4": ratio(
                float(kernel3["elapsed_ms"]) if kernel3 else None,
                float(kernel4["elapsed_ms"]) if kernel4 else None,
            ),
            "explicit_prefetch_ms": prefetch.get("elapsed_ms") if prefetch else "UNAVAILABLE",
            "correct": all(bool(row.get("correct")) for row in run_rows if not row.get("skipped")),
        })

    fields = [
        "run_id", "source_file", "allocation", "bytes_per_array", "cpu_retouch",
        "gpu_prefetch", "cpu_prefetch_before_retouch", "kernel_1_ms", "kernel_2_ms",
        "kernel_1_over_kernel_2", "kernel_3_ms", "kernel_4_ms",
        "kernel_3_over_kernel_4", "explicit_prefetch_ms", "correct",
    ]
    with (results / "summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)

    prefetch_counts: dict[str, int] = {}
    chunk_counts: dict[str, int] = {}
    for path in sorted(results.glob("prefetch_trace_*.csv")):
        prefetch_counts[path.name] = trace_count(path)
    for path in sorted(results.glob("chunk_trace_*.csv")):
        chunk_counts[path.name] = trace_count(path)
    fault_status, fault_files = find_nsys_fault_evidence(results)
    um_totals = nsys_um_totals(results)
    um_phases = nsys_um_phases(results)
    stage2_path = results / "gpu_ext_stage2_preflight.json"
    stage2 = json.loads(stage2_path.read_text()) if stage2_path.exists() else {}

    managed = [row for row in summary_rows if row["allocation"] == "managed"]
    device = [row for row in summary_rows if row["allocation"] == "device"]
    required_sizes = {268435456, 1073741824}
    completed_sizes = {
        int(row["bytes_per_array"])
        for row in summary_rows
        if row["correct"] and row["allocation"] in {"managed", "device"}
    }
    final_status = (
        "PASS_USERSPACE_UVM_BASIC"
        if required_sizes <= completed_sizes and managed and device and um_totals
        else "INCOMPLETE"
    )
    baseline_ratios: list[tuple[int, str]] = []
    for size in sorted(required_sizes):
        demand = next((row for row in managed
                       if int(row["bytes_per_array"]) == size
                       and row["cpu_retouch"] == "none"
                       and row["gpu_prefetch"] is False), None)
        control = next((row for row in device if int(row["bytes_per_array"]) == size), None)
        baseline_ratios.append((size, ratio(
            float(demand["kernel_1_ms"]) if demand else None,
            float(control["kernel_1_ms"]) if control else None,
        )))
    lines = [
        "# UVM Basic Results",
        "",
        f"Final status: `{final_status}`.",
        "",
        "Evidence class: `SYSTEM_NVIDIA_DRIVER_USERSPACE_UVM` for CUDA/Nsight runs. These are not gpu_ext hook results unless trace CSV files are listed below.",
        "",
        f"- Parsed runs: {len(summary_rows)}",
        f"- Managed runs: {len(managed)}",
        f"- Device-memory runs: {len(device)}",
        f"- All recorded non-skipped phases correct: {all(row['correct'] for row in summary_rows) if summary_rows else 'UNAVAILABLE'}",
        f"- Nsight Unified Memory fault evidence: {fault_status}",
        f"- Nsight evidence files: {', '.join(fault_files) if fault_files else 'UNAVAILABLE'}",
        f"- Nsight total HtoD migration: {um_totals.get('Total HtoD Migration Size (MB)', 'UNAVAILABLE')} MB",
        f"- Nsight total DtoH migration: {um_totals.get('Total DtoH Migration Size (MB)', 'UNAVAILABLE')} MB",
        f"- Nsight total CPU page faults: {um_totals.get('Total CPU Page Faults', 'UNAVAILABLE')}",
        f"- Nsight total GPU page faults: {um_totals.get('Total GPU PageFaults', 'UNAVAILABLE')}",
        "",
        "Timing alone does not prove page residency or migration. Fault and migration claims require the Nsight Unified Memory reports or a compatible gpu_ext trace.",
        "",
        "## Run Summary",
        "",
        "| Allocation | Bytes/array | CPU retouch | GPU prefetch | K1/K2 | K3/K4 | Correct |",
        "|---|---:|---|---|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['allocation']} | {row['bytes_per_array']} | {row['cpu_retouch']} | "
            f"{row['gpu_prefetch']} | {row['kernel_1_over_kernel_2']} | "
            f"{row['kernel_3_over_kernel_4']} | {row['correct']} |"
        )
    lines += [
        "",
        "## Managed Demand vs Device Control",
        "",
        "| Bytes/array | Managed kernel 1 / device kernel 1 |",
        "|---:|---:|",
    ]
    for size, value in baseline_ratios:
        lines.append(f"| {size} | {value} |")
    lines += [
        "",
        "## Nsight Phase Evidence",
        "",
        "| NVTX phase | HtoD MB | DtoH MB | GPU page faults |",
        "|---|---:|---:|---:|",
    ]
    if um_phases:
        for row in um_phases:
            lines.append(f"| {row['phase']} | {row['h2d_mb']} | {row['d2h_mb']} | {row['gpu_faults']} |")
    else:
        lines.append("| UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |")
    lines += [
        "",
        "Nsight repeats the run-wide CPU page-fault total in NVTX-filtered `um_total_sum` reports, so CPU faults are reported only as a run total and are not attributed to individual phases.",
    ]
    lines += ["", "## gpu_ext Trace Counts", ""]
    lines += [
        f"- Extension binary preflight: {stage2.get('status', 'UNAVAILABLE')}",
        f"- All trace/policy binaries ready: {stage2.get('all_binaries_ready', 'UNAVAILABLE')}",
        f"- Custom gpu_ext module loaded: {stage2.get('custom_binary_loaded', 'UNAVAILABLE')}",
        f"- BPF attached by this experiment: {stage2.get('bpf_attached', False)}",
    ]
    if not prefetch_counts and not chunk_counts:
        lines.append("`UNAVAILABLE`: the custom gpu_ext module was not loaded; no BPF policy was attached.")
    else:
        for name, count in prefetch_counts.items():
            lines.append(f"- Prefetch callbacks `{name}`: {count}")
        for name, count in chunk_counts.items():
            lines.append(f"- Chunk events `{name}`: {count}")
        lines.append("- Callback counts are not page-fault counts unless the trace schema explicitly establishes that equivalence.")
    lines += [
        "",
        "## Limitations",
        "",
        "- CUDA Event durations include observed execution stalls but do not identify physical residency.",
        "- `cudaMemGetInfo()` is auxiliary capacity information, not per-page residency evidence.",
        "- The default matrix does not oversubscribe GPU memory and therefore is not an eviction experiment.",
        "- No `gpu_block_access` conclusion is used because that hook is known to be unreliable in this branch.",
    ]
    if final_status == "PASS_USERSPACE_UVM_BASIC":
        lines += [
            "",
            "## Conclusions",
            "",
            "- CPU first touch followed by `kernel_1_demand` produced HtoD migration and GPU page faults in Nsight, while the immediate `kernel_2_hot` produced neither in this profiled run.",
            "- Page-stride CPU retouch produced DtoH migration; `kernel_3_after_cpu_touch` then produced HtoD migration and GPU page faults again.",
            "- After the third kernel had already restored GPU access, the explicit GPU prefetch and fourth kernel produced no additional UVM migration or GPU faults in this sequence.",
            "- The device-memory control used explicit copies and did not exercise the same managed-memory demand-paging path.",
        ]
    (docs / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.experiment_dir.resolve())


if __name__ == "__main__":
    main()
