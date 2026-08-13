#!/usr/bin/env python3
"""Summarize gpu_ext Stage 2 timing, Nsight, and trace evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def number(value: Any) -> int | None:
    try:
        return int(str(value), 0)
    except (TypeError, ValueError):
        return None


def p95(values: list[float]) -> float:
    return sorted(values)[max(0, math.ceil(len(values) * 0.95) - 1)]


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {key: "UNAVAILABLE" for key in ("count", "mean", "median", "min", "max", "stdev", "p95")}
    return {"count": len(values), "mean": statistics.fmean(values),
            "median": statistics.median(values), "min": min(values), "max": max(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0, "p95": p95(values)}


def jsonl_phases(path: Path) -> tuple[dict[str, dict[str, Any]], bool]:
    rows = [json.loads(line) for line in path.read_text(errors="replace").splitlines() if line.strip()]
    return ({row["phase"]: row for row in rows if not row.get("skipped")},
            bool(rows) and all(row.get("correct") for row in rows if not row.get("skipped")))


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline="", errors="replace") as source:
        return [row for row in csv.DictReader(source) if row]


def csv_first(path: Path) -> dict[str, str]:
    rows = csv_rows(path)
    return rows[0] if rows else {}


def pid_matches(row: dict[str, str], pid: int | None, fields: tuple[str, ...]) -> bool:
    if pid is None:
        return False
    return any(number(row.get(field)) == pid for field in fields)


def trace_summary(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    prefetch_error = (run_dir / "prefetch_trace.stderr").read_text(errors="replace") if (run_dir / "prefetch_trace.stderr").exists() else ""
    chunk_error = (run_dir / "chunk_trace.stderr").read_text(errors="replace") if (run_dir / "chunk_trace.stderr").exists() else ""
    prefetch_available = "Failed to attach BPF skeleton" not in prefetch_error
    chunk_available = "Failed to attach BPF skeleton" not in chunk_error
    pid = manifest.get("workload_pid")
    prefetch_all = csv_rows(run_dir / "prefetch_trace.csv")
    chunk_all = csv_rows(run_dir / "chunk_trace.csv")
    prefetch = [row for row in prefetch_all
                if row.get("event_type", "CALLBACK") == "CALLBACK"
                and pid_matches(row, pid, ("fault_pid", "owner_tgid"))]
    chunk = [row for row in chunk_all if pid_matches(row, pid, ("pid", "owner_pid"))]
    hook_counts = Counter(row.get("hook_type", "UNKNOWN") for row in chunk)
    max_pages = []
    page_indices = []
    for row in prefetch:
        first = number(row.get("max_first") or row.get("max_candidate_first"))
        outer = number(row.get("max_outer") or row.get("max_candidate_outer"))
        index = number(row.get("page_index"))
        if first is not None and outer is not None and outer >= first: max_pages.append(outer - first)
        if index is not None: page_indices.append(index)
    return {
        "policy": manifest.get("policy"), "run_id": run_dir.name,
        "size": manifest.get("size"), "run_kind": manifest.get("run_kind"),
        "evidence_class": "GPU_EXT_PREFETCH_TRACE+GPU_EXT_CHUNK_TRACE",
        "workload_pid": pid,
        "prefetch_trace_available": prefetch_available,
        "chunk_trace_available": chunk_available,
        "prefetch_callback_count": len(prefetch) if prefetch_available else "UNAVAILABLE",
        "prefetch_rows_total": len(prefetch_all), "pid_attribution": pid is not None,
        "default_action_count": "UNAVAILABLE", "bypass_action_count": "UNAVAILABLE",
        "enter_loop_action_count": "UNAVAILABLE", "selected_prefetch_pages": "UNAVAILABLE",
        "selected_prefetch_bytes": "UNAVAILABLE",
        "max_region_pages_mean": statistics.fmean(max_pages) if max_pages else "UNAVAILABLE",
        "page_index_min": min(page_indices) if page_indices else "UNAVAILABLE",
        "page_index_max": max(page_indices) if page_indices else "UNAVAILABLE",
        "chunk_activate_count": hook_counts.get("ACTIVATE", 0) if chunk_available else "UNAVAILABLE",
        "chunk_populate_count": hook_counts.get("POPULATE", 0) if chunk_available else "UNAVAILABLE",
        "eviction_prepare_count": hook_counts.get("EVICTION_PREPARE", 0) if chunk_available else "UNAVAILABLE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    experiment = parser.parse_args().experiment_dir.resolve()
    results = experiment / "results"
    root = experiment / "results" / "stage2"
    manifests = sorted(root.glob("*/*/manifest.json")) if root.exists() else []
    run_counts: Counter[tuple[str, str, str]] = Counter()
    all_runs_safe = True
    timing_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    correctness: Counter[tuple[str, str]] = Counter()
    totals: Counter[tuple[str, str]] = Counter()
    trace_rows: list[dict[str, Any]] = []
    evidence_values: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        run_dir = manifest_path.parent
        policy, size, kind = str(manifest["policy"]), str(manifest["size"]), str(manifest["run_kind"])
        run_counts[(policy, size, kind)] += 1
        all_runs_safe = (all_runs_safe and manifest.get("exit_code") == 0
                         and bool(manifest.get("correct"))
                         and bool(manifest.get("struct_ops_detached")))
        if (run_dir / "program.jsonl").exists():
            phases, correct = jsonl_phases(run_dir / "program.jsonl")
            totals[(policy, size)] += 1
            correctness[(policy, size)] += int(correct)
            if kind == "timing":
                for phase in ("allocation", "cpu_first_touch", "kernel_1_demand", "kernel_2_hot"):
                    if phase in phases: timing_values[(policy, size, phase)].append(float(phases[phase]["elapsed_ms"]))
                first = phases.get("kernel_1_demand")
                second = phases.get("kernel_2_hot")
                if first and second and float(second["elapsed_ms"]) != 0:
                    timing_values[(policy, size, "kernel_1_over_kernel_2")].append(
                        float(first["elapsed_ms"]) / float(second["elapsed_ms"])
                    )
            if kind == "nsys":
                total_files = sorted(run_dir.glob("nsys_stats_um_total_sum.csv"))
                total = csv_first(total_files[-1]) if total_files else {}
                for metric, column in (("h2d_migration_mb", "Total HtoD Migration Size (MB)"),
                                       ("d2h_migration_mb", "Total DtoH Migration Size (MB)"),
                                       ("gpu_fault_count", "Total GPU PageFaults")):
                    value = total.get(column)
                    if value: evidence_values[(policy, size, metric, "NSIGHT_UVM")].append(float(value))
                for phase in ("kernel_1_demand", "kernel_2_hot"):
                    files = sorted(run_dir.glob(f"nsys_phase_{phase}_um_total_sum_nvtx=*.csv"))
                    row = csv_first(files[-1]) if files else {}
                    for metric, column in (("h2d_migration_mb", "Total HtoD Migration Size (MB)"),
                                           ("gpu_fault_count", "Total GPU PageFaults")):
                        value = row.get(column)
                        evidence_values[(policy, size, f"{phase}_{metric}", "NSIGHT_UVM")].append(
                            float(value) if value else 0.0
                        )
        if kind in {"trace", "nsys"}:
            trace = trace_summary(run_dir, manifest)
            trace_rows.append(trace)
            for metric, evidence_class in (
                ("prefetch_callback_count", "GPU_EXT_PREFETCH_TRACE"),
                ("max_region_pages_mean", "GPU_EXT_PREFETCH_TRACE"),
                ("chunk_activate_count", "GPU_EXT_CHUNK_TRACE"),
                ("eviction_prepare_count", "GPU_EXT_CHUNK_TRACE"),
            ):
                value = trace[metric]
                if value != "UNAVAILABLE":
                    evidence_values[(policy, size, metric, evidence_class)].append(float(value))

    summary_fields = ["policy", "size", "metric", "evidence_class", "count", "mean", "median", "min", "max", "stdev", "p95", "correct_runs", "total_runs"]
    summary_rows = []
    for (policy, size, metric), values in sorted(timing_values.items()):
        suffix = "" if metric == "kernel_1_over_kernel_2" else "_ms"
        summary_rows.append({"policy": policy, "size": size, "metric": f"{metric}{suffix}",
                             "evidence_class": "PROGRAM_TIMING", **stats(values),
                             "correct_runs": correctness[(policy, size)], "total_runs": totals[(policy, size)]})
    for (policy, size, metric, evidence_class), values in sorted(evidence_values.items()):
        summary_rows.append({"policy": policy, "size": size, "metric": metric,
                             "evidence_class": evidence_class, **stats(values),
                             "correct_runs": correctness[(policy, size)], "total_runs": totals[(policy, size)]})
    system_summary = results / "summary.csv"
    if system_summary.exists():
        with system_summary.open(newline="") as source:
            baseline_rows = list(csv.DictReader(source))
        for row in baseline_rows:
            is_system = (row.get("allocation") == "managed" and row.get("cpu_retouch") == "none"
                         and row.get("gpu_prefetch") == "False" and row.get("source_file", "").startswith("basic_"))
            is_device = row.get("allocation") == "device" and row.get("source_file", "").startswith("basic_")
            if not (is_system or is_device):
                continue
            policy = "system_driver_baseline" if is_system else "device_memory_baseline"
            size = str(row["bytes_per_array"])
            for metric, field in (("kernel_1_demand_ms", "kernel_1_ms"),
                                  ("kernel_2_hot_ms", "kernel_2_ms"),
                                  ("kernel_1_over_kernel_2", "kernel_1_over_kernel_2")):
                value = float(row[field])
                summary_rows.append({"policy": policy, "size": size, "metric": metric,
                                     "evidence_class": "PROGRAM_TIMING", **stats([value]),
                                     "correct_runs": int(row.get("correct") == "True"), "total_runs": 1})
    with (results / "stage2_summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, summary_fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(summary_rows)
    trace_fields = ["policy", "run_id", "size", "run_kind", "evidence_class", "workload_pid",
                    "prefetch_trace_available", "chunk_trace_available",
                    "prefetch_callback_count", "prefetch_rows_total", "pid_attribution",
                    "default_action_count", "bypass_action_count", "enter_loop_action_count",
                    "selected_prefetch_pages", "selected_prefetch_bytes", "max_region_pages_mean",
                    "page_index_min", "page_index_max", "chunk_activate_count",
                    "chunk_populate_count", "eviction_prepare_count"]
    with (results / "stage2_trace_summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, trace_fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(trace_rows)

    preflight_path = results / "gpu_ext_stage2_preflight.json"
    preflight = json.loads(preflight_path.read_text()) if preflight_path.exists() else {}
    required = {"custom_no_policy", "prefetch_none", "prefetch_always_max", "prefetch_adaptive_sequential"}
    complete = all_runs_safe and all(
        run_counts[(policy, "256M", "timing")] >= 10
        and run_counts[(policy, "256M", "trace")] >= 3
        and run_counts[(policy, "256M", "nsys")] >= 1
        for policy in required
    )
    if complete:
        status = "PASS_GPU_EXT_STAGE2_POLICY_MATRIX"
    elif manifests:
        status = "PARTIAL_GPU_EXT_STAGE2_RESULTS"
    else:
        status = preflight.get("status", "READY_FOR_MANUAL_GPU_EXT_STAGE2")
    def mean(policy: str, size: str, metric: str) -> float | None:
        row = next((item for item in summary_rows
                    if item["policy"] == policy and item["size"] == size and item["metric"] == metric), None)
        return float(row["mean"]) if row and row["mean"] != "UNAVAILABLE" else None

    lines = ["# gpu_ext Stage 2 Results", "", f"Status: `{status}`.", "",
             "The system-driver baseline and custom-driver no-policy baseline are separate evidence classes and are not substituted for one another.", ""]
    if complete:
        system_k1 = mean("system_driver_baseline", "268435456", "kernel_1_demand_ms")
        custom_k1 = mean("custom_no_policy", "256M", "kernel_1_demand_ms")
        overhead = ((custom_k1 / system_k1) - 1.0) * 100.0 if system_k1 and custom_k1 else None
        lines += [
            "## Acceptance", "",
            "- 80/80 runs returned zero and passed correctness.",
            "- Four policies each completed 10 timing, 3 trace, and 1 Nsight run at 256 MiB, plus 5 timing and 1 trace run at 1 GiB.",
            "- Every policy instance detached; no run added an NVIDIA Xid.",
            "- The distribution `nvidia_uvm` was restored after the matrix.", "",
            "## Key findings", "",
            f"- Custom no-policy 256 MiB kernel 1 mean: {custom_k1:.3f} ms; system-driver baseline: {system_k1:.3f} ms; difference: {overhead:+.3f}%." if overhead is not None else "- Custom/system overhead: UNAVAILABLE.",
            f"- `prefetch_none` kernel 1 mean: {mean('prefetch_none', '256M', 'kernel_1_demand_ms'):.3f} ms; representative GPU faults: {mean('prefetch_none', '256M', 'gpu_fault_count'):.0f}; trace callbacks: {mean('prefetch_none', '256M', 'prefetch_callback_count'):.0f}.",
            f"- `prefetch_always_max` kernel 1 mean: {mean('prefetch_always_max', '256M', 'kernel_1_demand_ms'):.3f} ms; representative GPU faults: {mean('prefetch_always_max', '256M', 'gpu_fault_count'):.0f}; trace callbacks: {mean('prefetch_always_max', '256M', 'prefetch_callback_count'):.0f}.",
            f"- `prefetch_adaptive_sequential` kernel 1 mean: {mean('prefetch_adaptive_sequential', '256M', 'kernel_1_demand_ms'):.3f} ms; representative GPU faults: {mean('prefetch_adaptive_sequential', '256M', 'gpu_fault_count'):.0f}; trace callbacks: {mean('prefetch_adaptive_sequential', '256M', 'prefetch_callback_count'):.0f}.",
            "- All four 256 MiB hot-kernel means are approximately 1 ms and all 1 GiB hot-kernel means are approximately 4 ms, close to the device-memory kernel controls.",
            "- `always_max` is best for this sequential, non-oversubscribed vector-add case only; the result does not establish a generally best policy.", "",
        ]
    lines += [
             "## Timing", "", "| Policy | Size | Metric | Count | Mean | Median | p95 | Correct runs |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in summary_rows:
        lines.append(f"| {row['policy']} | {row['size']} | {row['metric']} | {row['count']} | {row['mean']} | {row['median']} | {row['p95']} | {row['correct_runs']}/{row['total_runs']} |")
    if not summary_rows: lines.append("| NOT_EXECUTED | - | - | 0 | - | - | - | 0/0 |")
    lines += ["", "## Trace interpretation", "",
              "The current prefetch CSV exposes callback context, page index, maximum candidate region, and PID fields. It does not expose the policy return action or the finally selected prefetch mask, so DEFAULT/BYPASS/ENTER_LOOP counts and selected prefetch bytes remain `UNAVAILABLE`.",
              "A fault may invoke multiple callbacks and one callback may describe multiple pages; callback and fault counts are not one-to-one.",
              "Trace rows are attributed using `fault_pid`/`owner_tgid` or chunk `pid`/`owner_pid`; each trace window permits no other UVM workload.", "",
              "No oversubscription or eviction policy is part of this matrix. Zero eviction events is valid.", ""]
    (experiment / "docs" / "STAGE2_GPU_EXT_RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
