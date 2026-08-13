#!/usr/bin/env python3
"""Aggregate Stage 3 timing, Linux resource, Nsight, and safety evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def stats(values: list[float]) -> dict[str, float | int | str]:
    if not values:
        return {key: "UNAVAILABLE" for key in ("count", "mean", "median", "stdev", "p95", "min", "max")}
    ordered = sorted(values)
    return {
        "count": len(values), "mean": statistics.fmean(values), "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p95": ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)],
        "min": ordered[0], "max": ordered[-1],
    }


def resource_usage(path: Path) -> dict[str, int | str]:
    result: dict[str, int | str] = {}
    if not path.exists():
        return result
    mapping = {
        "Minor (reclaiming a frame) page faults": "minor_faults",
        "Major (requiring I/O) page faults": "major_faults",
        "Voluntary context switches": "voluntary_context_switches",
        "Involuntary context switches": "involuntary_context_switches",
        "Maximum resident set size (kbytes)": "max_rss_kib",
    }
    for line in path.read_text(errors="replace").splitlines():
        for label, key in mapping.items():
            if label in line:
                try: result[key] = int(line.rsplit(":", 1)[1].strip())
                except ValueError: result[key] = "UNAVAILABLE"
    return result


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def aggregate(root: Path) -> None:
    stage3 = root / "results" / "stage3"
    manifests = []
    for path in sorted(stage3.glob("**/manifest.json")):
        data = json.loads(path.read_text())
        data["run_dir"] = str(path.parent.relative_to(root))
        data["program_rows"] = load_jsonl(path.parent / "program.jsonl")
        data["resource"] = resource_usage(path.parent / "resource_usage.txt")
        manifests.append(data)
    preflight_path = stage3 / "preflight.json"
    preflight = json.loads(preflight_path.read_text()) if preflight_path.exists() else {}

    first_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in manifests:
        if run.get("experiment") != "cpu_first_touch":
            continue
        phase = next((row for row in run["program_rows"] if row.get("phase") == "cpu_first_touch"), None)
        if phase:
            key = (str(run.get("policy")), str(run.get("first_touch_pattern")),
                   str(run.get("prefetch_cpu_before_first_touch")))
            first_groups[key].append({"phase": phase, "run": run})
    first_rows = []
    for (policy, pattern, prefetch), values in sorted(first_groups.items()):
        elapsed = [float(value["phase"]["elapsed_ms"]) for value in values]
        summary = stats(elapsed)
        resource_keys = ("minor_faults", "major_faults", "voluntary_context_switches",
                         "involuntary_context_switches", "max_rss_kib")
        row: dict[str, Any] = {
            "policy": policy, "first_touch_pattern": pattern,
            "prefetch_cpu_before_first_touch": prefetch,
            "evidence_class": "PROGRAM_TIMING+LINUX_RESOURCE_USAGE",
            **{f"elapsed_ms_{key}": value for key, value in summary.items()},
            "correctness_pass_rate": sum(value["run"].get("correct", False) for value in values) / len(values),
        }
        for resource_key in resource_keys:
            observed = [float(value["run"]["resource"][resource_key]) for value in values
                        if isinstance(value["run"]["resource"].get(resource_key), int)]
            row[f"{resource_key}_mean"] = statistics.fmean(observed) if observed else "UNAVAILABLE"
        first_rows.append(row)
    first_fields = [
        "policy", "first_touch_pattern", "prefetch_cpu_before_first_touch", "evidence_class",
        "elapsed_ms_count", "elapsed_ms_mean", "elapsed_ms_median", "elapsed_ms_stdev",
        "elapsed_ms_p95", "elapsed_ms_min", "elapsed_ms_max", "correctness_pass_rate",
        "minor_faults_mean", "major_faults_mean", "voluntary_context_switches_mean",
        "involuntary_context_switches_mean", "max_rss_kib_mean",
    ]
    write_csv(root / "results" / "stage3_first_touch_summary.csv", first_rows, first_fields)

    oversub_groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    correctness: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for run in manifests:
        if run.get("experiment") not in {"oversub", "joint_policy"}:
            continue
        if run.get("run_kind") != "timing":
            continue
        key_base = (str(run.get("policy")), str(run.get("ratio")))
        successful = bool(run.get("correct")) and run.get("exit_code") == 0
        correctness[key_base].append(successful)
        if not successful:
            continue
        for phase in run["program_rows"]:
            if str(phase.get("phase", "")).startswith(("phase_A_first", "phase_B_first", "phase_A_reuse", "phase_B_reuse")):
                oversub_groups[(key_base[0], key_base[1], str(phase["phase"]).split("_cycle_")[0])].append(
                    float(phase["elapsed_ms"]))
    oversub_rows = []
    for (policy, ratio, phase), values in sorted(oversub_groups.items()):
        oversub_rows.append({
            "policy": policy, "ratio": ratio, "phase": phase,
            "evidence_class": "PROGRAM_TIMING", **stats(values),
            "correctness_pass_rate": statistics.fmean(correctness[(policy, ratio)]),
        })
    write_csv(root / "results" / "stage3_oversub_summary.csv", oversub_rows,
              ["policy", "ratio", "phase", "evidence_class", "count", "mean", "median",
               "stdev", "p95", "min", "max", "correctness_pass_rate"])

    trace_summary = root / "results" / "stage3_trace_summary.csv"
    trace_rows = list(csv.DictReader(trace_summary.open())) if trace_summary.exists() else []
    refault_summary = root / "results" / "stage3_eviction_refault_summary.csv"
    refault_rows = list(csv.DictReader(refault_summary.open())) if refault_summary.exists() else []
    smoke_files = sorted((stage3 / "nonpriv_smoke").glob("*.jsonl"))
    smoke_rows = [row for path in smoke_files for row in load_jsonl(path)]
    smoke_correct = bool(smoke_rows) and all(row.get("correct", False) for row in smoke_rows)
    observed_ratios = sorted({str(row["ratio"]) for row in oversub_rows})
    required_policies = {
        "custom_no_policy", "prefetch_none", "prefetch_always_max",
        "prefetch_adaptive_sequential",
    }
    successful_timing_counts: dict[tuple[str, str], int] = defaultdict(int)
    for run in manifests:
        if (run.get("experiment") == "oversub" and run.get("run_kind") == "timing"
                and run.get("exit_code") == 0 and run.get("correct")
                and run.get("struct_ops_detached") and run.get("xid_delta") == 0):
            successful_timing_counts[(str(run.get("ratio")), str(run.get("policy")))] += 1
    completed_ratios = sorted(
        ratio for ratio in {"0.95", "1.05", "1.10"}
        if all(successful_timing_counts[(ratio, policy)] >= 3 for policy in required_policies)
    )
    all_safe = bool(manifests) and all(
        run.get("exit_code") == 0 and run.get("correct") and run.get("struct_ops_detached")
        and run.get("xid_delta") == 0 for run in manifests)
    full_pass = (
        {"0.95", "1.05", "1.10"} <= set(completed_ratios)
        and trace_rows and any(str(row.get("refaulted_block_count", "0")).isdigit()
                               and int(row["refaulted_block_count"]) > 0 for row in refault_rows)
        and all_safe
    )
    timed_out = any(run.get("exit_code") == 124 for run in manifests)
    if full_pass:
        status = "PASS_GPU_EXT_STAGE3_TRACE_AND_OVERSUBSCRIPTION"
    elif timed_out:
        status = "PARTIAL_GPU_EXT_STAGE3_STOPPED_AT_RUNTIME_LIMIT"
    elif manifests:
        status = "PARTIAL_GPU_EXT_STAGE3_RUNTIME"
    else:
        status = "READY_FOR_MANUAL_STAGE3"

    status_data = {
        "evidence_class": "GPU_EXT_STAGE3_STATUS",
        "status": status,
        "observed_ratios": observed_ratios,
        "full_four_policy_ratios": completed_ratios,
        "successful_timing_runs": {
            ratio: {
                policy: successful_timing_counts[(ratio, policy)]
                for policy in sorted(required_policies)
            }
            for ratio in ("0.95", "1.05", "1.10")
        },
        "runtime_limits": [
            {
                "policy": run.get("policy"), "ratio": run.get("ratio"),
                "run_kind": run.get("run_kind"), "exit_code": run.get("exit_code"),
                "run_dir": run.get("run_dir"),
            }
            for run in manifests if run.get("exit_code") == 124
        ],
        "joint_policy_executed": any(run.get("experiment") == "joint_policy" for run in manifests),
        "full_pass": full_pass,
    }
    (stage3 / "status.json").write_text(json.dumps(status_data, indent=2, sort_keys=True) + "\n")

    docs = root / "docs"
    if not (docs / "TRACE_OVERHEAD.md").exists():
        (docs / "TRACE_OVERHEAD.md").write_text(
        "# Stage 3 Trace Overhead\n\n"
        "Status: `NOT_MEASURED_WITH_NEW_MODULE`.\n\n"
        "The existing distribution-driver representative is 240.293 ms and the Stage 2 custom "
        "no-policy 10-run mean is 240.731 ms for the 256 MiB demand kernel. "
        "`run_trace_overhead.sh` collects ten new-module no-trace and ten trace-attached runs. "
            "No overhead claim is made until those runs exist; the acceptance threshold is approximately 1%.\n")
    if not (docs / "CPU_FIRST_TOUCH_DIAGNOSIS.md").exists():
        (docs / "CPU_FIRST_TOUCH_DIAGNOSIS.md").write_text(
        "# CPU First-Touch Diagnosis\n\n"
        "Status: `UNAVAILABLE_PENDING_CUSTOM_MODULE_RUN`.\n\n"
        "The runner compares full sequential touch, page-stride touch, and CPU-prefetch-then-full-touch "
        "under each policy with `/usr/bin/time -v` plus the enhanced decision trace. The prior 337 ms "
            "versus 2025 ms observation is retained as an anomaly, not yet assigned to CPU faults or callbacks.\n")
    if not (docs / "ALWAYS_MAX_MIGRATION_DIAGNOSIS.md").exists():
        (docs / "ALWAYS_MAX_MIGRATION_DIAGNOSIS.md").write_text(
        "# always_max Migration Diagnosis\n\n"
        "Status: `UNAVAILABLE_PENDING_ARRAY_ISOLATION_RUN`.\n\n"
        "The current Nsight SQLite schema exposes UVM copy `virtualAddress`, `bytes`, and copy kind, "
        "so address attribution is implementable. The CUDA workload now records exact A/B/C ranges and provides `read-a`, `read-b`, "
        "`write-c`, and `vector-add` NVTX-separated modes. Migration bytes will be assigned to an array "
        "only if the exported Nsight schema includes a virtual address within those ranges. The existing "
            "620.757 MB versus 805.306 MB total is not interpreted as an array-specific reduction.\n")

    report = [
        "# gpu_ext UVM Stage 3 Results", "", f"Status: `{status}`.", "",
        "This stage currently contains implementation and non-privileged validation only. No custom module switch, policy attach, or oversubscription run is inferred from absent result directories.", "",
        "## Evidence Inventory", "",
        f"- Stage 3 runtime manifests: {len(manifests)}",
        f"- Enhanced custom module built: {preflight.get('custom_binary_trace_symbols', 'UNAVAILABLE')}",
        f"- Enhanced custom module loaded: {preflight.get('custom_module_loaded', 'UNAVAILABLE')}",
        f"- Enhanced custom module SHA256: {preflight.get('custom_module_sha256', 'UNAVAILABLE')}",
        f"- Distribution-driver non-privileged smoke JSONL files: {len(smoke_files)}",
        f"- Non-privileged smoke correctness: {smoke_correct if smoke_rows else 'UNAVAILABLE'}",
        f"- Enhanced decision trace runs: {len(trace_rows)}",
        f"- Observed oversubscription ratios: {', '.join(observed_ratios) if observed_ratios else 'NONE'}",
        f"- Full four-policy oversubscription ratios: {', '.join(completed_ratios) if completed_ratios else 'NONE'}",
        f"- Proven eviction/refault runs: {sum(str(row.get('refaulted_block_count', '0')).isdigit() and int(row['refaulted_block_count']) > 0 for row in refault_rows)}",
        f"- All executed Stage 3 cases safe/correct/detached: {all_safe if manifests else 'UNAVAILABLE'}", "",
        "## Required Questions", "",
        "1. `prefetch_none` CPU first-touch slowdown: `UNAVAILABLE`; new fault/resource/decision evidence has not run.",
        "2. CPU first touch through the prefetch hook: `UNAVAILABLE`; the new first-touch-only trace will answer this.",
        "3. `always_max` 620 MB total: `UNAVAILABLE`; no array attribution is claimed yet.",
        "4. A/B/C migration bytes: `UNAVAILABLE` for Stage 3 runs; the installed Nsight SQLite schema is address-bearing and the classifier is implemented.",
        "5. 768 callbacks as 768 2 MiB regions: not proven; callback and final decision are distinct, and `final_pages` must be inspected.",
        "6. Action counts: available only after enhanced traces are collected.",
        "7. Policy region versus final region: available only after enhanced traces are collected.",
        "8. First eviction ratio: `UNAVAILABLE`; no oversubscription run occurred.",
        "9. A-B-A eviction/refault: `UNAVAILABLE`; selected-victim tracing is implemented but not run.",
        "10. Aggressive-prefetch thrashing: `UNAVAILABLE`.",
        "11. Adaptive low/high-pressure behavior: `UNAVAILABLE`.",
        "12. Workload generality: any future result applies only to this sequential scan unless independently reproduced.",
        "13. New trace overhead: `UNAVAILABLE`; dedicated 10+10 runner is ready.",
        "14. Policy detach: no Stage 3 policy was attached.",
        "15. Distribution `nvidia_uvm`: remains loaded at implementation time; no module operation was executed.", "",
        "## Safety", "",
        "No root command, module reload, policy attachment, oversubscription allocation, GPU setting change, or git commit was performed by the implementation/preflight path.",
        "The final `nvidia-smi` check reported the A30 normally with 0 MiB used and no compute process. Unprivileged kernel-log access was unavailable, so an Xid count is reported as `UNAVAILABLE`, not zero.",
    ]
    if not (docs / "STAGE3_RESULTS.md").exists():
        (docs / "STAGE3_RESULTS.md").write_text("\n".join(report) + "\n")
    print(status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.experiment_dir.resolve())


if __name__ == "__main__":
    main()
