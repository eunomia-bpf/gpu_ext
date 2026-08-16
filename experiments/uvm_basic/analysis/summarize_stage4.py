#!/usr/bin/env python3
"""Summarize Stage 4 program, decision, chunk, and safety evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path


PHASES = ("phase_A_first", "phase_B_first", "phase_A_reuse", "phase_B_reuse")
VECTOR_PHASES = ("allocation", "cpu_first_touch", "kernel_1_demand", "kernel_2_hot")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def stats(values: list[float]) -> dict[str, float | int | None]:
    mean = statistics.fmean(values) if values else None
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
    margin = 1.96 * stdev / math.sqrt(len(values)) if values and stdev is not None else None
    return {
        "count": len(values),
        "mean": mean,
        "median": statistics.median(values) if values else None,
        "stdev": stdev,
        "p95": percentile(values, 0.95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "ci95_low": mean - margin if mean is not None and margin is not None else None,
        "ci95_high": mean + margin if mean is not None and margin is not None else None,
    }


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def read_trace(path: Path) -> tuple[Counter[str], list[float]]:
    actions: Counter[str] = Counter()
    pages: list[float] = []
    if not path.exists() or path.stat().st_size == 0:
        return actions, pages
    try:
        with path.open(newline="", errors="replace") as stream:
            for row in csv.DictReader(stream):
                action = row.get("action_name") or row.get("action")
                if action:
                    actions[action.upper()] += 1
                raw_pages = row.get("final_pages")
                if raw_pages not in (None, "", "UNAVAILABLE"):
                    try:
                        pages.append(float(raw_pages))
                    except ValueError:
                        pass
    except (csv.Error, OSError):
        pass
    return actions, pages


def integer(value: object) -> int:
    try:
        return int(str(value or "0"), 0)
    except ValueError:
        return 0


def csv_rows(path: Path) -> Iterator[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return
    try:
        with path.open(newline="", errors="replace") as stream:
            yield from csv.DictReader(stream)
    except (csv.Error, OSError):
        return


def refault_evidence(program: list[dict[str, object]], root: Path) -> dict[str, object]:
    windows = {
        str(row.get("phase")): (integer(row.get("monotonic_start_ns")),
                                integer(row.get("monotonic_end_ns")))
        for row in program
        if str(row.get("phase", "")).startswith(
            ("phase_A_first", "phase_B_first", "phase_A_reuse")
        )
    }
    a_first = next((value for key, value in windows.items() if key.startswith("phase_A_first")), None)
    b_first = next((value for key, value in windows.items() if key.startswith("phase_B_first")), None)
    a_reuse = next((value for key, value in windows.items() if key.startswith("phase_A_reuse")), None)
    within = lambda stamp, window: bool(window and window[0] <= stamp <= window[1])

    evictions: dict[int, int] = {}
    selected = 0
    for row in csv_rows(root / "chunk_trace.csv"):
        if row.get("hook_type") != "EVICTION_SELECTED":
            continue
        selected += 1
        block, stamp = integer(row.get("va_start")), integer(row.get("timestamp_ns"))
        if block and within(stamp, b_first):
            evictions[block] = stamp
    first_blocks: set[int] = set()
    reuse: dict[int, int] = {}
    sizes: dict[int, int] = {}
    for row in csv_rows(root / "prefetch_decision_trace.csv"):
        if row.get("event_type") != "DECISION":
            continue
        block, stamp = integer(row.get("va_start")), integer(row.get("timestamp_ns"))
        if block and within(stamp, a_first):
            first_blocks.add(block)
        if block and within(stamp, a_reuse):
            reuse.setdefault(block, stamp)
            end = integer(row.get("va_end"))
            sizes[block] = end - block + 1 if end >= block else 0
    identities = first_blocks & set(evictions) & set(reuse)
    latencies = [(reuse[block] - evictions[block]) / 1000.0 for block in identities
                 if reuse[block] >= evictions[block]]
    available = selected > 0 and all((a_first, b_first, a_reuse))
    with (root / "refault_trace.csv").open("w", newline="") as stream:
        fields = ["va_block", "eviction_timestamp_ns", "refault_timestamp_ns",
                  "eviction_to_refault_us", "block_bytes", "evidence_class"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for block in sorted(identities):
            writer.writerow({
                "va_block": hex(block),
                "eviction_timestamp_ns": evictions[block],
                "refault_timestamp_ns": reuse[block],
                "eviction_to_refault_us": (reuse[block] - evictions[block]) / 1000.0,
                "block_bytes": sizes.get(block, 0),
                "evidence_class": "GPU_EXT_EVICTION_TRACE+GPU_EXT_PREFETCH_DECISION_TRACE",
            })
    return {
        "selected_evictions": selected,
        "same_block_refault_count": len(identities) if available else "UNAVAILABLE",
        "refaulted_bytes": sum(sizes.get(block, 0) for block in identities)
        if available else "UNAVAILABLE",
        "eviction_to_refault_us_mean": statistics.fmean(latencies)
        if latencies else "UNAVAILABLE",
        "identity_available": available,
    }


def collect(results: Path) -> list[dict[str, object]]:
    runs = []
    for manifest_path in sorted(results.rglob("manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not str(manifest.get("evidence_class", "")).startswith("GPU_EXT_STAGE4"):
            continue
        root = manifest_path.parent
        program = read_jsonl(root / "program.jsonl")
        capacity = next((row for row in program if row.get("phase") == "capacity_manifest"), {})
        capacity_model = capacity.get("evidence_class", "UNKNOWN")
        if capacity_model == "REDUCED_EFFECTIVE_GPU_CAPACITY":
            capacity_model = "LEGACY_MATHEMATICAL_HEADROOM_MODEL"
        snapshots = {
            str(row.get("checkpoint")): integer(row.get("gpu_free_bytes"))
            for row in program if row.get("phase") == "gpu_memory_snapshot"
        }
        phases = {
            phase: [float(row["elapsed_ms"]) for row in program if row.get("phase") == phase]
            for phase in (*PHASES, *VECTOR_PHASES)
        }
        actions, final_pages = read_trace(root / "prefetch_decision_trace.csv")
        refault = refault_evidence(program, root)
        runs.append(
            {
                "root": str(root),
                "experiment": manifest.get("experiment"),
                "policy": manifest.get("policy"),
                "ratio": str(manifest.get("ratio")),
                "kind": manifest.get("run_kind"),
                "capacity_model": capacity_model,
                "effective_capacity": capacity.get("effective_gpu_capacity_bytes"),
                "working_set": capacity.get("managed_working_set_bytes"),
                "actual_ratio": capacity.get("actual_working_set_ratio",
                                             capacity.get("working_set_ratio")),
                "main_reserve": capacity.get("main_reserve_allocated_bytes"),
                "guard": capacity.get("guard_allocated_bytes"),
                "gpu_free_initial": capacity.get("gpu_free_initial"),
                "gpu_free_after_main_reserve": capacity.get("gpu_free_after_main_reserve"),
                "gpu_free_after_guard": capacity.get("gpu_free_after_guard"),
                "capacity_target_relative_error": capacity.get("capacity_target_relative_error"),
                "working_set_ratio_error": capacity.get("working_set_ratio_error"),
                "region_a_bytes": capacity.get("region_a_bytes"),
                "region_b_bytes": capacity.get("region_b_bytes"),
                "snapshots": snapshots,
                "correct": bool(manifest.get("correct")),
                "detached": bool(manifest.get("struct_ops_detached")),
                "xid_delta": int(manifest.get("xid_delta", 0)),
                "phases": phases,
                "actions": actions,
                "final_pages": final_pages,
                **refault,
            }
        )
    return runs


def summarize(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for run in runs:
        key = (
            run["capacity_model"], run["experiment"], run["policy"],
            run["ratio"], run["kind"],
        )
        groups[key].append(run)
    output = []
    for key, members in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        row: dict[str, object] = {
            "capacity_model": key[0], "experiment": key[1], "policy": key[2],
            "ratio": key[3], "run_kind": key[4],
            "runs": len(members),
            "correctness_pass_rate": sum(bool(x["correct"]) for x in members) / len(members),
            "all_detached": all(bool(x["detached"]) for x in members),
            "xid_delta": sum(int(x["xid_delta"]) for x in members),
            "selected_eviction_count": sum(int(x["selected_evictions"]) for x in members),
            "evidence_class": "PROGRAM_TIMING|GPU_EXT_PREFETCH_DECISION_TRACE|GPU_EXT_CHUNK_TRACE",
        }
        scalar_fields = {
            "effective_capacity_bytes": "effective_capacity",
            "managed_working_set_bytes": "working_set",
            "actual_working_set_ratio": "actual_ratio",
            "main_reserve_allocated_bytes": "main_reserve",
            "guard_allocated_bytes": "guard",
            "gpu_free_initial": "gpu_free_initial",
            "gpu_free_after_main_reserve": "gpu_free_after_main_reserve",
            "gpu_free_after_guard": "gpu_free_after_guard",
            "capacity_target_relative_error": "capacity_target_relative_error",
            "working_set_ratio_error": "working_set_ratio_error",
            "region_a_bytes": "region_a_bytes",
            "region_b_bytes": "region_b_bytes",
        }
        for output_name, member_name in scalar_fields.items():
            values = [float(member[member_name]) for member in members
                      if member.get(member_name) not in (None, "", "UNAVAILABLE")]
            values_stats = stats(values)
            row[f"{output_name}_mean"] = values_stats["mean"]
            row[f"{output_name}_min"] = values_stats["min"]
            row[f"{output_name}_max"] = values_stats["max"]
        snapshot_names = (
            "after_managed_allocation", "after_cpu_first_touch", "after_phase_A_first",
            "after_phase_B_first", "after_phase_A_reuse", "after_phase_B_reuse",
            "before_cleanup", "after_cleanup",
        )
        for checkpoint in snapshot_names:
            values = [float(member["snapshots"][checkpoint]) for member in members
                      if checkpoint in member["snapshots"]]
            row[f"gpu_free_{checkpoint}_mean"] = stats(values)["mean"]
        for phase in PHASES:
            values = [value for member in members for value in member["phases"][phase]]
            for name, value in stats(values).items():
                row[f"{phase}_{name}"] = value
        page_values = [value for member in members for value in member["final_pages"]]
        page_stats = stats(page_values)
        row["final_pages_mean"] = page_stats["mean"]
        row["final_pages_median"] = page_stats["median"]
        row["final_pages_p95"] = page_stats["p95"]
        actions = Counter()
        for member in members:
            actions.update(member["actions"])
        for action in ("DEFAULT", "BYPASS", "ENTER_LOOP"):
            row[f"action_{action.lower()}_count"] = actions[action]
        available_refaults = [x for x in members if x["identity_available"]]
        if available_refaults:
            row["same_block_refault_count"] = sum(
                int(x["same_block_refault_count"]) for x in available_refaults
            )
            row["refaulted_bytes"] = sum(int(x["refaulted_bytes"]) for x in available_refaults)
            latency = [float(x["eviction_to_refault_us_mean"]) for x in available_refaults
                       if x["eviction_to_refault_us_mean"] != "UNAVAILABLE"]
            row["eviction_to_refault_us_mean"] = statistics.fmean(latency) if latency else "UNAVAILABLE"
        else:
            row["same_block_refault_count"] = "UNAVAILABLE"
            row["refaulted_bytes"] = "UNAVAILABLE"
            row["eviction_to_refault_us_mean"] = "UNAVAILABLE"
        output.append(row)
    return output


def summarize_trace_overhead(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize the independent vector-add runs used by Stage 4F."""
    selected = [run for run in runs if run["experiment"] == "trace_overhead"]
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for run in selected:
        groups[str(run["kind"])].append(run)

    rows: list[dict[str, object]] = []
    means: dict[str, float] = {}
    for kind, members in sorted(groups.items()):
        row: dict[str, object] = {
            "run_kind": kind,
            "runs": len(members),
            "correctness_pass_rate": sum(bool(x["correct"]) for x in members) / len(members),
            "all_detached": all(bool(x["detached"]) for x in members),
            "xid_delta": sum(int(x["xid_delta"]) for x in members),
            "evidence_class": "PROGRAM_TIMING",
        }
        for phase in VECTOR_PHASES:
            values = [value for member in members for value in member["phases"][phase]]
            phase_stats = stats(values)
            for name, value in phase_stats.items():
                row[f"{phase}_{name}"] = value
            if phase == "kernel_1_demand" and phase_stats["mean"] is not None:
                means[kind] = float(phase_stats["mean"])
        rows.append(row)

    if "timing" in means and "trace" in means and means["timing"]:
        overhead = (means["trace"] / means["timing"] - 1.0) * 100.0
        for row in rows:
            row["trace_attached_kernel_1_overhead_percent"] = overhead
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        if rows:
            writer.writerows(rows)
        else:
            writer.writerow({"status": "IMPLEMENTED_NOT_EXECUTED"})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runs = collect(args.results)
    rows = summarize(runs)
    output = args.output or args.results / "stage4_prefetch_summary.csv"
    write_csv(output, rows)
    write_csv(args.results / "stage4_eviction_refault_summary.csv", rows)
    canonical = args.results.parent
    write_csv(canonical / "stage4_summary.csv", rows)
    write_csv(canonical / "stage4_eviction_refault_summary.csv", rows)
    write_csv(canonical / "stage4_prefetch_summary.csv", [
        row for row in rows if row.get("experiment") == "prefetch_matrix_stage4"
    ])
    write_csv(canonical / "stage4_joint_summary.csv", [
        row for row in rows if row.get("experiment") == "joint_stage4"
    ])
    write_csv(canonical / "stage4_natural_confirmation.csv", [
        row for row in rows if row.get("experiment") == "natural_stage4"
    ])
    overhead_output = canonical / "stage4_trace_overhead.csv"
    write_csv(overhead_output, summarize_trace_overhead(runs))
    print(json.dumps({"runs": len(runs), "groups": len(rows), "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
