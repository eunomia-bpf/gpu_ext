#!/usr/bin/env python3
"""Analyze a complete fixed-work block-organization campaign CPU-only."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import stat
import statistics
from pathlib import Path
from typing import Any

import run_fixed_work as profile


runner = profile.runner
LOW_BLOCK_CELL = 0
HIGH_BLOCK_CELL = 4
EQUIVALENCE_MARGIN_PCT = 1.0
BOOTSTRAP_SAMPLES = 10_000
ALL_FIVE_FAMILYWISE_CONFIDENCE = 0.95
ALL_FIVE_CONTRAST_COUNT = 4
ALL_FIVE_CONTRAST_CONFIDENCE = (
    1.0 - (1.0 - ALL_FIVE_FAMILYWISE_CONFIDENCE) / ALL_FIVE_CONTRAST_COUNT
)


class AnalysisError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def median_interval(
    values: list[float], seed: int, confidence: float = 0.95,
) -> dict[str, float]:
    require(len(values) == runner.FULL_BLOCKS, "primary analysis requires ten pairs")
    require(all(math.isfinite(value) for value in values), "non-finite paired value")
    require(0.0 < confidence < 1.0, "invalid confidence level")
    rng = random.Random(seed)
    resampled = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [values[rng.randrange(len(values))] for _ in values]
        resampled.append(statistics.median(sample))
    tail = (1.0 - confidence) / 2.0
    return {
        "median": statistics.median(values),
        "ci_low": percentile(resampled, tail),
        "ci_high": percentile(resampled, 1.0 - tail),
        "confidence": confidence,
    }


def expected_measurement(cell: dict[str, int], phase: dict[str, Any]) -> dict[str, int]:
    return {
        "cell": cell["id"],
        "blocks": cell["blocks"],
        "threads_per_block": cell["threads_per_block"],
        "launched_threads": cell["blocks"] * cell["threads_per_block"],
        "active_threads": cell["active_threads"],
        "active_warps": cell["active_threads"] // 32,
        "counter_key": cell["counter_key"],
        "warmup": phase["warmup"],
        "launches": phase["launches"],
        "hook_repeats": phase["hook_repeats"],
        "checked_values": runner.MAX_THREADS,
        "mismatches": 0,
    }


def read_regular_text(path: Path, label: str) -> str:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise AnalysisError(f"missing raw {label}: {path}") from error
    require(stat.S_ISREG(info.st_mode), f"raw {label} is not a regular file: {path}")
    try:
        return path.read_text(errors="replace")
    except OSError as error:
        raise AnalysisError(f"cannot read raw {label}: {path}: {error}") from error


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    text = read_regular_text(path, label)
    try:
        value = json.loads(text)
    except ValueError as error:
        raise AnalysisError(f"malformed raw {label}: {path}: {error}") from error
    require(isinstance(value, dict), f"raw {label} is not a JSON object: {path}")
    return value


def run_gate(label: str, function: Any, *arguments: Any, **keywords: Any) -> Any:
    try:
        return function(*arguments, **keywords)
    except AnalysisError:
        raise
    except Exception as error:
        raise AnalysisError(f"raw {label} gate failed: {type(error).__name__}: {error}") from error


def validate_safety_pair(
    before: dict[str, Any], after: dict[str, Any], label: str,
) -> None:
    run_gate(f"{label} pre-safety", runner.safety.validate_pre_server_safety, before)
    run_gate(f"{label} post-safety", runner.safety.validate_post_server_safety, before, after)
    for name, snapshot in (("before", before), ("after", after)):
        timestamp = snapshot.get("timestamp_ns")
        require(type(timestamp) is int and timestamp > 0,
                f"raw {label} {name} safety timestamp is invalid")
        gpu = snapshot.get("gpu")
        require(isinstance(gpu, dict), f"raw {label} {name} safety GPU record is missing")
        require(
            gpu.get("driver") == runner.EXPECTED_DRIVER
            and gpu.get("name") == runner.EXPECTED_GPU,
            f"raw {label} {name} safety GPU/driver mismatch",
        )
    require(after["timestamp_ns"] > before["timestamp_ns"],
            f"raw {label} safety timestamps are not ordered")


def resolve_run_directory(
    output: Path, record: dict[str, Any], item: dict[str, Any],
) -> Path:
    stored = record.get("directory")
    require(isinstance(stored, str) and stored, "arm record has no raw directory locator")
    name = Path(stored).name
    stem = f"block-{item['block'] + 1:02d}-order-{item['order'] + 1}-{item['arm']}"
    require(
        re.fullmatch(re.escape(stem) + r"(?:-attempt-[2-9][0-9]*)?", name) is not None,
        f"raw directory name does not match scheduled arm: {name}",
    )
    run_dir = output / name
    try:
        info = run_dir.lstat()
    except FileNotFoundError as error:
        raise AnalysisError(f"missing raw arm directory: {run_dir}") from error
    require(stat.S_ISDIR(info.st_mode), f"raw arm path is not a directory: {run_dir}")
    require(run_dir.resolve().parent == output.resolve(),
            f"raw arm directory escaped campaign root: {run_dir}")
    return run_dir


def resolve_raw_reference(
    output: Path, run_dir: Path, value: Any, label: str,
) -> Path:
    require(isinstance(value, str) and value, f"missing raw {label} reference")
    relative = Path(value)
    require(not relative.is_absolute(), f"raw {label} reference must be relative")
    candidate = run_dir / relative
    try:
        info = candidate.lstat()
    except FileNotFoundError as error:
        raise AnalysisError(f"missing raw {label}: {candidate}") from error
    require(stat.S_ISREG(info.st_mode), f"raw {label} is not a regular file: {candidate}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(output.resolve())
    except ValueError as error:
        raise AnalysisError(f"raw {label} reference escaped campaign root") from error
    read_regular_text(resolved, label)
    return resolved


def replay_arm_evidence(
    output: Path, run_dir: Path, item: dict[str, Any], phase: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lifecycle = read_json_object(run_dir / "lifecycle.json", "lifecycle")
    cell_ids = tuple(item.get("cell_ids", phase["cell_ids"]))
    exact_identity = {
        "schema": runner.RAW_EVIDENCE_SCHEMA,
        "experiment_kind": runner.EXPERIMENT_KIND,
        "block": item["block"],
        "order": item["order"],
        "arm": item["arm"],
        "run_id": item["run_id"],
        "cell_ids": list(cell_ids),
        "application_command": runner.application_command(
            cell_ids, phase["warmup"], phase["launches"],
            phase["hook_repeats"], item["run_id"],
        ),
        "application_returncode": 0,
        "application_log": "application.log",
        "safety_before": "safety-before.json",
        "safety_after": "safety-after.json",
        "owned_group_survivors": {},
    }
    require(
        all(lifecycle.get(key) == value for key, value in exact_identity.items()),
        f"raw lifecycle identity/exit/cleanup mismatch for {run_dir.name}",
    )

    application_path = resolve_raw_reference(
        output, run_dir, lifecycle.get("application_log"), "application log",
    )
    application_text = read_regular_text(application_path, "application log")
    measurements = run_gate(
        "application", runner.validate_application_events,
        runner.json_events(application_path), cell_ids, phase["warmup"],
        phase["launches"], phase["hook_repeats"], item["run_id"],
    )

    before_path = resolve_raw_reference(
        output, run_dir, lifecycle.get("safety_before"), "pre-arm safety",
    )
    after_path = resolve_raw_reference(
        output, run_dir, lifecycle.get("safety_after"), "post-arm safety",
    )
    before = read_json_object(before_path, "pre-arm safety")
    after = read_json_object(after_path, "post-arm safety")
    validate_safety_pair(before, after, run_dir.name)

    telemetry_path = resolve_raw_reference(
        output, run_dir, lifecycle.get("telemetry_log"), "telemetry",
    )
    run_gate(
        "telemetry", runner.safety.validate_gpu_telemetry, telemetry_path,
        allow_fixed_power_cap=True,
    )

    attached = item["arm"] != "baseline"
    segment = lifecycle.get("private_segment")
    if not attached:
        require(
            lifecycle.get("loader_command") is None
            and lifecycle.get("loader_returncode") is None
            and lifecycle.get("loader_log") is None
            and lifecycle.get("agent_log") is None
            and segment is None
            and lifecycle.get("private_segment_removed") is None,
            f"native arm contains attached-runtime lifecycle evidence: {run_dir.name}",
        )
        require(not (run_dir / "loader.log").exists() and not (run_dir / "agent.log").exists(),
                f"native arm unexpectedly contains loader/agent logs: {run_dir.name}")
    else:
        object_path = runner.HERE / ".output" / f"{runner.BPF_OBJECT_PREFIX}-{item['arm']}.bpf.o"
        expected_loader = [
            str(runner.LOADER_BINARY), str(object_path), item["arm"],
            str(runner.MAX_THREADS), "300",
        ]
        require(
            lifecycle.get("loader_command") == expected_loader
            and lifecycle.get("loader_returncode") == 0
            and lifecycle.get("loader_log") == "loader.log"
            and lifecycle.get("agent_log") == "agent.log"
            and lifecycle.get("private_segment_removed") is True,
            f"attached lifecycle/exit/cleanup mismatch for {run_dir.name}",
        )
        require(
            isinstance(segment, str)
            and re.fullmatch(r"trampoline_scaling_[0-9]+_[0-9]+", segment) is not None,
            f"unsafe or missing private segment name for {run_dir.name}",
        )
        require(not os.path.lexists(runner.SHM_ROOT / segment),
                f"private shared-memory segment still exists: {segment}")
        loader_path = resolve_raw_reference(
            output, run_dir, lifecycle.get("loader_log"), "loader/map log",
        )
        run_gate(
            "loader/map engagement", runner.validate_loader_events,
            runner.json_events(loader_path), item["arm"], cell_ids,
            phase["warmup"], phase["launches"], phase["hook_repeats"],
        )
        run_gate("target/marker transform", runner.validate_agent_log, application_text)
        agent_path = resolve_raw_reference(
            output, run_dir, lifecycle.get("agent_log"), "agent bootstrap log",
        )
        run_gate(
            "agent bootstrap", runner.validate_agent_bootstrap_log,
            read_regular_text(agent_path, "agent bootstrap log"), segment,
        )

    return measurements, {
        "run_directory": run_dir,
        "telemetry_path": telemetry_path,
        "private_segment": segment,
        "attached": attached,
    }


def validate_result(
    result: dict[str, Any], result_path: Path,
) -> tuple[dict[tuple[int, str, int], float], dict[str, int]]:
    phase = runner.phase_parameters("full")
    require(result.get("kind") == runner.EXPERIMENT_KIND, "wrong experiment kind")
    require(result.get("status") == "complete", "campaign is not complete")
    params = result.get("params")
    require(isinstance(params, dict), "missing parameters")
    expected_params = {
        "kind": runner.EXPERIMENT_KIND,
        "phase": "full",
        "blocks": phase["blocks"],
        "cell_ids": list(phase["cell_ids"]),
        "warmup": phase["warmup"],
        "launches": phase["launches"],
        "hook_repeats": phase["hook_repeats"],
        "schedule_seed": runner.SEED,
        "expected_driver": runner.EXPECTED_DRIVER,
        "expected_gpu": runner.EXPECTED_GPU,
        "matrix": [dict(cell) for cell in runner.CELLS],
        "randomize_cell_order": True,
        "balance_arm_order": True,
        "independent_raw_evidence": True,
    }
    require(all(params.get(key) == value for key, value in expected_params.items()),
            "frozen parameter/profile mismatch")

    totals = {
        (cell["blocks"] * cell["threads_per_block"], cell["active_threads"],
         cell["active_threads"] // 32)
        for cell in runner.CELLS
    }
    require(totals == {(131_072, 131_072, 4096)},
            "matrix does not hold launched work and dynamic warps fixed")
    require(all(cell["threads_per_block"] % 32 == 0 for cell in runner.CELLS),
            "matrix contains a partial-warp block")

    schedule = runner.frozen_schedule("full")
    require(result.get("schedule") == schedule, "schedule or cell order mismatch")
    records = result.get("records")
    require(isinstance(records, list) and len(records) == 30,
            "full campaign must contain 30 arm records")
    scheduled = {(item["block"], item["order"], item["arm"]): item for item in schedule}
    lookup: dict[tuple[int, str, int], float] = {}
    seen_arms: set[tuple[int, int, str]] = set()
    seen_directories: set[Path] = set()
    seen_telemetry: set[Path] = set()
    seen_segments: set[str] = set()
    by_id = {cell["id"]: cell for cell in runner.CELLS}
    output = result_path.resolve().parent
    campaign_before = read_json_object(output / "safety-before.json", "campaign pre-safety")
    campaign_final = read_json_object(output / "safety-final.json", "campaign final safety")
    validate_safety_pair(campaign_before, campaign_final, "campaign")
    for record in records:
        require(isinstance(record, dict), "arm locator is not a JSON object")
        arm_key = (record.get("block"), record.get("order"), record.get("arm"))
        require(arm_key in scheduled and arm_key not in seen_arms,
                "missing, duplicate, or unscheduled arm record")
        seen_arms.add(arm_key)
        item = scheduled[arm_key]
        run_dir = resolve_run_directory(output, record, item)
        require(run_dir not in seen_directories, "raw arm directory was reused")
        seen_directories.add(run_dir)
        measurements, evidence = replay_arm_evidence(output, run_dir, item, phase)
        require(evidence["telemetry_path"] not in seen_telemetry,
                "raw telemetry file was reused")
        seen_telemetry.add(evidence["telemetry_path"])
        if evidence["private_segment"] is not None:
            require(evidence["private_segment"] not in seen_segments,
                    "private shared-memory segment name was reused")
            seen_segments.add(evidence["private_segment"])
        require([value.get("cell") for value in measurements] == item["cell_ids"],
                "raw measurement order differs from frozen per-block order")
        for measurement in measurements:
            cell_id = measurement["cell"]
            require(cell_id in by_id, "unknown fixed-work cell")
            expected = expected_measurement(by_id[cell_id], phase)
            require(all(measurement.get(key) == value for key, value in expected.items()),
                    f"raw measurement invariant failed for cell {cell_id}")
            elapsed = measurement.get("elapsed_ms")
            require(isinstance(elapsed, (int, float)) and math.isfinite(elapsed)
                    and elapsed > 0, "invalid CUDA-event time")
            key = (record["block"], record["arm"], cell_id)
            require(key not in lookup, "duplicate timed cell")
            lookup[key] = float(elapsed)
    require(len(seen_arms) == len(scheduled) and len(lookup) == 150,
            "paired measurement matrix is incomplete")
    return lookup, {
        "timed_cells": len(lookup),
        "arm_directories": len(seen_directories),
        "application_logs": len(seen_directories),
        "loader_map_logs": len(seen_segments),
        "agent_bootstrap_logs": len(seen_segments),
        "telemetry_logs": len(seen_telemetry),
        "arm_safety_pairs": len(seen_directories),
        "lifecycle_records": len(seen_directories),
        "unique_private_segments": len(seen_segments),
    }


def paired_organization_values(
    lookup: dict[tuple[int, str, int], float], cell_id: int,
) -> list[float]:
    values = []
    for block in range(runner.FULL_BLOCKS):
        native_reference = lookup[(block, "baseline", LOW_BLOCK_CELL)]
        native_cell = lookup[(block, "baseline", cell_id)]
        noop_reference = lookup[(block, "noop", LOW_BLOCK_CELL)]
        noop_cell = lookup[(block, "noop", cell_id)]
        denominator = (native_reference + native_cell) / 2.0
        require(denominator > 0, "non-positive organization native time")
        values.append(
            100.0
            * ((noop_cell - native_cell) - (noop_reference - native_reference))
            / denominator
        )
    return values


def bounded_status(interval: dict[str, float]) -> str:
    if (
        interval["ci_low"] >= -EQUIVALENCE_MARGIN_PCT
        and interval["ci_high"] <= EQUIVALENCE_MARGIN_PCT
    ):
        return "supported_within_predeclared_bound"
    if (
        interval["ci_high"] < -EQUIVALENCE_MARGIN_PCT
        or interval["ci_low"] > EQUIVALENCE_MARGIN_PCT
    ):
        return "contradicted"
    return "inconclusive"


def analyze(result: dict[str, Any], result_path: Path) -> dict[str, Any]:
    lookup, evidence_audit = validate_result(result, result_path)
    phase = runner.phase_parameters("full")
    endpoint_values = paired_organization_values(lookup, HIGH_BLOCK_CELL)
    primary = median_interval(endpoint_values, runner.SEED + 40_000)
    primary_status = bounded_status(primary)

    guard_contrasts = []
    for cell_id in range(LOW_BLOCK_CELL + 1, HIGH_BLOCK_CELL + 1):
        values = paired_organization_values(lookup, cell_id)
        interval = median_interval(
            values, runner.SEED + 50_000 + cell_id,
            confidence=ALL_FIVE_CONTRAST_CONFIDENCE,
        )
        guard_contrasts.append({
            "cell": cell_id,
            "blocks": runner.CELLS[cell_id]["blocks"],
            "threads_per_block": runner.CELLS[cell_id]["threads_per_block"],
            "status": bounded_status(interval),
            **interval,
            "paired_values_pct": values,
        })
    guard_statuses = {item["status"] for item in guard_contrasts}
    if guard_statuses == {"supported_within_predeclared_bound"}:
        guard_status = "supported_within_predeclared_bound"
    elif "contradicted" in guard_statuses:
        guard_status = "contradicted"
    else:
        guard_status = "inconclusive"

    if (
        primary_status == "supported_within_predeclared_bound"
        and guard_status == "supported_within_predeclared_bound"
    ):
        hypothesis = "supported_within_predeclared_bound"
    elif "contradicted" in (primary_status, guard_status):
        hypothesis = "contradicted"
    else:
        hypothesis = "inconclusive"

    cells = []
    dynamic_warp_calls = (
        phase["launches"] * phase["hook_repeats"] * 4096
    )
    for cell in runner.CELLS:
        row: dict[str, Any] = {
            "cell": cell["id"], "blocks": cell["blocks"],
            "threads_per_block": cell["threads_per_block"],
            "active_warps": 4096,
        }
        for offset, arm in enumerate(("noop", "counter")):
            deltas_ms = [
                lookup[(block, arm, cell["id"])]
                - lookup[(block, "baseline", cell["id"])]
                for block in range(runner.FULL_BLOCKS)
            ]
            interval = median_interval(deltas_ms, runner.SEED + cell["id"] * 10 + offset)
            row[arm] = {
                "median_delta_us": interval["median"] * 1000.0,
                "ci95_low_us": interval["ci_low"] * 1000.0,
                "ci95_high_us": interval["ci_high"] * 1000.0,
                "median_ns_per_dynamic_warp_call": (
                    interval["median"] * 1_000_000.0 / dynamic_warp_calls
                ),
            }
        cells.append(row)

    return {
        "kind": runner.EXPERIMENT_KIND,
        "run_status": "valid",
        "tested_hypothesis": hypothesis,
        "research_value": "supporting",
        "raw_evidence_audit": evidence_audit,
        "primary_metric": {
            "definition": (
                "paired endpoint difference-in-differences, normalized by the "
                "mean endpoint native batch time (percent)"
            ),
            "low_endpoint": "128 blocks x 1024 threads",
            "high_endpoint": "4096 blocks x 32 threads",
            "pairs": runner.FULL_BLOCKS,
            "equivalence_margin_pct": EQUIVALENCE_MARGIN_PCT,
            "status": primary_status,
            "median": primary["median"],
            "ci95_low": primary["ci_low"],
            "ci95_high": primary["ci_high"],
            "paired_values_pct": endpoint_values,
        },
        "organization_guard": {
            "definition": (
                "four paired no-op-minus-native organization contrasts versus "
                "128x1024, normalized by each contrast's mean native time"
            ),
            "status": guard_status,
            "reference_cell": LOW_BLOCK_CELL,
            "familywise_confidence": ALL_FIVE_FAMILYWISE_CONFIDENCE,
            "per_contrast_confidence": ALL_FIVE_CONTRAST_CONFIDENCE,
            "multiplicity_control": "Bonferroni over four predeclared contrasts",
            "equivalence_margin_pct": EQUIVALENCE_MARGIN_PCT,
            "contrasts": guard_contrasts,
        },
        "cells": cells,
        "claim_boundary": (
            "fixed total work and dynamic warps on this kernel and RTX 5090; "
            "not universal block-count independence or warp-leader execution"
        ),
    }


def render_markdown(analysis: dict[str, Any]) -> str:
    primary = analysis["primary_metric"]
    guard = analysis["organization_guard"]
    audit = analysis["raw_evidence_audit"]
    lines = [
        "# Fixed-work trampoline analysis", "",
        f"- Run status: **{analysis['run_status']}**",
        f"- Tested hypothesis: **{analysis['tested_hypothesis']}**",
        f"- Endpoint decision: **{primary['status']}**",
        f"- Endpoint effect: **{primary['median']:.4f}%** "
        f"(95% paired-bootstrap interval "
        f"[{primary['ci95_low']:.4f}%, {primary['ci95_high']:.4f}%])",
        f"- All-five organization guard: **{guard['status']}** "
        f"(four Bonferroni-adjusted "
        f"{100.0 * guard['per_contrast_confidence']:.2f}% intervals)",
        f"- Predeclared materiality interval: "
        f"**[-{primary['equivalence_margin_pct']:.1f}%, "
        f"+{primary['equivalence_margin_pct']:.1f}%]**",
        f"- Raw evidence replayed: **{audit['timed_cells']} cells in "
        f"{audit['arm_directories']} distinct arms**", "",
        "| Blocks | Threads/block | Contrast vs. 128x1,024 | "
        f"{100.0 * guard['per_contrast_confidence']:.2f}% interval | Decision |",
        "|---:|---:|---:|---:|---|",
    ]
    for contrast in guard["contrasts"]:
        lines.append(
            f"| {contrast['blocks']} | {contrast['threads_per_block']} | "
            f"{contrast['median']:.4f}% | "
            f"[{contrast['ci_low']:.4f}%, {contrast['ci_high']:.4f}%] | "
            f"{contrast['status']} |"
        )
    lines.extend([
        "", "| Blocks | Threads/block | No-op delta (us) | Counter delta (us) |",
        "|---:|---:|---:|---:|",
    ])
    for row in analysis["cells"]:
        lines.append(
            f"| {row['blocks']} | {row['threads_per_block']} | "
            f"{row['noop']['median_delta_us']:.4f} | "
            f"{row['counter']['median_delta_us']:.4f} |"
        )
    lines.extend(["", f"Claim boundary: {analysis['claim_boundary']}.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    analysis = analyze(result, args.result)
    prefix = args.output_prefix or args.result.parent / "fixed-work-analysis"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n"
    )
    prefix.with_suffix(".md").write_text(render_markdown(analysis))
    print(json.dumps({"status": analysis["run_status"], "output": str(prefix)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
