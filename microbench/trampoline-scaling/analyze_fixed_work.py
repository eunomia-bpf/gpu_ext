#!/usr/bin/env python3
"""Analyze a complete fixed-work block-organization campaign CPU-only."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

import run_fixed_work as profile


runner = profile.runner
LOW_BLOCK_CELL = 0
HIGH_BLOCK_CELL = 4
EQUIVALENCE_MARGIN_PCT = 1.0
BOOTSTRAP_SAMPLES = 10_000


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


def median_interval(values: list[float], seed: int) -> dict[str, float]:
    require(len(values) == runner.FULL_BLOCKS, "primary analysis requires ten pairs")
    require(all(math.isfinite(value) for value in values), "non-finite paired value")
    rng = random.Random(seed)
    resampled = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [values[rng.randrange(len(values))] for _ in values]
        resampled.append(statistics.median(sample))
    return {
        "median": statistics.median(values),
        "ci95_low": percentile(resampled, 0.025),
        "ci95_high": percentile(resampled, 0.975),
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


def validate_result(result: dict[str, Any]) -> dict[tuple[int, str, int], float]:
    phase = runner.phase_parameters("full")
    require(result.get("kind") == runner.EXPERIMENT_KIND, "wrong experiment kind")
    require(result.get("status") == "complete", "campaign is not complete")
    require(result.get("failures") == [], "campaign retains failures")
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
    by_id = {cell["id"]: cell for cell in runner.CELLS}
    for record in records:
        require(isinstance(record, dict) and record.get("valid") is True,
                "invalid arm record")
        arm_key = (record.get("block"), record.get("order"), record.get("arm"))
        require(arm_key in scheduled and arm_key not in seen_arms,
                "missing, duplicate, or unscheduled arm record")
        seen_arms.add(arm_key)
        item = scheduled[arm_key]
        measurements = record.get("measurements")
        require(isinstance(measurements, list), "missing measurements")
        require([value.get("cell") for value in measurements] == item["cell_ids"],
                "measurement order differs from frozen per-block order")
        for measurement in measurements:
            cell_id = measurement["cell"]
            require(cell_id in by_id, "unknown fixed-work cell")
            expected = expected_measurement(by_id[cell_id], phase)
            require(all(measurement.get(key) == value for key, value in expected.items()),
                    f"measurement invariant failed for cell {cell_id}")
            elapsed = measurement.get("elapsed_ms")
            require(isinstance(elapsed, (int, float)) and math.isfinite(elapsed)
                    and elapsed > 0, "invalid CUDA-event time")
            key = (record["block"], record["arm"], cell_id)
            require(key not in lookup, "duplicate timed cell")
            lookup[key] = float(elapsed)
        require(isinstance(record.get("telemetry", {}).get("summary"), dict),
                "missing validated telemetry")
        require(isinstance(record.get("safety_after"), dict),
                "missing post-arm safety record")
        if record["arm"] != "baseline":
            require(record.get("private_segment_removed") is True,
                    "private segment was not removed")
            require(record.get("owned_group_survivors") == {},
                    "owned process survived")
            require(record.get("engagement", {}).get("marker_callbacks") == 32,
                    "marker engagement failed")
            require(record.get("agent_gate", {}).get("routing_order_valid") is True,
                    "target/marker routing gate failed")
            if record["arm"] == "counter":
                require(record.get("engagement", {}).get("target_counter_exact") is True,
                        "counter target oracle failed")
    require(len(seen_arms) == len(scheduled) and len(lookup) == 150,
            "paired measurement matrix is incomplete")
    require(isinstance(result.get("safety_after"), dict), "missing final safety record")
    return lookup


def analyze(result: dict[str, Any]) -> dict[str, Any]:
    lookup = validate_result(result)
    phase = runner.phase_parameters("full")
    endpoint_values = []
    for block in range(runner.FULL_BLOCKS):
        native_low = lookup[(block, "baseline", LOW_BLOCK_CELL)]
        native_high = lookup[(block, "baseline", HIGH_BLOCK_CELL)]
        noop_low = lookup[(block, "noop", LOW_BLOCK_CELL)]
        noop_high = lookup[(block, "noop", HIGH_BLOCK_CELL)]
        denominator = (native_low + native_high) / 2.0
        require(denominator > 0, "non-positive endpoint native time")
        endpoint_values.append(
            100.0 * ((noop_high - native_high) - (noop_low - native_low))
            / denominator
        )
    primary = median_interval(endpoint_values, runner.SEED + 40_000)
    if (primary["ci95_low"] >= -EQUIVALENCE_MARGIN_PCT
            and primary["ci95_high"] <= EQUIVALENCE_MARGIN_PCT):
        hypothesis = "supported_within_predeclared_bound"
    elif (primary["ci95_high"] < -EQUIVALENCE_MARGIN_PCT
          or primary["ci95_low"] > EQUIVALENCE_MARGIN_PCT):
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
                "ci95_low_us": interval["ci95_low"] * 1000.0,
                "ci95_high_us": interval["ci95_high"] * 1000.0,
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
        "primary_metric": {
            "definition": (
                "paired endpoint difference-in-differences, normalized by the "
                "mean endpoint native batch time (percent)"
            ),
            "low_endpoint": "128 blocks x 1024 threads",
            "high_endpoint": "4096 blocks x 32 threads",
            "pairs": runner.FULL_BLOCKS,
            "equivalence_margin_pct": EQUIVALENCE_MARGIN_PCT,
            **primary,
            "paired_values_pct": endpoint_values,
        },
        "cells": cells,
        "claim_boundary": (
            "fixed total work and dynamic warps on this kernel and RTX 5090; "
            "not universal block-count independence or warp-leader execution"
        ),
    }


def render_markdown(analysis: dict[str, Any]) -> str:
    primary = analysis["primary_metric"]
    lines = [
        "# Fixed-work trampoline analysis", "",
        f"- Run status: **{analysis['run_status']}**",
        f"- Tested hypothesis: **{analysis['tested_hypothesis']}**",
        f"- Endpoint effect: **{primary['median']:.4f}%** "
        f"(95% paired-bootstrap interval "
        f"[{primary['ci95_low']:.4f}%, {primary['ci95_high']:.4f}%])",
        f"- Predeclared materiality interval: "
        f"**[-{primary['equivalence_margin_pct']:.1f}%, "
        f"+{primary['equivalence_margin_pct']:.1f}%]**", "",
        "| Blocks | Threads/block | No-op delta (us) | Counter delta (us) |",
        "|---:|---:|---:|---:|",
    ]
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
    analysis = analyze(result)
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
