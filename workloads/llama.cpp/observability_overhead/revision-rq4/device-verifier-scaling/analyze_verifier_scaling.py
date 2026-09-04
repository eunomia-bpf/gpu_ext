#!/usr/bin/env python3
"""Independent replay and statistics for device-verifier scaling."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Iterable


RESULT_SCHEMA = "device-verifier-scaling-run-v1"
PROBE_SCHEMA = "device-verifier-scaling-probe-v1"
SEED = 1797
CPU = 23
BLOCKS = 20
TIMEOUT_SECONDS = 60
BOOTSTRAP_SAMPLES = 20_000
SIZES = (16, 64, 256, 1024, 4096)
FAMILIES = ("linear", "diamonds")
ARMS = tuple((family, size) for size in SIZES for family in FAMILIES)
SECTION = "cuda__verifier_scaling"


def exact_json_equal(actual: Any, expected: Any) -> bool:
    """Compare JSON-like values without Python's bool/int equivalence."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            exact_json_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            exact_json_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _next_prng(state: int) -> int:
    return (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)


def full_schedule() -> list[dict[str, Any]]:
    state = SEED
    result: list[dict[str, Any]] = []
    sequence = 0
    for block in range(1, BLOCKS + 1):
        ordered = list(ARMS)
        for index in range(len(ordered) - 1, 0, -1):
            state = _next_prng(state)
            swap = state % (index + 1)
            ordered[index], ordered[swap] = ordered[swap], ordered[index]
        for position, (family, size) in enumerate(ordered, start=1):
            sequence += 1
            result.append(
                {
                    "sequence": sequence,
                    "block": block,
                    "position": position,
                    "family": family,
                    "instructions": size,
                }
            )
    return result


def preflight_schedule() -> list[dict[str, Any]]:
    return [
        {"sequence": 1, "block": 1, "position": 1, "family": "linear", "instructions": 16},
        {"sequence": 2, "block": 1, "position": 2, "family": "diamonds", "instructions": 4096},
    ]


def load_json(path: Path) -> Any:
    with path.open() as stream:
        return json.load(stream)


def load_stdout(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"{path}: expected one non-empty stdout line")
    value = json.loads(lines[0])
    if not isinstance(value, dict):
        raise ValueError(f"{path}: stdout JSON is not an object")
    return value


def expected_shape(family: str, size: int, mode: str) -> dict[str, Any]:
    branches = (size - 4) // 2 if family == "diamonds" else 0
    return {
        "schema": PROBE_SCHEMA,
        "build_type": "Release",
        "mode": mode,
        "family": family,
        "requested_instructions": size,
        "instruction_count": size,
        "conditional_branches": branches,
        "helper_calls": 1,
        "exits": 1,
        "minimum_branch_offset": 1 if family == "diamonds" else None,
        "maximum_branch_offset": 1 if family == "diamonds" else None,
        "section": SECTION,
    }


def expected_argv(probe: str, family: str, size: int, mode: str) -> list[str]:
    if mode == "describe":
        return [probe, "--describe", "--family", family, "--instructions", str(size)]
    prefix = [probe]
    if mode == "accept_only":
        prefix.append("--accept-only")
    return prefix + [
        "--family",
        family,
        "--instructions",
        str(size),
        "--require-cpu",
        str(CPU),
    ]


def validate_record(
    record: dict[str, Any], family: str, size: int, mode: str, revision: str
) -> list[str]:
    errors: list[str] = []
    for key, expected in expected_shape(family, size, mode).items():
        if not exact_json_equal(record.get(key), expected):
            errors.append(f"{family}/{size}/{mode}: {key} mismatch")
    if record.get("bpftime_source_revision") != revision:
        errors.append(f"{family}/{size}/{mode}: source revision mismatch")
    timed_keys = (
        "elapsed_ns",
        "process_cpu_ns",
        "cpu_before",
        "cpu_after",
        "minor_faults",
        "major_faults",
        "voluntary_context_switches",
        "involuntary_context_switches",
    )
    if mode == "describe":
        if record.get("accepted") is not None or record.get("error") != "":
            errors.append(f"{family}/{size}/describe: unexpected decision")
        if any(record.get(key) is not None for key in timed_keys):
            errors.append(f"{family}/{size}/describe: unexpected timing")
        return errors
    if record.get("accepted") is not True or record.get("error") != "":
        errors.append(f"{family}/{size}/{mode}: program was not accepted")
    if mode == "accept_only":
        if any(record.get(key) is not None for key in timed_keys):
            errors.append(f"{family}/{size}/accept_only: unexpected timing")
        return errors
    for key in timed_keys:
        if type(record.get(key)) is not int or record[key] < 0:
            errors.append(f"{family}/{size}/timed: invalid {key}")
    if type(record.get("elapsed_ns")) is int and record["elapsed_ns"] <= 0:
        errors.append(f"{family}/{size}/timed: non-positive elapsed_ns")
    if (
        type(record.get("process_cpu_ns")) is int
        and record["process_cpu_ns"] <= 0
    ):
        errors.append(f"{family}/{size}/timed: non-positive process_cpu_ns")
    if record.get("cpu_before") != CPU or record.get("cpu_after") != CPU:
        errors.append(f"{family}/{size}/timed: CPU mismatch")
    return errors


def validate_raw_call(
    run_dir: Path,
    relative: str,
    probe: str,
    family: str,
    size: int,
    mode: str,
    revision: str,
) -> tuple[list[str], dict[str, Any] | None]:
    errors: list[str] = []
    directory = run_dir / relative
    try:
        execution = load_json(directory / "execution.json")
        record = load_stdout(directory / "stdout.log")
        stderr = (directory / "stderr.log").read_text()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return [f"{relative}: raw call unreadable: {error}"], None
    if not isinstance(execution, dict):
        return [f"{relative}: execution JSON is not an object"], None
    if not exact_json_equal(
        execution.get("argv"), expected_argv(probe, family, size, mode)
    ):
        errors.append(f"{relative}: argv mismatch")
    if execution.get("cwd") != str(directory):
        errors.append(f"{relative}: cwd mismatch")
    if not exact_json_equal(execution.get("timeout_seconds"), TIMEOUT_SECONDS):
        errors.append(f"{relative}: timeout mismatch")
    if execution.get("timed_out") is not False or not exact_json_equal(
        execution.get("returncode"), 0
    ):
        errors.append(f"{relative}: execution did not complete successfully")
    if type(execution.get("duration_ns")) is not int or execution["duration_ns"] <= 0:
        errors.append(f"{relative}: invalid process duration")
    if not exact_json_equal(
        execution.get("environment"),
        {"CUDA_VISIBLE_DEVICES": "", "LD_PRELOAD": None},
    ):
        errors.append(f"{relative}: execution environment mismatch")
    if stderr != "":
        errors.append(f"{relative}: stderr is not empty")
    errors.extend(validate_record(record, family, size, mode, revision))
    if (
        mode == "timed"
        and type(record.get("elapsed_ns")) is int
        and type(execution.get("duration_ns")) is int
        and execution["duration_ns"] < record["elapsed_ns"]
    ):
        errors.append(f"{relative}: process duration is shorter than API interval")
    return errors, record


def quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty quantile")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def theil_sen(sizes: Iterable[int], latencies: Iterable[float]) -> float:
    points = list(zip(sizes, latencies, strict=True))
    slopes: list[float] = []
    for left in range(len(points)):
        for right in range(left + 1, len(points)):
            x1, y1 = points[left]
            x2, y2 = points[right]
            slopes.append(math.log(y2 / y1) / math.log(x2 / x1))
    return float(statistics.median(slopes))


def summarize(cells: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[tuple[str, int], dict[int, int]] = {}
    records_by_arm: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for cell in cells:
        key = (cell["family"], cell["instructions"])
        by_arm.setdefault(key, {})[cell["block"]] = cell["record"]["elapsed_ns"]
        records_by_arm.setdefault(key, []).append(cell["record"])

    rng = random.Random(SEED)
    resamples = [
        [rng.randrange(1, BLOCKS + 1) for _ in range(BLOCKS)]
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    arms: dict[str, Any] = {}
    for family, size in ARMS:
        values_by_block = by_arm[(family, size)]
        values = [values_by_block[block] for block in range(1, BLOCKS + 1)]
        boot = [
            statistics.median(values_by_block[block] for block in sample)
            for sample in resamples
        ]
        records = records_by_arm[(family, size)]
        arms[f"{family}-{size}"] = {
            "samples": len(values),
            "median_ns": statistics.median(values),
            "min_ns": min(values),
            "max_ns": max(values),
            "median_95_ci_ns": [quantile(boot, 0.025), quantile(boot, 0.975)],
            "median_process_cpu_ns": statistics.median(
                record["process_cpu_ns"] for record in records
            ),
            "major_faults": sum(record["major_faults"] for record in records),
            "involuntary_context_switches": sum(
                record["involuntary_context_switches"] for record in records
            ),
        }

    ratios: dict[str, Any] = {}
    for size in SIZES:
        values = [
            by_arm[("diamonds", size)][block] / by_arm[("linear", size)][block]
            for block in range(1, BLOCKS + 1)
        ]
        boot = [statistics.median(values[block - 1] for block in sample) for sample in resamples]
        ratios[str(size)] = {
            "paired_blocks": BLOCKS,
            "median_ratio": statistics.median(values),
            "median_95_ci": [quantile(boot, 0.025), quantile(boot, 0.975)],
        }

    exponents: dict[str, Any] = {}
    for family in FAMILIES:
        medians = [statistics.median(by_arm[(family, size)].values()) for size in SIZES]
        point = theil_sen(SIZES, medians)
        boot: list[float] = []
        for sample in resamples:
            sampled_medians = [
                statistics.median(by_arm[(family, size)][block] for block in sample)
                for size in SIZES
            ]
            boot.append(theil_sen(SIZES, sampled_medians))
        exponents[family] = {
            "theil_sen": point,
            "bootstrap_95_ci": [quantile(boot, 0.025), quantile(boot, 0.975)],
        }

    noisy = sum(
        1
        for cell in cells
        if cell["record"]["elapsed_ns"] > 1.25 * cell["record"]["process_cpu_ns"]
    )
    major_faults = sum(cell["record"]["major_faults"] for cell in cells)
    noise_veto = noisy > len(cells) * 0.10 or major_faults > 0
    if noise_veto:
        hypothesis = "inconclusive"
    elif all(exponents[family]["bootstrap_95_ci"][1] <= 1.25 for family in FAMILIES):
        hypothesis = "supported"
    elif any(exponents[family]["bootstrap_95_ci"][0] > 1.25 for family in FAMILIES):
        hypothesis = "contradicted"
    else:
        hypothesis = "inconclusive"
    return {
        "arms": arms,
        "diamonds_over_linear": ratios,
        "scaling_exponents": exponents,
        "diagnostics": {
            "wall_over_cpu_gt_1_25_cells": noisy,
            "wall_over_cpu_gt_1_25_fraction": noisy / len(cells),
            "major_faults": major_faults,
            "noise_veto": noise_veto,
        },
        "tested_hypothesis": hypothesis,
    }


def analyze(run_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    try:
        result = load_json(run_dir / "result.json")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {"complete": False, "errors": [f"result unreadable: {error}"]}
    if not isinstance(result, dict):
        return {"complete": False, "errors": ["result JSON is not an object"]}
    if result.get("schema") != RESULT_SCHEMA:
        errors.append("result schema mismatch")
    if result.get("status") != "complete":
        errors.append("run status is not complete")
    mode = result.get("mode")
    if mode not in ("preflight", "full"):
        errors.append("run mode is invalid")
        mode = "full"
    if not exact_json_equal(result.get("seed"), SEED) or not exact_json_equal(
        result.get("cpu"), CPU
    ):
        errors.append("seed or CPU differs from frozen plan")
    if not exact_json_equal(result.get("sizes"), list(SIZES)) or not exact_json_equal(
        result.get("families"), list(FAMILIES)
    ):
        errors.append("family/size set differs from frozen plan")
    if not exact_json_equal(result.get("timeout_seconds"), TIMEOUT_SECONDS):
        errors.append("timeout differs from frozen plan")
    expected_blocks = 1 if mode == "preflight" else BLOCKS
    if not exact_json_equal(result.get("blocks"), expected_blocks):
        errors.append("block count differs from frozen plan")
    if result.get("error") is not None:
        errors.append("completed run retains an error")

    environment = result.get("environment")
    if not isinstance(environment, dict):
        errors.append("missing start environment")
        environment = {}
    if not exact_json_equal(environment.get("runner_affinity"), [CPU]):
        errors.append("runner affinity mismatch")
    if environment.get("cuda_visible_devices") != "" or environment.get("ld_preload") is not None:
        errors.append("runner environment was not isolated")
    if environment.get("bpftime_verifier_status") != []:
        errors.append("bpftime verifier source was dirty")
    probe_metadata = environment.get("probe")
    if not isinstance(probe_metadata, dict):
        errors.append("probe metadata is not an object")
        probe_metadata = {}
    probe = probe_metadata.get("path")
    if not isinstance(probe, str) or not probe:
        errors.append("missing probe path")
        probe = ""
    for key in ("size", "mtime_ns"):
        if type(probe_metadata.get(key)) is not int or probe_metadata[key] <= 0:
            errors.append(f"invalid probe {key}")
    if probe_metadata.get("cmake_build_type") != "Release":
        errors.append("probe CMake build type mismatch")
    bpftime_root = environment.get("bpftime_root")
    if not isinstance(bpftime_root, str) or not bpftime_root:
        errors.append("missing bpftime source root")
    if probe_metadata.get("cmake_bpftime_root") != bpftime_root:
        errors.append("probe CMake/source root mismatch")
    revision = result.get("probe_source_revision")
    if not isinstance(revision, str) or not revision:
        errors.append("missing probe source revision")
        revision = ""
    if environment.get("bpftime_current_revision") != revision:
        errors.append("probe/current bpftime revision mismatch")

    end_environment = result.get("end_environment")
    if not isinstance(end_environment, dict):
        errors.append("missing end environment")
        end_environment = {}
    if not exact_json_equal(end_environment.get("runner_affinity"), [CPU]):
        errors.append("end affinity mismatch")
    if not exact_json_equal(environment.get("cpufreq"), end_environment.get("cpufreq")):
        errors.append("cpufreq policy changed during run")
    if end_environment.get("bpftime_current_revision") != revision:
        errors.append("bpftime revision changed during run")
    if end_environment.get("bpftime_verifier_status") != []:
        errors.append("bpftime verifier source changed during run")
    end_probe_metadata = end_environment.get("probe")
    if not isinstance(end_probe_metadata, dict):
        errors.append("end probe metadata is not an object")
        end_probe_metadata = {}
    for key in ("path", "size", "mtime_ns"):
        if end_probe_metadata.get(key) != probe_metadata.get(key):
            errors.append(f"probe {key} changed during run")

    expected_descriptions = (
        [("linear", 16), ("diamonds", 4096)] if mode == "preflight" else list(ARMS)
    )
    descriptions = result.get("descriptions")
    if not isinstance(descriptions, list) or len(descriptions) != len(expected_descriptions):
        errors.append("description cardinality mismatch")
        descriptions = []
    for index, (family, size) in enumerate(expected_descriptions):
        if index >= len(descriptions):
            break
        expected_relative = f"descriptions/{family}-{size}"
        if not exact_json_equal(
            descriptions[index],
            {
                "family": family,
                "instructions": size,
                "directory": expected_relative,
            },
        ):
            errors.append(f"description index {index} metadata mismatch")
        raw_errors, _ = validate_raw_call(
            run_dir, expected_relative, probe, family, size, "describe", revision
        )
        errors.extend(raw_errors)

    warmups = result.get("warmups")
    expected_warmups = [] if mode == "preflight" else list(ARMS)
    if not isinstance(warmups, list) or len(warmups) != len(expected_warmups):
        errors.append("warmup cardinality mismatch")
        warmups = []
    for index, (family, size) in enumerate(expected_warmups):
        if index >= len(warmups):
            break
        expected_relative = f"warmups/{family}-{size}"
        if not exact_json_equal(
            warmups[index],
            {
                "family": family,
                "instructions": size,
                "directory": expected_relative,
            },
        ):
            errors.append(f"warmup index {index} metadata mismatch")
        raw_errors, _ = validate_raw_call(
            run_dir, expected_relative, probe, family, size, "accept_only", revision
        )
        errors.extend(raw_errors)

    expected_cells = preflight_schedule() if mode == "preflight" else full_schedule()
    cells = result.get("cells")
    if not isinstance(cells, list) or len(cells) != len(expected_cells):
        errors.append("cell cardinality mismatch")
        cells = []
    analyzed_cells: list[dict[str, Any]] = []
    for index, expected in enumerate(expected_cells):
        if index >= len(cells):
            break
        relative = (
            f"cells/seq-{expected['sequence']:03d}-block-{expected['block']:02d}-"
            f"pos-{expected['position']:02d}-{expected['family']}-{expected['instructions']}"
        )
        expected_cell = dict(expected)
        expected_cell.update({"directory": relative, "valid": True})
        if not exact_json_equal(cells[index], expected_cell):
            errors.append(f"cell index {index} metadata/schedule mismatch")
        raw_errors, record = validate_raw_call(
            run_dir,
            relative,
            probe,
            expected["family"],
            expected["instructions"],
            "timed",
            revision,
        )
        errors.extend(raw_errors)
        if record is not None:
            analyzed = dict(expected)
            analyzed["record"] = record
            analyzed_cells.append(analyzed)

    summary = None
    if mode == "full" and not errors and len(analyzed_cells) == len(expected_cells):
        summary = summarize(analyzed_cells)
    analysis = {
        "schema": "device-verifier-scaling-analysis-v1",
        "complete": not errors,
        "run_status": "valid" if not errors else "invalid",
        "mode": mode,
        "errors": errors,
        "summary": summary,
        "claim_scope": (
            "one-time direct verify_gpu_program admission latency for accepted synthetic "
            "linear and warp-uniform-diamond programs at 16--4096 instructions"
        ),
        "excluded_claims": [
            "verifier soundness",
            "GPU execution or device overhead",
            "attach, JIT, or bootstrap latency",
            "cross-vendor portability",
            "the 65536-instruction runtime boundary",
        ],
        "research_value": "supporting" if not errors and mode == "full" else "dependency-only",
        "paper_impact": "additional RQ4 evidence" if not errors and mode == "full" else "none",
        "next_paper_decision": (
            "result review before any paper use"
            if not errors and mode == "full"
            else "do not use as a paper result"
        ),
    }
    return analysis


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    analysis = analyze(args.run_dir.resolve())
    if not args.no_write:
        (args.run_dir / "analysis.json").write_text(
            json.dumps(analysis, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(analysis, indent=2, sort_keys=True))
    return 0 if analysis["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
