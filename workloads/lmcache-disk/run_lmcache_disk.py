#!/usr/bin/env python3
"""Thin LMCache experiment adapter: one official vLLM cell at a time.

This module deliberately has no approval parser, promotion marker, resume
protocol, or attempt-budget controller. The experiment plan supplies ordinary
commands; this adapter starts one source-native server, retains raw output, and
recomputes validation/analysis from those outputs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
LEGACY_PATH = HERE / "historical_runner_v1.py"
SPEC = importlib.util.spec_from_file_location("lmcache_adapter_primitives", LEGACY_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load LMCache adapter primitives from {LEGACY_PATH}")
legacy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(legacy)

GateError = legacy.GateError
CONFIGS = legacy.CONFIGS
PROMPTS = legacy.PROMPTS
SCHEDULE = legacy.SCHEDULE
PREFIXES = legacy.PREFIXES
PREFIX_TOKENS = legacy.PREFIX_TOKENS
OUTPUT_TOKENS = legacy.OUTPUT_TOKENS
EXPECTED_DISK_BYTES = legacy.EXPECTED_DISK_BYTES
TARGET_BLOCKS = legacy.TARGET_BLOCKS
BOOTSTRAP_SEED = legacy.BOOTSTRAP_SEED
MODEL_REVISION = legacy.MODEL_REVISION
EXPECTED_DRIVER = legacy.EXPECTED_DRIVER
EXPECTED_LMCACHE_VERSION = legacy.EXPECTED_LMCACHE_VERSION
EXPECTED_VLLM_VERSION = legacy.EXPECTED_VLLM_VERSION

# Re-export the small primitives used by CPU structural tests.
server_environment = legacy.server_environment
server_argv = legacy.server_argv
start_server = legacy.start_server
request_log_values = legacy.request_log_values
validate_odirect = legacy.validate_odirect
validate_log = legacy.validate_log
load_prompts = legacy.load_prompts
prepare_prompts = legacy.prepare_prompts
validate_schedule = legacy.validate_schedule


def inspect_environment(port: int, storage_root: Path) -> dict[str, Any]:
    """Return read-only launch observations; this is not a promotion marker."""
    return legacy.admission(port, require_model=True, storage_path=storage_root)


def run_cell(config: str, output: Path, port: int, trace: bool) -> dict[str, Any]:
    """Run one cell through the official vLLM serve path and retain raw output."""
    if config not in CONFIGS:
        raise GateError(f"unknown configuration: {config}")
    observations = inspect_environment(port, output)
    prompts = load_prompts(PROMPTS)
    result = legacy.run_config(
        config,
        output,
        prompts,
        port,
        Path(observations["model_path"]),
        trace=trace,
    )
    legacy.atomic_write_json(output / "environment.json", observations)
    return result


def _validate_response(response: dict[str, Any], expected_tokens: int, request_id: str) -> None:
    if response.get("request_header") != request_id:
        raise GateError(f"request ID mismatch: expected {request_id}, got {response.get('request_header')}")
    if response.get("engine_request_id") != f"cmpl-{request_id}-0":
        raise GateError(f"engine request ID mismatch for {request_id}")
    usage = response.get("usage", {})
    if (
        response.get("status") != 200
        or response.get("input_tokens") != expected_tokens
        or usage.get("prompt_tokens") != expected_tokens
        or usage.get("completion_tokens") != OUTPUT_TOKENS
        or not isinstance(response.get("text"), str)
        or not response["text"]
    ):
        raise GateError(f"response semantics mismatch for {request_id}: {response}")
    for field in ("ttft_ms", "e2e_ms"):
        value = response.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise GateError(f"invalid {field} for {request_id}: {value}")


def validate_cell(run_dir: Path, require_trace: bool = False) -> dict[str, Any]:
    """Reparse one cell's result, server log, and optional raw trace."""
    result_path = run_dir / "result.json"
    log_path = run_dir / "server.log"
    environment_path = run_dir / "environment.json"
    if not result_path.is_file() or not log_path.is_file() or not environment_path.is_file():
        raise GateError(f"missing raw cell output under {run_dir}")
    result = json.loads(result_path.read_text())
    recorded_environment = json.loads(environment_path.read_text())
    if (
        recorded_environment.get("gpu", {}).get("driver") != EXPECTED_DRIVER
        or recorded_environment.get("gpu", {}).get("compute_apps") != []
        or recorded_environment.get("model_revision") != MODEL_REVISION
        or recorded_environment.get("runtime_imports", {}).get("lmcache_version")
        != EXPECTED_LMCACHE_VERSION
        or recorded_environment.get("runtime_imports", {}).get("vllm_version")
        != EXPECTED_VLLM_VERSION
    ):
        raise GateError(f"recorded launch environment mismatch under {run_dir}")
    config = result.get("config")
    if result.get("schema") != 2 or config not in CONFIGS:
        raise GateError(f"invalid result schema/config under {run_dir}")
    prompts = load_prompts(PROMPTS)
    observations = result.get("observations")
    if not isinstance(observations, list) or len(observations) != PREFIXES:
        raise GateError(f"expected {PREFIXES} observations under {run_dir}")
    for item, observation in zip(prompts["prefixes"], observations, strict=True):
        index = item["index"]
        if (
            observation.get("prefix_index") != index
            or observation.get("expected_hit_tokens") != item["expected_hit_tokens"]
        ):
            raise GateError(f"prompt/observation mismatch for prefix {index}")
        _validate_response(observation["cold"], len(item["cold_token_ids"]), f"lmc-p{index}-cold")
        _validate_response(observation["warm"], len(item["warm_token_ids"]), f"lmc-p{index}-warm")

    command = result.get("command")
    if not isinstance(command, list) or len(command) < 4 or "--port" not in command:
        raise GateError(f"missing raw server command under {run_dir}")
    if Path(command[2]).name != MODEL_REVISION or command[2] != recorded_environment.get("model_path"):
        raise GateError(f"model path differs from the pinned recorded revision under {run_dir}")
    expected_command = server_argv(config, Path(command[2]), command[command.index("--port") + 1])
    if result.get("command") != expected_command:
        raise GateError(f"server command differs from fixed cell configuration: {run_dir}")
    cache_dir = (run_dir / "cache").resolve()
    if result.get("environment") != server_environment(config, cache_dir):
        raise GateError(f"server environment differs from fixed cell configuration: {run_dir}")

    log = log_path.read_text(errors="replace")
    engagement = validate_log(config, log, observations, cache_dir)
    if result.get("engagement") != engagement:
        raise GateError(f"saved and recomputed engagement differ under {run_dir}")
    footprint = result.get("cache_footprint", {})
    if config == "lmcache_disk":
        if footprint != {"files": 48, "bytes": EXPECTED_DISK_BYTES}:
            raise GateError(f"disk footprint mismatch under {run_dir}: {footprint}")
    elif footprint != {"files": 0, "bytes": 0}:
        raise GateError(f"non-disk cell wrote disk chunks under {run_dir}: {footprint}")

    trace = None
    trace_dir = run_dir / "strace"
    if require_trace:
        if config != "lmcache_disk" or not trace_dir.is_dir():
            raise GateError("trace validation requires a traced lmcache_disk cell")
        trace = validate_odirect(trace_dir, cache_dir)
    elif trace_dir.exists():
        trace = validate_odirect(trace_dir, cache_dir)
    if result.get("odirect") != trace:
        raise GateError(f"saved and recomputed trace semantics differ under {run_dir}")
    launch = result.get("launch_command")
    if trace_dir.exists():
        expected_launch = [
            "/usr/bin/strace", "-ff", "-qq", "-s", "4096", "-e", "trace=open,openat",
            "-o", str((trace_dir.resolve() / "open.trace")), *expected_command,
        ]
    else:
        expected_launch = expected_command
    if launch != expected_launch:
        raise GateError(f"raw launch command mismatch under {run_dir}")

    warm_phase = result.get("warm_phase", {})
    elapsed = warm_phase.get("elapsed_s")
    if (
        warm_phase.get("requests") != PREFIXES
        or warm_phase.get("output_tokens") != PREFIXES * OUTPUT_TOKENS
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed <= 0
        or not math.isclose(warm_phase.get("requests_per_s", -1), PREFIXES / elapsed, rel_tol=1e-12)
        or not math.isclose(
            warm_phase.get("output_tokens_per_s", -1), PREFIXES * OUTPUT_TOKENS / elapsed,
            rel_tol=1e-12,
        )
    ):
        raise GateError(f"warm-phase metric derivation mismatch under {run_dir}")
    return {"result": result, "engagement": engagement, "trace": trace}


def exact_outputs(run_dir: Path) -> dict[str, str]:
    validated = validate_cell(run_dir)
    return legacy.output_texts(validated["result"])


def compare_outputs(run_dirs: list[Path]) -> dict[str, Any]:
    if len(run_dirs) < 2:
        raise GateError("exact-output comparison needs at least two cell directories")
    outputs = [(path, exact_outputs(path)) for path in run_dirs]
    reference = outputs[0][1]
    mismatched = [str(path) for path, value in outputs[1:] if value != reference]
    if mismatched:
        raise GateError(f"exact output text mismatch: {mismatched}")
    return {"cells": [str(path) for path, _ in outputs], "exact_text_equal": True}


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_ci(values: list[float], seed: int, draws: int = 10000) -> dict[str, Any]:
    import random

    rng = random.Random(seed)
    estimates = [statistics.median(rng.choices(values, k=len(values))) for _ in range(draws)]
    return {
        "median": statistics.median(values),
        "ci95": [percentile(estimates, 0.025), percentile(estimates, 0.975)],
    }


def classify_effect(latency_ci: list[float], rate_ci: list[float]) -> str:
    if latency_ci[1] < 0 and rate_ci[0] >= -0.05:
        return "beneficial"
    if latency_ci[1] < 0 and rate_ci[1] < -0.05:
        return "latency-throughput tradeoff"
    if latency_ci[0] >= 0:
        return "not beneficial"
    return "inconclusive"


def _attempt_cells(root: Path) -> dict[int, dict[str, tuple[int, Path]]]:
    pattern = re.compile(r"position-([0-2])-(recompute|lmcache_cpu|lmcache_disk)$")
    groups: dict[int, dict[str, tuple[int, Path]]] = {}
    for attempt_dir in sorted(root.glob("attempt-*")):
        match_attempt = re.fullmatch(r"attempt-(\d{2})", attempt_dir.name)
        if not match_attempt:
            continue
        attempt = int(match_attempt.group(1))
        for path in sorted(attempt_dir.iterdir() if attempt_dir.is_dir() else []):
            match = pattern.fullmatch(path.name)
            if match and (path / "result.json").is_file():
                groups.setdefault(attempt, {})[match.group(2)] = (int(match.group(1)), path)
    return groups


def analyze(root: Path) -> dict[str, Any]:
    """Revalidate raw cells, enforce the fixed schedule, and compute paired effects."""
    schedule = json.loads(SCHEDULE.read_text())
    validate_schedule(schedule)
    scheduled = {item["attempt"]: item["order"] for item in schedule["attempts"]}
    groups = _attempt_cells(root)
    complete = []
    reference_outputs = None
    for attempt in sorted(groups):
        cells = groups[attempt]
        if set(cells) != set(CONFIGS):
            continue
        if attempt not in scheduled:
            raise GateError(f"attempt {attempt} is outside the fixed schedule")
        positions = {config: position for config, (position, _) in cells.items()}
        expected_positions = {config: position for position, config in enumerate(scheduled[attempt])}
        if positions != expected_positions:
            raise GateError(f"attempt {attempt} position mismatch: {positions} != {expected_positions}")
        validated = {config: validate_cell(cells[config][1])["result"] for config in CONFIGS}
        for config, result in validated.items():
            outputs = legacy.output_texts(result)
            if reference_outputs is None:
                reference_outputs = outputs
            elif outputs != reference_outputs:
                raise GateError(f"exact output text drift in attempt {attempt}, config {config}")
        complete.append((attempt, validated))
    if len(complete) != TARGET_BLOCKS:
        raise GateError(f"analysis requires {TARGET_BLOCKS} complete attempts, found {len(complete)}")

    rows = []
    for attempt, cells in complete:
        row: dict[str, Any] = {"attempt": attempt}
        for config, result in cells.items():
            warm = [float(item["warm"]["ttft_ms"]) for item in result["observations"]]
            phase = result["warm_phase"]
            row[config] = {
                "warm_ttft_median_ms": statistics.median(warm),
                "warm_ttft_p95_ms": percentile(warm, 0.95),
                "warm_ttft_max_ms": max(warm),
                "warm_requests_per_s": float(phase["requests_per_s"]),
                "warm_output_tokens_per_s": float(phase["output_tokens_per_s"]),
            }
        rows.append(row)

    effects: dict[str, Any] = {}
    for baseline_index, baseline in enumerate(("recompute", "lmcache_cpu")):
        latency = [row["lmcache_disk"]["warm_ttft_median_ms"]
                   - row[baseline]["warm_ttft_median_ms"] for row in rows]
        request_rate = [row["lmcache_disk"]["warm_requests_per_s"]
                        / row[baseline]["warm_requests_per_s"] - 1 for row in rows]
        output_rate = [row["lmcache_disk"]["warm_output_tokens_per_s"]
                       / row[baseline]["warm_output_tokens_per_s"] - 1 for row in rows]
        effects[f"disk_vs_{baseline}"] = {
            "warm_ttft_difference_ms": bootstrap_ci(latency, BOOTSTRAP_SEED + baseline_index * 3),
            "warm_request_rate_relative": bootstrap_ci(request_rate, BOOTSTRAP_SEED + baseline_index * 3 + 1),
            "warm_output_token_rate_relative": bootstrap_ci(
                output_rate, BOOTSTRAP_SEED + baseline_index * 3 + 2
            ),
        }

    primary = effects["disk_vs_recompute"]
    latency_ci = primary["warm_ttft_difference_ms"]["ci95"]
    rate_ci = primary["warm_request_rate_relative"]["ci95"]
    decision = classify_effect(latency_ci, rate_ci)
    result = {
        "complete_attempts": [attempt for attempt, _ in complete],
        "rows": rows,
        "paired_percentile_bootstrap": {"draws": 10000, **effects},
        "decision": decision,
        "scope": "fixed model, eight prefixes, LMCache/vLLM runtime, RTX 5090, and local SSD",
    }
    legacy.atomic_write_json(root / "analysis.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--port", type=int, default=18080)
    inspect.add_argument("--storage-root", type=Path, default=HERE / "raw")
    prepare = sub.add_parser("prepare-prompts")
    prepare.add_argument("--output", type=Path, default=PROMPTS)
    cell = sub.add_parser("run-cell")
    cell.add_argument("--config", choices=CONFIGS, required=True)
    cell.add_argument("--output", type=Path, required=True)
    cell.add_argument("--port", type=int, default=18080)
    cell.add_argument("--trace", action="store_true")
    validate = sub.add_parser("validate-cell")
    validate.add_argument("run_dir", type=Path)
    validate.add_argument("--require-trace", action="store_true")
    compare = sub.add_parser("compare-outputs")
    compare.add_argument("run_dirs", type=Path, nargs="+")
    analysis = sub.add_parser("analyze")
    analysis.add_argument("root", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            value = inspect_environment(args.port, args.storage_root)
        elif args.command == "prepare-prompts":
            value = prepare_prompts(args.output)
        elif args.command == "run-cell":
            value = run_cell(args.config, args.output, args.port, args.trace)
        elif args.command == "validate-cell":
            value = validate_cell(args.run_dir, args.require_trace)
        elif args.command == "compare-outputs":
            value = compare_outputs(args.run_dirs)
        else:
            value = analyze(args.root)
        print(json.dumps(value, ensure_ascii=False, indent=2))
    except GateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
