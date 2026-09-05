#!/usr/bin/env python3
"""Eight-prefix native-vLLM correctness reference for LMCache V3.

This diagnostic is intentionally separate from the formal performance runner.
It uses native vLLM prefix caching to exercise the same cached-prefill path as
LMCache, then requires exact generated-token-ID and text equality with the
LMCache CPU and disk cells for every cold and warm request.

Frozen protocol: the eight prompt pairs from prompts.json, max_tokens=16,
temperature=0, seed=0, and return_token_ids=true; streamed choice.token_ids
and choice.text are parsed for every request. The native arm must report
cached_tokens=0 for every cold request and cached_tokens=1536 for every warm
request. Results whose responses lack exact generated token IDs (legacy
records) are rejected. The recompute arm remains a separate performance arm.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import diagnose_v3_warm_divergence as one_token
import run_lmcache_disk as runner


HERE = Path(__file__).resolve().parent
PREFIXES = runner.PREFIXES


def _cached_tokens(response: dict[str, Any]) -> int | None:
    details = response.get("usage", {}).get("prompt_tokens_details") or {}
    value = details.get("cached_tokens")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _validate_response(response: dict[str, Any], expected_tokens: int, request_id: str) -> None:
    runner._validate_response(response, expected_tokens, request_id)
    ids = response.get("generated_token_ids")
    if (
        not isinstance(ids, list)
        or len(ids) != runner.OUTPUT_TOKENS
        or any(not isinstance(value, int) or isinstance(value, bool) for value in ids)
    ):
        raise runner.GateError(
            f"{request_id} lacks the exact {runner.OUTPUT_TOKENS} generated token IDs; "
            "legacy results without token IDs are rejected"
        )


def _validate_prompt_observations(
    prompts: dict[str, Any],
    observations: list[dict[str, Any]],
) -> None:
    if (
        len(prompts["prefixes"]) != PREFIXES
        or not isinstance(observations, list)
        or len(observations) != PREFIXES
    ):
        raise runner.GateError(f"native-prefix reference requires exactly {PREFIXES} prompt/observation pairs")
    for item, observation in zip(prompts["prefixes"], observations, strict=True):
        index = item["index"]
        if (
            observation.get("prefix_index") != index
            or observation.get("expected_hit_tokens") != item["expected_hit_tokens"]
        ):
            raise runner.GateError(f"native-prefix prompt mismatch for prefix {index}")


def _validate_engagement(
    log: str,
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    fatal_log = runner.legacy._log_for_fatal_scan(log)
    fatal = [
        pattern
        for pattern in runner.legacy.FATAL_LOG_PATTERNS
        if re.search(pattern, fatal_log, re.I)
    ]
    if fatal:
        raise runner.GateError(f"fatal evidence in native-prefix server log: {fatal}")
    if "LMCache initialized" in log or "LMCache hit tokens:" in log:
        raise runner.GateError("native-prefix reference unexpectedly engaged LMCache")
    cached = []
    for observation in observations:
        expected = observation["expected_hit_tokens"]
        pair = [_cached_tokens(observation[phase]) for phase in ("cold", "warm")]
        if pair != [0, expected]:
            raise runner.GateError(
                f"native-prefix engagement mismatch for prefix "
                f"{observation['prefix_index']}: expected [0, {expected}], got {pair}"
            )
        cached.append({"prefix_index": observation["prefix_index"], "cold_warm": pair})
    return {"prompt_tokens_details_cached_tokens": cached}


def _server_argv(model_path: Path, port: int | str) -> list[str]:
    return one_token.diagnostic_server_argv("native_prefix", model_path, int(port))


def _run(output: Path, port: int, expected_driver: str) -> dict[str, Any]:
    with runner.managed_cell(output, expected_driver) as execution:
        environment = runner.inspect_environment(port, output, expected_driver)
        environment.update(
            {key: execution[key] for key in ("boot_id", "worker_cpu_affinity", "telemetry_cpu")}
        )
        prompts = runner.load_prompts(runner.PROMPTS)
        if len(prompts["prefixes"]) != PREFIXES:
            raise runner.GateError(f"native-prefix reference requires exactly {PREFIXES} prompts")
        cache_dir = (output / "cache").resolve()
        cache_dir.mkdir()
        server_environment = runner.server_environment("recompute", cache_dir, expected_driver)
        environment["server_environment"] = server_environment
        runner.legacy.atomic_write_json(output / "environment.json", environment)
        log_path = output / "server.log"
        proc, log_file, argv, launch = one_token._start_server(
            "native_prefix", Path(environment["model_path"]), cache_dir,
            port, log_path, expected_driver,
        )
        observations: list[dict[str, Any]] = []
        try:
            runner.legacy.wait_ready(proc, port, log_path)
            worker_affinity = sorted(os.sched_getaffinity(proc.pid))
            if worker_affinity != runner.WORKER_CPUS:
                raise runner.GateError(
                    f"server CPU affinity differs from 8-15: {worker_affinity}"
                )
            for item in prompts["prefixes"]:
                index = item["index"]
                cold = runner.legacy.streamed_completion(
                    port, item["cold_token_ids"], f"native-p{index}-cold"
                )
                observations.append(
                    {
                        "prefix_index": index,
                        "expected_hit_tokens": item["expected_hit_tokens"],
                        "cold": cold,
                    }
                )
            for item, observation in zip(
                prompts["prefixes"], observations, strict=True
            ):
                index = item["index"]
                observation["warm"] = runner.legacy.streamed_completion(
                    port, item["warm_token_ids"], f"native-p{index}-warm"
                )
        finally:
            try:
                runner.legacy.stop_owned_server(proc, log_file)
            finally:
                runner.legacy.wait_gpu_idle()
        log = log_path.read_text(errors="replace")
        engagement = _validate_engagement(log, observations)
        result = {
            "schema": 1,
            "config": "native_prefix_correctness",
            "purpose": (
                "candidate cached-path correctness reference only; excluded from "
                "formal performance analysis"
            ),
            "prefix_count": PREFIXES,
            "all_cold_before_all_warm": True,
            "max_output_tokens": runner.OUTPUT_TOKENS,
            "temperature": 0,
            "seed": 0,
            "return_token_ids": True,
            "worker_cpu_affinity": worker_affinity,
            "command": argv,
            "launch_command": launch,
            "environment": server_environment,
            "observations": observations,
            "engagement": engagement,
            "cache_footprint": {
                "files": sum(1 for path in cache_dir.rglob("*") if path.is_file()),
                "bytes": sum(path.stat().st_size for path in cache_dir.rglob("*") if path.is_file()),
            },
            "server_log": runner.legacy.file_identity(log_path),
        }
        runner.legacy.atomic_write_json(output / "result.json", result)
    return result


def _validate(run_dir: Path) -> dict[str, Any]:
    result_path = run_dir / "result.json"
    environment_path = run_dir / "environment.json"
    log_path = run_dir / "server.log"
    if not all(path.is_file() for path in (result_path, environment_path, log_path)):
        raise runner.GateError(f"missing native-prefix correctness evidence under {run_dir}")
    result = json.loads(result_path.read_text())
    environment = json.loads(environment_path.read_text())
    runner._validate_recorded_environment(environment)
    runner.validate_execution(run_dir, environment)
    if (
        result.get("schema") != 1
        or result.get("config") != "native_prefix_correctness"
        or result.get("prefix_count") != PREFIXES
        or result.get("all_cold_before_all_warm") is not True
        or result.get("max_output_tokens") != runner.OUTPUT_TOKENS
        or result.get("temperature") != 0
        or result.get("seed") != 0
        or result.get("return_token_ids") is not True
        or result.get("worker_cpu_affinity") != runner.WORKER_CPUS
    ):
        raise runner.GateError("native-prefix result does not match the frozen protocol")
    prompts = runner.load_prompts(runner.PROMPTS)
    observations = result.get("observations")
    _validate_prompt_observations(prompts, observations)
    for item, observation in zip(prompts["prefixes"], observations, strict=True):
        index = item["index"]
        _validate_response(
            observation["cold"], len(item["cold_token_ids"]), f"native-p{index}-cold"
        )
        _validate_response(
            observation["warm"], len(item["warm_token_ids"]), f"native-p{index}-warm"
        )
    command = result.get("command")
    if not isinstance(command, list) or "--port" not in command:
        raise runner.GateError("native-prefix result lacks the raw server command")
    if command[2] != environment.get("model_path"):
        raise runner.GateError("native-prefix model path differs from recorded admission")
    expected_command = _server_argv(
        Path(command[2]), command[command.index("--port") + 1]
    )
    if command != expected_command:
        raise runner.GateError("native-prefix server command differs from the frozen protocol")
    expected_launch = ["/usr/bin/taskset", "-c", "8-15", *expected_command]
    if result.get("launch_command") != expected_launch:
        raise runner.GateError("native-prefix launch command differs from the frozen protocol")
    cache_dir = (run_dir / "cache").resolve()
    expected_environment = runner.server_environment(
        "recompute", cache_dir, environment["expected_driver"]
    )
    if (
        result.get("environment") != expected_environment
        or environment.get("server_environment") != expected_environment
    ):
        raise runner.GateError("native-prefix server environment differs from the frozen protocol")
    footprint = {
        "files": sum(1 for path in cache_dir.rglob("*") if path.is_file()),
        "bytes": sum(path.stat().st_size for path in cache_dir.rglob("*") if path.is_file()),
    }
    if result.get("cache_footprint") != footprint or footprint != {"files": 0, "bytes": 0}:
        raise runner.GateError(f"native-prefix reference unexpectedly wrote storage: {footprint}")
    engagement = _validate_engagement(log_path.read_text(errors="replace"), observations)
    if result.get("engagement") != engagement:
        raise runner.GateError("saved native-prefix engagement differs from raw evidence")
    if result.get("server_log") != runner.legacy.file_identity(log_path):
        raise runner.GateError("native-prefix server-log inventory differs from the saved result")
    return {"result": result, "environment": environment, "engagement": engagement}


def _exact_outputs(result: dict[str, Any], arm: str) -> dict[str, dict[str, Any]]:
    observations = result.get("observations")
    if not isinstance(observations, list) or len(observations) != PREFIXES:
        raise runner.GateError(
            f"{arm} result requires exactly {PREFIXES} observations, "
            f"got {len(observations) if isinstance(observations, list) else 'non-list'}"
        )
    indices = [observation.get("prefix_index") for observation in observations]
    if (
        any(not isinstance(index, int) or isinstance(index, bool) for index in indices)
        or len(set(indices)) != PREFIXES
        or any(not 0 <= index < PREFIXES for index in indices)
    ):
        raise runner.GateError(
            f"{arm} result prefix indices must be the unique integers "
            f"0..{PREFIXES - 1}: {indices}"
        )
    outputs: dict[str, dict[str, Any]] = {}
    for observation in observations:
        index = observation["prefix_index"]
        for phase in ("cold", "warm"):
            if phase not in observation:
                raise runner.GateError(f"{arm} result lacks the {phase} request for prefix {index}")
            response = observation[phase]
            ids = response.get("generated_token_ids")
            if (
                not isinstance(ids, list)
                or len(ids) != runner.OUTPUT_TOKENS
                or any(not isinstance(value, int) or isinstance(value, bool) for value in ids)
            ):
                raise runner.GateError(
                    f"{arm} result for prefix {index} {phase} lacks exact generated token IDs; "
                    "legacy results without token IDs are rejected"
                )
            text = response.get("text")
            if not isinstance(text, str) or not text:
                raise runner.GateError(f"{arm} result for prefix {index} {phase} lacks generated text")
            outputs[f"{index}:{phase}"] = {"token_ids": list(ids), "text": text}
    return outputs


def _compare_outputs(
    native: dict[str, dict[str, Any]],
    cpu: dict[str, dict[str, Any]],
    disk: dict[str, dict[str, Any]],
    native_dir: str,
    cpu_dir: str,
    disk_dir: str,
) -> dict[str, Any]:
    if set(native) != set(cpu) or set(native) != set(disk):
        raise runner.GateError(
            f"comparison phases differ: native={sorted(native)} cpu={sorted(cpu)} disk={sorted(disk)}"
        )
    mismatched = []
    for name, outputs in (("lmcache_cpu", cpu), ("lmcache_disk", disk)):
        for key in sorted(native):
            reference = native[key]
            value = outputs[key]
            if value["token_ids"] != reference["token_ids"]:
                mismatched.append(f"{name}:{key}:token_ids")
            elif value["text"] != reference["text"]:
                mismatched.append(f"{name}:{key}:text")
    if mismatched:
        raise runner.GateError(f"exact cached-path output mismatch: {mismatched}")
    return {
        "native_prefix": native_dir,
        "lmcache_cpu": cpu_dir,
        "lmcache_disk": disk_dir,
        "prefixes": PREFIXES,
        "phases_per_prefix": ["cold", "warm"],
        "output_tokens_per_request": runner.OUTPUT_TOKENS,
        "exact_token_ids_equal": True,
        "exact_text_equal": True,
        "tolerance": None,
    }


def _compare(native_dir: Path, cpu_dir: Path, disk_dir: Path) -> dict[str, Any]:
    native = _validate(native_dir)["result"]
    cpu = runner.validate_cell(cpu_dir)["result"]
    disk = runner.validate_cell(disk_dir)["result"]
    if cpu.get("config") != "lmcache_cpu" or disk.get("config") != "lmcache_disk":
        raise runner.GateError("candidate comparison requires LMCache CPU then disk cells")
    return _compare_outputs(
        _exact_outputs(native, "native_prefix"),
        _exact_outputs(cpu, "lmcache_cpu"),
        _exact_outputs(disk, "lmcache_disk"),
        str(native_dir),
        str(cpu_dir),
        str(disk_dir),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("run")
    launch.add_argument("--output", type=Path, required=True)
    launch.add_argument("--port", type=int, default=18080)
    launch.add_argument(
        "--expected-driver", choices=runner.legacy.EXPERIMENT_DRIVERS,
        default="575.57.08",
    )
    validate = commands.add_parser("validate")
    validate.add_argument("run_dir", type=Path)
    compare = commands.add_parser("compare")
    compare.add_argument("native_dir", type=Path)
    compare.add_argument("cpu_dir", type=Path)
    compare.add_argument("disk_dir", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "run":
            value = _run(args.output, args.port, args.expected_driver)
        elif args.command == "validate":
            value = _validate(args.run_dir)
        else:
            value = _compare(args.native_dir, args.cpu_dir, args.disk_dir)
        print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    except runner.GateError as error:
        print(f"ERROR: {error}", file=__import__("sys").stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
