#!/usr/bin/env python3
"""One-prefix diagnostic for LMCache V3 warm-output divergence.

This is deliberately separate from the formal correctness and performance
runners.  It compares a full prefill with a cache hit using one generated
token and retains the generated token ID plus the requested top-20 logprobs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import run_lmcache_disk as runner


HERE = Path(__file__).resolve().parent
MODEL_ID = runner.legacy.MODEL_ID
TOP_LOGPROBS = 20
OUTPUT_TOKENS = 1


def diagnostic_server_argv(arm: str, model_path: Path, port: int) -> list[str]:
    if arm not in ("lmcache_cpu", "native_prefix"):
        raise runner.GateError(f"unknown diagnostic arm: {arm}")
    config = "lmcache_cpu" if arm == "lmcache_cpu" else "recompute"
    argv = runner.server_argv(config, model_path, port)
    native_flag = argv.index("--no-enable-prefix-caching")
    if arm == "native_prefix":
        argv[native_flag] = "--enable-prefix-caching"
    argv.extend(["--return-tokens-as-token-ids", "--enable-prompt-tokens-details"])
    return argv


def _start_server(
    arm: str,
    model_path: Path,
    cache_dir: Path,
    port: int,
    log_path: Path,
    expected_driver: str,
):
    argv = diagnostic_server_argv(arm, model_path, port)
    launch = ["/usr/bin/taskset", "-c", "8-15", *argv]
    config = "lmcache_cpu" if arm == "lmcache_cpu" else "recompute"
    log_file = log_path.open("x")
    try:
        proc = subprocess.Popen(
            launch,
            cwd=runner.legacy.VLLM_WORKLOAD,
            env=runner.server_environment(config, cache_dir, expected_driver),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except BaseException:
        log_file.close()
        raise
    return proc, log_file, argv, launch


def diagnostic_completion(port: int, token_ids: list[int], request_id: str) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": MODEL_ID,
            "prompt": token_ids,
            "max_tokens": OUTPUT_TOKENS,
            "temperature": 0,
            "seed": 0,
            "ignore_eos": True,
            "stream": True,
            "logprobs": TOP_LOGPROBS,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json", "X-Request-Id": request_id},
        method="POST",
    )
    start = time.perf_counter_ns()
    first = None
    status = None
    usage: dict[str, Any] = {}
    engine_ids: set[str] = set()
    texts: list[str] = []
    generated_ids: list[int] = []
    tokens: list[str] = []
    token_logprobs: list[float] = []
    top_logprobs: list[dict[str, float]] = []
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            status = response.status
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                engine_id = event.get("id")
                if isinstance(engine_id, str) and engine_id:
                    engine_ids.add(engine_id)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    piece_ids = choice.get("token_ids") or []
                    logprobs = choice.get("logprobs") or {}
                    piece_tokens = logprobs.get("tokens") or []
                    if (piece_ids or piece_tokens) and first is None:
                        first = time.perf_counter_ns()
                    texts.append(choice.get("text") or "")
                    generated_ids.extend(int(value) for value in piece_ids)
                    tokens.extend(str(value) for value in piece_tokens)
                    token_logprobs.extend(float(value) for value in (logprobs.get("token_logprobs") or []))
                    top_logprobs.extend(
                        {str(key): float(value) for key, value in values.items()}
                        for values in (logprobs.get("top_logprobs") or [])
                    )
    except urllib.error.HTTPError as error:
        raise runner.GateError(
            f"HTTP {error.code}: {error.read().decode(errors='replace')}"
        ) from error
    end = time.perf_counter_ns()
    expected_engine_id = f"cmpl-{request_id}"
    if status != 200 or first is None or engine_ids != {expected_engine_id}:
        raise runner.GateError(
            f"invalid response envelope: status={status}, first={first}, IDs={engine_ids}"
        )
    if usage.get("prompt_tokens") != len(token_ids) or usage.get("completion_tokens") != 1:
        raise runner.GateError(f"unexpected usage for {request_id}: {usage}")
    if len(generated_ids) != 1 or len(tokens) != 1 or len(token_logprobs) != 1 or len(top_logprobs) != 1:
        raise runner.GateError(
            "diagnostic response must expose one generated token ID and one logprob position: "
            f"ids={generated_ids}, tokens={tokens}, token_logprobs={token_logprobs}, "
            f"top_positions={len(top_logprobs)}"
        )
    if not top_logprobs[0] or len(top_logprobs[0]) > TOP_LOGPROBS + 1:
        raise runner.GateError(f"unexpected top-logprob response: {top_logprobs}")
    if any(not math.isfinite(value) for value in [*token_logprobs, *top_logprobs[0].values()]):
        raise runner.GateError("non-finite diagnostic logprob")
    return {
        "request_header": request_id,
        "engine_request_id": expected_engine_id,
        "status": status,
        "input_tokens": len(token_ids),
        "usage": usage,
        "text": "".join(texts),
        "generated_token_ids": generated_ids,
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "requested_top_logprobs": TOP_LOGPROBS,
        "ttft_ms": (first - start) / 1e6,
        "e2e_ms": (end - start) / 1e6,
    }


def _cached_tokens(response: dict[str, Any]) -> int | None:
    details = response.get("usage", {}).get("prompt_tokens_details") or {}
    value = details.get("cached_tokens")
    return value if isinstance(value, int) else None


def _validate_native_log(log: str, first: dict[str, Any], second: dict[str, Any], expected: int) -> dict[str, Any]:
    fatal_log = runner.legacy._log_for_fatal_scan(log)
    fatal = [
        pattern
        for pattern in runner.legacy.FATAL_LOG_PATTERNS
        if re.search(pattern, fatal_log, re.I)
    ]
    if fatal:
        raise runner.GateError(f"fatal evidence in native server log: {fatal}")
    if "LMCache initialized" in log or "LMCache hit tokens:" in log:
        raise runner.GateError("native-prefix arm unexpectedly engaged LMCache")
    observed = [_cached_tokens(first), _cached_tokens(second)]
    if observed != [0, expected]:
        raise runner.GateError(
            f"native-prefix engagement mismatch: expected [0, {expected}], got {observed}"
        )
    return {"prompt_tokens_details_cached_tokens": observed}


def _pair_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_top = left["top_logprobs"][0]
    right_top = right["top_logprobs"][0]
    common = sorted(set(left_top) & set(right_top))
    return {
        "same_generated_token_id": left["generated_token_ids"] == right["generated_token_ids"],
        "same_generated_text": left["text"] == right["text"],
        "common_top_token_ids": common,
        "common_top_token_count": len(common),
        "max_absolute_logprob_delta_on_common_tokens": (
            max(abs(left_top[token] - right_top[token]) for token in common) if common else None
        ),
        "selected_token_logprob_delta": right["token_logprobs"][0] - left["token_logprobs"][0],
    }


def _run_arm(
    arm: str,
    arm_dir: Path,
    model_path: Path,
    port: int,
    token_ids: list[int],
    expected_cached_tokens: int,
    expected_driver: str,
) -> dict[str, Any]:
    arm_dir.mkdir()
    cache_dir = arm_dir / "cache"
    cache_dir.mkdir()
    log_path = arm_dir / "server.log"
    proc, log_file, argv, launch = _start_server(
        arm, model_path, cache_dir, port, log_path, expected_driver
    )
    try:
        runner.legacy.wait_ready(proc, port, log_path)
        worker_affinity = sorted(os.sched_getaffinity(proc.pid))
        if worker_affinity != runner.WORKER_CPUS:
            raise runner.GateError(f"server CPU affinity differs from 8-15: {worker_affinity}")
        first = diagnostic_completion(port, token_ids, f"diag-p0-{arm}-miss")
        if arm == "lmcache_cpu":
            store_state = runner.legacy.wait_for_cold_store(
                "lmcache_cpu",
                log_path,
                cache_dir,
                first["engine_request_id"],
                0,
                expected_cached_tokens,
                len(token_ids),
            )
        else:
            store_state = {"durability": "native vLLM prefix-cache insertion"}
        second = diagnostic_completion(port, token_ids, f"diag-p0-{arm}-hit")
    finally:
        try:
            runner.legacy.stop_owned_server(proc, log_file)
        finally:
            runner.legacy.wait_gpu_idle()
    log = log_path.read_text(errors="replace")
    if arm == "lmcache_cpu":
        observation = {
            "expected_hit_tokens": expected_cached_tokens,
            "cold": first,
            "warm": second,
        }
        engagement = runner.legacy.validate_log(
            "lmcache_cpu", log, [observation], cache_dir
        )
    else:
        engagement = _validate_native_log(log, first, second, expected_cached_tokens)
    result = {
        "arm": arm,
        "command": argv,
        "launch_command": launch,
        "worker_cpu_affinity": worker_affinity,
        "first_full_prefill": first,
        "second_cache_hit": second,
        "store_state": store_state,
        "engagement": engagement,
        "comparison": _pair_delta(first, second),
        "server_log": runner.legacy.file_identity(log_path),
    }
    runner.legacy.atomic_write_json(arm_dir / "result.json", result)
    return result


def run(output: Path, port: int, expected_driver: str) -> dict[str, Any]:
    with runner.managed_cell(output, expected_driver) as execution:
        environment = runner.inspect_environment(port, output, expected_driver)
        runner.legacy.atomic_write_json(output / "environment.json", environment)
        prompt = runner.load_prompts(runner.PROMPTS)["prefixes"][0]
        token_ids = prompt["warm_token_ids"]
        expected = prompt["expected_hit_tokens"]
        arms = {
            arm: _run_arm(
                arm,
                output / arm,
                Path(environment["model_path"]),
                port,
                token_ids,
                expected,
                expected_driver,
            )
            for arm in ("lmcache_cpu", "native_prefix")
        }
        result = {
            "schema": 1,
            "purpose": "diagnostic only; excluded from formal correctness and performance analysis",
            "prefix_index": 0,
            "input_tokens": len(token_ids),
            "max_output_tokens": OUTPUT_TOKENS,
            "temperature": 0,
            "seed": 0,
            "requested_top_logprobs": TOP_LOGPROBS,
            "expected_cached_tokens": expected,
            "arms": arms,
            "cross_arm": {
                "full_prefill": _pair_delta(
                    arms["lmcache_cpu"]["first_full_prefill"],
                    arms["native_prefix"]["first_full_prefill"],
                ),
                "cache_hit": _pair_delta(
                    arms["lmcache_cpu"]["second_cache_hit"],
                    arms["native_prefix"]["second_cache_hit"],
                ),
            },
            "boot_id": execution["boot_id"],
        }
        runner.legacy.atomic_write_json(output / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument(
        "--expected-driver", choices=runner.legacy.EXPERIMENT_DRIVERS, default="575.57.08"
    )
    args = parser.parse_args()
    result = run(args.output, args.port, args.expected_driver)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
