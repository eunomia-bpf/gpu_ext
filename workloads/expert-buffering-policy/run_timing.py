#!/usr/bin/env python3
"""Run one frozen paired timing block after the approved correctness gate."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import run_correctness as common


HERE = Path(__file__).resolve().parent
SCHEDULE = HERE / "timing-schedule.json"
RAW = HERE / "raw/timing"
POLICY_CONFIGS = {"gpubpf_observe", "gpubpf_profile_protect"}


def start_trace(run_dir: Path) -> tuple[subprocess.Popen[Any], Any, Path]:
    path = run_dir / "trace.jsonl"
    log = path.open("x", buffering=1)
    process = subprocess.Popen(
        ["sudo", "-n", str(common.TRACE), str(common.LIBGGML_BASE), "0", "7200"],
        stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True,
    )
    try:
        common.wait_event(process, path, "ready")
        return process, log, path
    except Exception:
        common.stop_group(process)
        log.close()
        raise


def start_telemetry(run_dir: Path) -> tuple[subprocess.Popen[Any], Any, Path]:
    path = run_dir / "gpu-telemetry.csv"
    log = path.open("x", buffering=1)
    query = ",".join((
        "timestamp", "memory.used", "temperature.gpu", "power.draw",
        "clocks.current.sm", "clocks_event_reasons.hw_thermal_slowdown",
        "clocks_event_reasons.sw_thermal_slowdown",
        "clocks_event_reasons.hw_power_brake_slowdown",
    ))
    process = subprocess.Popen(
        ["taskset", "-c", "16", "nvidia-smi", f"--query-gpu={query}",
         "--format=csv,noheader,nounits", "--loop-ms=200"],
        stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True,
    )
    time.sleep(0.4)
    if process.poll() is not None:
        log.close()
        raise common.GateError(f"GPU telemetry exited early: {path.read_text()}")
    return process, log, path


def validate_telemetry(path: Path) -> dict[str, Any]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 8:
            continue
        rows.append(fields)
    if len(rows) < 2:
        raise common.GateError(f"insufficient GPU telemetry: {path}")
    throttled = [row for row in rows if any(value.lower() == "active" for value in row[5:])]
    if throttled:
        raise common.GateError(f"thermal/power-brake throttling observed: {throttled[:3]}")
    return {
        "samples": len(rows),
        "peak_memory_mib": max(float(row[1]) for row in rows),
        "peak_temperature_c": max(float(row[2]) for row in rows),
        "mean_power_w": sum(float(row[3]) for row in rows) / len(rows),
        "min_sm_clock_mhz": min(float(row[4]) for row in rows),
        "max_sm_clock_mhz": max(float(row[4]) for row in rows),
        "thermal_or_power_brake_throttled": False,
    }


def policy_metric(policy_path: Path, policy_ready: dict[str, Any],
                  class_table: Path, before_stats: dict[str, Any],
                  before_blocks: dict[int, int]) -> tuple[dict[str, int], dict[str, Any]]:
    after_stats = common.latest_event(policy_path, "policy_stats")
    delta = common.counter_delta(before_stats, after_stats, common.POLICY_STAT_KEYS)
    indices = common.hot_block_indices(class_table)
    after_blocks = common.request_block_snapshot(
        policy_ready, policy_path, indices, 2
    )
    counts = {}
    for index in sorted(indices):
        if after_blocks[index] < before_blocks[index]:
            raise common.GateError(f"hot block counter regressed at index {index}")
        counts[index] = after_blocks[index] - before_blocks[index]
    metric = {
        "block_bytes": 2 * 1024 * 1024,
        "hot_blocks": len(indices),
        "full_activation_bytes": (2 * 1024 * 1024) * sum(counts.values()),
        "repeated_activation_bytes": (2 * 1024 * 1024) * sum(
            max(0, value - 1) for value in counts.values()
        ),
        "counts_before": {str(key): value for key, value in before_blocks.items()},
        "counts_after": {str(key): value for key, value in after_blocks.items()},
    }
    return delta, metric


def validate_policy_engagement(config: str, delta: dict[str, int]) -> None:
    if config == "gpubpf_observe":
        if delta["observe_activate"] <= 0 or delta["observe_access"] <= 0:
            raise common.GateError(f"observe timing engagement failed: {delta}")
        reordered = sum(delta[key] for key in (
            "hot_tail", "cold_head", "shared_tail", "hot_access_tail",
            "shared_access_tail", "setter_failure",
        ))
        if reordered != 0:
            raise common.GateError(f"observe timing reordered pages: {delta}")
    elif config == "gpubpf_profile_protect":
        required = ("mapped", "hot_tail", "cold_native", "hot_access_tail")
        if any(delta[key] <= 0 for key in required):
            raise common.GateError(f"protect timing engagement failed: {delta}")
        if delta["cold_head"] != 0 or delta["setter_failure"] != 0:
            raise common.GateError(f"protect timing safety gate failed: {delta}")


def run_timing_cell(config: str, run_dir: Path, port: int,
                    prompts: dict[str, Any], prompt_order: list[int]) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    trace = server = policy = telemetry = None
    trace_log = server_log = policy_log = telemetry_log = None
    trace_path = run_dir / "trace.jsonl"
    server_path = run_dir / "server.log"
    class_table = run_dir / "class-table.txt"
    try:
        trace, trace_log, trace_path = start_trace(run_dir)
        command = common.server_command(config, port)
        environment = common.controlled_environment(config)
        common.atomic_write_json(run_dir / "launch.json", {
            "argv": command, "cwd": str(common.LLAMA_ROOT),
            "environment_overrides": {
                key: value for key, value in environment.items()
                if key.startswith(("CUDA_", "GGML_", "GPUBPF_", "HF_HUB_", "TRANSFORMERS_"))
            },
        })
        server_log = server_path.open("x", buffering=1)
        server = subprocess.Popen(
            command, cwd=common.LLAMA_ROOT, env=environment,
            stdout=server_log, stderr=subprocess.STDOUT, text=True,
            start_new_session=True,
        )
        common.wait_ready(server, port, server_path)
        common.compile_layout(trace_path, class_table, run_dir / "layout-report.json")
        if config != "llama_ncmoe32":
            common.stop_group(trace)
            trace = None
            trace_log.close()
            trace_log = None
        if config == "gpubpf_observe":
            policy, policy_log, policy_path, policy_ready = common.start_policy(
                "observe", class_table, run_dir
            )
        elif config == "gpubpf_profile_protect":
            policy, policy_log, policy_path, policy_ready = common.start_policy(
                "protect", class_table, run_dir
            )
        else:
            policy_path = None
            policy_ready = None

        warmup = common.completion(
            port, prompts["records"][0]["prompt_token_ids"], run_dir / "warmup.json"
        )
        correctness_outputs = []
        for pass_number in (1, 2):
            current = []
            for sequence, prompt_number in enumerate(prompt_order, start=1):
                response = common.completion(
                    port, prompts["records"][prompt_number]["prompt_token_ids"],
                    run_dir / f"untimed-pass-{pass_number}-request-{sequence:02d}-prompt-{prompt_number}.json",
                )
                current.append(response["text"])
            correctness_outputs.append(current)

        route_diagnostic = None
        trace_final = None
        if config == "llama_ncmoe32":
            common.stop_group(trace)
            trace = None
            trace_log.close()
            trace_log = None
            trace_final = common.latest_event(trace_path, "final")
            if (int(trace_final["layouts"]) != 216 or int(trace_final["routes"]) <= 0 or
                    int(trace_final["dropped"]) != 0):
                raise common.GateError(f"context timing route trace failed: {trace_final}")
            summary = common.run_checked([
                "python3", str(common.COMPILE_HOT_SET), "--input", str(trace_path),
                "--output", str(run_dir / "route-diagnostic-hot-set.txt"),
                "--report", str(run_dir / "route-diagnostic-report.json"),
                "--expected-layers", "36", "--expected-experts", "128",
                "--expected-route-layers", "32", "--top-k", "10",
            ])
            route_diagnostic = json.loads(summary)
            common.validate_context_route_summary(route_diagnostic)

        time.sleep(1.1)
        if policy_path is not None:
            before_stats = common.latest_event(policy_path, "policy_stats")
            indices = common.hot_block_indices(class_table)
            before_blocks = common.request_block_snapshot(
                policy_ready, policy_path, indices, 1
            )
        else:
            before_stats = before_blocks = None

        telemetry, telemetry_log, telemetry_path = start_telemetry(run_dir)
        block_start_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        requests = []
        for sequence, prompt_number in enumerate(prompt_order, start=1):
            requests.append(common.completion(
                port, prompts["records"][prompt_number]["prompt_token_ids"],
                run_dir / f"measured-request-{sequence:02d}-prompt-{prompt_number}.json",
            ))
        block_end_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        common.stop_group(telemetry)
        telemetry = None
        telemetry_log.close()
        telemetry_log = None
        gpu_telemetry = validate_telemetry(telemetry_path)

        policy_delta = activation_metric = None
        if policy_path is not None:
            policy_delta, activation_metric = policy_metric(
                policy_path, policy_ready, class_table, before_stats, before_blocks
            )
            validate_policy_engagement(config, policy_delta)
        duration_s = (block_end_ns - block_start_ns) / 1e9
        result = {
            "config": config, "warmup": warmup,
            "untimed_repeated_matching_prompts": sum(
                first == second for first, second in zip(*correctness_outputs)
            ),
            "prompt_order": prompt_order,
            "requests": requests,
            "block_start_ns": block_start_ns, "block_end_ns": block_end_ns,
            "duration_s": duration_s, "verified_output_tokens": 512,
            "output_throughput_tokens_per_s": 512 / duration_s,
            "policy_delta": policy_delta,
            "hot_activation_metric": activation_metric,
            "gpu_telemetry": gpu_telemetry,
            "route_diagnostic": route_diagnostic, "trace_final": trace_final,
        }
        common.atomic_write_json(run_dir / "result.json", result)
        return result
    finally:
        common.stop_group(telemetry)
        common.stop_group(policy)
        common.stop_group(trace)
        common.stop_group(server, 120)
        for stream in (telemetry_log, policy_log, trace_log, server_log):
            if stream is not None:
                stream.close()
        if server_path.exists():
            common.validate_server_log(server_path)
        if common.struct_ops_maps():
            raise common.GateError("owned timing policy did not detach cleanly")


def run_block(block: int, port: int) -> dict[str, Any]:
    correctness = json.loads((HERE / "correctness-result.json").read_text())
    if correctness.get("status") != "passed":
        raise common.GateError("approved four-cell correctness result is not passed")
    schedule = json.loads(SCHEDULE.read_text())
    entries = [entry for entry in schedule["blocks"] if int(entry["block"]) == block]
    if len(entries) != 1:
        raise common.GateError(f"timing block {block} is outside the frozen schedule")
    entry = entries[0]
    order = [str(value) for value in entry["configuration_order"]]
    prompt_order = [int(value) for value in entry["prompt_order"]]
    if sorted(order) != sorted(common.CONFIGS) or sorted(prompt_order) != list(range(1, 9)):
        raise common.GateError(f"invalid frozen timing schedule entry: {entry}")
    output = RAW / f"block-{block:02d}"
    if output.exists():
        raise common.GateError(f"timing block output already exists: {output}")
    for previous in range(1, block):
        previous_status = RAW / f"block-{previous:02d}/status.json"
        if (not previous_status.is_file() or
                json.loads(previous_status.read_text()).get("status") != "passed"):
            raise common.GateError(f"previous timing block {previous} is not passed")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise common.GateError(f"port {port} is in use")
    output.mkdir(parents=True, exist_ok=False)
    common.atomic_write_json(output / "status.json", {
        "status": "running", "block": block, "started_ns": time.time_ns(),
        "configuration_order": order, "prompt_order": prompt_order,
    })
    custom_loaded = False
    results = []
    try:
        admitted = common.require_idle()
        common.atomic_write_json(output / "admission.json", admitted)
        common.load_custom_uvm()
        custom_loaded = True
        prompts = json.loads(common.PROMPTS.read_text())
        for config in order:
            results.append(run_timing_cell(
                config, output / config, port, prompts, prompt_order
            ))
        final = {
            "status": "passed", "block": block,
            "configuration_order": order, "prompt_order": prompt_order,
            "completed_ns": time.time_ns(),
            "throughput_tokens_per_s": {
                result["config"]: result["output_throughput_tokens_per_s"]
                for result in results
            },
        }
        common.atomic_write_json(output / "status.json", final)
        return final
    except BaseException as exc:
        common.atomic_write_json(output / "status.json", {
            "status": "failed", "block": block,
            "error_type": type(exc).__name__, "error": str(exc),
            "completed_ns": time.time_ns(),
        })
        raise
    finally:
        if custom_loaded:
            common.restore_stock_uvm()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("admit", "block"))
    parser.add_argument("--block", type=int, default=1)
    parser.add_argument("--port", type=int, default=18082)
    args = parser.parse_args()
    try:
        with common.Lease():
            if args.action == "admit":
                print(json.dumps(common.require_idle(), sort_keys=True))
            else:
                print(json.dumps(run_block(args.block, args.port), sort_keys=True))
        return 0
    except BaseException as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__,
                          "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
