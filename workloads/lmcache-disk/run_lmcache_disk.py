#!/usr/bin/env python3
"""Thin LMCache experiment adapter: one official vLLM cell at a time.

This module deliberately has no approval parser, promotion marker, resume
protocol, or attempt-budget controller. The experiment plan supplies ordinary
commands; this adapter starts one source-native server, retains raw output, and
recomputes validation/analysis from those outputs.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import importlib.util
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PRIMITIVES_PATH = HERE / "lmcache_primitives.py"
SPEC = importlib.util.spec_from_file_location("lmcache_adapter_primitives", PRIMITIVES_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load LMCache adapter primitives from {PRIMITIVES_PATH}")
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
KV_CHUNK_BYTES = legacy.KV_CHUNK_BYTES
CHUNKS_PER_PREFIX = legacy.CHUNKS_PER_PREFIX
TARGET_BLOCKS = legacy.TARGET_BLOCKS
BOOTSTRAP_SEED = legacy.BOOTSTRAP_SEED
MODEL_REVISION = legacy.MODEL_REVISION
EXPECTED_DRIVER = legacy.EXPECTED_DRIVER
EXPECTED_LMCACHE_VERSION = legacy.EXPECTED_LMCACHE_VERSION
EXPECTED_VLLM_VERSION = legacy.EXPECTED_VLLM_VERSION
shared = legacy.shared
safety = shared.safety
WORKER_CPUS = list(range(8, 16))

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


@contextmanager
def managed_cell(output: Path, expected_driver: str):
    """Use the existing leases and safety checks; never clean global GPU state."""
    with ExitStack() as cleanup:
        lease = shared.Leases()
        cleanup.callback(lease.close)
        output.mkdir(parents=True, exist_ok=False)
        record = {"status": "failed", "expected_driver": expected_driver,
                  "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                  "worker_cpu_affinity": WORKER_CPUS, "telemetry_cpu": safety.TELEMETRY_CPU,
                  "coordinator_cpu_affinity": sorted(os.sched_getaffinity(0)), "started_ns": time.time_ns()}
        # Shared safety diagnostics use this cell's bounded, owned subprocess helper.
        previous_checked = safety.run_checked
        safety.run_checked = legacy.run_checked
        cleanup.callback(setattr, safety, "run_checked", previous_checked)
        previous_signal = signal.getsignal(signal.SIGTERM)
        def interrupted(signum, _frame):
            raise InterruptedError(f"signal {signum}; cleaning only owned processes")
        signal.signal(signal.SIGTERM, interrupted)
        cleanup.callback(signal.signal, signal.SIGTERM, previous_signal)
        processes, streams, errors = [], [], []
        before = None
        finished = False
        try:
            if not (set(WORKER_CPUS) | {safety.TELEMETRY_CPU}) <= set(record["coordinator_cpu_affinity"]):
                raise GateError("coordinator needs CPUs 8-16 available; only the server is pinned to 8-15")
            if any(os.environ.get(key) for key in ("LD_PRELOAD", "LD_AUDIT", "CUDA_INJECTION64_PATH", "CUDA_INJECTION32_PATH")):
                raise GateError("launch the coordinator without ambient injection")
            before = safety.safety_snapshot()
            record["safety_before"] = before
            safety.validate_pre_server_safety(before)
            legacy.validate_driver(before["gpu"]["driver"], expected_driver)
            telemetry, stream, telemetry_path = safety.start_gpu_telemetry(output)
            processes.append(telemetry)
            streams.append(stream)
            kernel_stream = (output / "kernel-follow.log").open("x")
            streams.append(kernel_stream)
            kernel_command = ["/usr/bin/taskset", "-c", str(safety.TELEMETRY_CPU), "/usr/bin/journalctl",
                              "-k", "-b", "--follow", "--no-pager", "-n", "0", "-o", "short-monotonic"]
            journal = subprocess.Popen(kernel_command, stdout=kernel_stream, stderr=subprocess.STDOUT,
                                       env=legacy.controlled_environment(""), start_new_session=True)
            processes.append(journal)
            record["kernel_command"] = kernel_command
            record["monitors_started_ns"] = time.time_ns()
            yield record
            finished = True
        except BaseException as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record["monitors_alive_before_stop"] = [process.poll() is None for process in processes]
            if finished and record["monitors_alive_before_stop"] != [True, True]:
                errors.append("continuous GPU/kernel monitor exited before the cell finished")
            # run_config has already reaped the server group before returning/raising.
            for process in reversed(processes):
                try:
                    shared.stop_owned(process)
                except BaseException as error:
                    errors.append(str(error))
            for stream in streams:
                stream.close()
            try:
                if before is not None:
                    try:
                        record["safety_after"] = safety.wait_for_post_server_safety(before)
                    except BaseException:
                        record["safety_after"] = safety.safety_snapshot()
                        raise
                    legacy.validate_driver(record["safety_after"]["gpu"]["driver"], expected_driver)
                if processes:
                    record["telemetry"] = safety.validate_gpu_telemetry(telemetry_path, allow_fixed_power_cap=True)
                if (output / "kernel-follow.log").exists():
                    record["kernel_abnormal"] = safety.filtered_kernel_records((output / "kernel-follow.log").read_text())
                    if record["kernel_abnormal"]:
                        errors.append("kernel/GPU error recorded during the cell")
                record["final_boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
                if record["final_boot_id"] != record["boot_id"]:
                    errors.append("boot identity changed during the cell")
            except BaseException as error:
                errors.append(str(error))
            record.update(finished_ns=time.time_ns(), cleanup_errors=errors)
            if finished and not errors:
                record["status"] = "passed"
            legacy.atomic_write_json(output / "execution.json", record)
            if errors:
                raise GateError("; ".join(errors))


def inspect_environment(port: int, storage_root: Path,
                        expected_driver: str = EXPECTED_DRIVER) -> dict[str, Any]:
    """Return read-only launch observations; this is not a promotion marker."""
    return legacy.admission(port, require_model=True, storage_path=storage_root,
                            expected_driver=expected_driver)


def run_cell(config: str, output: Path, port: int, trace: bool,
             prefix_limit: int = PREFIXES,
             expected_driver: str = EXPECTED_DRIVER) -> dict[str, Any]:
    """Run one cell through the official vLLM serve path and retain raw output."""
    if config not in CONFIGS:
        raise GateError(f"unknown configuration: {config}")
    if not 1 <= prefix_limit <= PREFIXES:
        raise GateError(f"prefix limit must be between 1 and {PREFIXES}")
    with managed_cell(output, expected_driver) as execution:
        observations = inspect_environment(port, output, expected_driver)
        observations.update({key: execution[key] for key in ("boot_id", "worker_cpu_affinity", "telemetry_cpu")})
        prompts = load_prompts(PROMPTS)
        prompts = {**prompts, "prefixes": prompts["prefixes"][:prefix_limit]}
        result = legacy.run_config(
            config, output, prompts, port, Path(observations["model_path"]),
            trace=trace, recorded_environment=observations, expected_driver=expected_driver,
        )
    return result


def _validate_recorded_environment(value: dict[str, Any]) -> None:
    gpu = value.get("gpu", {})
    # Records predating explicit selection belong to the original 610 protocol.
    legacy.validate_driver(gpu.get("driver"), value.get("expected_driver", EXPECTED_DRIVER))
    if (gpu.get("driver") == "575.57.08" or "worker_cpu_affinity" in value) and (
        not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", value.get("boot_id", ""))
        or value.get("worker_cpu_affinity") != WORKER_CPUS
        or value.get("telemetry_cpu") != safety.TELEMETRY_CPU
    ):
        raise GateError("575 records require the actual boot and fixed worker/telemetry CPU placement")
    if gpu.get("driver") == "575.57.08":
        compiler = value.get("triton_ptxas", {})
        if (compiler.get("path") != str(legacy.TRITON_PTXAS_575)
                or not isinstance(compiler.get("bytes"), int) or compiler["bytes"] <= 0
                or not re.search(r"release 12\.9, V" + re.escape(legacy.TRITON_PTXAS_575_VERSION) + r"\b",
                                 compiler.get("version_output", ""))
                or value.get("server_environment", {}).get("TRITON_PTXAS_BLACKWELL_PATH")
                != str(legacy.TRITON_PTXAS_575)):
            raise GateError("575 records require the explicit CUDA 12.9.86 Triton compiler pin and inventory")
    runtime = value.get("runtime_imports", {})
    source = value.get("lmcache_source", {})
    if (
        gpu.get("compute_apps") != []
        or not isinstance(gpu.get("memory_used_mib"), int)
        or gpu["memory_used_mib"] > 256
        or source.get("commit") != legacy.LMCACHE_COMMIT
        or Path(source.get("path", "")) != legacy.LMCACHE_REPO
        or runtime.get("lmcache_version") != EXPECTED_LMCACHE_VERSION
        or runtime.get("vllm_version") != EXPECTED_VLLM_VERSION
    ):
        raise GateError("recorded source/runtime/GPU semantics differ from the fixed experiment")

    frozen = legacy.load_artifacts()
    modules = runtime.get("modules", {})
    paths = {name: item.get("path") for name, item in modules.items()}
    if paths != frozen.get("runtime_import_paths"):
        raise GateError("recorded runtime import paths differ from the fixed environment")
    if any(not isinstance(item.get("bytes"), int) or item["bytes"] <= 0 for item in modules.values()):
        raise GateError("recorded runtime import inventory has an invalid file size")
    dependency_lines = runtime.get("dependency_lines")
    expected_dependencies = (HERE / frozen["environment_freeze"]["relative_path"]).read_text().splitlines()
    if dependency_lines != expected_dependencies:
        raise GateError("recorded dependency set differs from the fixed environment")

    storage = value.get("storage", {})
    filesystems = storage.get("mount", {}).get("filesystems", [])
    if len(filesystems) != 1:
        raise GateError("recorded storage mount is missing or ambiguous")
    mount = filesystems[0]
    if (
        Path(mount.get("source", "")).resolve() != Path(legacy.EXPECTED_MOUNT_SOURCE).resolve()
        or mount.get("fstype") != "ext4"
        or storage.get("free_bytes", 0) < 100 * 1024**3
    ):
        raise GateError("recorded storage semantics differ from the fixed NVMe filesystem")

    model_path = Path(value.get("model_path", ""))
    if value.get("model_revision") != MODEL_REVISION or model_path.name != MODEL_REVISION:
        raise GateError("recorded model revision differs from the fixed model")
    model_files = value.get("model_artifacts", [])
    by_name = {item.get("name"): item for item in model_files}
    required = {"config.json", "model.safetensors.index.json"} | {
        f"model-{index:05d}-of-00007.safetensors" for index in range(1, 8)
    }
    if not required.issubset(by_name):
        raise GateError("recorded model inventory is incomplete")
    for name, item in by_name.items():
        if (
            not isinstance(name, str)
            or not isinstance(item.get("bytes"), int)
            or item["bytes"] <= 0
            or Path(item.get("path", "")).parent != model_path
        ):
            raise GateError("recorded model filename/size inventory is invalid")

    workload = value.get("workload_artifacts", {})
    expected_paths = {
        "dataset": legacy.DATASET.absolute(),
        "prompts": PROMPTS.absolute(),
        "schedule": SCHEDULE.absolute(),
    }
    if {
        name: Path(item.get("path", "")) for name, item in workload.items()
    } != expected_paths:
        raise GateError("recorded workload artifact paths differ from the fixed inputs")


def _validate_response(response: dict[str, Any], expected_tokens: int, request_id: str) -> None:
    if response.get("request_header") != request_id:
        raise GateError(f"request ID mismatch: expected {request_id}, got {response.get('request_header')}")
    engine_request_id = response.get("engine_request_id")
    if engine_request_id != f"cmpl-{request_id}":
        raise GateError(f"engine request ID mismatch for {request_id}: {engine_request_id}")
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


def _expected_store_state(
    config: str,
    prefix_index: int,
    request_evidence: dict[str, Any],
) -> dict[str, Any]:
    if config == "recompute":
        return {"files": 0, "bytes": 0, "durability": "not applicable"}
    if config == "lmcache_cpu":
        return {
            "files": 0,
            "bytes": 0,
            "durability": "synchronous LocalCPUBackend insertion",
            "request_log": request_evidence,
        }
    return {
        "files": (prefix_index + 1) * CHUNKS_PER_PREFIX,
        "bytes": (prefix_index + 1) * CHUNKS_PER_PREFIX * KV_CHUNK_BYTES,
        "per_file_bytes": [KV_CHUNK_BYTES],
        "durability": "fsync(each file) + fsync(directory)",
        "request_log": request_evidence,
    }


def validate_execution(run_dir: Path, environment: dict[str, Any]) -> dict[str, Any]:
    """Reparse actual safety evidence, independently of the server's exit/result."""
    path = run_dir / "execution.json"
    if not path.is_file() or not (run_dir / "kernel-follow.log").is_file():
        raise GateError("new cells require execution and continuous kernel records")
    record = json.loads(path.read_text())
    if (record.get("status") != "passed" or record.get("cleanup_errors") != []
            or record.get("monitors_alive_before_stop") != [True, True]
            or record.get("worker_cpu_affinity") != WORKER_CPUS
            or record.get("telemetry_cpu") != safety.TELEMETRY_CPU
            or not (set(WORKER_CPUS) | {safety.TELEMETRY_CPU}) <= set(record.get("coordinator_cpu_affinity", []))
            or record.get("boot_id") != environment.get("boot_id")
            or record.get("final_boot_id") != record.get("boot_id")):
        raise GateError("cell safety/monitor/CPU/boot evidence is incomplete")
    try:
        before, after = record["safety_before"], record["safety_after"]
        expected = environment["expected_driver"]
        if record["expected_driver"] != expected:
            raise GateError("safety driver declaration differs from environment")
        for snapshot in (before, after):
            legacy.validate_driver(snapshot["gpu"]["driver"], expected)
        if not (record["started_ns"] <= before["timestamp_ns"] <= record["monitors_started_ns"]
                <= environment["timestamp_ns"] <= after["timestamp_ns"] <= record["finished_ns"]):
            raise GateError("safety evidence does not enclose the cell launch")
        safety.validate_post_server_safety(before, after)
        telemetry = safety.validate_gpu_telemetry(run_dir / "gpu-telemetry.csv", allow_fixed_power_cap=True)
        if telemetry != record.get("telemetry"):
            raise GateError("saved GPU telemetry differs from raw samples")
        abnormal = safety.filtered_kernel_records((run_dir / "kernel-follow.log").read_text())
        if abnormal or record.get("kernel_abnormal") != abnormal:
            raise GateError("kernel error or inconsistent kernel evidence during cell")
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise GateError(f"invalid execution evidence: {error}") from error
    return record


def validate_cell(run_dir: Path, require_trace: bool = False) -> dict[str, Any]:
    """Reparse one cell's result, server log, and optional raw trace."""
    result_path = run_dir / "result.json"
    log_path = run_dir / "server.log"
    environment_path = run_dir / "environment.json"
    if not result_path.is_file() or not log_path.is_file() or not environment_path.is_file():
        raise GateError(f"missing raw cell output under {run_dir}")
    result = json.loads(result_path.read_text())
    recorded_environment = json.loads(environment_path.read_text())
    _validate_recorded_environment(recorded_environment)
    guarded = "worker_cpu_affinity" in recorded_environment
    if guarded:
        validate_execution(run_dir, recorded_environment)
        if result.get("worker_cpu_affinity") != WORKER_CPUS:
            raise GateError("actual server CPU affinity differs from 8-15")
    config = result.get("config")
    if result.get("schema") != 2 or config not in CONFIGS:
        raise GateError(f"invalid result schema/config under {run_dir}")
    prompts = load_prompts(PROMPTS)
    log = log_path.read_text(errors="replace")
    observations = result.get("observations")
    prefix_count = result.get("prefix_count", PREFIXES)
    if (not isinstance(prefix_count, int) or isinstance(prefix_count, bool)
            or not 1 <= prefix_count <= PREFIXES):
        raise GateError(f"invalid prefix_count under {run_dir}: {prefix_count}")
    if not isinstance(observations, list) or len(observations) != prefix_count:
        raise GateError(f"expected {prefix_count} observations under {run_dir}")
    for item, observation in zip(prompts["prefixes"][:prefix_count], observations, strict=True):
        index = item["index"]
        if (
            observation.get("prefix_index") != index
            or observation.get("expected_hit_tokens") != item["expected_hit_tokens"]
        ):
            raise GateError(f"prompt/observation mismatch for prefix {index}")
        _validate_response(observation["cold"], len(item["cold_token_ids"]), f"lmc-p{index}-cold")
        _validate_response(observation["warm"], len(item["warm_token_ids"]), f"lmc-p{index}-warm")
        cold_id = observation["cold"]["engine_request_id"]
        request_evidence = request_log_values(log, cold_id)
        expected_store_state = _expected_store_state(config, index, request_evidence)
        if observation.get("store_state") != expected_store_state:
            raise GateError(f"incremental persistence evidence mismatch for prefix {index}")

    command = result.get("command")
    if not isinstance(command, list) or len(command) < 4 or "--port" not in command:
        raise GateError(f"missing raw server command under {run_dir}")
    if Path(command[2]).name != MODEL_REVISION or command[2] != recorded_environment.get("model_path"):
        raise GateError(f"model path differs from the pinned recorded revision under {run_dir}")
    expected_command = server_argv(config, Path(command[2]), command[command.index("--port") + 1])
    if result.get("command") != expected_command:
        raise GateError(f"server command differs from fixed cell configuration: {run_dir}")
    cache_dir = (run_dir / "cache").resolve()
    expected_environment = server_environment(
        config, cache_dir, recorded_environment.get("expected_driver", EXPECTED_DRIVER))
    if result.get("environment") != expected_environment:
        raise GateError(f"server environment differs from fixed cell configuration: {run_dir}")
    if ("server_environment" in recorded_environment
            and recorded_environment["server_environment"] != expected_environment):
        raise GateError(f"pre-launch server environment differs from the result: {run_dir}")

    engagement = validate_log(config, log, observations, cache_dir)
    if result.get("engagement") != engagement:
        raise GateError(f"saved and recomputed engagement differ under {run_dir}")
    footprint = result.get("cache_footprint", {})
    if config == "lmcache_disk":
        expected_files = prefix_count * CHUNKS_PER_PREFIX
        expected_bytes = expected_files * KV_CHUNK_BYTES
        if footprint != {"files": expected_files, "bytes": expected_bytes}:
            raise GateError(f"disk footprint mismatch under {run_dir}: {footprint}")
    elif footprint != {"files": 0, "bytes": 0}:
        raise GateError(f"non-disk cell wrote disk chunks under {run_dir}: {footprint}")

    trace = None
    trace_dir = run_dir / "strace"
    if require_trace:
        if config != "lmcache_disk" or not trace_dir.is_dir():
            raise GateError("trace validation requires a traced lmcache_disk cell")
        trace = validate_odirect(trace_dir, cache_dir, prefix_count)
    elif trace_dir.exists():
        trace = validate_odirect(trace_dir, cache_dir, prefix_count)
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
    if guarded:
        expected_launch = ["/usr/bin/taskset", "-c", "8-15", *expected_launch]
    if launch != expected_launch:
        raise GateError(f"raw launch command mismatch under {run_dir}")

    warm_phase = result.get("warm_phase", {})
    elapsed = warm_phase.get("elapsed_s")
    if (
        warm_phase.get("sequential") is not True
        or warm_phase.get("requests") != prefix_count
        or warm_phase.get("output_tokens") != prefix_count * OUTPUT_TOKENS
        or warm_phase.get("excludes")
        != ["server startup", "cold population", "persistence barriers", "shutdown"]
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed <= 0
        or not math.isclose(warm_phase.get("requests_per_s", -1), prefix_count / elapsed, rel_tol=1e-12)
        or not math.isclose(
            warm_phase.get("output_tokens_per_s", -1), prefix_count * OUTPUT_TOKENS / elapsed,
            rel_tol=1e-12,
        )
    ):
        raise GateError(f"warm-phase metric derivation mismatch under {run_dir}")
    return {"result": result, "engagement": engagement, "trace": trace,
            "environment": recorded_environment}


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


def _validate_execution_sequence(
    root: Path,
    groups: dict[int, dict[str, tuple[int, Path]]],
) -> tuple[list[int], list[tuple[int, int, int]]]:
    attempt_dirs: dict[int, Path] = {}
    for path in sorted(root.glob("attempt-*")):
        match = re.fullmatch(r"attempt-(\d{2})", path.name)
        if not match or not path.is_dir():
            raise GateError(f"malformed attempt path: {path}")
        attempt_dirs[int(match.group(1))] = path
    if not attempt_dirs:
        raise GateError("analysis found no attempt directories")
    observed_attempts = sorted(attempt_dirs)
    if observed_attempts != list(range(observed_attempts[-1] + 1)):
        raise GateError(f"attempt history has a gap: {observed_attempts}")
    if observed_attempts[-1] >= legacy.MAX_ATTEMPTS:
        raise GateError(f"attempt history exceeds the fixed schedule: {observed_attempts[-1]}")

    timeline: list[tuple[int, int, int]] = []
    for attempt, attempt_dir in attempt_dirs.items():
        position_dirs = sorted(attempt_dir.glob("position-*"))
        timestamps = []
        for path in position_dirs:
            match = re.fullmatch(r"position-([0-2])-(recompute|lmcache_cpu|lmcache_disk)", path.name)
            if not match or not path.is_dir():
                raise GateError(f"malformed cell path: {path}")
            environment_path = path / "environment.json"
            if not environment_path.is_file():
                raise GateError(f"attempted cell lacks preserved environment observations: {path}")
            timestamp = json.loads(environment_path.read_text()).get("timestamp_ns")
            if not isinstance(timestamp, int) or timestamp <= 0:
                raise GateError(f"cell lacks a valid launch-observation timestamp: {path}")
            position = int(match.group(1))
            timeline.append((attempt, position, timestamp))
            timestamps.append((position, timestamp))
        if not timestamps:
            raise GateError(f"attempt directory has no preserved cell evidence: {attempt_dir}")
        if [position for position, _ in timestamps] != list(range(len(timestamps))):
            raise GateError(f"attempt {attempt} did not execute a position prefix in order")
        if any(left[1] >= right[1] for left, right in zip(timestamps, timestamps[1:])):
            raise GateError(f"attempt {attempt} timestamps contradict position order")
        if set(groups.get(attempt, {})) != set(CONFIGS):
            failure = attempt_dir / "failure.md"
            if not failure.is_file() or not failure.read_text(errors="replace").strip():
                raise GateError(f"incomplete attempt lacks an ordinary failure record: {attempt_dir}")

    by_time = [(attempt, position) for attempt, position, _ in sorted(timeline, key=lambda item: item[2])]
    by_schedule = [(attempt, position) for attempt, position, _ in timeline]
    if by_time != by_schedule:
        raise GateError("recorded timestamps contradict global attempt/position execution order")
    return observed_attempts, timeline


def analyze(root: Path) -> dict[str, Any]:
    """Revalidate raw cells, enforce the fixed schedule, and compute paired effects."""
    schedule = json.loads(SCHEDULE.read_text())
    validate_schedule(schedule)
    scheduled = {item["attempt"]: item["order"] for item in schedule["attempts"]}
    groups = _attempt_cells(root)
    observed_attempts, _ = _validate_execution_sequence(root, groups)
    complete = []
    reference_outputs = None
    comparison_driver = None
    comparison_boot = None
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
        checked = {config: validate_cell(cells[config][1]) for config in CONFIGS}
        validated = {config: value["result"] for config, value in checked.items()}
        if any(value.get("prefix_count", PREFIXES) != PREFIXES for value in validated.values()):
            raise GateError("formal analysis requires all eight prefixes; smoke cells are excluded")
        drivers = {value["environment"]["gpu"]["driver"] for value in checked.values()}
        if len(drivers) != 1 or (comparison_driver is not None and drivers != {comparison_driver}):
            raise GateError("formal analysis cannot mix driver versions within or across blocks")
        comparison_driver = next(iter(drivers))
        boots = {value["environment"].get("boot_id") for value in checked.values()}
        if len(boots) != 1 or (comparison_boot is not None and boots != {comparison_boot}):
            raise GateError("formal analysis cannot mix boot identities within or across blocks")
        comparison_boot = next(iter(boots))
        for config, result in validated.items():
            outputs = legacy.output_texts(result)
            if reference_outputs is None:
                reference_outputs = outputs
            elif outputs != reference_outputs:
                raise GateError(f"exact output text drift in attempt {attempt}, config {config}")
        complete.append((attempt, validated))
    if len(complete) != TARGET_BLOCKS:
        raise GateError(f"analysis requires {TARGET_BLOCKS} complete attempts, found {len(complete)}")
    if complete[-1][0] != observed_attempts[-1]:
        raise GateError("attempts continued after the tenth complete attempt")

    position_counts = {
        config: [sum(groups[attempt][config][0] == position for attempt, _ in complete)
                 for position in range(len(CONFIGS))]
        for config in CONFIGS
    }
    if any(max(counts) - min(counts) > 1 for counts in position_counts.values()):
        raise GateError(f"actual complete attempts are position-imbalanced: {position_counts}")

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
        "driver": comparison_driver,
        "boot_id": comparison_boot,
        "complete_attempts": [attempt for attempt, _ in complete],
        "observed_attempts": observed_attempts,
        "actual_position_counts": position_counts,
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
    inspect.add_argument("--expected-driver", choices=legacy.EXPERIMENT_DRIVERS,
                         default=EXPECTED_DRIVER)
    prepare = sub.add_parser("prepare-prompts")
    prepare.add_argument("--output", type=Path, default=PROMPTS)
    cell = sub.add_parser("run-cell")
    cell.add_argument("--config", choices=CONFIGS, required=True)
    cell.add_argument("--output", type=Path, required=True)
    cell.add_argument("--port", type=int, default=18080)
    cell.add_argument("--trace", action="store_true")
    cell.add_argument("--prefix-limit", type=int, default=PREFIXES)
    cell.add_argument("--expected-driver", choices=legacy.EXPERIMENT_DRIVERS,
                      default=EXPECTED_DRIVER)
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
            value = inspect_environment(args.port, args.storage_root, args.expected_driver)
        elif args.command == "prepare-prompts":
            value = prepare_prompts(args.output)
        elif args.command == "run-cell":
            value = run_cell(args.config, args.output, args.port, args.trace,
                             args.prefix_limit, args.expected_driver)
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
