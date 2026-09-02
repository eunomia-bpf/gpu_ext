#!/usr/bin/env python3
"""Audit and safely replay the historical Figure 6 llama-bench protocol.

``audit`` is read-only and explains which old results are already supported by
surviving evidence. ``run`` executes selected current-stack replay cells one at
a time. It acquires the same exclusive leases and applies the same fail-closed
host checks as the MoE head-to-head runner; it never kills an unowned process.

This runner deliberately distinguishes a protocol replay from an exact
reproduction. The old UVM variants were not preserved as separately buildable
source states, so current UVM results must not be relabelled as the old plain
UVM or user-hint cells.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
GPU_EXT = HERE.parents[1]
MOE_ROOT = GPU_EXT / "workloads/moe-infinity"
sys.path.insert(0, str(MOE_ROOT))

import run_moe_head_to_head as safety  # noqa: E402
from run_575_head_to_head import admit as current_stack_admission  # noqa: E402


REFERENCE_PATH = HERE / "legacy_exp1_reference.json"
DEFAULT_BENCH = HERE / "build/bin/llama-bench"
DEFAULT_MODEL = safety.GGUF_MODEL
ARCHIVED_CONTROL_LOGS = {
    "ncmoe64": HERE / "results/exp1_config1_ncmoe64.log",
    "ncmoe32": HERE / "results/exp1_config2_ncmoe32.log",
    "uvm_historical_attempt": HERE / "results/exp1_config3_uvm_baseline.log",
}


@dataclass(frozen=True)
class ReplayConfig:
    name: str
    n_cpu_moe: int
    unified_memory: bool
    attach_policy: bool
    comparable_reference: str | None
    equivalence: str


REPLAY_CONFIGS = {
    "ncmoe64": ReplayConfig(
        "ncmoe64", 64, False, False, "ncmoe64", "same benchmark semantics"
    ),
    "ncmoe32": ReplayConfig(
        "ncmoe32", 32, False, False, "ncmoe32", "same benchmark semantics"
    ),
    "uvm_current": ReplayConfig(
        "uvm_current", 0, True, False, None,
        "current adaptive UVM implementation; not equivalent to either old UVM cell",
    ),
    "gpubpf_current": ReplayConfig(
        "gpubpf_current", 0, True, True, None,
        "current combined stride plus sampled-LFU policy; not the archived policy binary",
    ),
}


class ReplayError(RuntimeError):
    pass


def load_reference() -> dict[str, Any]:
    value = json.loads(REFERENCE_PATH.read_text())
    if value.get("schema") != 1:
        raise ReplayError(f"unsupported reference schema: {value.get('schema')}")
    return value


def parse_markdown_bench(text: str) -> dict[str, Any]:
    build = re.search(r"^build:\s+(\S+)\s+\((\d+)\)\s*$", text, re.MULTILINE)
    rows: dict[str, dict[str, float]] = {}
    for test in ("pp512", "tg128"):
        match = re.search(
            rf"\|[^\n|]+\|[^\n|]+\|[^\n|]+\|[^\n|]+\|[^\n|]+\|\s*{test}\s*"
            rf"\|\s*([0-9]+(?:\.[0-9]+)?)\s*(?:±\s*([0-9]+(?:\.[0-9]+)?))?\s*\|",
            text,
        )
        if match:
            rows[test] = {
                "tokens_per_second": float(match.group(1)),
                "stddev_tokens_per_second": (
                    float(match.group(2)) if match.group(2) is not None else 0.0
                ),
            }
    return {
        "build_commit": build.group(1) if build else None,
        "build_number": int(build.group(2)) if build else None,
        "tests": rows,
    }


def percent_delta(observed: float, reference: float) -> float:
    return 100.0 * (observed / reference - 1.0)


def audit_archived_controls(reference: dict[str, Any]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, path in ARCHIVED_CONTROL_LOGS.items():
        if not path.is_file():
            records[name] = {"path": str(path), "available": False}
            continue
        parsed = parse_markdown_bench(path.read_text(errors="replace"))
        record: dict[str, Any] = {
            "path": str(path),
            "available": True,
            "parsed": parsed,
        }
        reference_name = name if name in {"ncmoe64", "ncmoe32"} else None
        if reference_name:
            expected = reference["reported_results_tokens_per_second"][reference_name]
            record["delta_percent"] = {
                test: percent_delta(parsed["tests"][test]["tokens_per_second"], value)
                for test, value in expected.items()
                if test in parsed["tests"]
            }
            record["status"] = (
                "within_5_percent"
                if set(record["delta_percent"]) == {"pp512", "tg128"}
                and all(abs(value) <= 5.0 for value in record["delta_percent"].values())
                else "outside_5_percent"
            )
        else:
            record["status"] = "not_comparable_to_reported_plain_uvm"
        records[name] = record
    return records


def admission_record(value: dict[str, Any]) -> dict[str, Any]:
    """Keep auditable facts without persisting runtime identity metadata."""
    models = value.get("models", {})
    gguf = Path(models["gguf"]) if models.get("gguf") else None
    return {
        "admitted": bool(value.get("admitted")),
        "errors": value.get("errors", []),
        "gpu": value.get("gpu"),
        "struct_ops": value.get("struct_ops"),
        "loaded_uvm_interface": value.get("loaded_uvm_interface"),
        "mount": value.get("mount"),
        "llama_source": value.get("llama_source"),
        "moe_source": value.get("moe_source"),
        "gguf_model": (
            {"path": str(gguf), "size": gguf.stat().st_size}
            if gguf is not None and gguf.is_file() else None
        ),
    }


def audit(port: int) -> dict[str, Any]:
    reference = load_reference()
    admission = current_stack_admission(port)
    archived = audit_archived_controls(reference)
    reproduced_controls = [
        name for name in ("ncmoe64", "ncmoe32")
        if archived.get(name, {}).get("status") == "within_5_percent"
    ]
    return {
        "schema": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reference": reference,
        "current_stack_admission": admission_record(admission),
        "archived_current_host_controls": archived,
        "assessment": {
            "basic_current_workflow_ready": bool(admission["admitted"]),
            "old_cpu_offload_controls_reproduced_within_5_percent": reproduced_controls,
            "exact_five_cell_reproduction_ready": False,
            "current_replay_cells": list(REPLAY_CONFIGS),
            "blocking_gaps": reference["evidence_limits"],
        },
    }


def controlled_environment(config: ReplayConfig) -> dict[str, str]:
    env = safety.controlled_environment(
        "gpubpf_host_stride_lfu" if config.unified_memory else "llama_ncmoe32"
    )
    if not config.unified_memory:
        env.pop("GGML_CUDA_ENABLE_UNIFIED_MEMORY", None)
    return env


def benchmark_command(
    bench: Path, model: Path, config: ReplayConfig, repetitions: int
) -> list[str]:
    command = [
        str(bench), "--model", str(model),
        "--repetitions", str(repetitions), "--n-prompt", "512",
        "--n-gen", "128", "--output", "json",
    ]
    if config.n_cpu_moe:
        command.extend(["--n-cpu-moe", str(config.n_cpu_moe)])
    return command


def parse_json_result(path: Path, config: ReplayConfig, repetitions: int, *,
                      expected_model: Path | None = None) -> dict[str, Any]:
    try:
        rows = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayError(f"llama-bench did not produce valid JSON: {path}") from exc
    if not isinstance(rows, list) or len(rows) != 2:
        raise ReplayError(f"expected exactly pp512 and tg128 rows, found {rows!r}")
    tests: dict[str, Any] = {}
    identities = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ReplayError(f"benchmark row is not an object: {row!r}")
        n_prompt = int(row.get("n_prompt", -1))
        n_gen = int(row.get("n_gen", -1))
        if (n_prompt, n_gen) == (512, 0):
            name = "pp512"
        elif (n_prompt, n_gen) == (0, 128):
            name = "tg128"
        else:
            raise ReplayError(f"unexpected benchmark dimensions: {(n_prompt, n_gen)}")
        samples = row.get("samples_ts")
        if not isinstance(samples, list) or len(samples) != repetitions:
            raise ReplayError(
                f"{name} has {len(samples) if isinstance(samples, list) else 'invalid'} "
                f"samples, expected {repetitions}"
            )
        if int(row.get("n_cpu_moe", -1)) != config.n_cpu_moe:
            raise ReplayError(f"{name} n_cpu_moe disagrees with requested config")
        if "CUDA" not in str(row.get("backends", "")):
            raise ReplayError(f"{name} did not use the CUDA backend")
        if expected_model is not None:
            commit = str(row.get("build_commit", ""))
            if len(commit) < 8 or not safety.EXPECTED_LLAMA_COMMIT.startswith(commit):
                raise ReplayError(f"{name} build differs from the admitted llama source: {commit}")
            if row.get("model_filename") != str(expected_model):
                raise ReplayError(f"{name} model differs from the requested admitted model")
            if safety.EXPECTED_GPU not in str(row.get("gpu_info", "")):
                raise ReplayError(f"{name} GPU differs from the admitted RTX 5090")
        identities.add((row.get("build_commit"), row.get("build_number"), row.get("gpu_info")))
        tests[name] = {
            "tokens_per_second": float(row["avg_ts"]),
            "stddev_tokens_per_second": float(row["stddev_ts"]),
            "samples_tokens_per_second": [float(value) for value in samples],
        }
    if set(tests) != {"pp512", "tg128"} or len(identities) != 1:
        raise ReplayError(f"benchmark identity or row set changed: rows={rows!r}")
    build_commit, build_number, gpu_info = identities.pop()
    return {
        "build_commit": build_commit,
        "build_number": build_number,
        "gpu_info": gpu_info,
        "tests": tests,
    }


def latest_json_event(path: Path, event: str) -> dict[str, Any]:
    matches = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("event") == event:
            matches.append(value)
    if not matches:
        raise ReplayError(f"{path} has no {event!r} event")
    return matches[-1]


def run_cell(
    config: ReplayConfig,
    cell_dir: Path,
    bench: Path,
    model: Path,
    repetitions: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    cell_dir.mkdir(parents=True, exist_ok=False)
    before = safety.safety_snapshot()
    safety.validate_pre_server_safety(before)
    policy = telemetry = benchmark = None
    policy_log = telemetry_log = stdout_log = stderr_log = None
    telemetry_path = None
    parsed = None
    execution_error: Exception | None = None
    cleanup_errors: list[str] = []
    command = benchmark_command(bench, model, config, repetitions)
    started_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    try:
        if config.attach_policy:
            policy, policy_log, policy_ready = safety.start_policy(cell_dir)
        else:
            policy_ready = None
        telemetry, telemetry_log, telemetry_path = safety.start_gpu_telemetry(cell_dir)
        stdout_path = cell_dir / "llama-bench.json"
        stderr_path = cell_dir / "llama-bench.stderr.log"
        stdout_log = stdout_path.open("x")
        stderr_log = stderr_path.open("x")
        safety.atomic_write_json(cell_dir / "launch.json", {
            "argv": command,
            "environment": controlled_environment(config),
            "equivalence": config.equivalence,
            "policy_ready": policy_ready,
            "timeout_seconds": timeout_seconds,
        })
        benchmark = subprocess.Popen(
            command,
            cwd=HERE,
            env=controlled_environment(config),
            stdout=stdout_log,
            stderr=stderr_log,
            text=True,
            start_new_session=True,
        )
        try:
            returncode = benchmark.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            safety.stop_owned_process_group(benchmark)
            raise ReplayError(
                f"{config.name} exceeded its {timeout_seconds}-second hard timeout"
            ) from exc
        stdout_log.close()
        stderr_log.close()
        stdout_log = stderr_log = None
        if returncode:
            tail = stderr_path.read_text(errors="replace")[-4000:]
            raise ReplayError(f"{config.name} exited with {returncode}:\n{tail}")
        safety.validate_log(stderr_path)
        stderr_text = stderr_path.read_text(errors="replace")
        if config.unified_memory and "[UVM] Unified Memory ENABLED" not in stderr_text:
            raise ReplayError("current UVM replay did not engage managed allocation")
        if not config.unified_memory and "[UVM] Unified Memory ENABLED" in stderr_text:
            raise ReplayError("CPU-offload control unexpectedly enabled UVM")
        parsed = parse_json_result(stdout_path, config, repetitions, expected_model=model)
    except Exception as exc:
        execution_error = exc
    finally:
        if benchmark is not None:
            try:
                safety.stop_owned_process_group(benchmark)
            except Exception as exc:
                cleanup_errors.append(f"benchmark process group: {exc}")
        if stdout_log is not None:
            try:
                stdout_log.close()
            except Exception as exc:
                cleanup_errors.append(f"benchmark stdout: {exc}")
        if stderr_log is not None:
            try:
                stderr_log.close()
            except Exception as exc:
                cleanup_errors.append(f"benchmark stderr: {exc}")
        if telemetry is not None:
            try:
                safety.stop_exact_process(telemetry)
            except Exception as exc:
                cleanup_errors.append(f"telemetry process: {exc}")
        if telemetry_log is not None:
            try:
                telemetry_log.close()
            except Exception as exc:
                cleanup_errors.append(f"telemetry log: {exc}")
        if policy is not None:
            try:
                safety.stop_exact_process(policy)
            except Exception as exc:
                cleanup_errors.append(f"policy process: {exc}")
        if policy_log is not None:
            try:
                policy_log.close()
            except Exception as exc:
                cleanup_errors.append(f"policy log: {exc}")
    # The post-run safety check is mandatory even when policy admission,
    # telemetry, benchmark launch, or result validation failed.
    try:
        after = safety.wait_for_post_server_safety(before)
    except Exception as post_error:
        preceding = []
        if execution_error is not None:
            preceding.append(f"execution={execution_error}")
        if cleanup_errors:
            preceding.append(f"cleanup={cleanup_errors}")
        context = f"; preceding failures: {', '.join(preceding)}" if preceding else ""
        raise ReplayError(f"post-run safety gate failed: {post_error}{context}") from post_error
    if cleanup_errors:
        raise ReplayError(f"owned cleanup failed: {cleanup_errors}")
    if execution_error is not None:
        raise execution_error
    if telemetry_path is None:
        raise ReplayError("GPU telemetry did not start")
    telemetry_summary = safety.validate_gpu_telemetry(telemetry_path, allow_fixed_power_cap=True)
    if parsed is None:
        raise ReplayError("benchmark result validation did not complete")
    if config.attach_policy:
        policy_final = latest_json_event(cell_dir / "policy.jsonl", "final_engagement")
        required = ("page_fault_calls", "prefetches_issued", "lfu_accesses", "eviction_prepares")
        missing = [name for name in required if int(policy_final.get(name, 0)) <= 0]
        if missing:
            raise ReplayError(f"gpubpf policy did not engage required paths: {missing}")
    else:
        policy_final = None
    ended_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    result = {
        "config": config.name,
        "equivalence": config.equivalence,
        "exact_reproduction": False,
        "reference_comparison_scope": (
            "same llama-bench dimensions and options on the current source/driver stack"
            if config.comparable_reference else None
        ),
        "comparable_reference": config.comparable_reference,
        "duration_seconds": (ended_ns - started_ns) / 1e9,
        "benchmark": parsed,
        "gpu_telemetry": telemetry_summary,
        "policy_final_engagement": policy_final,
        "safety_before": before,
        "safety_after": after,
    }
    if config.comparable_reference:
        expected = load_reference()["reported_results_tokens_per_second"][config.comparable_reference]
        result["delta_percent"] = {
            name: percent_delta(parsed["tests"][name]["tokens_per_second"], value)
            for name, value in expected.items()
        }
    safety.atomic_write_json(cell_dir / "result.json", result)
    return result


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    names = [name.strip() for name in args.configs.split(",") if name.strip()]
    unknown = sorted(set(names) - set(REPLAY_CONFIGS))
    if not names or unknown or len(names) != len(set(names)):
        raise ReplayError(
            f"configs must be a non-empty unique subset of {sorted(REPLAY_CONFIGS)}; "
            f"unknown={unknown}"
        )
    bench = args.bench.resolve()
    # Preserve the named upstream snapshot path. Resolving the model symlink
    # would expose an internal object-store filename that is not experiment
    # evidence and must not become part of the record.
    model = args.model.absolute()
    if not bench.is_file() or not os.access(bench, os.X_OK):
        raise ReplayError(f"llama-bench is missing or not executable: {bench}")
    if not model.is_file():
        raise ReplayError(f"model is missing: {model}")
    if not model.samefile(DEFAULT_MODEL):
        raise ReplayError("historical replay requires the admitted GPT-OSS-120B MXFP4 model")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    lease = safety.LeaseSet.acquire()
    result: dict[str, Any] = {
        "schema": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "requested_configs": names,
        "repetitions": args.repetitions,
        "timeout_seconds_per_cell": args.timeout_seconds,
        "results": {},
    }
    try:
        for index, name in enumerate(names):
            admitted = current_stack_admission(args.port)
            if not admitted["admitted"]:
                raise ReplayError("admission refused:\n- " + "\n- ".join(admitted["errors"]))
            safety.atomic_write_json(
                output / f"admission-{index + 1:02d}-{name}.json",
                admission_record(admitted),
            )
            result["results"][name] = run_cell(
                REPLAY_CONFIGS[name], output / name, bench, model,
                args.repetitions, args.timeout_seconds,
            )
            if index + 1 < len(names):
                time.sleep(args.cooldown_seconds)
        result["status"] = "passed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        if isinstance(exc, (ReplayError, safety.GateError)):
            raise
        raise ReplayError(str(exc)) from exc
    finally:
        safety.atomic_write_json(output / "run-result.json", result)
        lease.close()
    return result


def bounded_integer(minimum: int, maximum: int):
    def parse(value: str) -> int:
        try:
            result = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
        if not minimum <= result <= maximum:
            raise argparse.ArgumentTypeError(
                f"expected a value from {minimum} through {maximum}, got {result}"
            )
        return result

    return parse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    audit_parser = subparsers.add_parser("audit", help="read-only readiness and evidence audit")
    audit_parser.add_argument("--port", type=int, default=18080)
    audit_parser.add_argument("--output", type=Path)
    run_parser = subparsers.add_parser("run", help="run selected current-stack replay cells")
    run_parser.add_argument(
        "--configs", default="ncmoe64",
        help="comma-separated replay cells; default: ncmoe64",
    )
    run_parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    run_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    run_parser.add_argument("--repetitions", type=bounded_integer(1, 10), default=1)
    run_parser.add_argument("--timeout-seconds", type=bounded_integer(60, 900), default=180)
    run_parser.add_argument("--cooldown-seconds", type=bounded_integer(0, 300), default=60)
    run_parser.add_argument("--port", type=int, default=18080)
    run_parser.add_argument(
        "--output", type=Path,
        default=HERE / "results/exp1_reproduction" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.action == "audit":
            result = audit(args.port)
            if args.output:
                safety.atomic_write_json(args.output.resolve(), result)
        else:
            result = run_matrix(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ReplayError, safety.GateError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
