#!/usr/bin/env python3
"""Current-stack continuation of the four-cell deployment comparison.

Reuses the original request, telemetry, policy ownership, and analysis code.
Historical 610 attempts are never admitted as current correctness or timing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import time
from typing import Any

import run_moe_head_to_head as base

PROTOCOL = "proposal-3-revision-8-575"
DRIVER = "575.57.08"
KERNEL = "6.15.11-061511-generic"
MODULES = Path("/opt/gpubpf/modules") / DRIVER / KERNEL
DEFAULT_OUTPUT = base.HERE / "raw/head-to-head-575"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def admit(port: int) -> dict[str, Any]:
    result = base.admission(port, driver_version=DRIVER, module_root=MODULES,
                            kernel_release=KERNEL, minimum_free_bytes=80 * 1024**3)
    result["kernel"] = os.uname().release
    if result["kernel"] != KERNEL:
        result["errors"].append(f"kernel must be {KERNEL}")
    try:
        result["safety"] = base.safety_snapshot()
        base.validate_pre_server_safety(result["safety"])
    except Exception as exc:
        result["errors"].append(str(exc))
    result["admitted"] = not result["errors"]
    return result


def require_admission(port: int, runtime: dict[str, Any] | None = None) -> dict[str, Any]:
    observed = admit(port)
    if not observed["admitted"]:
        raise base.GateError("575 admission refused: " + "; ".join(observed["errors"]))
    if runtime is not None:
        base.require_runtime_continuity(runtime, observed["runtime_files"])
    return observed


def emit(message: str) -> None:
    print(json.dumps({"time_ns": time.time_ns(), "progress": message}), flush=True)


def run_cell(config: str, directory: Path, port: int, prompts: dict[str, Any],
             store: Path, runtime: dict[str, Any], *,
             prompt_order: list[int] | None = None,
             goldens: list[str] | None = None) -> dict[str, Any]:
    admitted = require_admission(port, runtime)
    safety_before = admitted["safety"]
    emit(f"starting {'correctness' if prompt_order is None else 'timing'} {config}")
    result = None
    error = None
    try:
        if prompt_order is None:
            result = base.run_correctness_config(config, directory, port, prompts,
                                                  current_deployment=True, offload_dir=store)
        else:
            result = base.run_measured_config(config, directory, port, prompts,
                                               prompt_order, goldens,
                                               current_deployment=True, offload_dir=store)
        return result
    except BaseException as exc:
        error = str(exc)
        raise
    finally:
        after = base.wait_for_post_server_safety(safety_before)
        base.atomic_write_json(directory / "safety.json", {
            "before": safety_before, "after": after, "passed": True,
        })
        if error is not None:
            base.atomic_write_json(directory / "failure.json", {"error": error})
        emit(f"finished {config}: {'passed' if result is not None else 'failed'}")


def preflight(output: Path, port: int) -> dict[str, Any]:
    admitted = require_admission(port)
    output.mkdir(parents=True, exist_ok=False)
    base.atomic_write_json(output / "admission.json", admitted)
    runtime = admitted["runtime_files"]
    result: dict[str, Any] = {
        "protocol": PROTOCOL, "driver": DRIVER, "kernel": KERNEL,
        "source_commit": base.run_checked(["git", "rev-parse", "HEAD"], cwd=base.GPU_EXT),
        "status": "running", "runtime_files": runtime,
        "configuration_order": list(base.FROZEN_CORRECTNESS_ORDER), "results": {},
        "cross_configuration_output_equality_required": False,
        "completed_evictions_claimed": False,
        "moe_storage": "buffered NVMe hydration followed by CPU expert offload/cache",
    }
    base.atomic_write_json(output / "preflight-result.json", result)
    try:
        before = admitted["safety"]
        result["row_chunking_numerical_gate"] = base.run_row_chunking_numerical_gate()
        result["numerical_safety"] = base.wait_for_post_server_safety(before)
        prompts = read_json(base.PROMPTS)
        for config in base.FROZEN_CORRECTNESS_ORDER:
            result["results"][config] = run_cell(
                config, output / config, port, prompts, output / "expert-store", runtime,
            )
            base.atomic_write_json(output / "preflight-result.json", result)
            time.sleep(60)
        result["status"] = "passed"
    except BaseException as exc:
        result.update(status="failed", error=str(exc))
        raise
    finally:
        base.atomic_write_json(output / "preflight-result.json", result)
    return result


def load_preflight(path: Path) -> dict[str, Any]:
    result = read_json(path / "preflight-result.json")
    if (result.get("protocol") != PROTOCOL or result.get("status") != "passed"
            or result.get("driver") != DRIVER or result.get("kernel") != KERNEL
            or set(result.get("results", {})) != set(base.CONFIGS)
            or not result.get("row_chunking_numerical_gate")):
        raise base.GateError("a complete passing four-cell 575 correctness preflight is required")
    for config in base.CONFIGS:
        passes = result["results"][config].get("passes", [])
        if (len(passes) != 2 or any(len(part) != 8 for part in passes)
                or any(a["text"] != b["text"] for a, b in zip(*passes))):
            raise base.GateError(f"incomplete exact two-pass correctness: {config}")
    return result


def full_schedule(output: Path, preflight_path: Path, port: int,
                  max_blocks: int) -> dict[str, Any]:
    checked = load_preflight(preflight_path)
    runtime = checked["runtime_files"]
    admitted = require_admission(port, runtime)
    output.mkdir(parents=True, exist_ok=True)
    session_file = output / "session.json"
    session = {"protocol": PROTOCOL, "preflight": str(preflight_path.absolute()),
               "runtime_files": runtime, "target_valid_blocks": 5,
               "source_commit": checked["source_commit"],
               "maximum_attempts": 8, "schedule": read_json(base.SCHEDULE)}
    if session_file.exists() and read_json(session_file) != session:
        raise base.GateError("continuation session does not match the frozen preflight and schedule")
    base.atomic_write_json(session_file, session)
    base.atomic_write_json(output / "latest-admission.json", admitted)
    attempts = []
    valid = []
    prompts = read_json(base.PROMPTS)
    goldens = {c: checked["results"][c]["goldens"] for c in base.CONFIGS}
    for scheduled in session["schedule"]["attempts"]:
        if len(valid) >= max_blocks:
            break
        number = int(scheduled["attempt"])
        directory = output / f"attempt-{number:02d}"
        if directory.exists():
            block_file = directory / "block.json"
            if not block_file.exists():
                raise base.GateError(f"unfinished attempt retained; inspect before resuming: {directory}")
            block = read_json(block_file)
            if block.get("protocol") != PROTOCOL:
                raise base.GateError("attempt protocol changed")
        else:
            directory.mkdir()
            block = {"protocol": PROTOCOL, "attempt": number, "results": {},
                     "configuration_order": scheduled["configuration_order"], "valid": False}
            try:
                for config in scheduled["configuration_order"]:
                    block["results"][config] = run_cell(
                        config, directory / config, port, prompts,
                        preflight_path / "expert-store", runtime,
                        prompt_order=scheduled["prompt_order"], goldens=goldens[config],
                    )
                    base.atomic_write_json(directory / "progress.json", block)
                    time.sleep(60)
                block["valid"] = set(block["results"]) == set(base.CONFIGS)
            except BaseException as exc:
                block["error"] = str(exc)
                base.atomic_write_json(directory / "block.json", block)
                raise  # Do not blindly repeat a deterministic error or any GPU anomaly.
            base.atomic_write_json(directory / "block.json", block)
        attempts.append(block)
        if block["valid"]:
            valid.append(block)
        result = {"protocol": PROTOCOL, "attempts": attempts, "valid_blocks": len(valid),
                  "target_valid_blocks": 5, "stage_limit": max_blocks,
                  "full_experiment_complete": len(valid) == 5,
                  "descriptive": descriptive_summary(valid),
                  "analysis": base.analyze_valid_blocks(valid)}
        base.atomic_write_json(output / "experiment-result.json", result)
        emit(f"accepted complete paired blocks: {len(valid)}/5")
    if not attempts:
        raise base.GateError("no completed attempted block")
    return result


def descriptive_summary(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Show actual completed-block values without an underpowered interval claim."""
    if not blocks:
        return {}
    result = {}
    for config in base.CONFIGS:
        values = [block["results"][config]["output_throughput_tokens_per_s"] for block in blocks]
        ratios = [value / block["results"]["moe_infinity_075"]["output_throughput_tokens_per_s"]
                  for value, block in zip(values, blocks)]
        result[config] = {
            "block_output_throughput_tokens_per_s": values,
            "paired_geometric_mean_ratio_vs_moe": math.exp(sum(map(math.log, ratios)) / len(ratios)),
            "preliminary": len(blocks) < 5,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("admit", "preflight", "run"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_OUTPUT / "preflight")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--max-blocks", type=int, choices=(1, 2, 3, 4, 5), default=5,
                        help="fixed staged stopping point; fewer than five remains preliminary")
    args = parser.parse_args()
    if args.action == "admit":
        result = admit(args.port)
        print(json.dumps(result, indent=2), flush=True)
        return 0 if result["admitted"] else 1
    def interrupted(signum: int, _frame: Any) -> None:
        raise InterruptedError(f"received signal {signum}; cleaning owned processes")
    signal.signal(signal.SIGTERM, interrupted)
    lease = base.LeaseSet.acquire()
    try:
        result = (preflight(args.output, args.port) if args.action == "preflight" else
                  full_schedule(args.output, args.preflight, args.port, args.max_blocks))
        print(json.dumps(result, indent=2), flush=True)
        return 0
    finally:
        lease.close()


if __name__ == "__main__":
    raise SystemExit(main())
