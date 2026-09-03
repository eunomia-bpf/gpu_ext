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

PROTOCOL = "proposal-3-revision-10-575-lossless-stream"
DRIVER = "575.57.08"
KERNEL = "6.15.11-061511-generic"
MODULES = Path("/opt/gpubpf/modules") / DRIVER / KERNEL
DEFAULT_OUTPUT = base.HERE / "raw/head-to-head-575-lossless"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def admit(port: int) -> dict[str, Any]:
    result = base.admission(port, driver_version=DRIVER, module_root=MODULES,
                            kernel_release=KERNEL, minimum_free_bytes=80 * 1024**3)
    result["kernel"] = os.uname().release
    if result["kernel"] != KERNEL:
        result["errors"].append(f"kernel must be {KERNEL}")
    try:
        compiler = Path("/usr/local/cuda-12.9/bin/ptxas")
        result["triton_compiler_version"] = base.run_checked([str(compiler), "--version"])
        result["runtime_files"]["triton_ptxas"] = base.file_metadata(compiler)
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
                                                  current_deployment=True, offload_dir=store,
                                                  stream_parity=config == "moe_infinity_075")
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


def validate_saved_correctness(path: Path, result: dict[str, Any]) -> None:
    """Re-read the real responses and cleanup record before reusing a passed cell."""
    if read_json(path / "result.json") != result:
        raise base.GateError(f"saved correctness result differs: {path}")
    safety = read_json(path / "safety.json")
    if safety.get("passed") is not True:
        raise base.GateError(f"saved correctness cleanup did not pass: {path}")
    base.validate_pre_server_safety(safety["after"])
    passes = result.get("passes", [])
    if len(passes) != 2 or any(len(part) != 8 for part in passes):
        raise base.GateError(f"incomplete saved correctness: {path}")
    base.validate_completion_response(read_json(path / "warmup.json"), 512)
    for pass_number, part in enumerate(passes, start=1):
        for prompt_number, item in enumerate(part, start=1):
            raw = base.validate_completion_response(
                read_json(path / f"smoke-pass{pass_number}-prompt{prompt_number}.json"), 512)
            if any(raw[key] != item[key] for key in raw):
                raise base.GateError(f"saved response differs: {path}, {pass_number}/{prompt_number}")
    if ([item["text"] for item in passes[0]] != result.get("goldens")
            or any(a["text"] != b["text"] for a, b in zip(*passes))):
        raise base.GateError(f"saved exact-output gate failed: {path}")


def preflight(output: Path, port: int, expert_store: Path | None = None, *,
              resume: bool = False, reuse_llama_from: Path | None = None) -> dict[str, Any]:
    if resume and reuse_llama_from:
        raise base.GateError("choose continuation or transport-repair inheritance, not both")
    admitted = require_admission(port)
    if resume:
        result = read_json(output / "preflight-result.json")
        if result.get("protocol") != PROTOCOL or result.get("status") != "failed":
            raise base.GateError("only a failed current-protocol preflight can be resumed")
        base.require_runtime_continuity(result["runtime_files"], admitted["runtime_files"])
        if not result.get("row_chunking_numerical_gate"):
            raise base.GateError("the numerical gate must have completed before cell continuation")
        if expert_store and expert_store.absolute() != Path(result["expert_store"]):
            raise base.GateError("cannot change the expert store during correctness continuation")
        for config, saved in result["results"].items():
            cell = result.get("cell_directories", {}).get(config, config)
            validate_saved_correctness(output / cell, saved)
        number = len(list(output.glob("failed-preflight-*.json"))) + 1
        base.atomic_write_json(output / f"failed-preflight-{number:02d}.json", result)
        result.setdefault("continuations", []).append({
            "number": number, "admission": admitted,
            "source_commit": base.run_checked(["git", "rev-parse", "HEAD"], cwd=base.GPU_EXT),
            "reused_complete_cells": list(result["results"]),
        })
        result.update(status="running")
        result.pop("error", None)
        runtime = result["runtime_files"]
        expert_store = Path(result["expert_store"])
    else:
        output.mkdir(parents=True, exist_ok=False)
        base.atomic_write_json(output / "admission.json", admitted)
        runtime = admitted["runtime_files"]
        expert_store = (expert_store or output / "expert-store").absolute()
        result = {
            "protocol": PROTOCOL, "driver": DRIVER, "kernel": KERNEL,
            "source_commit": base.run_checked(["git", "rev-parse", "HEAD"], cwd=base.GPU_EXT),
            "status": "running", "runtime_files": runtime,
            "configuration_order": list(base.FROZEN_CORRECTNESS_ORDER), "results": {},
            "cross_configuration_output_equality_required": False,
            "completed_evictions_claimed": False,
            "moe_storage": "buffered NVMe hydration followed by CPU expert offload/cache",
            "expert_store": str(expert_store),
            "triton_compiler_version": admitted["triton_compiler_version"],
        }
        if reuse_llama_from:
            previous = read_json(reuse_llama_from / "preflight-result.json")
            if (previous.get("protocol") != "proposal-3-revision-9-575-cuda129"
                    or previous.get("status") != "passed"):
                raise base.GateError("transport repair requires the complete revision-9 preflight")
            base.require_runtime_continuity(
                {k: v for k, v in previous["runtime_files"].items() if k != "revision_server"},
                {k: v for k, v in runtime.items() if k != "revision_server"})
            result["cell_directories"] = {}
            for config in base.CONFIGS:
                if config == "moe_infinity_075":
                    continue
                cell = reuse_llama_from / previous.get("cell_directories", {}).get(config, config)
                saved = previous["results"][config]
                validate_saved_correctness(cell, saved)
                launch = read_json(cell / "launch.json")
                expected, _ = base.server_command(config, port, cell, Path(previous["expert_store"]))
                if (launch["argv"] != expected or launch["environment"] !=
                        base.controlled_environment(config, cuda129_triton=True)):
                    raise base.GateError(f"llama command/environment changed: {config}")
                result["results"][config] = saved
                result["cell_directories"][config] = os.path.relpath(cell, output)
            result["transport_repair_inheritance"] = {
                "source_preflight": str(reuse_llama_from.absolute()),
                "reused_complete_cells": list(result["results"]),
                "changed_runtime_files": ["revision_server"],
                "moe_correctness_and_stream_parity_rerun": True,
            }
            result["row_chunking_numerical_gate"] = previous["row_chunking_numerical_gate"]
            result["numerical_safety"] = previous["numerical_safety"]
    base.atomic_write_json(output / "preflight-result.json", result)
    try:
        if not resume and not reuse_llama_from:
            before = admitted["safety"]
            result["row_chunking_numerical_gate"] = base.run_row_chunking_numerical_gate()
            result["numerical_safety"] = base.wait_for_post_server_safety(before)
        prompts = read_json(base.PROMPTS)
        for config in base.FROZEN_CORRECTNESS_ORDER:
            if config in result["results"]:
                emit(f"retaining verified complete correctness {config}")
                continue
            cell = config if not (output / config).exists() else f"{config}-retry-{number:02d}"
            result.setdefault("cell_directories", {})[config] = cell
            result["results"][config] = run_cell(
                config, output / cell, port, prompts, expert_store, runtime,
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
        validate_saved_correctness(
            path / result.get("cell_directories", {}).get(config, config), result["results"][config])
        passes = result["results"][config].get("passes", [])
        if (len(passes) != 2 or any(len(part) != 8 for part in passes)
                or any(a["text"] != b["text"] for a, b in zip(*passes))):
            raise base.GateError(f"incomplete exact two-pass correctness: {config}")
    parity = result["results"]["moe_infinity_075"].get("stream_parity", {})
    if len(parity.get("requests", [])) != 8 or parity.get("verified_output_tokens") != 512:
        raise base.GateError("MoE full stream/nonstream parity is required after the transport repair")
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
               "expert_store": checked["expert_store"],
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
                        Path(checked["expert_store"]), runtime,
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
    parser.add_argument("--expert-store", type=Path,
                        help="reuse generated disk tensor data, never live process cache state")
    parser.add_argument("--resume-preflight", action="store_true",
                        help="retain failures and revalidate saved complete cells before continuation")
    parser.add_argument("--reuse-llama-from", type=Path,
                        help="revalidate unchanged llama cells after the MoE-only streaming repair")
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
        result = (preflight(args.output, args.port, args.expert_store,
                            resume=args.resume_preflight, reuse_llama_from=args.reuse_llama_from)
                  if args.action == "preflight" else
                  full_schedule(args.output, args.preflight, args.port, args.max_blocks))
        print(json.dumps(result, indent=2), flush=True)
        return 0
    finally:
        lease.close()


if __name__ == "__main__":
    raise SystemExit(main())
