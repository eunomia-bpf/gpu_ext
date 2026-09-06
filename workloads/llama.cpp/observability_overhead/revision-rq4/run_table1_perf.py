#!/usr/bin/env python3
"""RTX 5090 Table 1 performance runner (llama.cpp pp512, seven arms, ten blocks).

Records the Table 1 performance cells for the seven fixed arms: the no-probe
baseline plus gpubpf and NVBit kernelretsnoop, threadhist, and launchlate.
Within every block the arm order rotates by one, and each cell is attempted
exactly once.  Every cell always records its exact command, raw stdout, raw
stderr, return code, elapsed seconds, and the parsed pp512 throughput whenever
llama-bench emitted one.  Same-block overhead versus the baseline is computed
wherever both throughputs are numeric.  Build and command helpers are reused
from run_observability_overhead.py and run_revision_rq4.py; correctness,
verifier, safety, driver, idle, retry, filtering, clock-control, provenance,
and manifest machinery is absent by design.  --dry-run prints the JSON
schedule without performing builds or GPU work.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
OBS_ROOT = HERE.parent
sys.path.insert(0, str(OBS_ROOT))
import run_observability_overhead as core  # noqa: E402
sys.path.insert(0, str(HERE))
import run_revision_rq4 as runner  # noqa: E402


KIND = "table1_perf"
PP = 512
TG = 0
ARMS = (
    "baseline",
    "gpubpf_kernelretsnoop",
    "nvbit_kernelretsnoop",
    "gpubpf_threadhist",
    "nvbit_threadhist",
    "gpubpf_launchlate",
    "nvbit_launchlate",
)
TASKS = ("kernelretsnoop", "threadhist", "launchlate")


def block_schedule(block: int) -> list[str]:
    offset = (block - 1) % len(ARMS)
    return list(ARMS[offset:] + ARMS[:offset])


def build_schedule(blocks: int) -> dict[str, list[str]]:
    return {str(block): block_schedule(block) for block in range(1, blocks + 1)}


def dry_run_plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dry_run": True,
        "kind": KIND,
        "metric": "llama.cpp pp512 prefill token/s and same-block percent "
                  "overhead versus the no-probe baseline",
        "arms": list(ARMS),
        "blocks": args.blocks,
        "schedule": build_schedule(args.blocks),
        "cell_count": len(ARMS) * args.blocks,
        "attempts_per_cell": 1,
        "pp": PP,
        "tg": TG,
        "model": str(args.model),
        "llama_bench": str(args.llama_bench),
        "output_dir": str(args.output_dir) if args.output_dir is not None else None,
    }


def bench_base_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    if args.uvm:
        env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"
    return env


def nvbit_env(args: argparse.Namespace, tool: str) -> dict[str, str]:
    slots = runner.kernelretsnoop_layout(PP, correctness=False)["thread_slots"]
    thread_count = args.threadhist_gpu_thread_count if tool == "threadhist" else slots
    return {
        "LD_PRELOAD": str(args.nvbit_tool),
        "NOBANNER": "1",
        "OBS_MODE": tool,
        "OBS_TARGET_SYMBOL": args.target_symbol,
        "OBS_GPU_THREAD_COUNT": str(thread_count),
    }


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def cell_throughput(result: dict[str, Any] | None) -> float | None:
    value = ((result or {}).get("metrics") or {}).get("pp_tok_s")
    if is_number(value) and math.isfinite(value):
        return float(value)
    return None


def split_bench_log(text: str) -> tuple[str, str]:
    _, stdout_marker, rest = text.partition("## stdout\n")
    if not stdout_marker:
        return "", ""
    stdout_text, stderr_marker, stderr_text = rest.partition("## stderr\n")
    if stderr_marker:
        stderr_text = stderr_text.partition("\n# exit:")[0]
    return stdout_text, stderr_text


def build_tools(
    args: argparse.Namespace, output_dir: Path
) -> tuple[dict[str, Path], Path]:
    build_root = output_dir / "gpubpf_tool_build"
    build_root.mkdir(exist_ok=True)
    tool_dirs: dict[str, Path] = {}
    for tool in TASKS:
        tool_dir = runner.prepare_tool_source(
            core.TOOLS[tool],
            bpftime_root=args.bpftime_root,
            build_root=build_root,
            target_symbol=args.target_symbol,
        )
        core.build_tool(core.TOOLS[tool], tool_dir)
        tool_dirs[tool] = tool_dir
    nvbit_build_dir = output_dir / "nvbit_tool_build"
    shutil.copytree(
        runner.NVBIT_SOURCE_DIR,
        nvbit_build_dir,
        ignore=shutil.ignore_patterns("*.o", "*.so", "*.fatbin", "flush_channel.c"),
    )
    nvbit_tool = runner.build_nvbit(nvbit_build_dir, output_dir)
    return tool_dirs, nvbit_tool


def run_arm_cell(
    arm: str,
    block: int,
    args: argparse.Namespace,
    output_dir: Path,
    tool_dirs: dict[str, Path],
    nvbit_tool: Path,
) -> dict[str, Any]:
    cell_dir = output_dir / f"{arm}_run_{block:02d}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    tool = arm.partition("_")[2]
    is_gpubpf = arm.startswith("gpubpf_")
    is_nvbit = arm.startswith("nvbit_")
    env_extra = nvbit_env(args, tool) if is_nvbit else None

    started = time.monotonic()
    probe_teardown_error: str | None = None
    if is_gpubpf:
        result = None
        bench_env = bench_base_env(args)
        try:
            with runner.private_probe(tool, args, tool_dirs[tool], cell_dir) as probe_env:
                result = runner.run_bench(arm, block, args, output_dir, env_extra=probe_env)
                bench_env = {**bench_base_env(args), **probe_env}
        except runner.OwnedCleanupError:
            raise
        except RuntimeError as exc:
            if result is None:
                raise
            probe_teardown_error = f"{type(exc).__name__}: {exc}"
    else:
        result = runner.run_bench(arm, block, args, output_dir, env_extra=env_extra)
        bench_env = {**bench_base_env(args), **(env_extra or {})}
    elapsed = time.monotonic() - started

    record: dict[str, Any] = {
        "command": runner.target_launch(core.make_llama_cmd(args), bench_env)[0],
        "cwd": str(core.WORKLOAD_DIR),
        "stdout": None,
        "stderr": None,
        "returncode": result.get("returncode"),
        "timed_out": result.get("returncode") == -1,
        "elapsed_s": elapsed,
        "throughput_tok_s": cell_throughput(result),
    }
    if result.get("metrics") is not None:
        record["metrics"] = result["metrics"]
    if result.get("error"):
        record["bench_error"] = result["error"]
    if probe_teardown_error is not None:
        record["probe_teardown_error"] = probe_teardown_error

    log_path = output_dir / result["log"] if result.get("log") else cell_dir / "llama_bench.log"
    if log_path.exists():
        record["log"] = str(log_path.relative_to(output_dir))
        record["stdout"], record["stderr"] = split_bench_log(
            log_path.read_text(errors="replace")
        )

    if is_gpubpf:
        probe_log = cell_dir / "probe.log"
        if probe_log.exists():
            record["probe_log"] = str(probe_log.relative_to(output_dir))
            try:
                record["probe"] = runner.parse_gpubpf(
                    tool, probe_log.read_text(errors="replace")
                )
            except Exception as exc:  # noqa: BLE001
                record["probe_parse_error"] = f"{type(exc).__name__}: {exc}"
        agent_log = cell_dir / "agent.log"
        if agent_log.exists():
            record["agent_log"] = str(agent_log.relative_to(output_dir))
    elif is_nvbit:
        try:
            record["probe"] = runner.parse_nvbit(tool, record["stdout"] or "")
        except Exception as exc:  # noqa: BLE001
            record["probe_parse_error"] = f"{type(exc).__name__}: {exc}"
    return record


def attach_overheads(cells: list[dict[str, Any]]) -> None:
    baselines = {
        cell["block"]: cell.get("throughput_tok_s")
        for cell in cells
        if cell.get("arm") == "baseline"
    }
    for cell in cells:
        base = baselines.get(cell.get("block"))
        value = cell.get("throughput_tok_s")
        if (
            cell.get("arm") != "baseline"
            and is_number(base)
            and is_number(value)
            and base != 0
        ):
            cell["overhead_pct"] = (base - float(value)) / float(base) * 100.0
        else:
            cell["overhead_pct"] = None


def summarize(cells: list[dict[str, Any]]) -> dict[str, Any]:
    arms = []
    for arm in ARMS:
        arm_cells = [cell for cell in cells if cell.get("arm") == arm]
        values = [
            cell["throughput_tok_s"]
            for cell in arm_cells
            if is_number(cell.get("throughput_tok_s"))
        ]
        overheads = [
            cell["overhead_pct"] for cell in arm_cells if is_number(cell.get("overhead_pct"))
        ]
        arms.append(
            {
                "arm": arm,
                "cells": len(arm_cells),
                "throughput_tok_s_mean": sum(values) / len(values) if values else None,
                "mean_overhead_pct": sum(overheads) / len(overheads) if overheads else None,
            }
        )
    return {"arms": arms}


def write_records(output_dir: Path, cells: list[dict[str, Any]]) -> None:
    attach_overheads(cells)
    (output_dir / "cells.json").write_text(
        json.dumps(cells, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summarize(cells), indent=2) + "\n", encoding="utf-8"
    )


def run_campaign(args: argparse.Namespace) -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir or (HERE / "raw" / f"{KIND}-{timestamp}")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tool_dirs, nvbit_tool = build_tools(args, output_dir)
    args.nvbit_tool = nvbit_tool
    cells: list[dict[str, Any]] = []
    for block in range(1, args.blocks + 1):
        for arm in block_schedule(block):
            record: dict[str, Any] = {
                "block": block,
                "arm": arm,
                "command": None,
                "cwd": None,
                "stdout": None,
                "stderr": None,
                "returncode": None,
                "timed_out": False,
                "elapsed_s": None,
                "throughput_tok_s": None,
                "overhead_pct": None,
            }
            print(f"block={block} arm={arm}", flush=True)
            try:
                record.update(run_arm_cell(arm, block, args, output_dir, tool_dirs, nvbit_tool))
            except runner.OwnedCleanupError as exc:
                record["error"] = str(exc)
                record["fatal_cleanup"] = exc.details
                cells.append(record)
                write_records(output_dir, cells)
                return 3
            except Exception as exc:  # noqa: BLE001
                record["error"] = f"{type(exc).__name__}: {exc}"
            cells.append(record)
            write_records(output_dir, cells)
    print(f"wrote {len(cells)} cells under {output_dir}", flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=core.DEFAULT_MODEL)
    parser.add_argument("--llama-bench", type=Path, default=core.DEFAULT_LLAMA_BENCH)
    parser.add_argument("--bpftime-root", type=Path, default=core.DEFAULT_BPFTIME_ROOT)
    parser.add_argument(
        "--bpftime-build-dir",
        type=Path,
        default=core.DEFAULT_BPFTIME_BUILD_DIR,
        help="CUDA-enabled bpftime CMake build directory",
    )
    parser.add_argument("--target-symbol", default=core.DEFAULT_TARGET_SYMBOL)
    parser.add_argument("--uprobe-binary", type=Path, default=core.DEFAULT_LAUNCH_STUB_LIBRARY)
    parser.add_argument("--uprobe-symbol-hint", default=core.DEFAULT_TARGET_SYMBOL)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--probe-startup-s", type=float, default=3.0)
    parser.add_argument("--gpu-thread-count", type=int, default=22528)
    parser.add_argument("--threadhist-gpu-thread-count", type=int, default=1048576)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--uvm", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the JSON rotation schedule without performing builds or GPU work",
    )
    args = parser.parse_args(argv)
    args.pp = PP
    args.tg = TG
    for field in ("model", "llama_bench", "bpftime_root", "bpftime_build_dir", "uprobe_binary"):
        setattr(args, field, getattr(args, field).resolve())
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps(dry_run_plan(args), indent=2), flush=True)
        return 0
    return run_campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
