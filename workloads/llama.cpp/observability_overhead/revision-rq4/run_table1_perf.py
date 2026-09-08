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
schedule without performing builds or GPU work.  The opt-in
--auto-warp-three-arm mode records the same-object
baseline/auto_warp_off/auto_warp_on comparison for one selected gpubpf
tool, switching BPFTIME_GPU_AUTO_WARP_EXECUTION per off/on cell.
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

AUTO_WARP_ENV_KEY = "BPFTIME_GPU_AUTO_WARP_EXECUTION"
AUTO_WARP_TRANSPORT_ENV_KEY = "BPFTIME_GPU_RINGBUF_TRANSPORT"
AUTO_WARP_TRANSPORT_ALIGNED = 1
AUTO_WARP_TRANSPORT_ENCODED = 2
AUTO_WARP_ARMS = ("baseline", "auto_warp_off", "auto_warp_on")
AUTO_WARP_TRANSPORT_MARKER = "GPU ring-buffer aligned-word output enabled"
# rope_norm launch geometry for the default target/model at pp512, from
# ggml/src/ggml-cuda/rope.cu: block_dims(1, CUDA_ROPE_BLOCK_SIZE=256, 1) and
# block_nums(nr, ceil(ne0/(2*256)), 1). The default TinyLlama GGUF has
# head dimension 64 and 4 KV heads: this float-to-half K target has nr=PP*4.
ORIGINAL_ROPE_BLOCK_Y = 256
ORIGINAL_ROPE_ROWS = PP * 4
ORIGINAL_ROPE_NE0 = 64
ORIGINAL_THREAD_SLOTS = ORIGINAL_ROPE_BLOCK_Y * ORIGINAL_ROPE_ROWS * (
    (ORIGINAL_ROPE_NE0 + 2 * ORIGINAL_ROPE_BLOCK_Y - 1)
    // (2 * ORIGINAL_ROPE_BLOCK_Y)
)
ORIGINAL_VALUE_BYTES = 80
ORIGINAL_MAP_ENTRIES = 256
ORIGINAL_SHM_MARGIN_MB = 1024
SOURCE_CONTRACTS = {
    "seven_arm_default": "legacy_runner_prepared_kernelretsnoop_capacity_patched",
    "auto_warp_three_arm": "original_example_source_no_capacity_patch",
}


def selected_arms(args: argparse.Namespace) -> tuple[str, ...]:
    if args.auto_warp_three_arm:
        return AUTO_WARP_ARMS
    return ARMS


def campaign_mode(args: argparse.Namespace) -> str:
    return "auto_warp_three_arm" if args.auto_warp_three_arm else "seven_arm_default"

def block_schedule(block: int, arms: tuple[str, ...] = ARMS) -> list[str]:
    offset = (block - 1) % len(arms)
    return list(arms[offset:] + arms[:offset])


def build_schedule(blocks: int, arms: tuple[str, ...] = ARMS) -> dict[str, list[str]]:
    return {str(block): block_schedule(block, arms) for block in range(1, blocks + 1)}


def dry_run_plan(args: argparse.Namespace) -> dict[str, Any]:
    arms = selected_arms(args)
    return {
        "dry_run": True,
        "kind": KIND,
        "metric": "llama.cpp pp512 prefill token/s and same-block percent "
                "overhead versus the no-probe baseline",
        "mode": campaign_mode(args),
        "arms": list(arms),
        "blocks": args.blocks,
        "auto_warp_task": args.auto_warp_task if args.auto_warp_three_arm else None,
        "auto_warp_transport": args.auto_warp_transport if args.auto_warp_three_arm else None,
        "source_contract": SOURCE_CONTRACTS[campaign_mode(args)],
        "schedule": build_schedule(args.blocks, arms),
        "cell_count": len(arms) * args.blocks,
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


def auto_warp_observed_marker(cell_dir: Path, record: dict[str, Any]) -> str:
    texts = [record.get("stdout") or ""]
    agent_log = cell_dir / "agent.log"
    if agent_log.is_file():
        texts.append(agent_log.read_text(errors="replace"))
    combined = "\n".join(texts)
    if "GPU automatic warp execution admitted for" in combined:
        return "admitted"
    if "GPU automatic warp execution not admitted for" in combined:
        return "not_admitted"
    return "absent"


def auto_warp_transport_marker(cell_dir: Path, record: dict[str, Any]) -> str:
    texts = [record.get("stdout") or ""]
    agent_log = cell_dir / "agent.log"
    if agent_log.is_file():
        texts.append(agent_log.read_text(errors="replace"))
    combined = "\n".join(texts)
    return "enabled" if AUTO_WARP_TRANSPORT_MARKER in combined else "absent"


def original_ring_bytes(
    thread_slots: int = ORIGINAL_THREAD_SLOTS,
    value_bytes: int = ORIGINAL_VALUE_BYTES,
    max_entries: int = ORIGINAL_MAP_ENTRIES,
) -> int:
    aligned_record = ((value_bytes + 8 + 7) // 8) * 8
    return thread_slots * (24 + aligned_record * max_entries) + 32


def auto_warp_probe_env(arm: str, args: argparse.Namespace) -> dict[str, str]:
    env = {
        AUTO_WARP_ENV_KEY: "1" if arm == "auto_warp_on" else "0",
        AUTO_WARP_TRANSPORT_ENV_KEY: str(args.auto_warp_transport),
    }
    if args.auto_warp_task == "kernelretsnoop":
        ring_bytes = original_ring_bytes()
        shm_mb = (
            ring_bytes + ORIGINAL_SHM_MARGIN_MB * 1024 * 1024
        ) // (1024 * 1024) + 1
        env["BPFTIME_MAP_GPU_THREAD_COUNT"] = str(ORIGINAL_THREAD_SLOTS)
        env["BPFTIME_SHM_MEMORY_MB"] = str(shm_mb)
    return env


def relocate_runtime_includes(tool_dir: Path, bpftime_root: Path) -> None:
    """Redirect the relative runtime include to the configured bpftime tree."""
    runtime_include = str((bpftime_root / "runtime" / "include").resolve())
    relative = runner.RELATIVE_RUNTIME_INCLUDE
    for path in tool_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if relative in text:
            path.write_text(text.replace(relative, runtime_include), encoding="utf-8")
    for path in tool_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if runner.RELATIVE_RUNTIME_INCLUDE_PATTERN.search(text):
            raise RuntimeError(f"relative runtime include remains in {path}")


def build_tools(
    args: argparse.Namespace, output_dir: Path,
    *, tasks: tuple[str, ...] = TASKS, nvbit: bool = True,
    auto_warp: bool = False
) -> tuple[dict[str, Path], Path | None]:
    build_root = output_dir / "gpubpf_tool_build"
    build_root.mkdir(exist_ok=True)
    tool_dirs: dict[str, Path] = {}
    for tool in tasks:
        prepare = core.prepare_tool_source if auto_warp else runner.prepare_tool_source
        tool_dir = prepare(
            core.TOOLS[tool],
            bpftime_root=args.bpftime_root,
            build_root=build_root,
            target_symbol=args.target_symbol,
        )
        if auto_warp:
            relocate_runtime_includes(tool_dir, args.bpftime_root)
        core.build_tool(core.TOOLS[tool], tool_dir)
        tool_dirs[tool] = tool_dir
    nvbit_tool: Path | None = None
    if nvbit:
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
    nvbit_tool: Path | None,
) -> dict[str, Any]:
    cell_dir = output_dir / f"{arm}_run_{block:02d}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    if arm in ("auto_warp_off", "auto_warp_on"):
        tool = args.auto_warp_task
        is_gpubpf = True
        is_nvbit = False
        auto_warp_env = auto_warp_probe_env(arm, args)
    else:
        tool = arm.partition("_")[2]
        is_gpubpf = arm.startswith("gpubpf_")
        is_nvbit = arm.startswith("nvbit_")
        auto_warp_env = {}
    env_extra = nvbit_env(args, tool) if is_nvbit else None

    started = time.monotonic()
    probe_teardown_error: str | None = None
    probe_kwargs = {"extra_probe_env": auto_warp_env} if auto_warp_env else {}
    if is_gpubpf:
        result = None
        bench_env = bench_base_env(args)
        try:
            with runner.private_probe(
                tool, args, tool_dirs[tool], cell_dir, **probe_kwargs
            ) as probe_env:
                merged_probe_env = {**probe_env, **auto_warp_env}
                result = runner.run_bench(
                    arm, block, args, output_dir,
                    env_extra=merged_probe_env,
                )
                bench_env = {**bench_base_env(args), **probe_env, **auto_warp_env}
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
    if auto_warp_env:
        record["auto_warp_requested"] = "off" if arm.endswith("_off") else "on"
        record["auto_warp_env"] = dict(auto_warp_env)
        record["auto_warp_observed"] = auto_warp_observed_marker(cell_dir, record)
        record["auto_warp_transport_observed"] = auto_warp_transport_marker(cell_dir, record)
        if args.auto_warp_task == "kernelretsnoop":
            record["auto_warp_ring_bytes"] = original_ring_bytes()
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


def summarize(cells: list[dict[str, Any]], arms: tuple[str, ...] = ARMS,
            mode: str = "seven_arm_default") -> dict[str, Any]:
    arms_summary = []
    for arm in arms:
        arm_cells = [cell for cell in cells if cell.get("arm") == arm]
        values = [
            cell["throughput_tok_s"]
            for cell in arm_cells
            if is_number(cell.get("throughput_tok_s"))
        ]
        overheads = [
            cell["overhead_pct"] for cell in arm_cells if is_number(cell.get("overhead_pct"))
        ]
        arms_summary.append(
            {
                "arm": arm,
                "cells": len(arm_cells),
                "throughput_tok_s_mean": sum(values) / len(values) if values else None,
                "mean_overhead_pct": sum(overheads) / len(overheads) if overheads else None,
            }
        )
    return {
        "mode": mode,
        "source_contract": SOURCE_CONTRACTS[mode],
        "arms": arms_summary,
    }


def write_records(output_dir: Path, cells: list[dict[str, Any]],
                arms: tuple[str, ...] = ARMS,
                mode: str = "seven_arm_default") -> None:
    attach_overheads(cells)
    (output_dir / "cells.json").write_text(
        json.dumps(cells, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summarize(cells, arms, mode), indent=2) + "\n", encoding="utf-8"
    )


def run_campaign(args: argparse.Namespace) -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir or (HERE / "raw" / f"{KIND}-{timestamp}")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    arms = selected_arms(args)
    mode = campaign_mode(args)
    if args.auto_warp_three_arm:
        if args.target_symbol != core.DEFAULT_TARGET_SYMBOL:
            raise SystemExit(
                "verified rope_norm geometry applies to "
                f"{core.DEFAULT_TARGET_SYMBOL}; got {args.target_symbol}"
            )
        tool_dirs, nvbit_tool = build_tools(
            args, output_dir, tasks=(args.auto_warp_task,),
            nvbit=False, auto_warp=True
        )
    else:
        tool_dirs, nvbit_tool = build_tools(args, output_dir)
    args.nvbit_tool = nvbit_tool
    cells: list[dict[str, Any]] = []
    for block in range(1, args.blocks + 1):
        for arm in block_schedule(block, arms):
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
                write_records(output_dir, cells, arms, mode)
                return 3
            except Exception as exc:  # noqa: BLE001
                record["error"] = f"{type(exc).__name__}: {exc}"
            cells.append(record)
            write_records(output_dir, cells, arms, mode)
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
        "--auto-warp-three-arm",
        action="store_true",
        help="opt-in same-object baseline/off/on auto-warp comparison for one gpubpf tool",
    )
    parser.add_argument(
        "--auto-warp-task",
        default="kernelretsnoop",
        choices=list(TASKS),
        help="gpubpf tool object shared by the auto_warp_off/auto_warp_on arms",
    )
    parser.add_argument(
        "--auto-warp-transport",
        type=int,
        default=AUTO_WARP_TRANSPORT_ALIGNED,
        choices=(AUTO_WARP_TRANSPORT_ALIGNED, AUTO_WARP_TRANSPORT_ENCODED),
        help="loader/agent ring transport for the three-arm mode: 1 aligned-word, 2 encoded-tail",
    )
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
