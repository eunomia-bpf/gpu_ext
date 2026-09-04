#!/usr/bin/env python3
"""Run the Table 1 device-verifier A1 admission-latency experiment.

This is intentionally separate from the frozen revision-rq4 throughput runner.
It executes correctness-sized llama-cli clients and records only the runtime's
explicit verifier-call latency; whole-application elapsed time is not a metric.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
RQ4_ROOT = HERE.parent
sys.path.insert(0, str(RQ4_ROOT))
import run_revision_rq4 as rq4  # noqa: E402


TOOLS = ("kernelretsnoop", "threadhist")
MODES = ("STRICT", "NO_VERIFY")
SCHEMA = "device-verifier-a1-v1"
SCHEDULE_SEED = 1797
ANALYSIS_SEED = 9173
BOOTSTRAP_SAMPLES = 10000
MIN_PAIRS = 10
PROGRAM = "cuda__retprobe"
RUNTIME_TIMING_MARKER = (
    "GPU eBPF verification timing: program={} verification_elapsed_ns={}"
)
DEFAULT_BPFTIME_ROOT = rq4.core.GPU_WORKSPACE / "bpftime-table1-575"
DEFAULT_BPFTIME_BUILD = DEFAULT_BPFTIME_ROOT / "build-table1-575-strict"


def schedule(pairs: int, seed: int = SCHEDULE_SEED) -> list[dict[str, Any]]:
    """Return balanced randomized four-cell blocks, one pair per tool/block."""
    if pairs < MIN_PAIRS:
        raise ValueError(f"A1 requires at least {MIN_PAIRS} pairs per tool")
    rng = random.Random(seed)
    first_modes: dict[str, list[str]] = {}
    for tool in TOOLS:
        values = [MODES[index % 2] for index in range(pairs)]
        rng.shuffle(values)
        first_modes[tool] = values

    result: list[dict[str, Any]] = []
    sequence = 0
    for pair in range(1, pairs + 1):
        cells = [
            {"pair": pair, "tool": tool, "mode": mode}
            for tool in TOOLS
            for mode in MODES
        ]
        desired_first = {tool: first_modes[tool][pair - 1] for tool in TOOLS}
        while True:
            rng.shuffle(cells)
            positions = {
                (cell["tool"], cell["mode"]): index
                for index, cell in enumerate(cells)
            }
            if all(
                positions[(tool, desired_first[tool])]
                < positions[(tool, MODES[1 - MODES.index(desired_first[tool])])]
                for tool in TOOLS
            ):
                break
        for cell in cells:
            sequence += 1
            result.append({"sequence": sequence, **cell})
    return result


def expected_map(tool: str) -> dict[str, int]:
    if tool == "kernelretsnoop":
        return {"type": 1527, "key_size": 4, "value_size": 32, "max_entries": 256}
    if tool == "threadhist":
        return {"type": 1502, "key_size": 4, "value_size": 8, "max_entries": 1}
    raise ValueError(f"unsupported A1 tool: {tool}")


def _target_pid(execution_path: Path) -> tuple[int | None, str | None]:
    try:
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        pid = execution["identity"]["pid"]
        if type(pid) is not int or pid <= 0:
            raise ValueError("identity.pid is not a positive integer")
        if execution.get("cleanup_passed") is not True:
            raise ValueError("target cleanup was not confirmed")
        if execution.get("timed_out") is not False or execution.get("returncode") != 0:
            raise ValueError("target did not complete successfully")
        return pid, None
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        return None, f"{type(error).__name__}: {error}"


def parse_target_admission(
    log_path: Path,
    execution_path: Path,
    *,
    tool: str,
    mode: str,
    target_symbol: str,
) -> dict[str, Any]:
    """Parse admission evidence exclusively from one target llama-cli log."""
    target_pid, execution_error = _target_pid(execution_path)
    text = log_path.read_text(errors="replace") if log_path.is_file() else ""
    prefix = r"^\[[^\]\r\n]+\]\[[^\]\r\n]+\]\[(?P<pid>[1-9][0-9]*)\] "
    accepted_re = re.compile(
        prefix
        + r"GPU eBPF verification accepted: mode=(?P<mode>[^ \r\n]+) "
        + rf"program={PROGRAM} attach=(?P<attach>[^ \r\n]+) "
        + r"instructions=(?P<instructions>[1-9][0-9]*)\r?$"
    )
    timing_re = re.compile(
        prefix
        + rf"GPU eBPF verification timing: program={PROGRAM} "
        + r"verification_elapsed_ns=(?P<elapsed>[1-9][0-9]*)\r?$"
    )
    map_re = re.compile(
        prefix
        + rf"GPU eBPF verified map: program={PROGRAM} fd=(?P<fd>[0-9]+) "
        + r"type=(?P<type>[0-9]+) key_size=(?P<key_size>[0-9]+) "
        + r"value_size=(?P<value_size>[0-9]+) max_entries=(?P<max_entries>[0-9]+)\r?$"
    )
    skip_re = re.compile(
        prefix + rf"Skipping GPU eBPF verification for {PROGRAM}\r?$"
    )
    reject_re = re.compile(
        prefix + rf"GPU eBPF verification failed for {PROGRAM}:.*$"
    )
    unavailable_re = re.compile(prefix + r".*verifier unavailable.*$")
    fragments = (
        "GPU eBPF verification accepted:",
        "GPU eBPF verification timing:",
        "GPU eBPF verified map:",
        "Skipping GPU eBPF verification",
        "GPU eBPF verification failed",
        "verifier unavailable",
        "verification_elapsed_ns=",
    )
    accepted: list[dict[str, Any]] = []
    timings: list[int] = []
    maps: list[dict[str, int]] = []
    skipped = rejected = unavailable = foreign = unparsed = 0
    for line in text.splitlines():
        if not any(fragment in line for fragment in fragments):
            continue
        match = accepted_re.fullmatch(line)
        kind = "accepted"
        if match is None:
            match, kind = timing_re.fullmatch(line), "timing"
        if match is None:
            match, kind = map_re.fullmatch(line), "map"
        if match is None:
            match, kind = skip_re.fullmatch(line), "skip"
        if match is None:
            match, kind = reject_re.fullmatch(line), "reject"
        if match is None:
            match, kind = unavailable_re.fullmatch(line), "unavailable"
        if match is None:
            unparsed += 1
            continue
        pid = int(match.group("pid"))
        if target_pid is None or pid != target_pid:
            foreign += 1
            continue
        if kind == "accepted":
            accepted.append(
                {
                    "mode": match.group("mode"),
                    "attach": match.group("attach"),
                    "instructions": int(match.group("instructions")),
                }
            )
        elif kind == "timing":
            timings.append(int(match.group("elapsed")))
        elif kind == "map":
            maps.append(
                {
                    name: int(match.group(name))
                    for name in ("fd", "type", "key_size", "value_size", "max_entries")
                }
            )
        elif kind == "skip":
            skipped += 1
        elif kind == "reject":
            rejected += 1
        else:
            unavailable += 1

    map_contract = expected_map(tool)
    attach = f"kretprobe/{target_symbol}"
    common = (
        execution_error is None
        and log_path.is_file()
        and foreign == 0
        and unparsed == 0
        and rejected == 0
        and unavailable == 0
    )
    if mode == "STRICT":
        passed = (
            common
            and len(accepted) == 1
            and accepted[0]["mode"] == "STRICT"
            and accepted[0]["attach"] == attach
            and accepted[0]["instructions"] > 0
            and len(timings) == 1
            and timings[0] > 0
            and len(maps) == 1
            and all(maps[0][key] == value for key, value in map_contract.items())
            and skipped == 0
        )
    elif mode == "NO_VERIFY":
        passed = common and skipped == 1 and not accepted and not timings and not maps
    else:
        raise ValueError(f"unexpected mode: {mode}")
    return {
        "passed": bool(passed),
        "mode": mode,
        "program": PROGRAM,
        "attach": attach,
        "target_pid": target_pid,
        "execution_record": execution_path.name,
        "execution_error": execution_error,
        "log": log_path.name,
        "accepted_records": len(accepted),
        "accepted": accepted,
        "timing_records": len(timings),
        "verification_elapsed_ns": timings[0] if len(timings) == 1 else None,
        "map_records": len(maps),
        "maps": maps,
        "expected_map": map_contract,
        "skipped_records": skipped,
        "rejected_records": rejected,
        "unavailable_records": unavailable,
        "foreign_pid_records": foreign,
        "unparsed_records": unparsed,
        "latency_source": "target_llama_cli_log_runtime_marker",
    }


def file_metadata(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        return {"path": str(path), "exists": False}
    info = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "bytes": info.st_size,
        "device": info.st_dev,
        "inode": info.st_ino,
        "mtime_ns": info.st_mtime_ns,
    }


def git_commit(path: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def runtime_source_contract(root: Path) -> dict[str, Any]:
    source = root / "attach/nv_attach_impl/nv_attach_impl.cpp"
    text = source.read_text(encoding="utf-8") if source.is_file() else ""
    required = {
        "timing_marker": RUNTIME_TIMING_MARKER,
        "strict_accept": "GPU eBPF verification accepted: mode=STRICT",
        "skip_marker": "Skipping GPU eBPF verification for {}",
        "map_marker": "GPU eBPF verified map: program={}",
        "unavailable_guard": "verifier unavailable",
    }
    present = {name: marker in text for name, marker in required.items()}
    return {"source": str(source.resolve()), "required": required, "present": present,
            "passed": source.is_file() and all(present.values())}


def validate_tool_source(tool: str, directory: Path, target_symbol: str) -> None:
    source = directory / f"{tool}.bpf.c"
    text = source.read_text(encoding="utf-8")
    if text.count(f'SEC("kretprobe/{target_symbol}")') != 1:
        raise RuntimeError(f"{tool} does not have exactly one requested kretprobe target")
    if text.count(f"int {PROGRAM}()") != 1:
        raise RuntimeError(f"{tool} does not have exactly one {PROGRAM} program")
    if tool == "kernelretsnoop":
        rq4.validate_kernelretsnoop_source_schema(directory)
    else:
        required = ("BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP 1502", "__uint(max_entries, 1)",
                    "__type(key, u32)", "__type(value, u64)")
        if any(marker not in text for marker in required):
            raise RuntimeError("threadhist source does not match the Table 1 map contract")


def build_tools(args: argparse.Namespace, output: Path) -> dict[str, Path]:
    build_root = output / "gpubpf_tool_build"
    build_root.mkdir()
    directories: dict[str, Path] = {}
    for tool in TOOLS:
        directory = rq4.prepare_tool_source(
            rq4.core.TOOLS[tool], bpftime_root=args.bpftime_root,
            build_root=build_root, target_symbol=args.target_symbol,
        )
        validate_tool_source(tool, directory, args.target_symbol)
        rq4.core.build_tool(rq4.core.TOOLS[tool], directory)
        object_path = directory / ".output" / f"{tool}.bpf.o"
        if not object_path.is_file() or object_path.stat().st_size <= 0:
            raise RuntimeError(f"missing real built Table 1 object: {object_path}")
        directories[tool] = directory
    return directories


def run_baseline(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    run_dir = output / "correctness-baseline"
    if run_dir.exists():
        raise RuntimeError(f"refusing to reuse baseline directory: {run_dir}")
    run_dir.mkdir()
    rq4.idle_gpu_or_error(rq4.core.nvidia_smi_snapshot())
    with rq4.cell_safety(run_dir) as safety:
        completed = rq4.run_cli_separate(
            rq4.llama_cli_cmd(args), cwd=rq4.core.WORKLOAD_DIR,
            env=rq4.correctness_env(args), timeout=args.timeout_s,
            log_path=run_dir / "llama_cli.log",
        )
    normalized = rq4.normalized_output(completed.stdout)
    valid = completed.returncode == 0 and normalized == rq4.EXPECTED_NORMALIZED_STDOUT
    return {
        "valid": valid, "returncode": completed.returncode,
        "normalized_stdout": normalized,
        "stdout_bytes": len(normalized.encode()), "safety": safety,
        "directory": str(run_dir.relative_to(output)),
    }


def run_instrumented(
    args: argparse.Namespace,
    output: Path,
    tool_dirs: dict[str, Path],
    *,
    stage: str,
    tool: str,
    mode: str,
    pair: int | None,
    sequence: int,
) -> dict[str, Any]:
    label = f"{sequence:03d}-{tool}-{mode.lower()}"
    run_dir = output / stage / label
    if run_dir.exists():
        raise RuntimeError(f"refusing to reuse cell directory: {run_dir}")
    run_dir.mkdir(parents=True)
    rq4.idle_gpu_or_error(rq4.core.nvidia_smi_snapshot())
    args.verifier_level = mode
    with rq4.cell_safety(run_dir) as safety, rq4.private_probe(
        tool, args, tool_dirs[tool], run_dir, exact_exit_oracle=(tool == "kernelretsnoop")
    ) as probe_env:
        env = rq4.correctness_env(args)
        env.update(probe_env)
        completed = rq4.run_cli_separate(
            rq4.llama_cli_cmd(args), cwd=rq4.core.WORKLOAD_DIR, env=env,
            timeout=args.timeout_s, log_path=run_dir / "llama_cli.log",
        )
    normalized = rq4.normalized_output(completed.stdout)
    probe_text = (run_dir / "probe.log").read_text(errors="replace")
    probe = rq4.parse_gpubpf(tool, probe_text)
    if tool == "kernelretsnoop":
        probe_valid = rq4.gpubpf_probe_valid(
            tool, probe, expected_thread_count=rq4.EXPECTED_GPU_THREAD_SLOTS,
            expected_ring_entries=rq4.CORRECTNESS_RING_ENTRIES_PER_THREAD,
            expected_exit_events=rq4.CORRECTNESS_EXIT_EVENTS,
            expected_exit_launches=rq4.CORRECTNESS_EXIT_LAUNCHES,
            expected_exit_coordinates=rq4.CORRECTNESS_EXIT_COORDINATES,
            exact_exit_oracle=True,
        )
    else:
        probe_valid = rq4.gpubpf_probe_valid(
            tool, probe, expected_thread_count=args.threadhist_gpu_thread_count,
            exact_exit_oracle=False,
        )
    admission = parse_target_admission(
        run_dir / "llama_cli.log", run_dir / "llama_cli.execution.json",
        tool=tool, mode=mode, target_symbol=args.target_symbol,
    )
    output_valid = (
        completed.returncode == 0
        and normalized == rq4.EXPECTED_NORMALIZED_STDOUT
    )
    return {
        "stage": stage, "pair": pair, "sequence": sequence, "tool": tool,
        "mode": mode, "directory": str(run_dir.relative_to(output)),
        "returncode": completed.returncode, "normalized_stdout": normalized,
        "stdout_bytes": len(normalized.encode()), "output_valid": output_valid,
        "matches_baseline": output_valid, "probe": probe,
        "probe_valid": bool(probe_valid), "admission": admission,
        "safety": safety,
        "valid": bool(output_valid and probe_valid and admission["passed"] and safety["passed"]),
    }


def plan(args: argparse.Namespace) -> dict[str, Any]:
    cells = schedule(args.pairs)
    return {
        "schema": SCHEMA,
        "hypothesis": (
            "Strict SIMT verification admits each real Table 1 device object with a "
            "finite one-time verifier-call latency; randomized NO_VERIFY arms confirm "
            "the matched runtime/object path skips the verifier."
        ),
        "metric": "target-log verification_elapsed_ns on STRICT only",
        "forbidden_metric": "whole-application elapsed time or throughput",
        "tools": list(TOOLS), "modes": list(MODES), "pairs_per_tool": args.pairs,
        "schedule_seed": SCHEDULE_SEED, "analysis_seed": ANALYSIS_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "pair_definition": (
            "For one tool and pair index, exactly one fresh STRICT process and one "
            "fresh NO_VERIFY process using the same built object, verifier-enabled "
            "runtime, llama-cli command, correctness oracle, and safety gates."
        ),
        "a0_definition": "one separate fresh STRICT admission/correctness cell per tool",
        "schedule": cells,
    }


def write_state(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def defining_inputs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": str(args.model), "llama_cli": str(args.llama_cli),
        "llama_bench": str(args.llama_bench), "bpftime_root": str(args.bpftime_root),
        "bpftime_build_dir": str(args.bpftime_build_dir),
        "target_symbol": args.target_symbol, "pairs": args.pairs,
        "n_gpu_layers": args.n_gpu_layers, "timeout_s": args.timeout_s,
        "probe_startup_s": args.probe_startup_s,
        "gpu_thread_count": args.gpu_thread_count,
        "threadhist_gpu_thread_count": args.threadhist_gpu_thread_count,
        "uvm": args.uvm,
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    if args.pairs < MIN_PAIRS:
        raise ValueError(f"--pairs must be at least {MIN_PAIRS}")
    if args.gpu_thread_count != rq4.EXPECTED_GPU_THREAD_SLOTS:
        raise ValueError(f"kernelretsnoop correctness requires {rq4.EXPECTED_GPU_THREAD_SLOTS} slots")
    if args.threadhist_gpu_thread_count != 1048576:
        raise ValueError("threadhist Table 1 contract requires 1048576 entries")
    args.tools = list(TOOLS)
    rq4.core.validate(args)
    build_config = rq4.verifier_runtime_configuration(args.bpftime_build_dir)
    if any(value.upper() not in {"ON", "YES", "TRUE", "1"} for value in build_config.values()):
        raise RuntimeError("A1 requires one verifier-enabled CUDA/LLVM runtime build")
    contract = runtime_source_contract(args.bpftime_root)
    if not contract["passed"]:
        raise RuntimeError("runtime source does not provide the fixed A1 timing/admission contract")
    return {"build_configuration": build_config, "source_contract": contract}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pairs", type=int, default=MIN_PAIRS)
    parser.add_argument("--model", type=Path, default=rq4.core.DEFAULT_MODEL)
    parser.add_argument("--llama-cli", type=Path, default=rq4.core.DEFAULT_LLAMA_BENCH.parent / "llama-cli")
    parser.add_argument("--llama-bench", type=Path, default=rq4.core.DEFAULT_LLAMA_BENCH)
    parser.add_argument("--bpftime-root", type=Path, default=DEFAULT_BPFTIME_ROOT)
    parser.add_argument("--bpftime-build-dir", type=Path, default=DEFAULT_BPFTIME_BUILD)
    parser.add_argument("--target-symbol", default=rq4.core.DEFAULT_TARGET_SYMBOL)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--probe-startup-s", type=float, default=3.0)
    parser.add_argument("--gpu-thread-count", type=int, default=rq4.EXPECTED_GPU_THREAD_SLOTS)
    parser.add_argument("--threadhist-gpu-thread-count", type=int, default=1048576)
    parser.add_argument("--uvm", action="store_true")
    args = parser.parse_args(argv)
    for field in ("model", "llama_cli", "llama_bench", "bpftime_root", "bpftime_build_dir"):
        setattr(args, field, getattr(args, field).resolve())
    args.pp = 32
    args.tg = 0
    args.no_warmup = True
    args.uprobe_binary = rq4.core.DEFAULT_LAUNCH_STUB_LIBRARY.resolve()
    args.uprobe_symbol_hint = args.target_symbol
    args.tools = list(TOOLS)
    args.verifier_level = "STRICT"
    return args


def run(args: argparse.Namespace) -> int:
    rq4.reject_ambient_injection()
    validation = validate(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (args.output_dir or HERE / "raw" / f"full-{timestamp}").resolve()
    result_path = output / "result.json"
    current_plan = plan(args)
    if args.resume:
        state = json.loads(result_path.read_text(encoding="utf-8"))
        if state.get("schema") != SCHEMA or state.get("plan") != current_plan:
            raise RuntimeError("resume plan differs from the fixed A1 plan")
        if state.get("defining_inputs") != defining_inputs(args):
            raise RuntimeError("resume defining inputs differ")
        tool_dirs = {
            tool: output / "gpubpf_tool_build" / tool for tool in TOOLS
        }
        if any(not (path / tool).is_file() for tool, path in tool_dirs.items()):
            raise RuntimeError("resume tool build is incomplete")
    else:
        if output.exists() and any(output.iterdir()):
            raise RuntimeError("refusing to reuse a nonempty output directory")
        output.mkdir(parents=True, exist_ok=True)
        snapshot = rq4.core.nvidia_smi_snapshot()
        if rq4.parse_driver(snapshot) != rq4.EXPECTED_DRIVER:
            raise RuntimeError(f"A1 requires driver {rq4.EXPECTED_DRIVER}")
        rq4.idle_gpu_or_error(snapshot)
        tool_dirs = build_tools(args, output)
        state = {
            "schema": SCHEMA, "status": "running", "created": timestamp,
            "plan": current_plan, "defining_inputs": defining_inputs(args),
            "runtime": {
                **validation, "git_commit": git_commit(args.bpftime_root),
                "agent": file_metadata(args.bpftime_build_dir / "runtime/agent/libbpftime-agent.so"),
                "syscall_server": file_metadata(args.bpftime_build_dir / "runtime/syscall-server/libbpftime-syscall-server.so"),
            },
            "host": {
                "driver": rq4.parse_driver(snapshot),
                "expected_driver": rq4.EXPECTED_DRIVER,
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "nvidia_smi": snapshot,
            },
            "objects": {
                tool: file_metadata(tool_dirs[tool] / ".output" / f"{tool}.bpf.o")
                for tool in TOOLS
            },
            "baseline": None, "a0": [], "cells": [],
        }
        write_state(result_path, state)

    if state.get("baseline") is None:
        state["baseline"] = run_baseline(args, output)
        write_state(result_path, state)
    if not state["baseline"].get("valid"):
        state["status"] = "invalid_baseline"
        write_state(result_path, state)
        return 2

    for sequence, tool in enumerate(TOOLS, start=1):
        existing = [cell for cell in state["a0"] if cell.get("tool") == tool]
        if existing:
            if len(existing) != 1 or not existing[0].get("valid"):
                state["status"] = "invalid_a0"
                write_state(result_path, state)
                return 2
            continue
        cell = run_instrumented(
            args, output, tool_dirs, stage="a0", tool=tool, mode="STRICT",
            pair=None, sequence=sequence,
        )
        cell["matches_baseline"] = cell["normalized_stdout"] == state["baseline"]["normalized_stdout"]
        cell["valid"] = bool(cell["valid"] and cell["matches_baseline"])
        state["a0"].append(cell)
        write_state(result_path, state)
        if not cell["valid"]:
            state["status"] = "invalid_a0"
            write_state(result_path, state)
            return 2

    for specification in current_plan["schedule"]:
        existing = [cell for cell in state["cells"] if cell.get("sequence") == specification["sequence"]]
        if existing:
            if len(existing) != 1 or any(existing[0].get(key) != specification[key] for key in specification):
                raise RuntimeError("resume cell does not match the fixed schedule")
            if not existing[0].get("valid"):
                state["status"] = "invalid_cell"
                write_state(result_path, state)
                return 2
            continue
        cell = run_instrumented(args, output, tool_dirs, stage="a1", **specification)
        cell["matches_baseline"] = cell["normalized_stdout"] == state["baseline"]["normalized_stdout"]
        cell["valid"] = bool(cell["valid"] and cell["matches_baseline"])
        state["cells"].append(cell)
        write_state(result_path, state)
        if not cell["valid"]:
            state["status"] = "invalid_cell"
            write_state(result_path, state)
            return 2

    state["status"] = "complete"
    write_state(result_path, state)
    print(result_path)
    return 0


def main() -> int:
    args = parse_args()
    if args.dry_run:
        print(json.dumps(plan(args), indent=2))
        return 0
    lease = rq4.ReadOnlyLeases()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupted)
    previous_run_cmd = rq4.core.run_cmd
    try:
        rq4.core.run_cmd = rq4.run_cmd_owned
        return run(args)
    finally:
        rq4.core.run_cmd = previous_run_cmd
        signal.signal(signal.SIGTERM, previous)
        lease.close()


if __name__ == "__main__":
    raise SystemExit(main())
