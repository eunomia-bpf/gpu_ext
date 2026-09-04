#!/usr/bin/env python3
"""Run matched STRICT/NO_VERIFY/control steady-state Table 1 blocks."""

from __future__ import annotations

import argparse
import itertools
import json
import math
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
TREATMENTS = ("control", "STRICT", "NO_VERIFY")
SCHEMA = "device-verifier-s0-v1"
SCHEDULE_SEED = 1797
ANALYSIS_SEED = 9174
BOOTSTRAP_SAMPLES = 10000
BLOCKS = 10
CORRECTNESS_PP = 32
TIMING_PP = 512
PROGRAM = "cuda__retprobe"
DEFAULT_BPFTIME_ROOT = rq4.core.GPU_WORKSPACE / "bpftime-table1-575"
DEFAULT_BPFTIME_BUILD = DEFAULT_BPFTIME_ROOT / "build-table1-575-strict"
TIMING_MARKER = "GPU eBPF verification timing: program={} verification_elapsed_ns={}"
RUNTIME_BINARY_MARKERS = (
    TIMING_MARKER,
    "GPU eBPF verification accepted: mode=STRICT",
    "Skipping GPU eBPF verification for {}",
    "GPU eBPF verified map: program={}",
)


def randomized_orders(count: int, rng: random.Random) -> list[list[str]]:
    result: list[list[str]] = []
    while len(result) < count:
        cycle = [list(order) for order in itertools.permutations(TREATMENTS)]
        rng.shuffle(cycle)
        result.extend(cycle)
    return result[:count]


def fixed_plan() -> dict[str, Any]:
    rng = random.Random(SCHEDULE_SEED)
    correctness: list[dict[str, Any]] = []
    timing: list[dict[str, Any]] = []
    sequence = 0
    tool_order = list(TOOLS)
    rng.shuffle(tool_order)
    for tool in tool_order:
        order = list(TREATMENTS)
        rng.shuffle(order)
        for position, treatment in enumerate(order, start=1):
            sequence += 1
            correctness.append({
                "sequence": sequence, "tool": tool, "block": 0,
                "position": position, "treatment": treatment, "pp": CORRECTNESS_PP,
            })
    orders = {tool: randomized_orders(BLOCKS, rng) for tool in TOOLS}
    for block in range(1, BLOCKS + 1):
        block_tools = list(TOOLS)
        rng.shuffle(block_tools)
        for tool in block_tools:
            for position, treatment in enumerate(orders[tool][block - 1], start=1):
                sequence += 1
                timing.append({
                    "sequence": sequence, "tool": tool, "block": block,
                    "position": position, "treatment": treatment, "pp": TIMING_PP,
                })
    return {
        "schema": SCHEMA,
        "hypothesis": (
            "After admission, STRICT and NO_VERIFY have the same steady-state Table 1 "
            "throughput distribution within paired randomized blocks."
        ),
        "primary_effect": "100 * (STRICT_pp_tok_s / NO_VERIFY_pp_tok_s - 1) per tool/block",
        "secondary_effects": [
            "100 * (STRICT_pp_tok_s / control_pp_tok_s - 1)",
            "100 * (NO_VERIFY_pp_tok_s / control_pp_tok_s - 1)",
        ],
        "admission_timing_use": "gate_only_never_a_throughput_input",
        "schedule_seed": SCHEDULE_SEED, "analysis_seed": ANALYSIS_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "tools": list(TOOLS), "treatments": list(TREATMENTS),
        "correctness_pp": CORRECTNESS_PP, "timing_pp": TIMING_PP,
        "timing_blocks_per_tool": BLOCKS,
        "block_definition": (
            "For one tool and block, exactly one fresh control, STRICT, and NO_VERIFY "
            "llama-bench process; instrumented arms use the same built object and "
            "verifier-enabled runtime."
        ),
        "correctness_schedule": correctness,
        "timing_schedule": timing,
    }


def expected_map(tool: str) -> dict[str, int]:
    if tool == "kernelretsnoop":
        return {"type": 1527, "key_size": 4, "value_size": 32, "max_entries": 44}
    if tool == "threadhist":
        return {"type": 1502, "key_size": 4, "value_size": 8, "max_entries": 1}
    raise ValueError(f"unsupported S0 tool: {tool}")


def read_execution(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        pid = record["identity"]["pid"]
        start_ticks = record["identity"]["start_ticks"]
        if type(pid) is not int or pid <= 0:
            raise ValueError("target pid is not positive")
        if type(start_ticks) is not int or start_ticks <= 0:
            raise ValueError("target start_ticks is not positive")
        if (record.get("cleanup_passed") is not True
                or record.get("timed_out") is not False
                or record.get("returncode") != 0):
            raise ValueError("target completion/cleanup did not pass")
        return record, None
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        return None, f"{type(error).__name__}: {error}"


def parse_admission(log_path: Path, execution_path: Path, *, tool: str,
                    treatment: str, target_symbol: str) -> dict[str, Any]:
    """Audit admission records in the target log; elapsed time is gate-only."""
    execution, execution_error = read_execution(execution_path)
    target_pid = execution["identity"]["pid"] if execution is not None else None
    text = log_path.read_text(errors="replace") if log_path.is_file() else ""
    prefix = r"^\[[^\]\r\n]+\]\[[^\]\r\n]+\]\[(?P<pid>[1-9][0-9]*)\] "
    patterns = {
        "accepted": re.compile(
            prefix + r"GPU eBPF verification accepted: mode=(?P<mode>[^ \r\n]+) "
            + rf"program={PROGRAM} attach=(?P<attach>[^ \r\n]+) "
            + r"instructions=(?P<instructions>[1-9][0-9]*)\r?$"
        ),
        "timing": re.compile(
            prefix + rf"GPU eBPF verification timing: program={PROGRAM} "
            + r"verification_elapsed_ns=(?P<elapsed>[1-9][0-9]*)\r?$"
        ),
        "map": re.compile(
            prefix + rf"GPU eBPF verified map: program={PROGRAM} fd=(?P<fd>[0-9]+) "
            + r"type=(?P<type>[0-9]+) key_size=(?P<key_size>[0-9]+) "
            + r"value_size=(?P<value_size>[0-9]+) max_entries=(?P<max_entries>[0-9]+)\r?$"
        ),
        "skip": re.compile(prefix + rf"Skipping GPU eBPF verification for {PROGRAM}\r?$"),
        "reject": re.compile(prefix + rf"GPU eBPF verification failed for {PROGRAM}:.*$"),
        "unavailable": re.compile(prefix + r".*verifier unavailable.*$"),
    }
    fragments = (
        "GPU eBPF verification accepted:", "GPU eBPF verification timing:",
        "GPU eBPF verified map:", "Skipping GPU eBPF verification",
        "GPU eBPF verification failed", "verifier unavailable", "verification_elapsed_ns=",
    )
    records: dict[str, list[re.Match[str]]] = {name: [] for name in patterns}
    foreign = unparsed = 0
    for line in text.splitlines():
        if not any(fragment in line for fragment in fragments):
            continue
        matches = [(name, pattern.fullmatch(line)) for name, pattern in patterns.items()]
        matches = [(name, match) for name, match in matches if match is not None]
        if len(matches) != 1:
            unparsed += 1
            continue
        name, match = matches[0]
        assert match is not None
        if target_pid is None or int(match.group("pid")) != target_pid:
            foreign += 1
            continue
        records[name].append(match)
    accepted = [{
        "mode": match.group("mode"), "attach": match.group("attach"),
        "instructions": int(match.group("instructions")),
    } for match in records["accepted"]]
    maps = [{name: int(match.group(name))
             for name in ("fd", "type", "key_size", "value_size", "max_entries")}
            for match in records["map"]]
    timing_positive = all(int(match.group("elapsed")) > 0 for match in records["timing"])
    common = (
        execution_error is None and log_path.is_file() and foreign == 0 and unparsed == 0
        and not records["reject"] and not records["unavailable"]
    )
    attach = f"kretprobe/{target_symbol}"
    if treatment == "STRICT":
        passed = (
            common and len(accepted) == 1 and accepted[0]["mode"] == "STRICT"
            and accepted[0]["attach"] == attach and accepted[0]["instructions"] > 0
            and len(records["timing"]) == 1 and timing_positive and len(maps) == 1
            and all(maps[0][key] == value for key, value in expected_map(tool).items())
            and not records["skip"]
        )
    elif treatment == "NO_VERIFY":
        passed = (common and len(records["skip"]) == 1 and not accepted
                  and not records["timing"] and not maps)
    elif treatment == "control":
        passed = (common and not accepted and not records["timing"] and not maps
                  and not records["skip"])
    else:
        raise ValueError(f"unknown treatment: {treatment}")
    return {
        "passed": bool(passed), "treatment": treatment, "program": PROGRAM,
        "attach": attach, "target_pid": target_pid, "execution_error": execution_error,
        "accepted_records": len(accepted), "accepted": accepted,
        "timing_records": len(records["timing"]), "timing_positive": timing_positive,
        "map_records": len(maps), "maps": maps, "expected_map": expected_map(tool),
        "skip_records": len(records["skip"]), "reject_records": len(records["reject"]),
        "unavailable_records": len(records["unavailable"]),
        "foreign_pid_records": foreign, "unparsed_records": unparsed,
        "timing_role": "admission_gate_only_not_steady_state_metric",
    }


def validate_bench_output(parsed: dict[str, Any], args: argparse.Namespace) -> bool:
    raw = parsed.get("raw")
    metrics = parsed.get("metrics", {})
    if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], dict):
        return False
    row = raw[0]
    samples_ns = row.get("samples_ns")
    samples_ts = row.get("samples_ts")
    avg_ts = row.get("avg_ts")
    derived_ts = (
        args.pp * 1e9 / row["avg_ns"]
        if type(row.get("avg_ns")) is int and row["avg_ns"] > 0 else math.nan
    )
    return (
        row.get("n_prompt") == args.pp and row.get("n_gen") == 0
        and row.get("n_gpu_layers") == args.n_gpu_layers
        and Path(str(row.get("model_filename", ""))).resolve() == args.model
        and type(row.get("avg_ns")) is int and row["avg_ns"] > 0
        and isinstance(avg_ts, (int, float)) and math.isfinite(avg_ts) and avg_ts > 0
        and isinstance(samples_ns, list) and len(samples_ns) == 1
        and type(samples_ns[0]) is int and samples_ns[0] > 0
        and samples_ns[0] == row["avg_ns"]
        and isinstance(samples_ts, list) and len(samples_ts) == 1
        and isinstance(samples_ts[0], (int, float)) and math.isfinite(samples_ts[0])
        and samples_ts[0] > 0
        and math.isclose(float(avg_ts), float(samples_ts[0]), rel_tol=1e-6, abs_tol=1e-3)
        and math.isclose(float(avg_ts), derived_ts, rel_tol=1e-6, abs_tol=1e-3)
        and metrics.get("pp_tokens") == args.pp
        and isinstance(metrics.get("pp_tok_s"), float)
        and math.isfinite(metrics["pp_tok_s"]) and metrics["pp_tok_s"] > 0
    )


def injection_environment(environment: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in environment.items()
            if key == "LD_PRELOAD" or key.startswith("BPFTIME_")}


def expected_target_command(args: argparse.Namespace, treatment: str) -> list[str]:
    command = ["taskset", "-c", rq4.CLIENT_CPUS, "/usr/bin/env"]
    if treatment != "control":
        command.append(
            f"LD_PRELOAD={args.bpftime_build_dir / 'runtime/agent/libbpftime-agent.so'}"
        )
    command.extend([
        str(args.llama_bench), "-m", str(args.model), "-r", "1", "-o", "json",
        "-p", str(args.pp), "-n", "0", "-ngl", str(args.n_gpu_layers),
    ])
    return command


def file_metadata(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        return {"path": str(path), "exists": False}
    info = path.stat()
    return {"path": str(path), "exists": True, "bytes": info.st_size,
            "device": info.st_dev, "inode": info.st_ino, "mtime_ns": info.st_mtime_ns}


def git_commit(path: Path) -> str:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def runtime_contract(root: Path) -> dict[str, Any]:
    source = root / "attach/nv_attach_impl/nv_attach_impl.cpp"
    text = source.read_text(encoding="utf-8") if source.is_file() else ""
    required = (TIMING_MARKER, "GPU eBPF verification accepted: mode=STRICT",
                "Skipping GPU eBPF verification for {}", "GPU eBPF verified map: program={}",
                "verifier unavailable")
    return {"source": str(source.resolve()), "required_present": [item in text for item in required],
            "passed": source.is_file() and all(item in text for item in required)}


def binary_markers(path: Path) -> dict[str, bool]:
    encoded = {marker: marker.encode("utf-8") for marker in RUNTIME_BINARY_MARKERS}
    present = {marker: False for marker in RUNTIME_BINARY_MARKERS}
    longest = max(map(len, encoded.values()), default=1)
    overlap = b""
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                window = overlap + chunk
                for marker, value in encoded.items():
                    if not present[marker] and value in window:
                        present[marker] = True
                if all(present.values()):
                    break
                overlap = window[-(longest - 1):] if longest > 1 else b""
    except OSError:
        pass
    return present


def runtime_binary_contract(root: Path, build_dir: Path) -> dict[str, Any]:
    source = root / "attach/nv_attach_impl/nv_attach_impl.cpp"
    source_mtime = source.stat().st_mtime_ns if source.is_file() else None
    binaries: dict[str, Any] = {}
    paths = {
        "agent": build_dir / "runtime/agent/libbpftime-agent.so",
        "syscall_server": build_dir / "runtime/syscall-server/libbpftime-syscall-server.so",
    }
    for name, path in paths.items():
        exists = path.is_file()
        mtime = path.stat().st_mtime_ns if exists else None
        markers = binary_markers(path)
        fresh = bool(source_mtime is not None and mtime is not None and mtime >= source_mtime)
        binaries[name] = {"path": str(path.resolve()), "exists": exists, "mtime_ns": mtime,
                          "fresh_vs_source": fresh, "present": markers,
                          "passed": exists and fresh and all(markers.values())}
    return {"source": str(source.resolve()), "source_mtime_ns": source_mtime,
            "required_markers": list(RUNTIME_BINARY_MARKERS), "binaries": binaries,
            "passed": source_mtime is not None and all(row["passed"] for row in binaries.values())}


def validate_tool_source(tool: str, directory: Path, target_symbol: str) -> None:
    source = directory / f"{tool}.bpf.c"
    text = source.read_text(encoding="utf-8")
    if text.count(f'SEC("kretprobe/{target_symbol}")') != 1:
        raise RuntimeError(f"{tool} does not have the exact requested attach")
    if text.count(f"int {PROGRAM}()") != 1:
        raise RuntimeError(f"{tool} does not have exactly one {PROGRAM}")
    if tool == "kernelretsnoop":
        rq4.validate_kernelretsnoop_source_schema(directory)
    elif any(marker not in text for marker in (
        "BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP 1502", "__uint(max_entries, 1)",
        "__type(key, u32)", "__type(value, u64)",
    )):
        raise RuntimeError("threadhist map contract differs")


def build_tools(args: argparse.Namespace, output: Path) -> dict[str, Path]:
    build_root = output / "gpubpf_tool_build"
    build_root.mkdir()
    result: dict[str, Path] = {}
    for tool in TOOLS:
        directory = rq4.prepare_tool_source(
            rq4.core.TOOLS[tool], bpftime_root=args.bpftime_root,
            build_root=build_root, target_symbol=args.target_symbol,
        )
        validate_tool_source(tool, directory, args.target_symbol)
        rq4.core.build_tool(rq4.core.TOOLS[tool], directory)
        object_path = directory / ".output" / f"{tool}.bpf.o"
        if not object_path.is_file() or object_path.stat().st_size <= 0:
            raise RuntimeError(f"missing built Table 1 object: {object_path}")
        result[tool] = directory
    return result


def artifact_snapshot(args: argparse.Namespace, tool_dirs: dict[str, Path], tool: str) -> dict[str, Any]:
    return {
        "agent": file_metadata(args.bpftime_build_dir / "runtime/agent/libbpftime-agent.so"),
        "syscall_server": file_metadata(
            args.bpftime_build_dir / "runtime/syscall-server/libbpftime-syscall-server.so"
        ),
        "object": file_metadata(tool_dirs[tool] / ".output" / f"{tool}.bpf.o"),
    }


def run_cell(args: argparse.Namespace, output: Path, tool_dirs: dict[str, Path],
             specification: dict[str, Any], stage: str,
             expected_artifacts: dict[str, Any]) -> dict[str, Any]:
    tool = specification["tool"]
    treatment = specification["treatment"]
    args.pp = specification["pp"]
    label = f"{specification['sequence']:03d}-{tool}-{treatment.lower()}"
    directory = output / stage / label
    if directory.exists():
        raise RuntimeError(f"refusing to reuse cell directory: {directory}")
    artifacts_before = artifact_snapshot(args, tool_dirs, tool)
    if artifacts_before != expected_artifacts:
        raise RuntimeError("runtime or object metadata changed before cell")
    directory.mkdir(parents=True)
    rq4.reject_ambient_injection()
    rq4.idle_gpu_or_error(rq4.core.nvidia_smi_snapshot())
    env = rq4.correctness_env(args)
    probe_context = None
    expected_injection: dict[str, str] = {}
    if treatment != "control":
        args.verifier_level = treatment
        probe_context = rq4.private_probe(tool, args, tool_dirs[tool], directory,
                                          exact_exit_oracle=False)
    with rq4.cell_safety(directory) as safety:
        if probe_context is None:
            target_injection = injection_environment(env)
            (directory / "target-environment.json").write_text(
                json.dumps(target_injection, indent=2) + "\n", encoding="utf-8"
            )
            completed = rq4.run_cli_separate(
                rq4.core.make_llama_cmd(args), cwd=rq4.core.WORKLOAD_DIR, env=env,
                timeout=args.timeout_s, log_path=directory / "llama_bench.log",
            )
        else:
            with probe_context as probe_env:
                env.update(probe_env)
                expected_injection = injection_environment(probe_env)
                target_injection = injection_environment(env)
                (directory / "target-environment.json").write_text(
                    json.dumps(target_injection, indent=2) + "\n", encoding="utf-8"
                )
                completed = rq4.run_cli_separate(
                    rq4.core.make_llama_cmd(args), cwd=rq4.core.WORKLOAD_DIR, env=env,
                    timeout=args.timeout_s, log_path=directory / "llama_bench.log",
                )
    try:
        parsed = rq4.core.parse_llama_bench(completed.stdout + "\n" + completed.stderr)
    except (ValueError, KeyError, TypeError) as error:
        parsed = {"raw": [], "metrics": {}, "parse_error": f"{type(error).__name__}: {error}"}
    output_valid = completed.returncode == 0 and validate_bench_output(parsed, args)
    admission = parse_admission(
        directory / "llama_bench.log", directory / "llama_bench.execution.json",
        tool=tool, treatment=treatment, target_symbol=args.target_symbol,
    )
    execution, execution_error = read_execution(directory / "llama_bench.execution.json")
    execution_identity = execution.get("identity") if execution is not None else None
    command_valid = (
        execution_error is None
        and execution is not None
        and execution.get("command") == expected_target_command(args, treatment)
    )
    if treatment == "control":
        environment_valid = target_injection == expected_injection == {}
    else:
        environment_valid = (
            target_injection == expected_injection
            and target_injection.get("LD_PRELOAD")
            == str(args.bpftime_build_dir / "runtime/agent/libbpftime-agent.so")
            and target_injection.get("BPFTIME_VERIFIER_LEVEL") == treatment
            and target_injection.get("BPFTIME_GLOBAL_SHM_NAME") is not None
        )
    probe: dict[str, Any] | None = None
    probe_valid = treatment == "control"
    if treatment != "control":
        probe = rq4.parse_gpubpf(tool, (directory / "probe.log").read_text(errors="replace"))
        if tool == "kernelretsnoop":
            layout = rq4.kernelretsnoop_layout(args.pp, correctness=False)
            probe_valid = rq4.gpubpf_probe_valid(
                tool, probe, expected_thread_count=layout["thread_slots"],
                expected_ring_entries=layout["entries_per_thread"],
                expected_exit_events=layout["events"], expected_exit_launches=layout["launches"],
                expected_exit_coordinates=layout["coordinates"], exact_exit_oracle=False,
            )
        else:
            probe_valid = rq4.gpubpf_probe_valid(
                tool, probe, expected_thread_count=args.threadhist_gpu_thread_count,
                exact_exit_oracle=False,
            )
    valid = (output_valid and admission["passed"] and probe_valid and safety["passed"]
             and command_valid and environment_valid)
    artifacts_after = artifact_snapshot(args, tool_dirs, tool)
    artifacts_stable = artifacts_before == artifacts_after == expected_artifacts
    return {
        "stage": stage, **specification, "directory": str(directory.relative_to(output)),
        "returncode": completed.returncode, "output_valid": bool(output_valid),
        "execution_identity": execution_identity, "command_valid": command_valid,
        "target_injection_environment": target_injection,
        "environment_valid": environment_valid,
        "metrics": parsed.get("metrics", {}), "raw": parsed.get("raw", []),
        "admission": admission, "probe": probe, "probe_valid": bool(probe_valid),
        "safety": safety, "artifacts_before": artifacts_before,
        "artifacts_after": artifacts_after, "artifacts_stable": artifacts_stable,
        "valid": bool(valid and artifacts_stable),
    }


def reconcile_complete_block(cells: list[dict[str, Any]], specification: dict[str, Any]) -> bool:
    block = [cell for cell in cells
             if cell.get("tool") == specification["tool"]
             and cell.get("block") == specification["block"]]
    if len(block) < len(TREATMENTS):
        return True
    if len(block) != len(TREATMENTS) or {cell.get("treatment") for cell in block} != set(TREATMENTS):
        return False
    matched = True
    if specification["tool"] == "threadhist":
        instrumented = {cell["treatment"]: cell for cell in block
                        if cell["treatment"] != "control"}
        strict_count = instrumented["STRICT"].get("probe", {}).get("sample_count")
        off_count = instrumented["NO_VERIFY"].get("probe", {}).get("sample_count")
        matched = type(strict_count) is int and strict_count > 0 and strict_count == off_count
    for cell in block:
        cell["block_event_count_match"] = matched
        cell["valid"] = bool(cell.get("valid")) and matched
    return matched


def defining_inputs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": str(args.model), "llama_bench": str(args.llama_bench),
        "bpftime_root": str(args.bpftime_root), "bpftime_build_dir": str(args.bpftime_build_dir),
        "target_symbol": args.target_symbol, "n_gpu_layers": args.n_gpu_layers,
        "timeout_s": args.timeout_s, "probe_startup_s": args.probe_startup_s,
        "gpu_thread_count": args.gpu_thread_count,
        "threadhist_gpu_thread_count": args.threadhist_gpu_thread_count,
        "uvm": args.uvm, "warmup": True,
    }


def canonical_cell_directory(stage: str, specification: dict[str, Any]) -> str:
    return (f"{stage}/{specification['sequence']:03d}-{specification['tool']}-"
            f"{specification['treatment'].lower()}")


def validate(args: argparse.Namespace) -> dict[str, Any]:
    if args.gpu_thread_count != rq4.EXPECTED_GPU_THREAD_SLOTS:
        raise ValueError(f"kernelretsnoop source build requires {rq4.EXPECTED_GPU_THREAD_SLOTS} slots")
    if args.threadhist_gpu_thread_count != 1048576:
        raise ValueError("threadhist Table 1 contract requires 1048576 entries")
    args.tools = list(TOOLS)
    rq4.core.validate(args)
    build = rq4.verifier_runtime_configuration(args.bpftime_build_dir)
    if any(value.upper() not in {"ON", "YES", "TRUE", "1"} for value in build.values()):
        raise RuntimeError("S0 requires one verifier-enabled CUDA/LLVM build")
    contract = runtime_contract(args.bpftime_root)
    if not contract["passed"]:
        raise RuntimeError("runtime source lacks the fixed admission/timing contract")
    binary_contract = runtime_binary_contract(args.bpftime_root, args.bpftime_build_dir)
    if not binary_contract["passed"]:
        raise RuntimeError(
            "runtime libraries are stale or lack the fixed S0 binary contract; rebuild first"
        )
    if args.no_warmup:
        raise ValueError("S0 requires llama-bench warmup")
    return {"build_configuration": build, "source_contract": contract,
            "binary_contract": binary_contract}


def write_state(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", type=Path, default=rq4.core.DEFAULT_MODEL)
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
    for field in ("model", "llama_bench", "bpftime_root", "bpftime_build_dir"):
        setattr(args, field, getattr(args, field).resolve())
    args.pp = CORRECTNESS_PP
    args.tg = 0
    args.no_warmup = False
    args.uprobe_binary = rq4.core.DEFAULT_LAUNCH_STUB_LIBRARY.resolve()
    args.uprobe_symbol_hint = args.target_symbol
    args.tools = list(TOOLS)
    args.verifier_level = "STRICT"
    return args


def execute(args: argparse.Namespace) -> int:
    rq4.reject_ambient_injection()
    validation = validate(args)
    plan = fixed_plan()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (args.output_dir or HERE / "raw" / f"full-{timestamp}").resolve()
    result_path = output / "result.json"
    if args.resume:
        state = json.loads(result_path.read_text(encoding="utf-8"))
        if state.get("schema") != SCHEMA or state.get("plan") != plan:
            raise RuntimeError("resume plan differs")
        if state.get("defining_inputs") != defining_inputs(args):
            raise RuntimeError("resume inputs differ")
        for key, schedule in (("correctness_cells", plan["correctness_schedule"]),
                              ("timing_cells", plan["timing_schedule"])):
            cells = state.get(key)
            if not isinstance(cells, list):
                raise RuntimeError(f"resume {key} is not a list")
            sequences = [cell.get("sequence") for cell in cells if isinstance(cell, dict)]
            expected_prefix = [cell["sequence"] for cell in schedule[:len(cells)]]
            if len(sequences) != len(cells) or sequences != expected_prefix:
                raise RuntimeError(f"resume {key} is not an exact schedule prefix")
            for cell, specification in zip(cells, schedule):
                if cell.get("directory") != canonical_cell_directory(
                    "correctness" if key == "correctness_cells" else "timing",
                    specification,
                ):
                    raise RuntimeError(f"resume {key} has a non-canonical directory")
        existing = state["correctness_cells"] + state["timing_cells"]
        directories = [cell.get("directory") for cell in existing]
        identities = [cell.get("execution_identity") for cell in existing]
        identity_pairs = [
            (identity.get("pid"), identity.get("start_ticks"))
            for identity in identities if isinstance(identity, dict)
        ]
        if (len(set(directories)) != len(directories)
                or len(identity_pairs) != len(identities)
                or len(set(identity_pairs)) != len(identity_pairs)
                or any(type(pid) is not int or type(ticks) is not int or pid <= 0 or ticks <= 0
                       for pid, ticks in identity_pairs)):
            raise RuntimeError("resume reuses a raw directory or target execution identity")
        tool_dirs = {tool: output / "gpubpf_tool_build" / tool for tool in TOOLS}
        if any(not (directory / tool).is_file() for tool, directory in tool_dirs.items()):
            raise RuntimeError("resume tool build is incomplete")
    else:
        if output.exists() and any(output.iterdir()):
            raise RuntimeError("refusing nonempty output directory")
        output.mkdir(parents=True, exist_ok=True)
        snapshot = rq4.core.nvidia_smi_snapshot()
        if rq4.parse_driver(snapshot) != rq4.EXPECTED_DRIVER:
            raise RuntimeError(f"S0 requires driver {rq4.EXPECTED_DRIVER}")
        rq4.idle_gpu_or_error(snapshot)
        tool_dirs = build_tools(args, output)
        state = {
            "schema": SCHEMA, "status": "running", "created": timestamp,
            "plan": plan, "defining_inputs": defining_inputs(args),
            "host": {"driver": rq4.parse_driver(snapshot), "expected_driver": rq4.EXPECTED_DRIVER,
                     "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                     "nvidia_smi": snapshot},
            "runtime": {**validation, "git_commit": git_commit(args.bpftime_root),
                        "agent": file_metadata(args.bpftime_build_dir / "runtime/agent/libbpftime-agent.so"),
                        "syscall_server": file_metadata(args.bpftime_build_dir / "runtime/syscall-server/libbpftime-syscall-server.so")},
            "objects": {tool: file_metadata(tool_dirs[tool] / ".output" / f"{tool}.bpf.o")
                        for tool in TOOLS},
            "correctness_cells": [], "timing_cells": [],
        }
        write_state(result_path, state)
    for stage, key, specifications in (
        ("correctness", "correctness_cells", plan["correctness_schedule"]),
        ("timing", "timing_cells", plan["timing_schedule"]),
    ):
        for specification in specifications:
            matches = [cell for cell in state[key]
                       if cell.get("sequence") == specification["sequence"]]
            if matches:
                if (len(matches) != 1
                        or any(matches[0].get(name) != value for name, value in specification.items())
                        or matches[0].get("stage") != stage):
                    raise RuntimeError("recorded cell differs from fixed schedule")
                if matches[0].get("valid") is not True:
                    state["status"] = f"invalid_{stage}"
                    write_state(result_path, state)
                    return 2
                continue
            expected_artifacts = {
                "agent": state["runtime"]["agent"],
                "syscall_server": state["runtime"]["syscall_server"],
                "object": state["objects"][specification["tool"]],
            }
            cell = run_cell(
                args, output, tool_dirs, specification, stage, expected_artifacts
            )
            prior_cells = state["correctness_cells"] + state["timing_cells"]
            prior_identities = {
                (row["execution_identity"]["pid"], row["execution_identity"]["start_ticks"])
                for row in prior_cells
                if isinstance(row.get("execution_identity"), dict)
            }
            identity = cell.get("execution_identity")
            identity_pair = (
                (identity.get("pid"), identity.get("start_ticks"))
                if isinstance(identity, dict) else None
            )
            if (cell.get("directory") != canonical_cell_directory(stage, specification)
                    or identity_pair is None
                    or type(identity_pair[0]) is not int or identity_pair[0] <= 0
                    or type(identity_pair[1]) is not int or identity_pair[1] <= 0
                    or identity_pair in prior_identities):
                cell["identity_directory_valid"] = False
                cell["valid"] = False
            else:
                cell["identity_directory_valid"] = True
            state[key].append(cell)
            block_valid = reconcile_complete_block(state[key], specification)
            write_state(result_path, state)
            if not cell["valid"] or not block_valid:
                state["status"] = f"invalid_{stage}"
                write_state(result_path, state)
                return 2
    state["status"] = "complete"
    write_state(result_path, state)
    print(result_path)
    return 0


def main() -> int:
    args = parse_args()
    if args.dry_run:
        print(json.dumps(fixed_plan(), indent=2))
        return 0
    leases = rq4.ReadOnlyLeases()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    previous_run_cmd = rq4.core.run_cmd
    try:
        rq4.core.run_cmd = rq4.run_cmd_owned
        return execute(args)
    finally:
        rq4.core.run_cmd = previous_run_cmd
        signal.signal(signal.SIGTERM, previous_handler)
        leases.close()


if __name__ == "__main__":
    raise SystemExit(main())
