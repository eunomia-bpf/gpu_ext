#!/usr/bin/env python3
"""Independently audit raw S0 logs and compute paired throughput effects."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any


TOOLS = ("kernelretsnoop", "threadhist")
TREATMENTS = ("control", "STRICT", "NO_VERIFY")
SCHEMA = "device-verifier-s0-v1"
SCHEDULE_SEED = 1797
ANALYSIS_SEED = 9174
BOOTSTRAP_SAMPLES = 10000
BLOCKS = 10
CORRECTNESS_PP = 32
TIMING_PP = 512
EXPECTED_DRIVER = "575.57.08"
EXPECTED_WORKER_CPUS = "8-15"
EXPECTED_POWER_LIMIT_W = 400.0
MAX_IDLE_MEMORY_MIB = 256
PROGRAM = "cuda__retprobe"
SAMPLE_TS_SIGNIFICANT_DIGITS = 6
BUILD_KEYS = ("ENABLE_EBPF_VERIFIER", "BPFTIME_ENABLE_CUDA_ATTACH", "BPFTIME_LLVM_JIT")
TELEMETRY_HEADERS = (
    "timestamp", "memory.used [MiB]", "temperature.gpu", "power.draw [W]",
    "clocks.current.sm [MHz]", "clocks.current.memory [MHz]",
    "clocks_event_reasons.sw_power_cap", "clocks_event_reasons.hw_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.hw_power_brake_slowdown",
    "clocks_event_reasons.sw_thermal_slowdown",
)


def randomized_orders(count: int, rng: random.Random) -> list[list[str]]:
    result: list[list[str]] = []
    while len(result) < count:
        cycle = [list(order) for order in itertools.permutations(TREATMENTS)]
        rng.shuffle(cycle)
        result.extend(cycle)
    return result[:count]


def fixed_schedules() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
            correctness.append({"sequence": sequence, "tool": tool, "block": 0,
                                "position": position, "treatment": treatment,
                                "pp": CORRECTNESS_PP})
    orders = {tool: randomized_orders(BLOCKS, rng) for tool in TOOLS}
    for block in range(1, BLOCKS + 1):
        block_tools = list(TOOLS)
        rng.shuffle(block_tools)
        for tool in block_tools:
            for position, treatment in enumerate(orders[tool][block - 1], start=1):
                sequence += 1
                timing.append({"sequence": sequence, "tool": tool, "block": block,
                               "position": position, "treatment": treatment,
                               "pp": TIMING_PP})
    return correctness, timing


def safe_directory(root: Path, value: Any) -> Path:
    if not isinstance(value, str):
        raise ValueError("cell directory is not a string")
    path = (root / value).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("cell directory escapes result root")
    return path


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def log_sections(path: Path) -> tuple[str, str, int]:
    text = path.read_text(errors="replace")
    if text.count("\n## stdout\n") != 1 or text.count("\n## stderr\n") != 1:
        raise ValueError("target log lacks unique stdout/stderr sections")
    after = text.split("\n## stdout\n", 1)[1]
    stdout, stderr_tail = after.split("\n## stderr\n", 1)
    if len(re.findall(r"(?m)^# exit: ", stderr_tail)) != 1:
        raise ValueError("target log lacks one exit footer")
    match = re.fullmatch(r"(?s)(.*)\n# exit: ([+-]?[0-9]+)\n", stderr_tail)
    if match is None:
        raise ValueError("target log exit footer is not one final signed integer")
    return stdout, match.group(1), int(match.group(2))


def finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def safety_snapshot_valid(snapshot: Any) -> bool:
    if not isinstance(snapshot, dict):
        return False
    gpu = snapshot.get("gpu")
    struct_ops = snapshot.get("struct_ops")
    return (
        type(snapshot.get("timestamp_ns")) is int and snapshot["timestamp_ns"] > 0
        and snapshot.get("power_limit_service") == "active"
        and finite_number(snapshot.get("power_limit_w"))
        and abs(float(snapshot["power_limit_w"]) - EXPECTED_POWER_LIMIT_W) <= 0.01
        and type(snapshot.get("uvm_refcount")) is int and snapshot["uvm_refcount"] == 0
        and snapshot.get("dmesg_abnormal") == []
        and snapshot.get("journal_abnormal") == []
        and snapshot.get("xids") == []
        and isinstance(struct_ops, dict)
        and struct_ops.get("maps") == [] and struct_ops.get("links") == []
        and isinstance(gpu, dict)
        and gpu.get("driver") == EXPECTED_DRIVER
        and gpu.get("compute_apps") == []
        and finite_number(gpu.get("memory_used_mib"))
        and 0 <= float(gpu["memory_used_mib"]) <= MAX_IDLE_MEMORY_MIB
        and type(gpu.get("utilization_gpu_percent")) is int
        and gpu["utilization_gpu_percent"] == 0
    )


def telemetry_number(row: dict[str, str], header: str) -> float:
    value = float(row[header].split()[0])
    if not math.isfinite(value):
        raise ValueError(f"non-finite telemetry value for {header}")
    return value


def replay_gpu_telemetry(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8", errors="replace") as stream:
        reader = csv.DictReader(stream, skipinitialspace=True)
        if tuple(reader.fieldnames or ()) != TELEMETRY_HEADERS:
            raise ValueError("GPU telemetry headers differ from the fixed query")
        rows = list(reader)
    if not rows:
        raise ValueError("GPU telemetry has no samples")
    reason_headers = TELEMETRY_HEADERS[6:]
    fixed_power_cap_samples = 0
    for row in rows:
        if (None in row or set(row) != set(TELEMETRY_HEADERS)
                or any(value is None for value in row.values())):
            raise ValueError("GPU telemetry row has the wrong number of fields")
        if not row["timestamp"].strip():
            raise ValueError("GPU telemetry timestamp is empty")
        for header in reason_headers:
            reason = row[header].strip().lower()
            allowed = {"not active", "n/a"}
            if header == "clocks_event_reasons.sw_power_cap":
                allowed.add("active")
                fixed_power_cap_samples += reason == "active"
            if reason not in allowed:
                raise ValueError(f"GPU throttling observed in {header}: {row[header]}")
    memory = [telemetry_number(row, TELEMETRY_HEADERS[1]) for row in rows]
    temperature = [telemetry_number(row, TELEMETRY_HEADERS[2]) for row in rows]
    power = [telemetry_number(row, TELEMETRY_HEADERS[3]) for row in rows]
    sm_clock = [telemetry_number(row, TELEMETRY_HEADERS[4]) for row in rows]
    if (any(value < 0 for value in memory + power + sm_clock)
            or any(telemetry_number(row, TELEMETRY_HEADERS[5]) < 0 for row in rows)):
        raise ValueError("GPU telemetry contains a negative physical value")
    return {
        "samples": len(rows), "peak_memory_mib": max(memory),
        "peak_temperature_c": max(temperature),
        "mean_power_w": sum(power) / len(rows),
        "min_sm_clock_mhz": min(sm_clock), "max_sm_clock_mhz": max(sm_clock),
        "throttled": False, "fixed_power_cap_samples": fixed_power_cap_samples,
    }


def safety_gate(directory: Path, safety: dict[str, Any], *, boot_id: str) -> dict[str, Any]:
    forbidden = ("error", "cleanup_errors", "fatal_cleanup")
    before = safety.get("before")
    after = safety.get("after")
    replayed_telemetry = replay_gpu_telemetry(directory / "gpu-telemetry.csv")
    valid = (
        safety.get("passed") is True
        and not any(key in safety for key in forbidden)
        and isinstance(boot_id, str) and bool(boot_id)
        and safety.get("boot_id") == boot_id
        and safety.get("worker_cpus") == EXPECTED_WORKER_CPUS
        and safety_snapshot_valid(before) and safety_snapshot_valid(after)
        and after["timestamp_ns"] >= before["timestamp_ns"]
        and safety.get("telemetry") == replayed_telemetry
    )
    return {"valid": bool(valid), "replayed_telemetry": replayed_telemetry}


def parse_one_bench_array(stdout: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    candidates: list[list[dict[str, Any]]] = []
    for match in re.finditer(r"(?m)^\s*\[", stdout):
        try:
            value, _ = decoder.raw_decode(stdout[match.start():].lstrip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, list) and all(isinstance(row, dict) for row in value):
            candidates.append(value)
    if len(candidates) != 1:
        raise ValueError("target stdout does not contain exactly one JSON object array")
    return candidates[0]


def sample_ts_matches_average(average: float, printed_sample: float) -> bool:
    """Match llama-bench's default-stream, six-significant-digit sample output."""
    if not (math.isfinite(average) and average > 0
            and math.isfinite(printed_sample) and printed_sample > 0):
        return False
    exponent = math.floor(math.log10(abs(average)))
    half_print_unit = 0.5 * 10 ** (exponent - SAMPLE_TS_SIGNIFICANT_DIGITS + 1)
    return abs(average - printed_sample) <= half_print_unit * (1 + 1e-12)


def bench_gate(stdout: str, pp: int, model: Path, n_gpu_layers: int) -> dict[str, Any]:
    raw = parse_one_bench_array(stdout)
    if len(raw) != 1:
        return {"valid": False, "error": "expected exactly one benchmark row", "raw": raw}
    row = raw[0]
    samples_ns = row.get("samples_ns")
    samples_ts = row.get("samples_ts")
    throughput = row.get("avg_ts")
    build_commit = row.get("build_commit")
    build_number = row.get("build_number")
    derived = (pp * 1e9 / row["avg_ns"]
               if type(row.get("avg_ns")) is int and row["avg_ns"] > 0 else math.nan)
    valid = (
        row.get("n_prompt") == pp and row.get("n_gen") == 0
        and row.get("n_gpu_layers") == n_gpu_layers
        and Path(str(row.get("model_filename", ""))).resolve() == model.resolve()
        and isinstance(build_commit, str) and bool(build_commit)
        and type(build_number) is int and build_number > 0
        and type(row.get("avg_ns")) is int and row["avg_ns"] > 0
        and isinstance(throughput, (int, float)) and math.isfinite(throughput)
        and throughput > 0
        and isinstance(samples_ns, list) and len(samples_ns) == 1
        and type(samples_ns[0]) is int and samples_ns[0] > 0
        and samples_ns[0] == row["avg_ns"]
        and isinstance(samples_ts, list) and len(samples_ts) == 1
        and isinstance(samples_ts[0], (int, float)) and math.isfinite(samples_ts[0])
        and samples_ts[0] > 0
        and sample_ts_matches_average(float(throughput), float(samples_ts[0]))
        and math.isclose(float(throughput), derived, rel_tol=1e-6, abs_tol=1e-3)
    )
    return {"valid": bool(valid), "pp_tok_s": float(throughput) if valid else None,
            "avg_ns": row.get("avg_ns"), "build_commit": build_commit,
            "build_number": build_number, "raw": raw}


def injection_environment(environment: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in environment.items()
            if key == "LD_PRELOAD" or key.startswith("BPFTIME_")}


def expected_command(defining: dict[str, Any], treatment: str, pp: int) -> list[str]:
    command = ["taskset", "-c", EXPECTED_WORKER_CPUS, "/usr/bin/env"]
    if treatment != "control":
        build = Path(defining["bpftime_build_dir"])
        command.append(f"LD_PRELOAD={build / 'runtime/agent/libbpftime-agent.so'}")
    command.extend([
        defining["llama_bench"], "-m", defining["model"], "-r", "1", "-o", "json",
        "-p", str(pp), "-n", "0", "-ngl", str(defining["n_gpu_layers"]),
    ])
    return command


def canonical_directory(stage: str, expected: dict[str, Any]) -> str:
    return (f"{stage}/{expected['sequence']:03d}-{expected['tool']}-"
            f"{expected['treatment'].lower()}")


def expected_map(tool: str) -> dict[str, int]:
    return ({"type": 1527, "key_size": 4, "value_size": 32, "max_entries": 44}
            if tool == "kernelretsnoop"
            else {"type": 1502, "key_size": 4, "value_size": 8, "max_entries": 1})


def admission_gate(text: str, pid: int | None, tool: str, treatment: str,
                   target_symbol: str) -> dict[str, Any]:
    prefix = r"^\[[^\]\r\n]+\]\[[^\]\r\n]+\]\[(?P<pid>[1-9][0-9]*)\] "
    patterns = {
        "accepted": re.compile(
            prefix + r"GPU eBPF verification accepted: mode=(?P<mode>[^ \r\n]+) "
            + rf"program={PROGRAM} attach=(?P<attach>[^ \r\n]+) "
            + r"instructions=(?P<instructions>[1-9][0-9]*)\r?$"),
        "timing": re.compile(prefix + rf"GPU eBPF verification timing: program={PROGRAM} "
                             + r"verification_elapsed_ns=(?P<elapsed>[1-9][0-9]*)\r?$"),
        "map": re.compile(prefix + rf"GPU eBPF verified map: program={PROGRAM} fd=(?P<fd>[0-9]+) "
                          + r"type=(?P<type>[0-9]+) key_size=(?P<key_size>[0-9]+) "
                          + r"value_size=(?P<value_size>[0-9]+) max_entries=(?P<max_entries>[0-9]+)\r?$"),
        "skip": re.compile(prefix + rf"Skipping GPU eBPF verification for {PROGRAM}\r?$"),
        "reject": re.compile(prefix + rf"GPU eBPF verification failed for {PROGRAM}:.*$"),
        "unavailable": re.compile(prefix + r".*verifier unavailable.*$"),
    }
    fragments = ("GPU eBPF verification accepted:", "GPU eBPF verification timing:",
                 "GPU eBPF verified map:", "Skipping GPU eBPF verification",
                 "GPU eBPF verification failed", "verifier unavailable", "verification_elapsed_ns=")
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
        if pid is None or int(match.group("pid")) != pid:
            foreign += 1
        else:
            records[name].append(match)
    accepted = [{"mode": match.group("mode"), "attach": match.group("attach"),
                 "instructions": int(match.group("instructions"))}
                for match in records["accepted"]]
    maps = [{name: int(match.group(name))
             for name in ("fd", "type", "key_size", "value_size", "max_entries")}
            for match in records["map"]]
    common = (pid is not None and foreign == 0 and unparsed == 0
              and not records["reject"] and not records["unavailable"])
    attach = f"kretprobe/{target_symbol}"
    if treatment == "STRICT":
        valid = (common and len(accepted) == 1 and accepted[0]["mode"] == "STRICT"
                 and accepted[0]["attach"] == attach and accepted[0]["instructions"] > 0
                 and len(records["timing"]) == 1
                 and int(records["timing"][0].group("elapsed")) > 0
                 and len(maps) == 1
                 and all(maps[0][key] == value for key, value in expected_map(tool).items())
                 and not records["skip"])
    elif treatment == "NO_VERIFY":
        valid = (common and len(records["skip"]) == 1 and not accepted
                 and not records["timing"] and not maps)
    else:
        valid = (common and not records["skip"] and not accepted
                 and not records["timing"] and not maps)
    return {"valid": bool(valid), "accepted_records": len(accepted),
            "timing_records": len(records["timing"]),
            "timing_positive": (len(records["timing"]) == 1
                                and int(records["timing"][0].group("elapsed")) > 0),
            "map_records": len(maps), "maps": maps, "skip_records": len(records["skip"]),
            "reject_records": len(records["reject"]),
            "unavailable_records": len(records["unavailable"]),
            "foreign_pid_records": foreign, "unparsed_records": unparsed,
            "admission_timing_used_in_throughput": False}


def last_int(text: str, label: str) -> int:
    values = re.findall(rf"^{re.escape(label)}:\s*(\d+)$", text, re.MULTILINE)
    return int(values[-1]) if values else -1


def probe_gate(path: Path, tool: str, pp: int) -> dict[str, Any]:
    text = path.read_text(errors="replace") if path.is_file() else ""
    if tool == "threadhist":
        result = {"samples": last_int(text, "Total exit probes"),
                  "nonzero_threads": last_int(text, "Nonzero threads"),
                  "configured": last_int(text, "Configured thread entries"),
                  "readback": last_int(text, "Readback entries"),
                  "bytes": last_int(text, "Readback bytes"),
                  "complete": last_int(text, "Readback complete")}
        valid = (result["samples"] > 0 and result["nonzero_threads"] > 0
                 and result["configured"] == result["readback"] == 1048576
                 and result["bytes"] == 8388608 and result["complete"] == 1)
        return {"valid": valid, **result}
    labels = {
        "samples": "Total events collected", "nonzero": "Nonzero timestamps",
        "requested": "Requested thread slots", "allocated": "Allocated thread slots",
        "entries": "Ring entries per thread", "requested_entries": "Requested ring entries per thread",
        "record_bytes": "Record bytes", "committed": "Committed events",
        "runtime": "Runtime collected events", "oob": "OOB drops", "full": "Full drops",
        "bad_size": "Bad-size drops", "other": "Other drops", "dirty": "Dirty slots",
        "pending": "Pending events", "final_drain": "Final drain events",
        "second_drain": "Second drain events", "launches": "Cartesian launches",
        "coordinates": "Cartesian coordinates", "complete": "Cartesian complete",
        "extent_x": "Coordinate extent x", "extent_y": "Coordinate extent y",
        "extent_z": "Coordinate extent z", "m220": "Coordinate multiplicity 220",
        "m44": "Coordinate multiplicity 44", "m22": "Coordinate multiplicity 22",
        "mother": "Coordinate multiplicity other", "mismatch": "Coordinate segment mismatches",
        "invalid": "Invalid launch coordinates", "unique": "Unique coordinates",
        "oracle_enabled": "Multiplicity oracle enabled", "oracle_total": "Multiplicity oracle total events",
        "oracle_passed": "Multiplicity oracle passed", "collector": "Collector gate passed",
    }
    value = {name: last_int(text, label) for name, label in labels.items()}
    slots = pp * 1024
    events = slots * 44
    zero = ("oob", "full", "bad_size", "other", "dirty", "pending", "second_drain",
            "invalid", "m220", "m22", "mother", "mismatch", "oracle_enabled", "oracle_passed")
    valid = (value["samples"] == value["nonzero"] == value["committed"]
             == value["runtime"] == value["oracle_total"] == events
             and value["requested"] == value["allocated"] == value["coordinates"]
             == value["unique"] == slots
             and value["entries"] == value["requested_entries"] == 44
             and value["record_bytes"] == 32 and all(value[name] == 0 for name in zero)
             and 0 <= value["final_drain"] <= events and value["launches"] == 44
             and value["complete"] == value["collector"] == 1
             and (value["extent_x"], value["extent_y"], value["extent_z"])
             == (slots // 256, 256, 1) and value["m44"] == slots)
    return {"valid": valid, **value}


def audit_cell(root: Path, cell: dict[str, Any], expected: dict[str, Any],
               defining: dict[str, Any]) -> dict[str, Any]:
    identity_ok = cell.get("stage") == expected["stage"] and all(
        cell.get(key) == value for key, value in expected.items() if key != "stage")
    canonical = canonical_directory(expected["stage"], expected)
    directory_name_ok = cell.get("directory") == canonical
    try:
        directory = safe_directory(root, cell.get("directory"))
        execution = read_json(directory / "llama_bench.execution.json")
        stdout, stderr, log_exit = log_sections(directory / "llama_bench.log")
        pid = execution.get("identity", {}).get("pid")
        start_ticks = execution.get("identity", {}).get("start_ticks")
        execution_identity = (pid, start_ticks)
        execution_ok = (type(pid) is int and pid > 0
                        and type(start_ticks) is int and start_ticks > 0
                        and execution.get("cleanup_passed") is True
                        and execution.get("timed_out") is False
                        and execution.get("returncode") == log_exit == 0
                        and "error" not in execution
                        and "cleanup_failure" not in execution)
        command = execution.get("command", [])
        command_ok = command == expected_command(defining, expected["treatment"], expected["pp"])
        target_environment = read_json(directory / "target-environment.json")
        safety = read_json(directory / "gpu-safety.json")
        safety = safety_gate(directory, safety, boot_id=defining.get("host_boot_id"))
        safety_ok = safety["valid"]
        bench = bench_gate(stdout, expected["pp"], Path(defining["model"]),
                           int(defining["n_gpu_layers"]))
        admission = admission_gate(stdout + "\n" + stderr, pid, expected["tool"],
                                   expected["treatment"], defining["target_symbol"])
        private_ok = True
        environment_ok = True
        probe = None
        segment = None
        if expected["treatment"] == "control":
            private_ok = not (directory / "probe-execution.json").exists() and not (directory / "probe.log").exists()
            environment_ok = target_environment == {}
        else:
            probe_execution = read_json(directory / "probe-execution.json")
            segment = probe_execution.get("private_segment")
            agent = probe_execution.get("agent_environment", {})
            loader = probe_execution.get("loader_environment", {})
            build = Path(defining["bpftime_build_dir"])
            private_ok = (isinstance(segment, str) and segment.startswith("rq4_")
                          and probe_execution.get("private_segment_removed") is True
                          and probe_execution.get("loader_preserved") is not True)
            expected_injection = injection_environment(agent)
            environment_ok = (target_environment == expected_injection
                              and agent.get("BPFTIME_GLOBAL_SHM_NAME") == segment
                              and loader.get("BPFTIME_GLOBAL_SHM_NAME") == segment
                              and agent.get("BPFTIME_VERIFIER_LEVEL") == expected["treatment"]
                              and loader.get("BPFTIME_VERIFIER_LEVEL") == expected["treatment"]
                              and agent.get("SPDLOG_LEVEL") == loader.get("SPDLOG_LEVEL") == "info"
                              and agent.get("LD_PRELOAD") == str(build / "runtime/agent/libbpftime-agent.so")
                              and loader.get("LD_PRELOAD") == str(build / "runtime/syscall-server/libbpftime-syscall-server.so"))
            probe = probe_gate(directory / "probe.log", expected["tool"], expected["pp"])
        probe_ok = probe is None or probe["valid"]
        expected_artifacts = {
            "agent": defining.get("runtime_agent_metadata"),
            "syscall_server": defining.get("runtime_syscall_server_metadata"),
            "object": defining.get("object_metadata", {}).get(expected["tool"]),
        }
        artifacts_ok = (cell.get("artifacts_stable") is True
                        and cell.get("artifacts_before") == expected_artifacts
                        and cell.get("artifacts_after") == expected_artifacts)
        recorded_identity_ok = cell.get("execution_identity") == execution.get("identity")
        valid = (identity_ok and directory_name_ok and recorded_identity_ok
                 and cell.get("identity_directory_valid") is True
                 and cell.get("valid") is True and execution_ok and command_ok
                 and safety_ok and bench["valid"] and admission["valid"]
                 and private_ok and environment_ok and probe_ok and artifacts_ok)
        return {"valid": bool(valid), "identity_valid": identity_ok,
                "directory": str(directory), "canonical_directory_valid": directory_name_ok,
                "execution_identity": execution_identity,
                "recorded_execution_identity_valid": recorded_identity_ok,
                "recorded_valid": cell.get("valid") is True,
                "execution_valid": execution_ok, "log_exit": log_exit,
                "command_valid": command_ok, "safety_valid": safety_ok,
                "safety": safety, "bench": bench, "admission": admission,
                "private_shm_valid": private_ok, "environment_valid": environment_ok,
                "artifact_inventory_valid": artifacts_ok,
                "private_segment": segment, "probe": probe}
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        return {"valid": False, "error": f"{type(error).__name__}: {error}"}


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_effect(values: list[float], seed_offset: int) -> dict[str, Any]:
    rng = random.Random(ANALYSIS_SEED + seed_offset)
    bootstrap = [statistics.fmean(rng.choices(values, k=len(values)))
                 for _ in range(BOOTSTRAP_SAMPLES)]
    return {"samples_percent": values, "mean_percent": statistics.fmean(values),
            "median_percent": statistics.median(values),
            "bootstrap_mean_95_ci_percent": [quantile(bootstrap, 0.025), quantile(bootstrap, 0.975)]}


def analyze(root: Path) -> dict[str, Any]:
    root = root.resolve()
    errors: list[str] = []
    try:
        state = read_json(root / "result.json")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {"schema": SCHEMA, "complete": False, "run_status": "invalid",
                "errors": [f"cannot read result.json: {error}"]}
    correctness_schedule, timing_schedule = fixed_schedules()
    plan = state.get("plan", {})
    if state.get("schema") != SCHEMA or plan.get("schema") != SCHEMA:
        errors.append("schema mismatch")
    if plan.get("correctness_schedule") != correctness_schedule or plan.get("timing_schedule") != timing_schedule:
        errors.append("fixed schedule mismatch")
    if (plan.get("schedule_seed") != SCHEDULE_SEED or plan.get("analysis_seed") != ANALYSIS_SEED
            or plan.get("bootstrap_samples") != BOOTSTRAP_SAMPLES):
        errors.append("fixed analysis protocol mismatch")
    if plan.get("admission_timing_use") != "gate_only_never_a_throughput_input":
        errors.append("admission timing role mismatch")
    host = state.get("host", {})
    if (not isinstance(host, dict) or host.get("driver") != EXPECTED_DRIVER
            or host.get("expected_driver") != EXPECTED_DRIVER):
        errors.append("driver gate mismatch")
    host_boot_id = host.get("boot_id") if isinstance(host, dict) else None
    if not isinstance(host_boot_id, str) or not host_boot_id:
        errors.append("missing run boot identity")
    runtime = state.get("runtime", {})
    build = runtime.get("build_configuration", {})
    if any(str(build.get(key, "")).upper() not in {"ON", "YES", "TRUE", "1"} for key in BUILD_KEYS):
        errors.append("runtime is not verifier/CUDA/LLVM enabled")
    if runtime.get("source_contract", {}).get("passed") is not True:
        errors.append("runtime source contract failed")
    if runtime.get("binary_contract", {}).get("passed") is not True:
        errors.append("runtime binary marker/freshness contract failed")
    for name in ("agent", "syscall_server"):
        metadata = runtime.get(name, {})
        if metadata.get("exists") is not True or type(metadata.get("bytes")) is not int or metadata["bytes"] <= 0:
            errors.append(f"missing runtime {name} inventory")
    for tool in TOOLS:
        metadata = state.get("objects", {}).get(tool, {})
        if metadata.get("exists") is not True or type(metadata.get("bytes")) is not int or metadata["bytes"] <= 0:
            errors.append(f"missing {tool} object inventory")
    defining = state.get("defining_inputs", {})
    defining = dict(defining)
    defining["runtime_agent_metadata"] = runtime.get("agent")
    defining["runtime_syscall_server_metadata"] = runtime.get("syscall_server")
    defining["object_metadata"] = state.get("objects", {})
    defining["host_boot_id"] = host_boot_id
    if defining.get("warmup") is not True:
        errors.append("warmup was not fixed on")
    audited: dict[str, list[dict[str, Any]]] = {"correctness": [], "timing": []}
    segments: list[str] = []
    raw_directories: list[str] = []
    execution_identities: list[tuple[int, int]] = []
    for stage, key, schedule in (("correctness", "correctness_cells", correctness_schedule),
                                 ("timing", "timing_cells", timing_schedule)):
        cells = state.get(key, [])
        if not isinstance(cells, list):
            errors.append(f"{key} is not a list")
            continue
        expected_sequences = {item["sequence"] for item in schedule}
        actual_sequences = [cell.get("sequence") for cell in cells if isinstance(cell, dict)]
        if len(cells) != len(schedule) or set(actual_sequences) != expected_sequences or len(set(actual_sequences)) != len(actual_sequences):
            errors.append(f"{stage} cell inventory has missing, extra, or duplicate cells")
        for expected in schedule:
            matches = [cell for cell in cells if isinstance(cell, dict)
                       and cell.get("sequence") == expected["sequence"]]
            if len(matches) != 1:
                continue
            audit = audit_cell(root, matches[0], {"stage": stage, **expected}, defining)
            audited[stage].append({**expected, **audit})
            if not audit["valid"]:
                errors.append(f"{stage} sequence {expected['sequence']} failed")
            if isinstance(audit.get("private_segment"), str):
                segments.append(audit["private_segment"])
            if isinstance(audit.get("directory"), str):
                raw_directories.append(audit["directory"])
            identity = audit.get("execution_identity")
            if (isinstance(identity, tuple) and len(identity) == 2
                    and all(type(value) is int and value > 0 for value in identity)):
                execution_identities.append(identity)
    if len(segments) != 44 or len(set(segments)) != len(segments):
        errors.append("instrumented cells do not have 44 unique private SHM segments")
    if len(raw_directories) != 66 or len(set(raw_directories)) != 66:
        errors.append("the 66 cells do not have unique canonical raw directories")
    if len(execution_identities) != 66 or len(set(execution_identities)) != 66:
        errors.append("the 66 target (pid,start_ticks) identities are missing or reused")
    build_records = [
        (cell.get("bench", {}).get("build_commit"),
         cell.get("bench", {}).get("build_number"))
        for cells in audited.values() for cell in cells
    ]
    build_consistent = len(build_records) == 66 and len(set(build_records)) == 1
    if not build_consistent:
        errors.append("the 66 cells do not report one consistent llama-bench build")
    build_identity = {
        "consistent": build_consistent,
        "cells": len(build_records),
        "build_commit": build_records[0][0] if build_consistent else None,
        "build_number": build_records[0][1] if build_consistent else None,
    }
    for stage, blocks in (("correctness", (0,)), ("timing", range(1, BLOCKS + 1))):
        for block in blocks:
            cells = [cell for cell in audited[stage]
                     if cell["tool"] == "threadhist" and cell["block"] == block]
            if len(cells) != 3:
                continue
            instrumented = {cell["treatment"]: cell for cell in cells
                            if cell["treatment"] != "control"}
            strict_count = (instrumented.get("STRICT", {}).get("probe") or {}).get("samples")
            off_count = (instrumented.get("NO_VERIFY", {}).get("probe") or {}).get("samples")
            if type(strict_count) is not int or strict_count <= 0 or strict_count != off_count:
                errors.append(f"{stage} threadhist block {block} has unmatched event counts")
    summaries: dict[str, Any] = {}
    for tool_index, tool in enumerate(TOOLS):
        effects = {"strict_vs_no_verify": [], "strict_vs_control": [],
                   "no_verify_vs_control": []}
        complete_blocks = 0
        for block in range(1, BLOCKS + 1):
            cells = [cell for cell in audited["timing"]
                     if cell["tool"] == tool and cell["block"] == block and cell["valid"]]
            if len(cells) != 3 or {cell["treatment"] for cell in cells} != set(TREATMENTS):
                continue
            complete_blocks += 1
            throughput = {cell["treatment"]: cell["bench"]["pp_tok_s"] for cell in cells}
            effects["strict_vs_no_verify"].append(100 * (throughput["STRICT"] / throughput["NO_VERIFY"] - 1))
            effects["strict_vs_control"].append(100 * (throughput["STRICT"] / throughput["control"] - 1))
            effects["no_verify_vs_control"].append(100 * (throughput["NO_VERIFY"] / throughput["control"] - 1))
        summary: dict[str, Any] = {"complete_blocks": complete_blocks, "required_blocks": BLOCKS,
                                   "effect_sign": "positive means numerator treatment has higher throughput"}
        for offset, (name, values) in enumerate(effects.items()):
            summary[name] = summarize_effect(values, tool_index * 10 + offset) if len(values) == BLOCKS else None
        summaries[tool] = summary
        if complete_blocks != BLOCKS:
            errors.append(f"{tool} lacks ten complete timing blocks")
    state_status = state.get("status")
    complete = not errors and state_status == "complete"
    total_cells = len(state.get("correctness_cells", [])) + len(state.get("timing_cells", []))
    if complete:
        run_status = "valid"
    elif isinstance(state_status, str) and state_status.startswith("invalid_"):
        run_status = "invalid"
    elif total_cells < 66 and state_status == "running":
        run_status = "incomplete"
    else:
        run_status = "invalid"
    return {"schema": SCHEMA, "complete": complete, "run_status": run_status,
            "application_throughput_source": "target llama-bench JSON avg_ts only",
            "admission_timing_used_in_throughput": False,
            "llama_bench_build": build_identity,
            "errors": errors, "summary": summaries, "audited": audited}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.result_dir)
    output = args.output or args.result_dir / "analysis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("complete", "run_status", "summary", "errors")}, indent=2))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
