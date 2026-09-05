#!/usr/bin/env python3
"""Independently audit a revision-RQ4 result without launching GPU work."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any


TASKS = ("kernelretsnoop", "threadhist", "launchlate")
CONFIGS = (
    "baseline",
    "gpubpf_kernelretsnoop", "nvbit_kernelretsnoop",
    "gpubpf_threadhist", "nvbit_threadhist",
    "gpubpf_launchlate", "nvbit_launchlate",
)
SCHEDULE_SEED = 1797
BOOTSTRAP_SAMPLES = 10000
EXPECTED_OUTPUT = "Deterministic tests are essential\n> EOF by user"
EXPECTED_OUTPUT_BYTES = 47
EXPECTED_GPU_THREAD_SLOTS = 22528
CORRECTNESS_RING_ENTRIES_PER_THREAD = 256
TIMING_RING_ENTRIES_PER_THREAD = 44
TIMING_THREADS_PER_PROMPT_TOKEN = 1024
TIMING_EXIT_LAUNCHES = 44
RING_SLOT_HEADER_BYTES = 24
RING_ALIGNED_RECORD_BYTES = 40
RING_ERROR_COUNTER_BYTES = 32
EXIT_RECORD_BYTES = 32
CORRECTNESS_EXIT_EVENTS = 720896
CORRECTNESS_EXIT_LAUNCHES = 220
CORRECTNESS_EXIT_COORDINATES = 22528
CORRECTNESS_MULTIPLICITIES = (1024, 1024, 20480, 0)
LAUNCH_CLOCK_DRIFT_LIMIT_PPB = 10000
LAUNCH_MIN_CALIBRATION_SPAN_NS = 1_000_000_000
LAUNCH_UNCERTAIN_PERCENT_LIMIT = 10
LAUNCH_RM_CALIBRATION_SAMPLES = 32
LAUNCH_RM_MAX_BRACKET_NS = 1500
LAUNCH_CONTROL_SAMPLES = 200
GPUBPF_LAUNCH_CLOCK_METHOD = (
    "RM endpoints-v1 PTIMER intervals with three-anchor held-out affine validation"
)
NVBIT_LAUNCH_CLOCK_METHOD = (
    "rm_endpoints_v1_PTIMER_against_CLOCK_MONOTONIC_RAW_"
    "with_three_anchor_held_out_affine_validation"
)
EXPECTED_DRIVER = "575.57.08"
EXPECTED_THREADHIST_GPU_THREAD_COUNT = 1048576
EXPECTED_KERNELRETSNOOP_SHM_MEMORY_MB = 1000
EXPECTED_N_GPU_LAYERS = 99
EXPECTED_TG = 0
EXPECTED_WORKER_CPUS = "8-15"
EXPECTED_TELEMETRY_CPU = 16
VERIFIER_LEVELS = {"DEFAULT", "STRICT", "NO_VERIFY"}


def kernelretsnoop_layout(pp: int, *, correctness: bool) -> dict[str, int]:
    if correctness:
        slots, entries = CORRECTNESS_EXIT_COORDINATES, CORRECTNESS_RING_ENTRIES_PER_THREAD
        launches, events = CORRECTNESS_EXIT_LAUNCHES, CORRECTNESS_EXIT_EVENTS
    else:
        if pp not in (32, 512):
            raise ValueError("kernelretsnoop timing layout is defined only for pp32/pp512")
        slots, entries = pp * TIMING_THREADS_PER_PROMPT_TOKEN, TIMING_RING_ENTRIES_PER_THREAD
        launches, events = TIMING_EXIT_LAUNCHES, slots * TIMING_EXIT_LAUNCHES
    if slots % 256 != 0:
        raise ValueError("kernelretsnoop coordinates must form x-by-256-by-1 rope geometry")
    return {
        "thread_slots": slots, "entries_per_thread": entries,
        "launches": launches, "coordinates": slots, "events": events,
        "extent_x": slots // 256, "extent_y": 256, "extent_z": 1,
        "shared_bytes": RING_ERROR_COUNTER_BYTES + slots * (
            RING_SLOT_HEADER_BYTES + entries * RING_ALIGNED_RECORD_BYTES),
    }


def selected_tools(params: dict[str, Any]) -> tuple[str, ...]:
    requested = tuple(params.get("tools", TASKS))
    if (not requested or len(requested) != len(set(requested))
            or any(tool not in TASKS for tool in requested)):
        raise ValueError("invalid recorded tool selection")
    canonical = tuple(tool for tool in TASKS if tool in requested)
    if requested != canonical:
        raise ValueError("recorded tools are not in canonical predeclared order")
    return canonical


def absolute_path_param(params: dict[str, Any], key: str) -> Path:
    value = params.get(key)
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f"recorded {key} is not a nonempty absolute path")
    return Path(value)


def validate_frozen_params(params: dict[str, Any]) -> None:
    pp = params.get("pp")
    if type(pp) is not int:
        raise ValueError("recorded pp is not an integer")
    correctness_layout = kernelretsnoop_layout(pp, correctness=True)
    timing_layout = kernelretsnoop_layout(pp, correctness=False)
    exact = {
        "tg": EXPECTED_TG,
        "n_gpu_layers": EXPECTED_N_GPU_LAYERS,
        "gpu_thread_count": EXPECTED_GPU_THREAD_SLOTS,
        "threadhist_gpu_thread_count": EXPECTED_THREADHIST_GPU_THREAD_COUNT,
        "kernelretsnoop_shm_memory_mb": EXPECTED_KERNELRETSNOOP_SHM_MEMORY_MB,
        "kernelretsnoop_correctness_exact_oracle": True,
        "kernelretsnoop_timing_exact_oracle": False,
        "kernelretsnoop_correctness_thread_slots": correctness_layout["thread_slots"],
        "kernelretsnoop_correctness_ring_entries_per_thread": correctness_layout["entries_per_thread"],
        "kernelretsnoop_timing_thread_slots": timing_layout["thread_slots"],
        "kernelretsnoop_timing_ring_entries_per_thread": timing_layout["entries_per_thread"],
        "kernelretsnoop_timing_expected_launches": timing_layout["launches"],
        "kernelretsnoop_timing_expected_coordinates": timing_layout["coordinates"],
        "kernelretsnoop_timing_expected_events": timing_layout["events"],
        "kernelretsnoop_timing_shared_bytes": timing_layout["shared_bytes"],
        "uvm": False,
        "no_warmup": False,
        "cuda_graphs_disabled": True,
        "schedule_seed": SCHEDULE_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "expected_driver": EXPECTED_DRIVER,
        "worker_cpus": EXPECTED_WORKER_CPUS,
        "telemetry_cpu": EXPECTED_TELEMETRY_CPU,
    }
    for key, expected in exact.items():
        if type(params.get(key)) is not type(expected) or params[key] != expected:
            raise ValueError(f"recorded frozen parameter differs: {key}")
    for key in (
        "model", "llama_bench", "llama_cli", "bpftime_root",
        "bpftime_build_dir", "nvbit_root", "uprobe_binary",
    ):
        absolute_path_param(params, key)
    target = params.get("target_symbol")
    hint = params.get("uprobe_symbol_hint")
    if not isinstance(target, str) or not target or target != hint:
        raise ValueError("target_symbol and uprobe_symbol_hint must be the same nonempty string")
    timeout = params.get("timeout_s")
    startup = params.get("probe_startup_s")
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("recorded timeout_s must be a positive integer")
    if type(startup) not in (int, float) or not math.isfinite(startup) or startup < 0:
        raise ValueError("recorded probe_startup_s must be finite and nonnegative")
    verifier_level = params.get("verifier_level", "DEFAULT")
    if verifier_level not in VERIFIER_LEVELS:
        raise ValueError("recorded verifier_level is invalid")
    runtime = params.get("verifier_runtime_configuration")
    if verifier_level != "DEFAULT":
        required = ("ENABLE_EBPF_VERIFIER", "BPFTIME_ENABLE_CUDA_ATTACH", "BPFTIME_LLVM_JIT")
        if not isinstance(runtime, dict) or any(
            runtime.get(key, "").upper() not in {"ON", "YES", "TRUE", "1"}
            for key in required
        ):
            raise ValueError("explicit verifier treatment lacks an enabled runtime record")


def verifier_map_expectation(tool: str, *, correctness: bool) -> dict[str, int]:
    if tool == "kernelretsnoop":
        return {
            "type": 1527,
            "key_size": 4,
            "value_size": 32,
            "max_entries": (
                CORRECTNESS_RING_ENTRIES_PER_THREAD
                if correctness else TIMING_RING_ENTRIES_PER_THREAD
            ),
        }
    if tool == "threadhist":
        return {
            "type": 1502,
            "key_size": 4,
            "value_size": 8,
            "max_entries": 1,
        }
    raise ValueError(f"explicit verifier evidence is unsupported for {tool}")


def verifier_valid(
    cell: dict[str, Any], params: dict[str, Any], tool: str, *, correctness: bool
) -> bool:
    level = params.get("verifier_level", "DEFAULT")
    if level == "DEFAULT":
        return True
    evidence = cell.get("verifier")
    if not isinstance(evidence, dict) or evidence.get("level") != level:
        return False
    if evidence.get("required") is not True or evidence.get("passed") is not True:
        return False
    executable = "llama_cli" if correctness else "llama_bench"
    expected_log = f"{executable}.log"
    target = params.get("target_symbol")
    try:
        expected_map = verifier_map_expectation(tool, correctness=correctness)
    except ValueError:
        return False
    if (
        not isinstance(target, str)
        or not target
        or evidence.get("program") != "cuda__retprobe"
        or evidence.get("attach") != f"kretprobe/{target}"
        or type(evidence.get("target_pid")) is not int
        or evidence["target_pid"] <= 0
        or evidence.get("execution_record") != f"{executable}.execution.json"
        or evidence.get("execution_error") is not None
        or evidence.get("expected_map") != expected_map
        or evidence.get("logs_scanned") != [expected_log]
        or evidence.get("logs_missing") != []
        or evidence.get("matched_log_sources") != [expected_log]
        or evidence.get("foreign_pid_records") != 0
        or evidence.get("unexpected_target_records") != 0
        or evidence.get("unparsed_records") != 0
    ):
        return False
    if level == "STRICT":
        counts = evidence.get("instruction_counts")
        maps = evidence.get("verified_maps")
        return (
            evidence.get("accepted_records") == 1
            and isinstance(counts, list)
            and len(counts) == 1
            and type(counts[0]) is int
            and counts[0] > 0
            and evidence.get("verified_map_records") == 1
            and isinstance(maps, list)
            and len(maps) == 1
            and isinstance(maps[0], dict)
            and set(maps[0]) == {"fd", *expected_map}
            and type(maps[0].get("fd")) is int
            and maps[0]["fd"] >= 0
            and all(maps[0].get(field) == value for field, value in expected_map.items())
            and evidence.get("skipped_records") == 0
            and evidence.get("rejected") is False
        )
    return (
        evidence.get("skipped_records") == 1
        and evidence.get("accepted_records") == 0
        and evidence.get("instruction_counts") == []
        and evidence.get("verified_map_records") == 0
        and evidence.get("verified_maps") == []
        and evidence.get("rejected") is False
    )


def selected_configs(tools: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        config for config in CONFIGS
        if config == "baseline" or config.split("_", 1)[1] in tools
    )


def fixed_schedule(configs: tuple[str, ...], runs: int) -> dict[str, list[str]]:
    result = {}
    for block in range(1, runs + 1):
        order = list(configs)
        random.Random(SCHEDULE_SEED + block).shuffle(order)
        result[str(block)] = order
    return result


def require_disjoint_campaign_paths(first: Path, second: Path) -> None:
    first = first.resolve()
    second = second.resolve()
    if first == second or first in second.parents or second in first.parents:
        raise ValueError("preflight and full campaign paths must be distinct and mutually non-nested")


def safety_valid(safety: Any) -> bool:
    if not isinstance(safety, dict) or safety.get("passed") is not True:
        return False
    for point in ("before", "after"):
        snapshot = safety.get(point)
        if not isinstance(snapshot, dict):
            return False
        gpu = snapshot.get("gpu")
        struct_ops = snapshot.get("struct_ops")
        if (snapshot.get("power_limit_service") != "active"
                or type(snapshot.get("power_limit_w")) not in (int, float)
                or not math.isfinite(snapshot["power_limit_w"])
                or abs(snapshot["power_limit_w"] - 400.0) > .01
                or type(snapshot.get("uvm_refcount")) is not int
                or snapshot["uvm_refcount"] != 0
                or snapshot.get("dmesg_abnormal") != []
                or snapshot.get("journal_abnormal") != []
                or snapshot.get("xids") != []
                or not isinstance(struct_ops, dict)
                or struct_ops.get("maps") != [] or struct_ops.get("links") != []
                or not isinstance(gpu, dict)
                or gpu.get("driver") != EXPECTED_DRIVER
                or gpu.get("compute_apps") != []
                or type(gpu.get("memory_used_mib")) not in (int, float)
                or not math.isfinite(gpu["memory_used_mib"])
                or gpu["memory_used_mib"] > 256
                or type(gpu.get("utilization_gpu_percent")) is not int
                or gpu["utilization_gpu_percent"] != 0):
            return False
    telemetry = safety.get("telemetry")
    return (isinstance(telemetry, dict)
            and type(telemetry.get("samples")) is int
            and telemetry["samples"] > 0
            and telemetry.get("throttled") is False)


def raw_gpu_clock_state(path: Path) -> dict[str, Any]:
    """Independently recover exact P-state and clock-pair evidence from CSV."""
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, skipinitialspace=True)
        headers = tuple(reader.fieldnames or ())
        sm_header = next((key for key in headers if key.startswith("clocks.current.sm")), None)
        memory_header = next(
            (key for key in headers if key.startswith("clocks.current.memory")), None
        )
        if "pstate" not in headers or sm_header is None or memory_header is None:
            raise ValueError("GPU telemetry lacks exact P-state/clock fields")
        required_reasons = {
            "clocks_event_reasons.sw_power_cap",
            "clocks_event_reasons.hw_slowdown",
            "clocks_event_reasons.hw_thermal_slowdown",
            "clocks_event_reasons.hw_power_brake_slowdown",
            "clocks_event_reasons.sw_thermal_slowdown",
        }
        if not required_reasons <= set(headers):
            raise ValueError("GPU telemetry lacks throttle-reason fields")
        reason_headers = [
            key for key in headers
            if key.startswith("clocks_event_reasons.") and "sw_power_cap" not in key
        ]
        rows = list(reader)
    if not rows or any(None in row or set(row) != set(headers) for row in rows):
        raise ValueError("GPU telemetry has no complete rows")
    pairs = set()
    pstates = set()
    throttled = False
    for row in rows:
        pstates.add(row["pstate"].strip())
        values = []
        for header in (sm_header, memory_header):
            value = float(row[header].strip().split()[0])
            if not value.is_integer():
                raise ValueError("GPU clock telemetry is not integral MHz")
            values.append(int(value))
        pairs.add(tuple(values))
        throttled = throttled or any(
            row[key].strip().lower() not in {"not active", "n/a"}
            for key in reason_headers
        )
    return {"pstates": sorted(pstates),
            "clock_pairs_mhz": [list(pair) for pair in sorted(pairs)],
            "throttled": throttled}


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"missing raw JSON evidence: {path}")
    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=unique_object,
        parse_constant=reject_json_constant,
    )


def campaign_path(campaign: Path, value: Any, *, absolute_ok: bool = False) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("missing evidence path")
    recorded = Path(value)
    if recorded.is_absolute() and not absolute_ok:
        raise ValueError("cell evidence path must be campaign-relative")
    path = (recorded if recorded.is_absolute() else campaign / recorded).resolve()
    if not path.is_relative_to(campaign):
        raise ValueError("evidence path escapes the campaign directory")
    if not path.is_file():
        raise ValueError(f"missing raw evidence: {path}")
    return path


def wrapped_process_log(path: Path) -> tuple[str, str, int]:
    text = path.read_text(encoding="utf-8", errors="strict")
    stdout_marker = "\n## stdout\n"
    stderr_marker = "\n## stderr\n"
    if text.count(stdout_marker) != 1 or text.count(stderr_marker) != 1:
        raise ValueError("process log does not have exactly one stdout/stderr section")
    _, body = text.split(stdout_marker, 1)
    stdout, tail = body.split(stderr_marker, 1)
    exit_matches = list(re.finditer(r"\n# exit: (-?\d+)\n?", tail))
    if len(exit_matches) != 1 or exit_matches[0].end() != len(tail):
        raise ValueError("process log does not end with one exact exit record")
    match = exit_matches[0]
    return stdout, tail[:match.start()], int(match.group(1))


def execution_valid(path: Path, expected_returncode: int) -> bool:
    try:
        record = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError, UnicodeError):
        return False
    identity = record.get("identity") if isinstance(record, dict) else None
    return (
        isinstance(record, dict)
        and record.get("cleanup_passed") is True
        and record.get("timed_out") is False
        and record.get("returncode") == expected_returncode
        and "error" not in record
        and "cleanup_failure" not in record
        and isinstance(identity, dict)
        and type(identity.get("pid")) is int
        and identity["pid"] > 0
    )


def normalized_output(stdout: str) -> str:
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", stdout)
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def raw_bench_result(stdout: str, pp: int) -> tuple[dict[str, Any], list[Any]]:
    raw = json.loads(
        stdout, object_pairs_hook=unique_object, parse_constant=reject_json_constant
    )
    if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], dict):
        raise ValueError("llama-bench stdout is not one JSON result record")
    entry = raw[0]
    if type(entry.get("n_prompt")) is not int or entry["n_prompt"] != pp:
        raise ValueError("llama-bench prompt cardinality differs from frozen pp")
    if type(entry.get("n_gen", 0)) is not int or entry.get("n_gen", 0) != 0:
        raise ValueError("llama-bench raw record unexpectedly contains generation work")
    throughput = entry.get("avg_ts")
    if type(throughput) not in (int, float) or not math.isfinite(throughput) or throughput <= 0:
        raise ValueError("llama-bench raw throughput is invalid")
    stddev = entry.get("stddev_ts", 0.0)
    if type(stddev) not in (int, float) or not math.isfinite(stddev) or stddev < 0:
        raise ValueError("llama-bench raw standard deviation is invalid")
    samples = entry.get("samples_ts", [])
    if not isinstance(samples, list) or any(
        type(value) not in (int, float) or not math.isfinite(value) or value <= 0
        for value in samples
    ):
        raise ValueError("llama-bench raw sample list is invalid")
    metrics = {
        "pp_tok_s": float(throughput),
        "pp_stddev": float(stddev),
        "pp_tokens": entry["n_prompt"],
        "pp_samples_tok_s": samples,
    }
    return metrics, raw


def integer(probe: dict[str, Any], key: str) -> int:
    value = probe.get(key)
    if type(value) is not int:
        raise ValueError(f"missing integer probe field: {key}")
    return value


def launch_clock_valid(probe: dict[str, Any]) -> bool:
    try:
        start_low = integer(probe, "start_clock_offset_lower_ns")
        start_high = integer(probe, "start_clock_offset_upper_ns")
        middle_low = integer(probe, "measurement_end_clock_offset_lower_ns")
        middle_high = integer(probe, "measurement_end_clock_offset_upper_ns")
        end_low = integer(probe, "validation_end_clock_offset_lower_ns")
        end_high = integer(probe, "validation_end_clock_offset_upper_ns")
        start_anchor = integer(probe, "start_clock_host_anchor_ns")
        middle_anchor = integer(probe, "measurement_end_clock_host_anchor_ns")
        end_anchor = integer(probe, "validation_end_clock_host_anchor_ns")
        elapsed = end_anchor - start_anchor
        position = middle_anchor - start_anchor
        validation_span = end_anchor - middle_anchor
        change_low = end_low - start_high
        change_high = end_high - start_low
        if elapsed <= 0 or position <= 0 or position >= elapsed:
            return False
        rate = (max(abs(change_low), abs(change_high)) * 1_000_000_000 + elapsed - 1) // elapsed
        predicted_low = start_low + (end_low - start_low) * position // elapsed
        high_numerator = (end_high - start_high) * position
        predicted_high = start_high - (-high_numerator // elapsed)
        overlap_low = max(predicted_low, middle_low)
        overlap_high = min(predicted_high, middle_high)
        return (
            start_low <= start_high and middle_low <= middle_high
            and end_low <= end_high
            and integer(probe, "start_clock_uncertainty_ns") == (start_high - start_low + 1) // 2
            and integer(probe, "measurement_end_clock_uncertainty_ns") == (middle_high - middle_low + 1) // 2
            and integer(probe, "validation_end_clock_uncertainty_ns") == (end_high - end_low + 1) // 2
            and integer(probe, "clock_offset_change_lower_ns") == change_low
            and integer(probe, "clock_offset_change_upper_ns") == change_high
            and integer(probe, "clock_calibration_elapsed_ns") == elapsed
            and validation_span >= LAUNCH_MIN_CALIBRATION_SPAN_NS
            and integer(probe, "clock_drift_rate_bound_ppb") == rate
            and integer(probe, "clock_drift_limit_ppb") == LAUNCH_CLOCK_DRIFT_LIMIT_PPB
            and integer(probe, "clock_drift_bounded") == int(rate <= LAUNCH_CLOCK_DRIFT_LIMIT_PPB)
            and integer(probe, "clock_slope_diagnostic_only") == 1
            and overlap_low <= overlap_high
            and integer(probe, "held_out_predicted_lower_ns") == predicted_low
            and integer(probe, "held_out_predicted_upper_ns") == predicted_high
            and integer(probe, "held_out_overlap_lower_ns") == overlap_low
            and integer(probe, "held_out_overlap_upper_ns") == overlap_high
            and integer(probe, "validation_span_ns") == validation_span
            and integer(probe, "held_out_validation_passed") == 1
        )
    except ValueError:
        return False


def launch_rm_anchors_valid(probe: dict[str, Any]) -> bool:
    """Recompute all three endpoint-v1 RAW/PTIMER anchor intervals."""
    try:
        for phase in ("start", "measurement_end", "validation_end"):
            names = (
                "rm_samples_requested", "rm_samples_accepted",
                "rm_samples_rejected", "rm_outer_before_raw_ns",
                "rm_cpu_before_raw_ns", "rm_gpu_ptimer_ns",
                "rm_cpu_after_raw_ns", "rm_outer_after_raw_ns",
                "rm_outer_width_ns", "rm_selected_gap_ns",
                "rm_bracket_width_ns", "rm_status", "rm_cleanup_complete",
            )
            values = {name: integer(probe, f"{phase}_{name}") for name in names}
            outer_before = values["rm_outer_before_raw_ns"]
            cpu_before = values["rm_cpu_before_raw_ns"]
            gpu = values["rm_gpu_ptimer_ns"]
            cpu_after = values["rm_cpu_after_raw_ns"]
            outer_after = values["rm_outer_after_raw_ns"]
            selected_gap = cpu_after - cpu_before
            bracket = selected_gap + 64
            if not (
                values["rm_samples_requested"] == LAUNCH_RM_CALIBRATION_SAMPLES
                and values["rm_samples_accepted"] == LAUNCH_RM_CALIBRATION_SAMPLES
                and values["rm_samples_rejected"] == 0
                and values["rm_status"] == 0
                and values["rm_cleanup_complete"] == 1
                and 0 < outer_before <= cpu_before <= cpu_after <= outer_after
                and gpu > 0
                and values["rm_outer_width_ns"] == outer_after - outer_before
                and values["rm_outer_width_ns"] < 10_000_000
                and values["rm_selected_gap_ns"] == selected_gap
                and values["rm_bracket_width_ns"] == bracket
                and bracket <= LAUNCH_RM_MAX_BRACKET_NS
                and integer(probe, f"{phase}_clock_offset_lower_ns")
                    == gpu - cpu_after - 32
                and integer(probe, f"{phase}_clock_offset_upper_ns")
                    == gpu - cpu_before + 32
                and integer(probe, f"{phase}_clock_uncertainty_ns")
                    == (bracket + 1) // 2
                and integer(probe, f"{phase}_clock_host_anchor_ns")
                    == cpu_before + selected_gap // 2
            ):
                return False
        return True
    except ValueError:
        return False


def launch_uncertainty_valid(classified: int, uncertain: int, total: int) -> bool:
    return (classified >= 0 and uncertain >= 0 and total > 0
            and classified + uncertain == total
            and uncertain * 100 <= total * LAUNCH_UNCERTAIN_PERCENT_LIMIT)


def gpubpf_valid(tool: str, probe: dict[str, Any], params: dict[str, Any],
                 correctness: bool) -> bool:
    try:
        samples = integer(probe, "sample_count")
        if samples <= 0:
            return False
        if tool == "kernelretsnoop":
            layout = kernelretsnoop_layout(params["pp"], correctness=correctness)
            launches = integer(probe, "cartesian_launches")
            coordinates = integer(probe, "cartesian_coordinates")
            multiplicities = tuple(integer(probe, key) for key in (
                "multiplicity_220", "multiplicity_44", "multiplicity_22", "other_multiplicity"))
            exact = int(correctness)
            generic = (
                integer(probe, "requested_thread_slots") == layout["thread_slots"]
                and integer(probe, "allocated_thread_slots") == layout["thread_slots"]
                and integer(probe, "requested_entries_per_thread") == layout["entries_per_thread"]
                and integer(probe, "entries_per_thread") == layout["entries_per_thread"]
                and integer(probe, "record_bytes") == EXIT_RECORD_BYTES
                and integer(probe, "committed_events")
                == integer(probe, "runtime_collected_events")
                == integer(probe, "nonzero_timestamps") == samples
                and all(integer(probe, key) == 0 for key in (
                    "oob_drops", "full_drops", "bad_size_drops", "other_drops",
                    "dirty_slots", "pending_events", "second_drain_events",
                    "invalid_launch_coordinates"))
                and 0 <= integer(probe, "final_drain_events") <= samples
                and integer(probe, "cartesian_complete") == 1
                and integer(probe, "collector_gate_passed") == 1
                and launches > 0 and coordinates > 0
                and integer(probe, "extent_x") == layout["extent_x"]
                and integer(probe, "extent_y") == layout["extent_y"]
                and integer(probe, "extent_z") == layout["extent_z"]
                and coordinates == integer(probe, "unique_coordinates") == sum(multiplicities)
                and integer(probe, "segment_mismatches") == 0
                and integer(probe, "oracle_enabled") == exact
                and integer(probe, "oracle_total_events") == samples
                and integer(probe, "oracle_passed") == exact
                and samples == layout["events"]
                and launches == layout["launches"]
                and coordinates == layout["coordinates"]
                and (correctness or multiplicities == (0, coordinates, 0, 0))
            )
            if not generic:
                return False
            if not correctness:
                return True
            return (
                samples == CORRECTNESS_EXIT_EVENTS
                and launches == CORRECTNESS_EXIT_LAUNCHES
                and coordinates == CORRECTNESS_EXIT_COORDINATES
                and multiplicities == CORRECTNESS_MULTIPLICITIES
                and integer(probe, "segment_mismatches") == 0
            )
        if tool == "threadhist":
            count = params.get("threadhist_gpu_thread_count")
            return (
                type(count) is int and count > 0
                and integer(probe, "nonzero_threads") > 0
                and integer(probe, "configured_entries") == count
                and integer(probe, "readback_entries") == count
                and integer(probe, "readback_bytes") == count * 8
                and integer(probe, "readback_complete") == 1
            )
        classified = integer(probe, "classified_samples")
        uncertain = integer(probe, "uncertain_samples")
        matched = integer(probe, "matched_samples")
        return (
            probe.get("clock_calibration_method")
            == GPUBPF_LAUNCH_CLOCK_METHOD
            and launch_rm_anchors_valid(probe)
            and launch_clock_valid(probe)
            and integer(probe, "probes_detached_before_readback") == 1
            and all(integer(probe, key) == 0 for key in (
                "clock_errors", "queue_underflows", "queue_overflows", "queue_update_errors"))
            and all(integer(probe, key) == 1 for key in (
                "online_accounting_complete", "accounting_complete", "pairing_complete"))
            and launch_uncertainty_valid(classified, uncertain, matched)
            and integer(probe, "host_launches") == integer(probe, "host_enqueued")
            == integer(probe, "device_entries") == matched == samples
            and integer(probe, "histogram_samples") == classified
            and (not correctness or matched == CORRECTNESS_EXIT_LAUNCHES)
        )
    except ValueError:
        return False


def nvbit_valid(tool: str, probe: dict[str, Any], params: dict[str, Any],
                correctness: bool) -> bool:
    try:
        samples = integer(probe, "sample_count")
        selected = integer(probe, "selected_launches")
        if samples <= 0 or selected <= 0:
            return False
        if tool == "kernelretsnoop":
            layout = kernelretsnoop_layout(params["pp"], correctness=correctness)
            multiplicities = tuple(integer(probe, key) for key in (
                "multiplicity_220", "multiplicity_44", "multiplicity_22",
                "other_multiplicity"))
            expected_multiplicities = (
                CORRECTNESS_MULTIPLICITIES if correctness
                else (0, layout["coordinates"], 0, 0)
            )
            return (
                integer(probe, "nonzero_timestamps") == samples
                and integer(probe, "record_bytes") == EXIT_RECORD_BYTES
                and integer(probe, "bad_size_bytes") == 0
                and integer(probe, "cartesian_launches") == selected
                and integer(probe, "cartesian_coordinates") == layout["coordinates"]
                and integer(probe, "cartesian_complete") == 1
                and integer(probe, "extent_x") == layout["extent_x"]
                and integer(probe, "extent_y") == layout["extent_y"]
                and integer(probe, "extent_z") == layout["extent_z"]
                and multiplicities == expected_multiplicities
                and integer(probe, "segment_mismatches") == 0
                and integer(probe, "invalid_launch_coordinates") == 0
                and integer(probe, "unique_coordinates") == layout["coordinates"]
                and integer(probe, "collector_gate_passed") == 1
                and integer(probe, "validation_blocks") == 1
                and integer(probe, "process_selected_launches") == selected
                and samples == layout["events"]
                and selected == layout["launches"]
            )
        if tool == "threadhist":
            return integer(probe, "nonzero_threads") > 0
        histogram = probe.get("histogram")
        uncertain = integer(probe, "uncertain_samples")
        return (
            probe.get("clock_calibration_method")
            == NVBIT_LAUNCH_CLOCK_METHOD
            and integer(probe, "start_clock_calibration_valid") == 1
            and integer(probe, "measurement_end_clock_calibration_valid") == 1
            and integer(probe, "validation_end_clock_calibration_valid") == 1
            and launch_rm_anchors_valid(probe)
            and launch_clock_valid(probe)
            and integer(probe, "clock_errors") == 0
            and isinstance(histogram, list) and len(histogram) == 10
            and all(type(count) is int and count >= 0 for count in histogram)
            and sum(histogram) == samples == integer(probe, "histogram_sum")
            and integer(probe, "pair_capacity") >= selected
            and integer(probe, "stored_pairs") == selected
            and integer(probe, "device_entries") == selected
            and all(integer(probe, key) == 0 for key in (
                "pair_overflows", "capture_errors", "selected_counter_overflow"))
            and integer(probe, "accounting_complete") == 1
            and integer(probe, "process_selected_launches") == selected
            and integer(probe, "result_blocks") == 1
            and integer(probe, "calibration_blocks") == 1
            and selected == samples + uncertain
            and launch_uncertainty_valid(samples, uncertain, selected)
            and (not correctness or selected == CORRECTNESS_EXIT_LAUNCHES)
        )
    except ValueError:
        return False


def one_match(pattern: str, text: str, *, flags: int = re.MULTILINE) -> str:
    values = re.findall(pattern, text, flags)
    if len(values) != 1:
        raise ValueError(f"expected one raw marker, found {len(values)}: {pattern}")
    return values[0]


def parse_gpubpf_launch_raw(text: str) -> dict[str, Any]:
    labels = {
        "sample_count": "Total samples",
        "histogram_samples": "Histogram samples",
        "host_launches": "Host launches",
        "host_enqueued": "Host enqueued",
        "device_entries": "Device entries",
        "matched_samples": "Matched samples",
        "queue_underflows": "Queue underflows",
        "queue_overflows": "Queue overflows",
        "queue_update_errors": "Queue update errors",
        "classified_samples": "Classified samples",
        "uncertain_samples": "Uncertain samples",
        "clock_errors": "Clock errors",
        "online_accounting_complete": "Online accounting complete",
        "accounting_complete": "Accounting complete",
        "pairing_complete": "Pairing complete",
        "probes_detached_before_readback": "Probes detached before final readback",
        "start_clock_offset_lower_ns": "Start clock offset lower",
        "start_clock_offset_upper_ns": "Start clock offset upper",
        "start_clock_uncertainty_ns": "Start clock uncertainty",
        "start_clock_host_anchor_ns": "Start clock host anchor",
        "measurement_end_clock_offset_lower_ns": "Measurement-end clock offset lower",
        "measurement_end_clock_offset_upper_ns": "Measurement-end clock offset upper",
        "measurement_end_clock_uncertainty_ns": "Measurement-end clock uncertainty",
        "measurement_end_clock_host_anchor_ns": "Measurement-end clock host anchor",
        "validation_end_clock_offset_lower_ns": "Validation-end clock offset lower",
        "validation_end_clock_offset_upper_ns": "Validation-end clock offset upper",
        "validation_end_clock_uncertainty_ns": "Validation-end clock uncertainty",
        "validation_end_clock_host_anchor_ns": "Validation-end clock host anchor",
        "clock_offset_change_lower_ns": "Clock offset change lower",
        "clock_offset_change_upper_ns": "Clock offset change upper",
        "clock_calibration_elapsed_ns": "Clock calibration elapsed",
        "clock_drift_rate_bound_ppb": "Clock drift rate bound",
        "clock_drift_limit_ppb": "Clock drift limit",
        "clock_drift_bounded": "Clock drift bounded",
        "clock_slope_diagnostic_only": "Clock slope diagnostic only",
        "held_out_predicted_lower_ns": "Held-out predicted lower",
        "held_out_predicted_upper_ns": "Held-out predicted upper",
        "held_out_overlap_lower_ns": "Held-out overlap lower",
        "held_out_overlap_upper_ns": "Held-out overlap upper",
        "validation_span_ns": "Clock validation span",
        "held_out_validation_passed": "Held-out clock validation passed",
    }
    signed = {
        "start_clock_offset_lower_ns", "start_clock_offset_upper_ns",
        "measurement_end_clock_offset_lower_ns",
        "measurement_end_clock_offset_upper_ns",
        "validation_end_clock_offset_lower_ns",
        "validation_end_clock_offset_upper_ns",
        "clock_offset_change_lower_ns", "clock_offset_change_upper_ns",
        "held_out_predicted_lower_ns", "held_out_predicted_upper_ns",
        "held_out_overlap_lower_ns", "held_out_overlap_upper_ns",
    }
    result: dict[str, Any] = {}
    for key, label in labels.items():
        unit = (r"\s+ppb" if key.endswith("_ppb") else
                r"\s+ns" if key.endswith("_ns") else "")
        number = r"(-?\d+)" if key in signed else r"(\d+)"
        result[key] = int(one_match(
            rf"^{re.escape(label)}:\s*{number}{unit}$", text
        ))
    result["clock_calibration_method"] = one_match(
        r"^Clock calibration method:\s*(.+)$", text
    ).strip()
    rm_labels = {
        "rm_samples_requested": "RM samples requested",
        "rm_samples_accepted": "RM samples accepted",
        "rm_samples_rejected": "RM samples rejected",
        "rm_outer_before_raw_ns": "RM outer before RAW",
        "rm_cpu_before_raw_ns": "RM CPU before RAW",
        "rm_gpu_ptimer_ns": "RM GPU PTIMER",
        "rm_cpu_after_raw_ns": "RM CPU after RAW",
        "rm_outer_after_raw_ns": "RM outer after RAW",
        "rm_outer_width_ns": "RM outer width",
        "rm_selected_gap_ns": "RM selected gap",
        "rm_bracket_width_ns": "RM bracket width",
        "rm_cleanup_complete": "RM cleanup complete",
    }
    for phase, display in (
        ("start", "Start"),
        ("measurement_end", "Measurement-end"),
        ("validation_end", "Validation-end"),
    ):
        for suffix, label in rm_labels.items():
            unit = r"\s+ns" if suffix.endswith("_ns") else ""
            result[f"{phase}_{suffix}"] = int(one_match(
                rf"^{display} {re.escape(label)}:\s*(\d+){unit}$", text
            ))
        result[f"{phase}_rm_status"] = int(one_match(
            rf"^{display} RM status:\s*0x([0-9a-fA-F]+)$", text
        ), 16)
    return result


def parse_nvbit_launch_raw(text: str) -> dict[str, Any]:
    selected = int(one_match(r"^NVBIT selected_launches=(\d+)$", text))
    samples = int(one_match(
        r"^NVBIT launchlate samples=(\d+) clock_errors=\d+$", text
    ))
    errors = int(one_match(
        r"^NVBIT launchlate samples=\d+ clock_errors=(\d+)$", text
    ))
    bins = [int(one_match(
        rf"^NVBIT launchlate bin_{index}=(\d+)$", text
    )) for index in range(10)]
    result: dict[str, Any] = {
        "sample_count": samples,
        "clock_errors": errors,
        "histogram": bins,
        "histogram_sum": sum(bins),
        "selected_launches": selected,
        "process_selected_launches": int(one_match(
            r"^NVBIT_OBS process_selected_launches=(\d+)$", text
        )),
        "result_blocks": 1,
    }
    fields = (
        "uncertain_samples", "pair_capacity", "stored_pairs", "device_entries",
        "pair_overflows", "capture_errors", "selected_counter_overflow",
        "accounting_complete", "start_clock_offset_lower_ns",
        "start_clock_offset_upper_ns", "start_clock_uncertainty_ns",
        "start_clock_host_anchor_ns", "start_clock_calibration_valid",
        "measurement_end_clock_offset_lower_ns",
        "measurement_end_clock_offset_upper_ns",
        "measurement_end_clock_uncertainty_ns",
        "measurement_end_clock_host_anchor_ns",
        "measurement_end_clock_calibration_valid",
        "validation_end_clock_offset_lower_ns",
        "validation_end_clock_offset_upper_ns",
        "validation_end_clock_uncertainty_ns",
        "validation_end_clock_host_anchor_ns",
        "validation_end_clock_calibration_valid", "clock_offset_change_lower_ns",
        "clock_offset_change_upper_ns", "clock_calibration_elapsed_ns",
        "clock_drift_rate_bound_ppb", "clock_drift_limit_ppb",
        "clock_drift_bounded", "clock_slope_diagnostic_only",
        "held_out_predicted_lower_ns", "held_out_predicted_upper_ns",
        "held_out_overlap_lower_ns", "held_out_overlap_upper_ns",
        "validation_span_ns", "held_out_validation_passed",
        "start_rm_samples_requested",
        "start_rm_samples_accepted", "start_rm_samples_rejected",
        "start_rm_outer_before_raw_ns", "start_rm_cpu_before_raw_ns",
        "start_rm_gpu_ptimer_ns", "start_rm_cpu_after_raw_ns",
        "start_rm_outer_after_raw_ns", "start_rm_outer_width_ns",
        "start_rm_selected_gap_ns", "start_rm_bracket_width_ns",
        "start_rm_status", "start_rm_cleanup_complete",
        *(f"{phase}_{suffix}"
          for phase in ("measurement_end", "validation_end")
          for suffix in (
              "rm_samples_requested", "rm_samples_accepted",
              "rm_samples_rejected", "rm_outer_before_raw_ns",
              "rm_cpu_before_raw_ns", "rm_gpu_ptimer_ns",
              "rm_cpu_after_raw_ns", "rm_outer_after_raw_ns",
              "rm_outer_width_ns", "rm_selected_gap_ns",
              "rm_bracket_width_ns", "rm_status", "rm_cleanup_complete",
          )),
    )
    signed = {
        "start_clock_offset_lower_ns", "start_clock_offset_upper_ns",
        "measurement_end_clock_offset_lower_ns",
        "measurement_end_clock_offset_upper_ns",
        "validation_end_clock_offset_lower_ns",
        "validation_end_clock_offset_upper_ns",
        "clock_offset_change_lower_ns", "clock_offset_change_upper_ns",
        "held_out_predicted_lower_ns", "held_out_predicted_upper_ns",
        "held_out_overlap_lower_ns", "held_out_overlap_upper_ns",
    }
    for key in fields:
        number = r"(-?\d+)" if key in signed else r"(\d+)"
        result[key] = int(one_match(
            rf"^NVBIT launchlate {key}={number}$", text
        ))
    result["clock_calibration_method"] = one_match(
        r"^NVBIT launchlate clock_calibration_method=(\S+)$", text
    )
    result["calibration_blocks"] = 1
    return result


def cell_raw_valid(campaign: Path, cell: dict[str, Any], config: str,
                   params: dict[str, Any], correctness: bool) -> bool:
    """Reopen raw process, cleanup, safety, output, and launch engagement evidence."""
    try:
        log_path = campaign_path(campaign, cell.get("log"))
        stdout, stderr, raw_returncode = wrapped_process_log(log_path)
        if raw_returncode != cell.get("returncode"):
            return False
        if not execution_valid(log_path.with_suffix(".execution.json"), raw_returncode):
            return False
        raw_safety = read_json(log_path.parent / "gpu-safety.json")
        if raw_safety != cell.get("safety") or not safety_valid(raw_safety):
            return False
        if correctness:
            output = normalized_output(stdout)
            if (output != cell.get("normalized_stdout")
                    or len(output.encode()) != cell.get("stdout_bytes")):
                return False
        else:
            metrics, raw = raw_bench_result(stdout, params["pp"])
            if metrics != cell.get("metrics") or raw != cell.get("raw"):
                return False
        if config == "gpubpf_launchlate":
            expected_probe = log_path.parent / "probe.log"
            if correctness:
                probe_path = expected_probe
            else:
                probe_path = campaign_path(campaign, cell.get("probe_log"))
                if probe_path != expected_probe.resolve():
                    return False
            raw_probe = parse_gpubpf_launch_raw(
                probe_path.read_text(encoding="utf-8", errors="strict")
            )
            return raw_probe == cell.get("probe")
        if config == "nvbit_launchlate":
            return parse_nvbit_launch_raw(stderr) == cell.get("probe")
        return True
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, UnicodeError):
        return False


def cell_valid(cell: dict[str, Any], config: str, params: dict[str, Any],
               correctness: bool, campaign: Path | None = None,
               require_raw: bool = False) -> bool:
    if (cell.get("valid") is not True or cell.get("returncode") != 0
            or cell.get("error") or cell.get("fatal_cleanup")
            or not safety_valid(cell.get("safety"))):
        return False
    if require_raw and (
        campaign is None or not cell_raw_valid(campaign, cell, config, params, correctness)
    ):
        return False
    if correctness:
        if (cell.get("normalized_stdout") != EXPECTED_OUTPUT
                or cell.get("stdout_bytes") != EXPECTED_OUTPUT_BYTES):
            return False
        if config != "baseline" and cell.get("matches_baseline") is not True:
            return False
    else:
        metrics = cell.get("metrics", {})
        throughput = metrics.get("pp_tok_s")
        if (metrics.get("pp_tokens") != params.get("pp")
                or type(throughput) not in (int, float)
                or not math.isfinite(throughput) or throughput <= 0):
            return False
    if config == "baseline":
        return True
    system, tool = config.split("_", 1)
    probe = cell.get("probe")
    if not isinstance(probe, dict):
        return False
    return (
        gpubpf_valid(tool, probe, params, correctness)
        and verifier_valid(cell, params, tool, correctness=correctness)
        if system == "gpubpf" else nvbit_valid(tool, probe, params, correctness)
    )


def one_valid(entries: list[dict[str, Any]], config: str, params: dict[str, Any],
              correctness: bool, block: int | None = None,
              campaign: Path | None = None,
              require_raw: bool = False) -> dict[str, Any] | None:
    candidates = [entry for entry in entries
                  if (block is None or entry.get("block") == block)
                  and cell_valid(entry, config, params, correctness,
                                 campaign, require_raw)]
    if len(candidates) > 1:
        raise ValueError(f"multiple independently valid entries for {config}, block {block}")
    return candidates[0] if candidates else None


def launchlate_clock_block_valid(
        campaign: Path, cells: dict[str, dict[str, Any]],
        supported_clock_pairs: set[tuple[int, int]]) -> bool:
    names = ("baseline", "gpubpf_launchlate", "nvbit_launchlate")
    if not all(name in cells for name in names) or not supported_clock_pairs:
        return False
    observed_sets = []
    try:
        for name in names:
            cell = cells[name]
            log_path = campaign_path(campaign, cell.get("log"))
            raw = raw_gpu_clock_state(log_path.parent / "gpu-telemetry.csv")
            stored = cell.get("safety", {}).get("telemetry", {})
            pairs = {tuple(pair) for pair in raw["clock_pairs_mhz"]}
            if (raw["pstates"] != ["P0"] or raw["throttled"] is not False
                    or not pairs or not pairs <= supported_clock_pairs
                    or stored.get("pstates") != raw["pstates"]
                    or stored.get("clock_pairs_mhz") != raw["clock_pairs_mhz"]):
                return False
            observed_sets.append(pairs)
    except (OSError, ValueError, TypeError, KeyError, UnicodeError):
        return False
    return len({frozenset(value) for value in observed_sets}) == 1


def bootstrap_mean(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    rng = random.Random(SCHEDULE_SEED)
    samples = sorted(statistics.mean(values[rng.randrange(len(values))] for _ in values)
                     for _ in range(BOOTSTRAP_SAMPLES))
    def quantile(probability: float) -> float:
        location = (len(samples) - 1) * probability
        low, high = math.floor(location), math.ceil(location)
        if low == high:
            return samples[low]
        fraction = location - low
        return samples[low] * (1 - fraction) + samples[high] * fraction
    return {"mean": statistics.mean(values),
            "ci95_low": quantile(.025), "ci95_high": quantile(.975)}


def read_strict_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        raise ValueError(f"missing raw JSONL: {path}")
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank JSONL record at line {number}")
        record = json.loads(
            line, object_pairs_hook=unique_object, parse_constant=reject_json_constant
        )
        if not isinstance(record, dict):
            raise ValueError(f"non-object JSONL record at line {number}")
        records.append(record)
    return records


def integer_median(values: list[int]) -> int:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take the median of an empty list")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) // 2


def endpoint_control_valid(records: list[dict[str, Any]]) -> bool:
    if len(records) != LAUNCH_CONTROL_SAMPLES + 1:
        return False
    samples, summary = records[:-1], records[-1]
    widths: list[int] = []
    outer_widths: list[int] = []
    previous_cpu = previous_gpu = 0
    for index, sample in enumerate(samples):
        fields = (
            "index", "rm_status", "host_before_ns", "host_after_ns",
            "rm_cpu_before_ns", "rm_cpu_midpoint_ns", "rm_cpu_after_ns",
            "rm_gpu_ptimer_ns", "outer_width_ns", "max_selected_gap_ns",
            "cpu_lower_ns", "cpu_upper_ns", "offset_low_ns",
            "offset_high_ns", "bracket_width_ns",
        )
        if any(type(sample.get(key)) is not int for key in fields):
            return False
        before = sample["host_before_ns"]
        after = sample["host_after_ns"]
        cpu_before = sample["rm_cpu_before_ns"]
        cpu_after = sample["rm_cpu_after_ns"]
        midpoint = sample["rm_cpu_midpoint_ns"]
        gpu = sample["rm_gpu_ptimer_ns"]
        gap = cpu_after - cpu_before
        bracket = gap + 64
        if not (
            sample.get("record") == "sample"
            and sample["index"] == index
            and sample.get("control_transport") == "direct"
            and sample.get("correlation_command") == "endpoints-v1"
            and sample.get("valid") is True
            and sample.get("cpu_midpoint_regression") is False
            and sample.get("ptimer_regression") is False
            and sample["rm_status"] == 0
            and 0 < before <= cpu_before <= cpu_after <= after
            and gpu > 0
            and midpoint == cpu_before + gap // 2
            and sample["outer_width_ns"] == after - before
            and sample["outer_width_ns"] < 10_000_000
            and sample["max_selected_gap_ns"] == gap
            and sample["cpu_lower_ns"] == cpu_before
            and sample["cpu_upper_ns"] == cpu_after
            and sample["offset_low_ns"] == gpu - cpu_after - 32
            and sample["offset_high_ns"] == gpu - cpu_before + 32
            and sample["bracket_width_ns"] == bracket
            and (previous_cpu == 0 or midpoint >= previous_cpu)
            and (previous_gpu == 0 or gpu >= previous_gpu)
        ):
            return False
        previous_cpu, previous_gpu = midpoint, gpu
        widths.append(bracket)
        outer_widths.append(after - before)
    exact_summary = {
        "record": "summary", "setup_stage": "samples",
        "control_transport": "direct", "correlation_command": "endpoints-v1",
        "setup_error": 0, "cleanup_error": 0, "cleanup_rm_status": 0,
        "output_error": 0, "requested": LAUNCH_CONTROL_SAMPLES,
        "attempted": LAUNCH_CONTROL_SAMPLES, "accepted": LAUNCH_CONTROL_SAMPLES,
        "rejected": 0, "cpu_midpoint_regressions": 0,
        "ptimer_regressions": 0, "min_outer_width_ns": min(outer_widths),
        "median_outer_width_ns": integer_median(outer_widths),
        "max_outer_width_ns": max(outer_widths),
        "min_bracket_width_ns": min(widths),
        "median_bracket_width_ns": integer_median(widths),
        "max_bracket_width_ns": max(widths),
        "target_median_bracket_ns": LAUNCH_RM_MAX_BRACKET_NS,
        "gate_pass": True,
    }
    return summary == exact_summary and integer_median(widths) <= LAUNCH_RM_MAX_BRACKET_NS


def identity_control_valid(records: list[dict[str, Any]]) -> bool:
    if len(records) != LAUNCH_CONTROL_SAMPLES + 1:
        return False
    previous_raw = previous_ptimer = 0
    for index, sample in enumerate(records[:-1]):
        fields = (
            "trial", "rm_before_outer_before_raw_ns",
            "rm_before_cpu_before_raw_ns", "rm_before_gpu_ptimer_ns",
            "rm_before_cpu_after_raw_ns", "rm_before_outer_after_raw_ns",
            "rm_before_offset_low_ns", "rm_before_offset_high_ns",
            "kernel_before_raw_ns", "device_globaltimer_ns",
            "kernel_after_raw_ns", "rm_after_outer_before_raw_ns",
            "rm_after_cpu_before_raw_ns", "rm_after_gpu_ptimer_ns",
            "rm_after_cpu_after_raw_ns", "rm_after_outer_after_raw_ns",
            "rm_after_offset_low_ns", "rm_after_offset_high_ns",
            "before_bracket_width_ns", "after_bracket_width_ns",
        )
        if any(type(sample.get(key)) is not int for key in fields):
            return False
        bo, bc = sample["rm_before_outer_before_raw_ns"], sample["rm_before_cpu_before_raw_ns"]
        bg, ba = sample["rm_before_gpu_ptimer_ns"], sample["rm_before_cpu_after_raw_ns"]
        bx = sample["rm_before_outer_after_raw_ns"]
        ko, kg = sample["kernel_before_raw_ns"], sample["device_globaltimer_ns"]
        kx = sample["kernel_after_raw_ns"]
        ao, ac = sample["rm_after_outer_before_raw_ns"], sample["rm_after_cpu_before_raw_ns"]
        ag, aa = sample["rm_after_gpu_ptimer_ns"], sample["rm_after_cpu_after_raw_ns"]
        ax = sample["rm_after_outer_after_raw_ns"]
        if not (
            sample.get("type") == "identity_sample"
            and sample["trial"] == index
            and sample.get("contained") is True
            and sample.get("accepted") is True
            and 0 < bo <= bc <= ba <= bx <= ko <= kx <= ao <= ac <= aa <= ax
            and bx - bo < 10_000_000 and ax - ao < 10_000_000
            and 0 < bg <= kg <= ag
            and sample["rm_before_offset_low_ns"] == bg - ba - 32
            and sample["rm_before_offset_high_ns"] == bg - bc + 32
            and sample["rm_after_offset_low_ns"] == ag - aa - 32
            and sample["rm_after_offset_high_ns"] == ag - ac + 32
            and sample["before_bracket_width_ns"] == ba - bc + 64
            and sample["after_bracket_width_ns"] == aa - ac + 64
            and (previous_raw == 0 or bc >= previous_raw)
            and (previous_ptimer == 0 or bg >= previous_ptimer)
        ):
            return False
        previous_raw, previous_ptimer = ac, ag
    return records[-1] == {
        "type": "identity_summary", "requested": LAUNCH_CONTROL_SAMPLES,
        "attempted": LAUNCH_CONTROL_SAMPLES, "accepted": LAUNCH_CONTROL_SAMPLES,
        "rejected": 0, "containment_failures": 0, "raw_regressions": 0,
        "ptimer_regressions": 0, "cuda_errors": 0, "setup_complete": True,
        "cleanup_complete": True, "gate_passed": True,
    }


def launch_controls_valid(campaign: Path, state: dict[str, Any]) -> bool:
    try:
        stored = state.get("clock_controls")
        raw_record = read_json(campaign / "clock-controls.json")
        provenance = state.get("provenance", {})
        if (
            not isinstance(stored, dict) or raw_record != stored
            or stored.get("role") != "calibration_only"
            or stored.get("passed") is not True
            or stored.get("boot_id") != provenance.get("boot_id")
            or stored.get("driver") != provenance.get("driver")
        ):
            return False
        for name, validator, executable in (
            ("endpoint_precision", endpoint_control_valid, "rm_ptimer_correlation_sanity"),
            ("globaltimer_identity", identity_control_valid, "rm_globaltimer_identity"),
        ):
            item = stored.get(name)
            if (
                not isinstance(item, dict) or item.get("valid") is not True
                or item.get("returncode") != 0 or item.get("error") is not None
            ):
                return False
            stdout_path = campaign_path(campaign, item.get("stdout"), absolute_ok=True)
            stderr_path = campaign_path(campaign, item.get("stderr"), absolute_ok=True)
            safety_path = campaign_path(campaign, item.get("safety"), absolute_ok=True)
            if stdout_path.parent != stderr_path.parent or stdout_path.parent != safety_path.parent:
                return False
            process_path = stdout_path.parent / "process.log"
            process_stdout, process_stderr, returncode = wrapped_process_log(process_path)
            if (
                returncode != 0
                or process_stdout != stdout_path.read_text(encoding="utf-8")
                or process_stderr != stderr_path.read_text(encoding="utf-8")
                or not execution_valid(process_path.with_suffix(".execution.json"), 0)
            ):
                return False
            safety = read_json(safety_path)
            if (
                not safety_valid(safety)
                or safety.get("boot_id") != provenance.get("boot_id")
            ):
                return False
            command = item.get("command")
            command_path = (
                Path(command[0]).resolve()
                if isinstance(command, list) and command and isinstance(command[0], str)
                else None
            )
            if (
                not isinstance(command, list) or not command
                or Path(command[0]).name != executable
                or command_path is None
                or not command_path.is_relative_to(campaign)
                or not command_path.is_file()
                or command_path.stat().st_size <= 0
            ):
                return False
            if name == "endpoint_precision" and command[1:] != [
                "--samples", str(LAUNCH_CONTROL_SAMPLES), "--control-transport",
                "direct", "--correlation-command", "endpoints-v1",
            ]:
                return False
            if name == "globaltimer_identity" and command[1:] != [
                "--samples", str(LAUNCH_CONTROL_SAMPLES)
            ]:
                return False
            if not validator(read_strict_jsonl(stdout_path)):
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, UnicodeError):
        return False


def analyze(campaign: Path) -> dict[str, Any]:
    campaign = campaign.resolve()
    state = read_json(campaign / "result.json")
    if not isinstance(state, dict):
        raise ValueError("result.json is not an object")
    params = state.get("params")
    if not isinstance(params, dict):
        raise ValueError("missing recorded parameters")
    tools = selected_tools(params)
    validate_frozen_params(params)
    verifier_level = params.get("verifier_level", "DEFAULT")
    configs = selected_configs(tools)
    phase = params.get("phase")
    expected_runs, expected_pp = {"preflight": (1, 32), "full": (10, 512)}.get(
        phase, (None, None))
    if params.get("runs") != expected_runs or params.get("pp") != expected_pp:
        raise ValueError("campaign is not a fixed preflight/full matrix")
    if state.get("phase") != phase:
        raise ValueError("top-level and parameter phases differ")
    if state.get("provenance", {}).get("driver") != EXPECTED_DRIVER:
        raise ValueError("campaign was not admitted on the fixed supported driver")
    require_raw = "launchlate" in tools
    if require_raw and not isinstance(state.get("provenance", {}).get("boot_id"), str):
        raise ValueError("launchlate campaign lacks a recorded boot identity")
    inventory_raw = state.get("provenance", {}).get("supported_clock_pairs_mhz")
    supported_clock_pairs = {
        tuple(pair) for pair in inventory_raw
        if isinstance(pair, list) and len(pair) == 2
        and all(type(value) is int for value in pair)
    } if isinstance(inventory_raw, list) else set()
    if require_raw and (
            not supported_clock_pairs
            or len(supported_clock_pairs) != len(inventory_raw)):
        raise ValueError("launchlate campaign lacks a valid pre-recorded clock inventory")
    clock_controls_passed = (
        launch_controls_valid(campaign, state) if require_raw else None
    )
    preflight_gate = {"required": False, "campaign": None, "independently_complete": None}
    preflight_reference = params.get("preflight_campaign")
    if phase == "full" and tools != TASKS:
        if not isinstance(preflight_reference, str):
            raise ValueError("subset full is missing its absolute preflight campaign")
        preflight_path = Path(preflight_reference)
        if not preflight_path.is_absolute():
            raise ValueError("subset full preflight campaign is not absolute")
        require_disjoint_campaign_paths(preflight_path, campaign)
        preflight = analyze(preflight_path)
        if (preflight.get("phase") != "preflight"
                or tuple(preflight.get("tools", ())) != tools
                or preflight.get("configs") != list(configs)
                or preflight.get("verifier_level") != verifier_level
                or preflight.get("complete") is not True):
            raise ValueError("referenced subset preflight is not independently complete")
        preflight_gate = {"required": True, "campaign": str(preflight_path.resolve()),
                          "independently_complete": True}
    elif preflight_reference is not None:
        raise ValueError("preflight campaign references are only valid for subset full")
    if set(state.get("correctness", {})) != set(configs):
        raise ValueError("correctness matrix differs from selected tools")
    if set(state.get("configs", {})) != set(configs):
        raise ValueError("timing matrix differs from selected tools")
    if state.get("schedule") != fixed_schedule(configs, expected_runs):
        raise ValueError("schedule differs from fixed selected-tool matrix")

    correctness = {}
    for config in configs:
        attempts = state["correctness"][config].get("attempts", [])
        correctness[config] = one_valid(
            attempts, config, params, True, campaign=campaign,
            require_raw=require_raw,
        ) is not None

    complete_blocks = []
    cells_by_block = {}
    rejected_cells = []
    for block in range(1, expected_runs + 1):
        cells = {}
        for config in configs:
            runs = state["configs"][config].get("runs", [])
            cell = one_valid(
                runs, config, params, False, block,
                campaign=campaign, require_raw=require_raw,
            )
            if cell is None:
                rejected_cells.append({"block": block, "config": config})
            else:
                cells[config] = cell
        if "kernelretsnoop" in tools and all(
                config in cells for config in ("gpubpf_kernelretsnoop", "nvbit_kernelretsnoop")):
            gp = cells["gpubpf_kernelretsnoop"]["probe"]
            nv = cells["nvbit_kernelretsnoop"]["probe"]
            matched_fields = (
                "sample_count", "nonzero_timestamps", "record_bytes",
                "cartesian_launches", "cartesian_coordinates", "cartesian_complete",
                "extent_x", "extent_y", "extent_z", "multiplicity_220",
                "multiplicity_44", "multiplicity_22", "other_multiplicity",
                "segment_mismatches", "invalid_launch_coordinates",
                "unique_coordinates", "collector_gate_passed",
            )
            if (any(gp.get(key) != nv.get(key) for key in matched_fields)
                    or gp.get("cartesian_launches") != nv.get("selected_launches")):
                rejected_cells.extend([
                    {"block": block, "config": "gpubpf_kernelretsnoop", "reason": "pair mismatch"},
                    {"block": block, "config": "nvbit_kernelretsnoop", "reason": "pair mismatch"},
                ])
                cells.pop("gpubpf_kernelretsnoop")
                cells.pop("nvbit_kernelretsnoop")
        if "launchlate" in tools and all(
                config in cells for config in
                ("baseline", "gpubpf_launchlate", "nvbit_launchlate")):
            if not launchlate_clock_block_valid(
                    campaign, cells, supported_clock_pairs):
                rejected_cells.extend([
                    {"block": block, "config": config,
                     "reason": "launchlate clock-state fairness mismatch"}
                    for config in
                    ("baseline", "gpubpf_launchlate", "nvbit_launchlate")
                ])
                cells.pop("baseline")
                cells.pop("gpubpf_launchlate")
                cells.pop("nvbit_launchlate")
        if len(cells) == len(configs):
            complete_blocks.append({"block": block, "cells": cells})
        cells_by_block[block] = cells

    comparisons = []
    for tool in tools:
        paired = []
        raw_triples = []
        for block_number in range(1, expected_runs + 1):
            cells = cells_by_block[block_number]
            required = ("baseline", f"gpubpf_{tool}", f"nvbit_{tool}")
            if not all(config in cells for config in required):
                continue
            baseline = cells["baseline"]["metrics"]["pp_tok_s"]
            gpubpf = cells[f"gpubpf_{tool}"]["metrics"]["pp_tok_s"]
            nvbit = cells[f"nvbit_{tool}"]["metrics"]["pp_tok_s"]
            gp_overhead = (baseline - gpubpf) / baseline * 100
            nv_overhead = (baseline - nvbit) / baseline * 100
            effect = nv_overhead - gp_overhead
            paired.append(effect)
            raw_triples.append({
                "block": block_number, "baseline_pp_tok_s": baseline,
                "gpubpf_pp_tok_s": gpubpf, "nvbit_pp_tok_s": nvbit,
                "gpubpf_degradation_percent": gp_overhead,
                "nvbit_degradation_percent": nv_overhead,
                "paired_effect_percentage_points": effect,
            })
        comparisons.append({"task": tool, "paired_blocks": len(paired),
                            "effect_definition": "NVBit overhead - gpubpf overhead (percentage points)",
                            "raw_paired_triples": raw_triples,
                            "median_paired_effect": statistics.median(paired) if paired else None,
                            "bootstrap": bootstrap_mean(paired)})

    config_rows = []
    for config in configs:
        values = [cells_by_block[block][config]["metrics"]["pp_tok_s"]
                  for block in range(1, expected_runs + 1)
                  if config in cells_by_block[block]]
        geomean = (math.exp(statistics.mean(math.log(value) for value in values))
                   if values else None)
        config_rows.append({"config": config, "valid_blocks": len(values),
                            "pp_tok_s_geomean": geomean})

    complete = (
        all(correctness.values()) and len(complete_blocks) == expected_runs
        and not rejected_cells and clock_controls_passed is not False
    )
    return {
        "phase": phase,
        "tools": list(tools),
        "configs": list(configs),
        "verifier_level": verifier_level,
        "correctness": correctness,
        "valid_complete_blocks": len(complete_blocks),
        "required_blocks": expected_runs,
        "preflight_gate": preflight_gate,
        "launch_clock_controls": {
            "required": require_raw,
            "role": "calibration_only" if require_raw else None,
            "independently_passed": clock_controls_passed,
        },
        "complete": complete,
        "rejected_cells": rejected_cells,
        "config_summaries": config_rows,
        "comparisons": comparisons,
        "scope_policy": (
            "Unselected tools are outside this predeclared campaign only; prior failures "
            "remain failures and are not reclassified by this analysis."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.campaign.resolve())
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
