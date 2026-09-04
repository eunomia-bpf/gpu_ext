#!/usr/bin/env python3
"""Independently audit a revision-RQ4 result without launching GPU work."""

from __future__ import annotations

import argparse
import json
import math
import random
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


def verifier_valid(cell: dict[str, Any], params: dict[str, Any]) -> bool:
    level = params.get("verifier_level", "DEFAULT")
    if level == "DEFAULT":
        return True
    evidence = cell.get("verifier")
    if not isinstance(evidence, dict) or evidence.get("level") != level:
        return False
    if evidence.get("required") is not True or evidence.get("passed") is not True:
        return False
    logs = evidence.get("logs_scanned")
    matched = evidence.get("matched_log_sources")
    if (
        not isinstance(logs, list)
        or not logs
        or not all(isinstance(name, str) and name for name in logs)
        or not isinstance(matched, list)
        or not matched
        or not all(isinstance(name, str) and name in logs for name in matched)
    ):
        return False
    if level == "STRICT":
        counts = evidence.get("instruction_counts")
        return (
            type(evidence.get("accepted_records")) is int
            and evidence["accepted_records"] >= 1
            and isinstance(counts, list)
            and bool(counts)
            and all(type(count) is int and count > 0 for count in counts)
            and type(evidence.get("verified_map_records")) is int
            and evidence["verified_map_records"] >= 1
            and evidence.get("skipped_records") == 0
            and evidence.get("rejected") is False
        )
    return (
        type(evidence.get("skipped_records")) is int
        and evidence["skipped_records"] >= 1
        and evidence.get("accepted_records") == 0
        and evidence.get("verified_map_records") == 0
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


def integer(probe: dict[str, Any], key: str) -> int:
    value = probe.get(key)
    if type(value) is not int:
        raise ValueError(f"missing integer probe field: {key}")
    return value


def launch_clock_valid(probe: dict[str, Any]) -> bool:
    try:
        start_low = integer(probe, "start_clock_offset_lower_ns")
        start_high = integer(probe, "start_clock_offset_upper_ns")
        end_low = integer(probe, "end_clock_offset_lower_ns")
        end_high = integer(probe, "end_clock_offset_upper_ns")
        start_anchor = integer(probe, "start_clock_host_anchor_ns")
        end_anchor = integer(probe, "end_clock_host_anchor_ns")
        elapsed = end_anchor - start_anchor
        change_low = end_low - start_high
        change_high = end_high - start_low
        if elapsed <= 0:
            return False
        rate = (max(abs(change_low), abs(change_high)) * 1_000_000_000 + elapsed - 1) // elapsed
        return (
            start_low <= start_high and end_low <= end_high
            and integer(probe, "start_clock_uncertainty_ns") == (start_high - start_low + 1) // 2
            and integer(probe, "end_clock_uncertainty_ns") == (end_high - end_low + 1) // 2
            and integer(probe, "clock_offset_change_lower_ns") == change_low
            and integer(probe, "clock_offset_change_upper_ns") == change_high
            and integer(probe, "clock_calibration_elapsed_ns") == elapsed
            and elapsed >= LAUNCH_MIN_CALIBRATION_SPAN_NS
            and integer(probe, "clock_drift_rate_bound_ppb") == rate
            and integer(probe, "clock_drift_limit_ppb") == LAUNCH_CLOCK_DRIFT_LIMIT_PPB
            and rate <= LAUNCH_CLOCK_DRIFT_LIMIT_PPB
            and integer(probe, "clock_drift_bounded") == 1
        )
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
            == "bracketed %globaltimer endpoint intervals with affine CLOCK_MONOTONIC interpolation"
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
            == "bracketed_globaltimer_endpoints_against_CLOCK_MONOTONIC_with_affine_interpolation_and_drift_bound"
            and integer(probe, "start_clock_calibration_valid") == 1
            and integer(probe, "end_clock_calibration_valid") == 1
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
        )
    except ValueError:
        return False


def cell_valid(cell: dict[str, Any], config: str, params: dict[str, Any],
               correctness: bool) -> bool:
    if (cell.get("valid") is not True or cell.get("returncode") != 0
            or cell.get("error") or cell.get("fatal_cleanup")
            or not safety_valid(cell.get("safety"))):
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
        gpubpf_valid(tool, probe, params, correctness) and verifier_valid(cell, params)
        if system == "gpubpf" else nvbit_valid(tool, probe, params, correctness)
    )


def one_valid(entries: list[dict[str, Any]], config: str, params: dict[str, Any],
              correctness: bool, block: int | None = None) -> dict[str, Any] | None:
    candidates = [entry for entry in entries
                  if (block is None or entry.get("block") == block)
                  and cell_valid(entry, config, params, correctness)]
    if len(candidates) > 1:
        raise ValueError(f"multiple independently valid entries for {config}, block {block}")
    return candidates[0] if candidates else None


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


def analyze(campaign: Path) -> dict[str, Any]:
    campaign = campaign.resolve()
    state = json.loads((campaign / "result.json").read_text(encoding="utf-8"))
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
        correctness[config] = one_valid(attempts, config, params, True) is not None

    complete_blocks = []
    cells_by_block = {}
    rejected_cells = []
    for block in range(1, expected_runs + 1):
        cells = {}
        for config in configs:
            runs = state["configs"][config].get("runs", [])
            cell = one_valid(runs, config, params, False, block)
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
        if len(cells) == len(configs):
            complete_blocks.append({"block": block, "cells": cells})
        cells_by_block[block] = cells

    comparisons = []
    for tool in tools:
        paired = []
        for block_number in range(1, expected_runs + 1):
            cells = cells_by_block[block_number]
            required = ("baseline", f"gpubpf_{tool}", f"nvbit_{tool}")
            if not all(config in cells for config in required):
                continue
            baseline = cells["baseline"]["metrics"]["pp_tok_s"]
            gp_overhead = (baseline - cells[f"gpubpf_{tool}"]["metrics"]["pp_tok_s"]) / baseline * 100
            nv_overhead = (baseline - cells[f"nvbit_{tool}"]["metrics"]["pp_tok_s"]) / baseline * 100
            paired.append(nv_overhead - gp_overhead)
        comparisons.append({"task": tool, "paired_blocks": len(paired),
                            "effect_definition": "NVBit overhead - gpubpf overhead (percentage points)",
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

    complete = (all(correctness.values()) and len(complete_blocks) == expected_runs
                and not rejected_cells)
    return {
        "phase": phase,
        "tools": list(tools),
        "configs": list(configs),
        "verifier_level": verifier_level,
        "correctness": correctness,
        "valid_complete_blocks": len(complete_blocks),
        "required_blocks": expected_runs,
        "preflight_gate": preflight_gate,
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
