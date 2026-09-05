#!/usr/bin/env python3
"""Run the matched RTX 5090 gpubpf/NVBit observability experiment."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import csv
import errno
import fcntl
import json
import math
import os
import random
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
OBS_ROOT = HERE.parent
sys.path.insert(0, str(OBS_ROOT))
import run_observability_overhead as core  # noqa: E402
sys.path.insert(0, str(core.WORKLOADS_DIR / "gpreempt"))
import run_three_way as shared  # noqa: E402


NVBIT_ROOT = HERE / "deps/nvbit_release_x86_64"
NVBIT_SOURCE_DIR = HERE / "nvbit_adapters/observability"
CLOCK_CONTROL_SOURCE_DIR = HERE / "launch-clock-recovery"
CONFIGS = [
    "baseline",
    "gpubpf_kernelretsnoop",
    "nvbit_kernelretsnoop",
    "gpubpf_threadhist",
    "nvbit_threadhist",
    "gpubpf_launchlate",
    "nvbit_launchlate",
]
TASKS = ("kernelretsnoop", "threadhist", "launchlate")
VERIFIER_LEVELS = ("DEFAULT", "STRICT", "NO_VERIFY")
SCHEDULE_SEED = 1797
BOOTSTRAP_SAMPLES = 10000
EXPECTED_DRIVER = "575.57.08"
SUPPORTED_CLOCK_COMMAND = (
    "nvidia-smi", "--query-supported-clocks=memory,graphics",
    "--format=csv,noheader,nounits",
)
SHM_ROOT = Path("/dev/shm")
CLIENT_CPUS = "8-15"
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
CORRECTNESS_MULTIPLICITY_220 = 1024
CORRECTNESS_MULTIPLICITY_44 = 1024
CORRECTNESS_MULTIPLICITY_22 = 20480
KERNELRETSNOOP_SHM_MEMORY_MB = 1000
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
EXPECTED_NORMALIZED_STDOUT = "Deterministic tests are essential\n> EOF by user"
EXPECTED_NORMALIZED_STDOUT_BYTES = 47
LEASE_PATHS = (
    Path("/tmp/gpubpf-revision-gpu0.lock"),
    Path("/tmp/gpubpf-revision-struct-ops.lock"),
)
RELATIVE_RUNTIME_INCLUDE = "../../../runtime/include"
RELATIVE_RUNTIME_INCLUDE_PATTERN = re.compile(r"(?:\.\./)+runtime/include")
KERNELRETSNOOP_CAPACITY_PATCH = HERE / "kernelretsnoop-phase-capacity.patch"
LATE_BOOTSTRAP_TARGET_FILTER_PATCH = (
    HERE / "runtime-575/late-bootstrap-target-filter.patch"
)


def kernelretsnoop_layout(pp: int, *, correctness: bool) -> dict[str, int]:
    """Return the frozen phase-specific dense-ring geometry and exact event count."""
    if correctness:
        slots = CORRECTNESS_EXIT_COORDINATES
        entries = CORRECTNESS_RING_ENTRIES_PER_THREAD
        launches = CORRECTNESS_EXIT_LAUNCHES
        events = CORRECTNESS_EXIT_EVENTS
    else:
        if pp not in (32, 512):
            raise ValueError("kernelretsnoop timing layout is defined only for pp32/pp512")
        slots = pp * TIMING_THREADS_PER_PROMPT_TOKEN
        entries = TIMING_RING_ENTRIES_PER_THREAD
        launches = TIMING_EXIT_LAUNCHES
        events = slots * launches
    shared_bytes = RING_ERROR_COUNTER_BYTES + slots * (
        RING_SLOT_HEADER_BYTES + entries * RING_ALIGNED_RECORD_BYTES
    )
    if slots % 256 != 0:
        raise ValueError("kernelretsnoop coordinates must form x-by-256-by-1 rope geometry")
    return {"thread_slots": slots, "entries_per_thread": entries,
            "launches": launches, "coordinates": slots, "events": events,
            "extent_x": slots // 256, "extent_y": 256, "extent_z": 1,
            "shared_bytes": shared_bytes}


def selected_tools(args: argparse.Namespace) -> tuple[str, ...]:
    """Return a unique, canonical, predeclared tool selection."""
    requested = tuple(getattr(args, "tools", TASKS))
    unknown = [tool for tool in requested if tool not in TASKS]
    duplicates = [tool for tool in TASKS if requested.count(tool) > 1]
    if not requested:
        raise ValueError("--tools requires at least one tool")
    if unknown:
        raise ValueError("unknown tool(s): " + ", ".join(unknown))
    if duplicates:
        raise ValueError("duplicate tool(s): " + ", ".join(duplicates))
    return tuple(tool for tool in TASKS if tool in requested)


def selected_configs(args: argparse.Namespace) -> tuple[str, ...]:
    tools = selected_tools(args)
    return tuple(
        config
        for config in CONFIGS
        if config == "baseline" or config.split("_", 1)[1] in tools
    )


def selected_verifier_level(args: argparse.Namespace) -> str:
    """Return the explicit verifier treatment without changing legacy runs."""
    level = str(getattr(args, "verifier_level", "DEFAULT")).upper()
    if level not in VERIFIER_LEVELS:
        raise ValueError(f"unsupported verifier level: {level}")
    return level


def verifier_environment(args: argparse.Namespace) -> dict[str, str]:
    """Environment required to make verifier treatment observable and auditable."""
    level = selected_verifier_level(args)
    if level == "DEFAULT":
        return {}
    return {"BPFTIME_VERIFIER_LEVEL": level, "SPDLOG_LEVEL": "info"}


def verifier_runtime_configuration(build: Path) -> dict[str, str]:
    cache = build / "CMakeCache.txt"
    config: dict[str, str] = {}
    if cache.is_file():
        for line in cache.read_text().splitlines():
            key, separator, value = line.partition("=")
            if separator and ":" in key:
                config[key.partition(":")[0]] = value
    keys = ("ENABLE_EBPF_VERIFIER", "BPFTIME_ENABLE_CUDA_ATTACH", "BPFTIME_LLVM_JIT")
    return {key: config.get(key, "unknown") for key in keys}


def require_explicit_verifier_build(args: argparse.Namespace) -> dict[str, str]:
    """Fail closed unless STRICT/NO_VERIFY use the same verifier-capable runtime."""
    config = verifier_runtime_configuration(args.bpftime_build_dir)
    if selected_verifier_level(args) != "DEFAULT" and any(
        config[key].upper() not in {"ON", "YES", "TRUE", "1"} for key in config
    ):
        raise RuntimeError(
            "explicit verifier treatment requires one verifier-enabled CUDA/LLVM runtime build"
        )
    return config


def verifier_map_expectation(
    args: argparse.Namespace, tool: str, *, correctness: bool
) -> dict[str, int]:
    """Return the one GPU map expected for each supported Table 1 policy."""
    if tool == "kernelretsnoop":
        return {
            "type": 1527,
            "key_size": 4,
            "value_size": 32,
            "max_entries": kernelretsnoop_layout(
                args.pp, correctness=correctness
            )["entries_per_thread"],
        }
    if tool == "threadhist":
        return {
            "type": 1502,
            "key_size": 4,
            "value_size": 8,
            "max_entries": 1,
        }
    raise ValueError(f"explicit verifier evidence is unsupported for {tool}")


def verifier_evidence(
    args: argparse.Namespace, run_dir: Path, tool: str, *, correctness: bool
) -> dict[str, Any]:
    """Bind one policy admission and its map to the recorded target process."""
    level = selected_verifier_level(args)
    if level == "DEFAULT":
        return {"level": level, "required": False, "passed": True}
    executable = "llama_cli" if correctness else "llama_bench"
    execution_path = run_dir / f"{executable}.execution.json"
    evidence_path = run_dir / f"{executable}.log"
    execution_error: str | None = None
    target_pid: int | None = None
    try:
        execution = json.loads(execution_path.read_text())
        recorded_pid = execution["identity"]["pid"]
        if type(recorded_pid) is not int or recorded_pid <= 0:
            raise ValueError("target pid is not a positive integer")
        target_pid = recorded_pid
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        execution_error = f"{type(error).__name__}: {error}"
    log_text = evidence_path.read_text(errors="replace") if evidence_path.is_file() else ""
    expected_map = verifier_map_expectation(args, tool, correctness=correctness)
    program = "cuda__retprobe"
    expected_attach = f"kretprobe/{args.target_symbol}"
    accepted_pattern = re.compile(
        rf"^\[[^\]\r\n]+\]\[[^\]\r\n]+\]\[(?P<pid>[1-9][0-9]*)\] "
        rf"GPU eBPF verification accepted: mode=(?P<mode>[^ \r\n]+) "
        rf"program={program} attach=(?P<attach>[^ \r\n]+) "
        rf"instructions=(?P<instructions>[0-9]+)\r?$"
    )
    map_pattern = re.compile(
        rf"^\[[^\]\r\n]+\]\[[^\]\r\n]+\]\[(?P<pid>[1-9][0-9]*)\] "
        rf"GPU eBPF verified map: program={program} fd=(?P<fd>[0-9]+) "
        r"type=(?P<type>[0-9]+) key_size=(?P<key_size>[0-9]+) "
        r"value_size=(?P<value_size>[0-9]+) "
        r"max_entries=(?P<max_entries>[0-9]+)\r?$"
    )
    skip_pattern = re.compile(
        rf"^\[[^\]\r\n]+\]\[[^\]\r\n]+\]\[(?P<pid>[1-9][0-9]*)\] "
        rf"Skipping GPU eBPF verification for {program}\r?$"
    )
    reject_pattern = re.compile(
        rf"^\[[^\]\r\n]+\]\[[^\]\r\n]+\]\[(?P<pid>[1-9][0-9]*)\] "
        rf"GPU eBPF verification failed for {program}:.*$"
    )
    accepted: list[dict[str, Any]] = []
    verified_maps: list[dict[str, int]] = []
    skipped = 0
    rejected = 0
    foreign_pid_records = 0
    unexpected_target_records = 0
    unparsed_records = 0
    marker_fragments = (
        f"GPU eBPF verification accepted:",
        f"GPU eBPF verified map: program={program}",
        f"Skipping GPU eBPF verification for {program}",
        f"GPU eBPF verification failed for {program}:",
    )
    for line in log_text.splitlines():
        if not any(fragment in line for fragment in marker_fragments):
            continue
        match = accepted_pattern.fullmatch(line)
        kind = "accepted"
        if match is None:
            match = map_pattern.fullmatch(line)
            kind = "map"
        if match is None:
            match = skip_pattern.fullmatch(line)
            kind = "skipped"
        if match is None:
            match = reject_pattern.fullmatch(line)
            kind = "rejected"
        if match is None:
            unparsed_records += 1
            continue
        pid = int(match.group("pid"))
        if target_pid is None or pid != target_pid:
            foreign_pid_records += 1
            continue
        if kind == "accepted":
            record = {
                "mode": match.group("mode"),
                "attach": match.group("attach"),
                "instructions": int(match.group("instructions")),
            }
            accepted.append(record)
            if (
                record["mode"] != "STRICT"
                or record["attach"] != expected_attach
                or record["instructions"] <= 0
            ):
                unexpected_target_records += 1
        elif kind == "map":
            record = {
                field: int(match.group(field))
                for field in (
                    "fd", "type", "key_size", "value_size", "max_entries"
                )
            }
            verified_maps.append(record)
            if any(record[field] != value for field, value in expected_map.items()):
                unexpected_target_records += 1
        elif kind == "skipped":
            skipped += 1
        else:
            rejected += 1
    exact_accepted = (
        len(accepted) == 1
        and accepted[0]["mode"] == "STRICT"
        and accepted[0]["attach"] == expected_attach
        and accepted[0]["instructions"] > 0
    )
    exact_map = (
        len(verified_maps) == 1
        and all(
            verified_maps[0][field] == value
            for field, value in expected_map.items()
        )
    )
    common = (
        execution_error is None
        and evidence_path.is_file()
        and foreign_pid_records == 0
        and unexpected_target_records == 0
        and unparsed_records == 0
    )
    if level == "STRICT":
        passed = common and exact_accepted and exact_map and skipped == 0 and rejected == 0
    else:
        passed = (
            common and skipped == 1 and not accepted and not verified_maps
            and rejected == 0
        )
    matched_sources = [evidence_path.name] if any(
        (accepted, verified_maps, skipped, rejected)
    ) else []
    return {
        "level": level,
        "required": True,
        "passed": passed,
        "program": program,
        "attach": expected_attach,
        "target_pid": target_pid,
        "execution_record": execution_path.name,
        "execution_error": execution_error,
        "expected_map": expected_map,
        "accepted_records": len(accepted),
        "instruction_counts": [record["instructions"] for record in accepted],
        "verified_map_records": len(verified_maps),
        "verified_maps": verified_maps,
        "skipped_records": skipped,
        "rejected": rejected > 0,
        "foreign_pid_records": foreign_pid_records,
        "unexpected_target_records": unexpected_target_records,
        "unparsed_records": unparsed_records,
        "logs_scanned": [evidence_path.name] if evidence_path.is_file() else [],
        "logs_missing": [] if evidence_path.is_file() else [evidence_path.name],
        "matched_log_sources": matched_sources,
    }


def fixed_schedule(args: argparse.Namespace) -> dict[str, list[str]]:
    configs = selected_configs(args)
    schedules = {}
    for block in range(1, args.runs + 1):
        order = list(configs)
        random.Random(SCHEDULE_SEED + block).shuffle(order)
        schedules[str(block)] = order
    return schedules


def engagement_gate_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Machine-readable declaration of unchanged, exact per-tool gates."""
    gates: dict[str, Any] = {
        "all_correctness_cells": {
            "returncode": 0,
            "normalized_stdout": EXPECTED_NORMALIZED_STDOUT,
            "stdout_bytes": EXPECTED_NORMALIZED_STDOUT_BYTES,
            "must_match_baseline": True,
        },
        "all_timing_cells": {
            "returncode": 0,
            "pp_tokens": args.pp,
            "pp_tok_s": "finite and > 0",
        },
    }
    if "kernelretsnoop" in selected_tools(args):
        correctness_layout = kernelretsnoop_layout(args.pp, correctness=True)
        timing_layout = kernelretsnoop_layout(args.pp, correctness=False)
        gates["kernelretsnoop_correctness"] = {
            "gpubpf": {
                "events": CORRECTNESS_EXIT_EVENTS,
                "nonzero_timestamps": CORRECTNESS_EXIT_EVENTS,
                "selected_launches": CORRECTNESS_EXIT_LAUNCHES,
                "unique_coordinates": CORRECTNESS_EXIT_COORDINATES,
                "exact_extents_xyz": [correctness_layout["extent_x"],
                                       correctness_layout["extent_y"],
                                       correctness_layout["extent_z"]],
                "requested_and_allocated_thread_slots": EXPECTED_GPU_THREAD_SLOTS,
                "exact_ring_entries_per_thread": correctness_layout["entries_per_thread"],
                "record_bytes": EXIT_RECORD_BYTES,
                "zero_drop_dirty_pending_second_drain_and_invalid_coordinate_counters": True,
                "cartesian_complete": True,
                "exact_multiplicity": {
                    "220": CORRECTNESS_MULTIPLICITY_220,
                    "44": CORRECTNESS_MULTIPLICITY_44,
                    "22": CORRECTNESS_MULTIPLICITY_22,
                    "other": 0,
                    "segment_mismatches": 0,
                },
            },
            "nvbit": {
                "events": CORRECTNESS_EXIT_EVENTS,
                "nonzero_timestamps": CORRECTNESS_EXIT_EVENTS,
                "selected_launches": CORRECTNESS_EXIT_LAUNCHES,
                "record_bytes": EXIT_RECORD_BYTES,
                "exact_extents_xyz": [correctness_layout["extent_x"],
                                       correctness_layout["extent_y"],
                                       correctness_layout["extent_z"]],
                "same_coordinate_multiplicity_and_collector_gate": True,
            },
        }
        gates["kernelretsnoop_timing"] = {
            "gpubpf_requested_and_allocated_thread_slots": timing_layout["thread_slots"],
            "gpubpf_exact_ring_entries_per_thread": timing_layout["entries_per_thread"],
            "gpubpf_exact_coordinates": timing_layout["coordinates"],
            "gpubpf_exact_events": timing_layout["events"],
            "gpubpf_exact_selected_launches": timing_layout["launches"],
            "both_exact_extents_xyz": [timing_layout["extent_x"],
                                       timing_layout["extent_y"],
                                       timing_layout["extent_z"]],
            "gpubpf_exact_coordinate_multiplicity": {"44": timing_layout["coordinates"],
                                                       "220": 0, "22": 0, "other": 0},
            "nvbit_exact_events": timing_layout["events"],
            "nvbit_exact_selected_launches": timing_layout["launches"],
            "gpubpf_internal_lossless_and_cartesian_complete": True,
            "gpubpf_exact_correctness_multiplicity_oracle": False,
            "pair_events_equal": True,
            "pair_selected_launches_equal": True,
        }
    if "threadhist" in selected_tools(args):
        gates["threadhist_all_cells"] = {
            "gpubpf_configured_entries": args.threadhist_gpu_thread_count,
            "gpubpf_readback_entries": args.threadhist_gpu_thread_count,
            "gpubpf_readback_bytes": args.threadhist_gpu_thread_count * 8,
            "gpubpf_readback_complete": 1,
            "gpubpf_nonzero_threads": "> 0",
            "gpubpf_total_exit_probes": "> 0",
            "nvbit_selected_launches": "> 0",
            "nvbit_nonzero_threads": "> 0",
            "nvbit_total_exit_probes": "> 0",
        }
    if "launchlate" in selected_tools(args):
        gates["launchlate_all_cells"] = {
            "clock_method": "CLOCK_MONOTONIC_RAW translated to RM PTIMER/globaltimer intervals",
            "rm_endpoint_trials_per_anchor": LAUNCH_RM_CALIBRATION_SAMPLES,
            "maximum_anchor_bracket_ns": LAUNCH_RM_MAX_BRACKET_NS,
            "zero_clock_queue_capture_overflow_errors": True,
            "bounded_clock_drift_and_uncertainty": True,
            "complete_host_device_pairing": True,
        }
        gates["launchlate_timing_block_clock_fairness"] = {
            "arms": ["baseline", "gpubpf_launchlate", "nvbit_launchlate"],
            "all_samples_pstate": "P0",
            "exact_observed_sm_memory_sets": "nonempty and identical",
            "all_observed_pairs_in_prerecorded_supported_inventory": True,
            "numeric_tolerance_or_frequency_normalization": False,
        }
        gates["launchlate_correctness"] = {
            "gpubpf_exact_launches": CORRECTNESS_EXIT_LAUNCHES,
            "nvbit_exact_launches": CORRECTNESS_EXIT_LAUNCHES,
            "minimum_classified": 198,
            "maximum_uncertain": 22,
        }
        gates["launchlate_calibration_controls"] = {
            "endpoint_precision_samples": LAUNCH_CONTROL_SAMPLES,
            "globaltimer_identity_samples": LAUNCH_CONTROL_SAMPLES,
            "controls_are_not_performance_results": True,
        }
    return gates


def dry_run_plan(args: argparse.Namespace) -> dict[str, Any]:
    configs = selected_configs(args)
    subset_full = args.phase == "full" and selected_tools(args) != TASKS
    return {
        "dry_run": True,
        "phase": args.phase,
        "tools": list(selected_tools(args)),
        "configs": list(configs),
        "runs": args.runs,
        "pp": args.pp,
        "verifier_level": selected_verifier_level(args),
        "correctness_cells": list(configs),
        "timing_schedule": fixed_schedule(args),
        "timing_cell_count": len(configs) * args.runs,
        "completion_rule": {
            "required_valid_correctness_cells": len(configs),
            "required_valid_timing_cells": len(configs) * args.runs,
            "all_selected_configs_required_in_every_block": True,
        },
        "preflight_gate": {
            "required": subset_full,
            "campaign": (
                str(args.preflight_dir.resolve())
                if getattr(args, "preflight_dir", None) else None
            ),
            "condition": (
                "independent analyzer reports complete=true for the same exact tool selection"
                if subset_full else "unchanged default phase admission"
            ),
        },
        "engagement_gates": engagement_gate_manifest(args),
        "scope_policy": (
            "Only the predeclared selected tools are planned. Existing results for "
            "unselected tools are not relabeled, repaired, or counted."
        ),
    }


class OwnedCleanupError(RuntimeError):
    """Unsafe to continue the campaign; an owned resource may still be live."""

    def __init__(self, message: str, details: dict[str, Any]):
        super().__init__(message)
        self.details = details


class ReadOnlyLeases:
    """Lock pre-created coordination inodes without write or create access."""

    def __init__(self, paths: tuple[Path, ...] = LEASE_PATHS):
        self.files = []
        try:
            for path in paths:
                before = path.lstat()
                if not stat.S_ISREG(before.st_mode):
                    raise RuntimeError(f"lease path is not a regular file: {path}")
                stream = path.open("r")
                try:
                    opened = os.fstat(stream.fileno())
                    current = path.lstat()
                    expected = (before.st_dev, before.st_ino)
                    if ((opened.st_dev, opened.st_ino) != expected
                            or (current.st_dev, current.st_ino) != expected
                            or not stat.S_ISREG(opened.st_mode)
                            or not stat.S_ISREG(current.st_mode)):
                        raise RuntimeError(f"lease inode changed while opening: {path}")
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.files.append(stream)
                except BaseException:
                    stream.close()
                    raise
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        for stream in reversed(self.files):
            stream.close()
        self.files.clear()


def validate_inherited_lease_fds(descriptors: tuple[int, ...]) -> list[dict[str, Any]]:
    """Validate the exact read-only lease descriptors handed off by a parent."""
    if len(descriptors) != len(LEASE_PATHS) or len(set(descriptors)) != len(descriptors):
        raise RuntimeError("exactly two distinct inherited lease descriptors are required")
    inventory = []
    for path, descriptor in zip(LEASE_PATHS, descriptors, strict=True):
        if descriptor < 0:
            raise RuntimeError(f"invalid inherited lease descriptor: {descriptor}")
        before = path.lstat()
        opened = os.fstat(descriptor)
        current = path.lstat()
        expected = (before.st_dev, before.st_ino)
        if (not stat.S_ISREG(before.st_mode)
                or not stat.S_ISREG(opened.st_mode)
                or not stat.S_ISREG(current.st_mode)
                or (opened.st_dev, opened.st_ino) != expected
                or (current.st_dev, current.st_ino) != expected):
            raise RuntimeError(f"inherited lease identity differs: {path}")
        if (fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE) != os.O_RDONLY:
            raise RuntimeError(f"inherited lease is not read-only: {path}")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inventory.append({"path": str(path), "fd": descriptor,
                          "device": opened.st_dev, "inode": opened.st_ino})
    return inventory


def process_identity(process) -> dict[str, Any]:
    identity = {"pid": process.pid}
    try:
        fields = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()
        identity.update(pgid=int(fields[2]), sid=int(fields[3]), start_ticks=int(fields[19]))
    except (OSError, IndexError, ValueError):
        identity["proc_stat_unavailable"] = True
    return identity


def stop_owned(process, role: str, identity: dict[str, Any] | None = None) -> None:
    try:
        shared.stop_owned(process)
    except BaseException as error:
        details = {"role": role, "identity": identity or (process_identity(process) if process else None),
                   "reason": f"{type(error).__name__}: {error}"}
        if process is not None:
            details["live_group_members"] = shared.group_members(process.pid)
        raise OwnedCleanupError(f"{role} cleanup failed: {error}", details) from error


def run_cmd_owned(
    cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None,
    timeout: int | None = None, log_path: Path | None = None, check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """The legacy CPU-helper contract, with owned teardown on every exit."""
    started = datetime.now().isoformat(timespec="seconds")
    process = reader = log_file = None
    output: list[str] = []
    lock = threading.Lock()
    returncode = None
    timed_out = False
    failure = None
    try:
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w", encoding="utf-8")
            log_file.write(f"$ {' '.join(cmd)}\n# cwd: {cwd or Path.cwd()}\n# started: {started}\n\n## output\n")
            log_file.flush()
        process = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)

        def read_output() -> None:
            with process.stdout:
                for line in process.stdout:
                    with lock:
                        output.append(line)
                        if log_file is not None:
                            log_file.write(line)
                            log_file.flush()

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            stop_owned(process, "CPU helper")
        except BaseException as error:
            failure = error
            raise
        finally:
            if process is not None and returncode is None:
                returncode = process.returncode
            if reader is not None:
                reader.join(timeout=5)
            with lock:
                if log_file is not None:
                    log_file.write(f"\n# exit: {returncode}\n")
                    if timed_out:
                        log_file.write(f"# timeout_s: {timeout}\n")
                    if failure is not None:
                        log_file.write(f"# error: {type(failure).__name__}: {failure}\n")
                        if isinstance(failure, OwnedCleanupError):
                            log_file.write(f"# cleanup: {json.dumps(failure.details)}\n")
                    log_file.close()
                    log_file = None
    completed = subprocess.CompletedProcess(cmd, returncode, "".join(output), "")
    if timed_out:
        raise subprocess.TimeoutExpired(cmd, timeout, output=completed.stdout)
    if check and returncode != 0:
        raise RuntimeError(f"command failed ({returncode}): {' '.join(cmd)}")
    return completed


def target_launch(command: list[str], environment: dict[str, str]):
    # Apply affinity before loading either instrumentation runtime. In particular,
    # do not inject the GPU agent into taskset itself.
    launch_env = dict(environment)
    preload = launch_env.pop("LD_PRELOAD", None)
    prefix = ["taskset", "-c", CLIENT_CPUS, "/usr/bin/env"]
    if preload is not None:
        prefix.append(f"LD_PRELOAD={preload}")
    return prefix + command, launch_env


@contextmanager
def cell_safety(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    record = {"passed": False, "worker_cpus": CLIENT_CPUS,
              "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    process = stream = path = before = failure = None
    try:
        before = shared.safety.safety_snapshot()
        record["before"] = before
        shared.safety.validate_pre_server_safety(before)
        if before["gpu"]["driver"] != EXPECTED_DRIVER:
            raise RuntimeError("driver changed before cell")
        process, stream, path = shared.safety.start_gpu_telemetry(
            directory, include_pstate=True
        )
        yield record
        if process.poll() is not None:
            raise RuntimeError("GPU telemetry stopped before cell completion")
        record["passed"] = True
    except BaseException as error:
        failure = error
        record["error"] = str(error)
        if isinstance(error, OwnedCleanupError):
            record["fatal_cleanup"] = error.details
        raise
    finally:
        errors = []
        try:
            stop_owned(process, "GPU telemetry")
        except BaseException as error:
            errors.append(str(error))
            if not isinstance(failure, OwnedCleanupError):
                failure = error
        if stream is not None:
            stream.close()
        try:
            if before is not None:
                record["after"] = shared.safety.wait_for_post_server_safety(before)
                if record["after"]["gpu"]["driver"] != EXPECTED_DRIVER:
                    raise RuntimeError("driver changed during cell")
            if path is not None:
                record["telemetry"] = shared.safety.validate_gpu_telemetry(path, allow_fixed_power_cap=True)
            if record["boot_id"] != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
                raise RuntimeError("boot changed during cell")
        except BaseException as error:
            errors.append(str(error))
        if errors:
            record.update(passed=False, cleanup_errors=errors)
            if isinstance(failure, OwnedCleanupError):
                record["fatal_cleanup"] = failure.details
        (directory / "gpu-safety.json").write_text(json.dumps(record, indent=2) + "\n")
        if errors:
            if isinstance(failure, OwnedCleanupError):
                raise failure
            raise RuntimeError("; ".join(errors))


def reject_ambient_injection() -> None:
    forbidden = [key for key in os.environ if key.startswith(("BPFTIME_", "OBS_", "NVBIT_", "GGML_"))
                 or key in ("LD_PRELOAD", "LD_AUDIT", "CUDA_INJECTION64_PATH", "CUDA_INJECTION32_PATH")]
    if forbidden or os.environ.get("CUDA_VISIBLE_DEVICES", "0") != "0":
        raise RuntimeError(f"use an uninjected GPU-0 launch environment; conflicting keys: {forbidden}")


def segment_identity(path: Path) -> tuple[int, int, int]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise RuntimeError("private segment has unexpected type/owner")
    return info.st_dev, info.st_ino, info.st_uid


@contextmanager
def private_probe(tool: str, args: argparse.Namespace, tool_dir: Path, run_dir: Path,
                  *, diagnostic_log_level: str | None = None,
                  exact_exit_oracle: bool = False):
    """Keep an owned loader alive until its direct CUDA client has returned."""
    name = f"rq4_{os.getpid()}_{time.monotonic_ns()}"
    segment = SHM_ROOT / name
    if segment.exists() or segment.is_symlink():
        raise RuntimeError("private segment already exists; refusing loader start")
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [str(tool_dir / tool)]
    if tool == "launchlate":
        command += [str(args.uprobe_binary), args.uprobe_symbol_hint]
    env = core.probe_env(args, tool)
    env["BPFTIME_GLOBAL_SHM_NAME"] = name
    command, loader_env = target_launch(command, env)
    target_env = {**core.agent_env(args, run_dir, tool), "BPFTIME_GLOBAL_SHM_NAME": name}
    explicit_verifier_env = verifier_environment(args)
    env.update(explicit_verifier_env)
    loader_env.update(explicit_verifier_env)
    target_env.update(explicit_verifier_env)
    if tool == "kernelretsnoop":
        layout = kernelretsnoop_layout(args.pp, correctness=exact_exit_oracle)
        exit_environment = {
            "BPFTIME_MAP_GPU_THREAD_COUNT": str(layout["thread_slots"]),
            "BPFTIME_KERNELRETSNOOP_RING_ENTRIES": str(layout["entries_per_thread"]),
            "BPFTIME_SHM_MEMORY_MB": str(KERNELRETSNOOP_SHM_MEMORY_MB),
            "BPFTIME_KERNELRETSNOOP_EXACT_ORACLE": "1" if exact_exit_oracle else "0",
        }
        env.update(exit_environment)
        loader_env.update(exit_environment)
        target_env.update(exit_environment)
    if diagnostic_log_level is not None:
        if diagnostic_log_level != "info":
            raise ValueError("the optional untimed diagnostic logging level is info")
        env["SPDLOG_LEVEL"] = loader_env["SPDLOG_LEVEL"] = "info"
        target_env["SPDLOG_LEVEL"] = "info"
    recorded_env = {key: value for key, value in env.items()
                    if key.startswith("BPFTIME_") or key in ("LD_PRELOAD", "SPDLOG_LEVEL")}
    record = {"private_segment": name, "command": command, "loader_environment": recorded_env,
              "agent_environment": target_env, "private_segment_removed": False}
    process = identity = None
    preserve_loader = False
    with (run_dir / "probe.log").open("x") as stream:
        try:
            process = subprocess.Popen(command, cwd=core.WORKLOAD_DIR, env=loader_env,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            record["loader_identity"] = process_identity(process)
            time.sleep(args.probe_startup_s)
            if process.poll() is not None:
                raise RuntimeError("private probe exited before the CUDA client")
            identity = segment_identity(segment)
            record["segment_identity"] = identity
            yield target_env
            if process.poll() is not None:
                raise RuntimeError("private probe exited before its CUDA client finished")
        except OwnedCleanupError as error:
            if error.details.get("role") == "CUDA client":
                preserve_loader = True
                record.update(client_cleanup_failure=error.details, loader_preserved=True,
                              preservation_reason="CUDA client cleanup is unconfirmed; its agent state must remain live")
            raise
        finally:
            try:
                if not preserve_loader:
                    stop_owned(process, "private loader", record.get("loader_identity"))
                    record["loader_returncode"] = process.returncode if process is not None else None
                    if segment.exists() or segment.is_symlink():
                        actual = segment_identity(segment)
                        if identity is not None and actual != identity:
                            raise OwnedCleanupError("private segment changed identity; refusing removal", record)
                        if process is None or shared.group_members(process.pid):
                            raise OwnedCleanupError("private loader is not stopped; refusing removal", record)
                        segment.unlink()
                        record["private_segment_removed"] = True
                    if process is not None and process.returncode != 0:
                        probe_text = (run_dir / "probe.log").read_text(errors="replace")
                        semantic = launchlate_unbounded_drift_exit(
                            tool, process.returncode, probe_text)
                        record["semantic_gate_failure"] = semantic
                        if not semantic:
                            raise RuntimeError(
                                f"private probe exited unsuccessfully: {process.returncode}")
            except OwnedCleanupError as error:
                record["cleanup_error"] = str(error)
                raise
            finally:
                (run_dir / "probe-execution.json").write_text(json.dumps(record, indent=2) + "\n")


def validate_launchlate_source_schema(directory: Path) -> None:
    """Reject a stale launchlate copy instead of rewriting it at run time."""
    required = {
        "Makefile": (
            "rm_ptimer_575.c",
            "rm_ptimer_575.o",
        ),
        "launchlate.bpf.c": (
            "BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP",
            "LAUNCHLATE_TARGET_SYMBOL",
            "MATCHED_SAMPLES",
            "UNCERTAIN_SAMPLES",
            "gpu_entry_ns",
            "host_raw_ns",
            "bpftime_ktime_get_raw_ns",
        ),
        "launchlate.c": (
            "RM endpoints-v1 PTIMER intervals with three-anchor held-out affine validation",
            "rm_ptimer_575_sample",
            "RM cleanup complete:",
            "Host enqueued:",
            "Matched samples:",
            "Queue update errors:",
            "Uncertain samples:",
            "Accounting complete:",
            "Online accounting complete:",
            "Pairing complete:",
            "Probes detached before final readback:",
            "Clock drift rate bound:",
            "Clock drift bounded:",
            "Clock slope diagnostic only:",
            "Held-out clock validation passed:",
            'print_calibration("Measurement-end"',
            'print_calibration("Validation-end"',
            "held_out_affine_clock_validation(",
            "classify_affine_sample(",
        ),
    }
    for name, markers in required.items():
        text = (directory / name).read_text()
        missing = [marker for marker in markers if marker not in text]
        if missing:
            raise RuntimeError(
                f"launchlate source lacks native accounting/calibration fields in {name}: "
                + ", ".join(missing)
            )


def validate_nvbit_kernelretsnoop_source_schema(directory: Path) -> None:
    """Require the compact global-coordinate ABI used by both tool arms."""
    required = {
        "common.h": (
            "uint64_t coordinate_x;",
            "uint64_t coordinate_y;",
            "uint64_t coordinate_z;",
            "static_assert(sizeof(exit_record_t) == 4 * sizeof(uint64_t)",
        ),
        "inject_funcs.cu": (
            "static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x",
            "static_cast<uint64_t>(blockIdx.y) * blockDim.y + threadIdx.y",
            "static_cast<uint64_t>(blockIdx.z) * blockDim.z + threadIdx.z",
            "sizeof(exit_record_t)",
        ),
        "observability.cu": (
            "state->exit_bad_size_bytes += bytes % sizeof(exit_record_t);",
            "state->exit_events.push_back(*record);",
            "validate_exit_events(state, launches)",
            "NVBIT kernelretsnoop record_bytes=%zu",
            "NVBIT kernelretsnoop cartesian_complete=%u",
            "NVBIT kernelretsnoop collector_gate_passed=%u",
        ),
    }
    for name, markers in required.items():
        text = (directory / name).read_text()
        missing = [marker for marker in markers if marker not in text]
        if missing:
            raise RuntimeError(
                f"NVBit kernelretsnoop source lacks the compact coordinate ABI in {name}: "
                + ", ".join(missing)
            )


def validate_nvbit_launchlate_source_schema(directory: Path) -> None:
    """Require the fail-closed raw-pair affine ABI in the copied NVBit tool."""
    required = {
        "Makefile": (
            "observability.o: observability.cu common.h clock_domain.h",
            "inject_funcs.o: inject_funcs.cu common.h clock_domain.h",
            "rm_ptimer_575.o",
        ),
        "clock_domain.h": (
            "int64_t offset_low_ns;",
            "int64_t offset_high_ns;",
            "uint64_t uncertainty_ns;",
            "uint64_t host_anchor_ns;",
            "uint64_t valid;",
            "LAUNCH_SAMPLE_UNCERTAIN",
            "clock_calibration_valid",
            "clock_calibration_drift",
            "CLOCK_MIN_CALIBRATION_SPAN_NS",
            "CLOCK_MAX_ANCHOR_BRACKET_NS",
            "minimum_end_calibration_deadline",
            "affine_clock_offset_interval",
            "held_out_affine_clock_validation",
            "classify_affine_launch_latency",
            "if (latency_high_ns < 0)",
        ),
        "common.h": (
            "LAUNCH_PAIR_CAPACITY",
            "struct launch_pair_t",
            "uint64_t host_raw_ns;",
            "uint64_t gpu_entry_ns;",
            "uint64_t sequence;",
        ),
        "inject_funcs.cu": (
            'asm volatile("mov.u64 %0, %%globaltimer;"',
            "pair_ptr",
            "device_entry_count_ptr",
            "capture_error_count_ptr",
            "pair->gpu_entry_ns = gpu_ns",
            "__threadfence_system()",
        ),
        "observability.cu": (
            "rm_endpoints_v1_PTIMER_against_CLOCK_MONOTONIC_RAW_",
            "with_three_anchor_held_out_affine_validation",
            "rm_ptimer_575_sample",
            "monotonic_raw_ns",
            "cudaMallocManaged(\n            &state->launch_pairs",
            "nvbit_set_at_launch(ctx, func, pair_ptr)",
            "state->launch_pair_overflows++",
            "print_launchlate_results(state, measurement_end_calibration)",
            "classify_affine_launch_latency(",
            "accounting_complete=",
            "calibrate_gpu_clock(state->start_calibration, state->start_rm)",
            "calibrate_gpu_clock(\n            &measurement_end_calibration, &measurement_end_rm)",
            "calibrate_gpu_clock(\n            &validation_end_calibration, &validation_end_rm)",
            "clock_calibration_drift(",
            "wait_for_minimum_clock_span(",
            'print_clock_calibration("start"',
            'print_clock_calibration("measurement_end"',
            'print_clock_calibration("validation_end"',
            "%s_clock_offset_lower_ns=",
            "%s_clock_offset_upper_ns=",
            "%s_clock_uncertainty_ns=",
            "%s_clock_host_anchor_ns=",
            "%s_clock_calibration_valid=",
            "clock_offset_change_lower_ns=",
            "clock_offset_change_upper_ns=",
            "clock_calibration_elapsed_ns=",
            "clock_drift_rate_bound_ppb=",
            "clock_drift_limit_ppb=",
            "clock_drift_bounded=",
            "clock_slope_diagnostic_only=1",
            "held_out_predicted_lower_ns=",
            "held_out_predicted_upper_ns=",
            "held_out_overlap_lower_ns=",
            "held_out_overlap_upper_ns=",
            "validation_span_ns=",
            "held_out_validation_passed=",
            "pair_capacity=",
            "stored_pairs=",
            "device_entries=",
            "pair_overflows=",
            "capture_errors=",
            "selected_counter_overflow=",
            "uncertain_samples=",
            "samples=%llu clock_errors=%llu",
        ),
        "rm_ptimer_575.c": (
            "CLOCK_MONOTONIC_RAW",
            "RM_ENDPOINTS_V1_COMMAND",
            "RM_PTIMER_QUANTIZATION_NS",
        ),
    }
    for name, markers in required.items():
        text = (directory / name).read_text()
        missing = [marker for marker in markers if marker not in text]
        if missing:
            raise RuntimeError(
                f"NVBit launchlate source lacks bounded calibration fields in {name}: "
                + ", ".join(missing)
            )


def validate_kernelretsnoop_source_schema(directory: Path) -> None:
    """Require the compact global-coordinate record ABI used by the oracle."""
    required = {
        "kernelretsnoop.bpf.c": (
            "u64 coordinate_x, coordinate_y, coordinate_z;",
            "data.coordinate_x = block_x * block_dim_x + thread_x;",
            "data.coordinate_y = block_y * block_dim_y + thread_y;",
            "data.coordinate_z = block_z * block_dim_z + thread_z;",
            "sizeof(struct data)",
        ),
        "kernelretsnoop.c": (
            "uint64_t coordinate_x, coordinate_y, coordinate_z;",
            "event_coordinate(&state->events[i]",
            "Invalid launch coordinates:",
            "sizeof(struct data)",
            "BPFTIME_KERNELRETSNOOP_RING_ENTRIES",
            "bpf_map__set_max_entries(skel->maps.rb, requested_entries)",
            "Requested ring entries per thread:",
        ),
    }
    for name, markers in required.items():
        text = (directory / name).read_text()
        missing = [marker for marker in markers if marker not in text]
        if missing:
            raise RuntimeError(
                f"kernelretsnoop source lacks the compact coordinate record ABI in {name}: "
                + ", ".join(missing)
            )


def prepare_tool_source(
    spec: core.ToolSpec,
    *,
    bpftime_root: Path,
    build_root: Path,
    target_symbol: str,
) -> Path:
    """Copy a tool while freezing runtime includes to its source tree."""
    directory = core.prepare_tool_source(
        spec,
        bpftime_root=bpftime_root,
        build_root=build_root,
        target_symbol=target_symbol,
    )
    if spec.name == "kernelretsnoop" and (directory / spec.user_file).exists():
        completed = subprocess.run(
            ["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i",
             str(KERNELRETSNOOP_CAPACITY_PATCH)],
            cwd=directory, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "failed to apply declared kernelretsnoop capacity patch:\n" + completed.stdout
            )
    makefile = directory / "Makefile"
    text = makefile.read_text()
    relative_includes = set(RELATIVE_RUNTIME_INCLUDE_PATTERN.findall(text))
    stale_includes = relative_includes - {RELATIVE_RUNTIME_INCLUDE}
    if stale_includes:
        raise RuntimeError(
            f"{spec.name} Makefile has stale runtime include marker(s): "
            + ", ".join(sorted(stale_includes))
        )
    rewritten = text.replace(
        RELATIVE_RUNTIME_INCLUDE,
        str((bpftime_root / "runtime/include").resolve()),
    )
    if RELATIVE_RUNTIME_INCLUDE_PATTERN.search(rewritten):
        raise RuntimeError(f"{spec.name} Makefile retains a relative runtime include")
    if rewritten != text:
        makefile.write_text(rewritten)
    return directory


def parse_gpubpf(tool: str, text: str) -> dict[str, Any]:
    result = core.parse_probe_samples(tool, text)
    if tool == "kernelretsnoop":
        labels = {
            "sample_count": "Total events collected",
            "nonzero_timestamps": "Nonzero timestamps",
            "requested_thread_slots": "Requested thread slots",
            "allocated_thread_slots": "Allocated thread slots",
            "entries_per_thread": "Ring entries per thread",
            "requested_entries_per_thread": "Requested ring entries per thread",
            "record_bytes": "Record bytes",
            "committed_events": "Committed events",
            "runtime_collected_events": "Runtime collected events",
            "oob_drops": "OOB drops",
            "full_drops": "Full drops",
            "bad_size_drops": "Bad-size drops",
            "other_drops": "Other drops",
            "dirty_slots": "Dirty slots",
            "pending_events": "Pending events",
            "final_drain_events": "Final drain events",
            "second_drain_events": "Second drain events",
            "cartesian_launches": "Cartesian launches",
            "cartesian_coordinates": "Cartesian coordinates",
            "cartesian_complete": "Cartesian complete",
            "extent_x": "Coordinate extent x",
            "extent_y": "Coordinate extent y",
            "extent_z": "Coordinate extent z",
            "multiplicity_220": "Coordinate multiplicity 220",
            "multiplicity_44": "Coordinate multiplicity 44",
            "multiplicity_22": "Coordinate multiplicity 22",
            "other_multiplicity": "Coordinate multiplicity other",
            "segment_mismatches": "Coordinate segment mismatches",
            "invalid_launch_coordinates": "Invalid launch coordinates",
            "unique_coordinates": "Unique coordinates",
            "oracle_enabled": "Multiplicity oracle enabled",
            "oracle_total_events": "Multiplicity oracle total events",
            "oracle_passed": "Multiplicity oracle passed",
            "collector_gate_passed": "Collector gate passed",
        }
        for key, label in labels.items():
            values = re.findall(rf"^{re.escape(label)}:\s*(\d+)$", text, re.MULTILINE)
            result[key] = int(values[-1]) if values else -1
    if tool == "threadhist":
        for key, label in (("configured_entries", "Configured thread entries"),
                           ("readback_entries", "Readback entries"),
                           ("readback_bytes", "Readback bytes"),
                           ("readback_complete", "Readback complete")):
            values = re.findall(rf"^{label}:\s*(\d+)$", text, re.MULTILINE)
            result[key] = int(values[-1]) if values else -1
    if tool == "launchlate":
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
        for key, label in labels.items():
            unit = (r"\s+ppb" if key.endswith("_ppb") else
                    r"\s+ns" if key.endswith("_ns") else "")
            number = r"(-?\d+)" if key in signed else r"(\d+)"
            values = re.findall(
                rf"^{re.escape(label)}:\s*{number}{unit}$", text, re.MULTILINE
            )
            result[key] = int(values[0]) if len(values) == 1 else None
        methods = re.findall(
            r"^Clock calibration method:\s*(.+)$", text, re.MULTILINE
        )
        result["clock_calibration_method"] = (
            methods[0].strip() if len(methods) == 1 else ""
        )
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
                values = re.findall(
                    rf"^{display} {re.escape(label)}:\s*(\d+){unit}$",
                    text,
                    re.MULTILINE,
                )
                result[f"{phase}_{suffix}"] = (
                    int(values[0]) if len(values) == 1 else None
                )
            statuses = re.findall(
                rf"^{display} RM status:\s*0x([0-9a-fA-F]+)$", text, re.MULTILINE
            )
            result[f"{phase}_rm_status"] = (
                int(statuses[0], 16) if len(statuses) == 1 else None
            )
    return result


def file_metadata(path: Path) -> dict[str, Any]:
    logical = path.absolute()
    if not logical.is_file():
        return {"path": str(logical), "exists": False}
    stat = logical.stat()
    return {
        "path": str(logical),
        "exists": True,
        "bytes": stat.st_size,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def source_manifest(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    paths = [
        Path(__file__).resolve(),
        HERE / "analyze_revision_rq4.py",
        OBS_ROOT / "run_observability_overhead.py",
        Path(shared.__file__),
        Path(shared.run_smoke.__file__),
        Path(shared.safety.__file__),
        KERNELRETSNOOP_CAPACITY_PATCH,
        LATE_BOOTSTRAP_TARGET_FILTER_PATCH,
        NVBIT_SOURCE_DIR / "Makefile",
        NVBIT_SOURCE_DIR / "clock_domain.h",
        NVBIT_SOURCE_DIR / "common.h",
        NVBIT_SOURCE_DIR / "inject_funcs.cu",
        NVBIT_SOURCE_DIR / "observability.cu",
        NVBIT_SOURCE_DIR / "rm_ptimer_575.c",
        NVBIT_SOURCE_DIR / "rm_ptimer_575.h",
        NVBIT_SOURCE_DIR / "tool_func/flush_channel.cu",
        CLOCK_CONTROL_SOURCE_DIR / "Makefile",
        CLOCK_CONTROL_SOURCE_DIR / "rm_ptimer_correlation_sanity.c",
        CLOCK_CONTROL_SOURCE_DIR / "rm_globaltimer_identity.cu",
        CLOCK_CONTROL_SOURCE_DIR / "launchlate-frozen-plan.md",
        CLOCK_CONTROL_SOURCE_DIR / "launchlate-frozen-plan-v2.md",
        CLOCK_CONTROL_SOURCE_DIR / "launchlate-attempt09-frozen-path.md",
        CLOCK_CONTROL_SOURCE_DIR / "launchlate-attempt09-pre-live-amendment.md",
        args.bpftime_root / "runtime/include/bpf_attach_ctx.hpp",
        args.bpftime_root / "runtime/include/bpftime_helper_group.hpp",
        args.bpftime_root / "runtime/include/bpftime_gpu_ringbuf.h",
        args.bpftime_root / "runtime/src/bpf_helper.cpp",
        args.bpftime_root / "runtime/src/bpf_map/gpu/nv_gpu_ringbuf_map.cpp",
        args.bpftime_root / "runtime/src/bpf_map/gpu/nv_gpu_ringbuf_map.hpp",
        args.bpftime_root / "runtime/syscall-server/syscall_server_main.cpp",
        args.bpftime_root / "runtime/syscall-server/syscall_server_utils.cpp",
        args.bpftime_root / "attach/nv_attach_impl/trampoline/default_trampoline.cu",
        args.bpftime_root / "attach/nv_attach_impl/nv_attach_impl.cpp",
        args.bpftime_root
        / "attach/nv_attach_impl/test/test_late_attach_source_invariants.py",
    ]
    for tool in selected_tools(args):
        spec = core.TOOLS[tool]
        paths.extend(
            [
                args.bpftime_root / spec.example_dir / "Makefile",
                args.bpftime_root / spec.example_dir / spec.bpf_file,
                args.bpftime_root / spec.example_dir / spec.user_file,
            ]
        )
        if tool == "launchlate":
            paths.extend((
                args.bpftime_root / spec.example_dir / "rm_ptimer_575.c",
                args.bpftime_root / spec.example_dir / "rm_ptimer_575.h",
            ))
    return {str(path): file_metadata(path) for path in paths}


def defining_params(args: argparse.Namespace) -> dict[str, Any]:
    correctness_layout = kernelretsnoop_layout(args.pp, correctness=True)
    timing_layout = kernelretsnoop_layout(args.pp, correctness=False)
    return {
        "phase": args.phase,
        "tools": list(selected_tools(args)),
        "preflight_campaign": (
            str(args.preflight_dir.resolve())
            if getattr(args, "preflight_dir", None) else None
        ),
        "model": str(args.model),
        "llama_bench": str(args.llama_bench),
        "llama_cli": str(args.llama_cli),
        "bpftime_root": str(args.bpftime_root),
        "bpftime_build_dir": str(args.bpftime_build_dir),
        "verifier_level": selected_verifier_level(args),
        "verifier_runtime_configuration": require_explicit_verifier_build(args),
        "nvbit_root": str(NVBIT_ROOT),
        "target_symbol": args.target_symbol,
        "runs": args.runs,
        "pp": args.pp,
        "tg": args.tg,
        "n_gpu_layers": args.n_gpu_layers,
        "timeout_s": args.timeout_s,
        "probe_startup_s": args.probe_startup_s,
        "gpu_thread_count": args.gpu_thread_count,
        "threadhist_gpu_thread_count": args.threadhist_gpu_thread_count,
        "kernelretsnoop_shm_memory_mb": KERNELRETSNOOP_SHM_MEMORY_MB,
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
        "uprobe_binary": str(args.uprobe_binary),
        "uprobe_symbol_hint": args.uprobe_symbol_hint,
        "uvm": args.uvm,
        "no_warmup": args.no_warmup,
        "cuda_graphs_disabled": core.CUDA_GRAPHS_DISABLED,
        "schedule_seed": SCHEDULE_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "expected_driver": EXPECTED_DRIVER,
        "worker_cpus": CLIENT_CPUS,
        "telemetry_cpu": shared.safety.TELEMETRY_CPU,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "launch_environment": {key: os.environ.get(key) for key in
                               ("PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
                                "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }


def parse_driver(snapshot: dict[str, Any]) -> str:
    gpu = str(snapshot.get("gpu", ""))
    fields = [field.strip() for field in gpu.split(",")]
    return fields[1] if len(fields) > 1 else "unknown"


def query_supported_clock_pairs() -> list[list[int]]:
    completed = subprocess.run(
        list(SUPPORTED_CLOCK_COMMAND), check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"supported-clock query failed ({completed.returncode}): {completed.stderr.strip()}"
        )
    pairs = set()
    for line in completed.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            memory, sm = (int(value) for value in fields)
        except ValueError:
            continue
        pairs.add((sm, memory))
    if not pairs:
        raise RuntimeError("supported-clock query returned no exact clock pairs")
    return [list(pair) for pair in sorted(pairs)]


def nvbit_driver_supported(driver: str) -> bool:
    match = re.match(r"(\d+)", driver)
    return bool(match and int(match.group(1)) <= 575)


def idle_gpu_or_error(snapshot: dict[str, Any]) -> None:
    applications = str(snapshot.get("compute_apps", "")).strip()
    if applications:
        raise RuntimeError(
            "GPU is not idle; refusing to terminate or overlap external CUDA "
            f"processes:\n{applications}"
        )


def build_nvbit(source_dir: Path, log_dir: Path) -> Path:
    core.run_cmd(
        [
            "make",
            "CXX=g++",
            f"NVBIT_ROOT={NVBIT_ROOT}",
            "ARCH=sm_120",
        ],
        cwd=source_dir,
        log_path=log_dir / "build_nvbit.log",
    )
    tool = source_dir / "observability.so"
    if not tool.exists():
        raise FileNotFoundError(tool)
    return tool


def read_strict_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL control log while rejecting duplicate keys and non-objects."""
    if not path.is_file():
        raise ValueError(f"missing raw JSONL: {path}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank JSONL record at line {number}")
        record = json.loads(line, object_pairs_hook=unique_object)
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
    return ordered[middle - 1] // 2 + ordered[middle] // 2 + (
        ordered[middle - 1] % 2 + ordered[middle] % 2
    ) // 2


def endpoint_control_valid(records: list[dict[str, Any]]) -> bool:
    if len(records) != LAUNCH_CONTROL_SAMPLES + 1:
        return False
    samples, summary = records[:-1], records[-1]
    widths: list[int] = []
    outer_widths: list[int] = []
    previous_cpu = previous_gpu = 0
    for index, sample in enumerate(samples):
        integer_fields = (
            "index", "rm_status", "host_before_ns", "host_after_ns",
            "rm_cpu_before_ns", "rm_cpu_midpoint_ns", "rm_cpu_after_ns",
            "rm_gpu_ptimer_ns", "outer_width_ns", "max_selected_gap_ns",
            "cpu_lower_ns", "cpu_upper_ns", "offset_low_ns",
            "offset_high_ns", "bracket_width_ns",
        )
        if any(type(sample.get(key)) is not int for key in integer_fields):
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
        "record": "summary",
        "setup_stage": "samples",
        "control_transport": "direct",
        "correlation_command": "endpoints-v1",
        "setup_error": 0,
        "cleanup_error": 0,
        "cleanup_rm_status": 0,
        "output_error": 0,
        "requested": LAUNCH_CONTROL_SAMPLES,
        "attempted": LAUNCH_CONTROL_SAMPLES,
        "accepted": LAUNCH_CONTROL_SAMPLES,
        "rejected": 0,
        "cpu_midpoint_regressions": 0,
        "ptimer_regressions": 0,
        "min_outer_width_ns": min(outer_widths),
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
        integer_fields = (
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
        if any(type(sample.get(key)) is not int for key in integer_fields):
            return False
        bo = sample["rm_before_outer_before_raw_ns"]
        bc = sample["rm_before_cpu_before_raw_ns"]
        bg = sample["rm_before_gpu_ptimer_ns"]
        ba = sample["rm_before_cpu_after_raw_ns"]
        bx = sample["rm_before_outer_after_raw_ns"]
        ko = sample["kernel_before_raw_ns"]
        kg = sample["device_globaltimer_ns"]
        kx = sample["kernel_after_raw_ns"]
        ao = sample["rm_after_outer_before_raw_ns"]
        ac = sample["rm_after_cpu_before_raw_ns"]
        ag = sample["rm_after_gpu_ptimer_ns"]
        aa = sample["rm_after_cpu_after_raw_ns"]
        ax = sample["rm_after_outer_after_raw_ns"]
        if not (
            sample.get("type") == "identity_sample"
            and sample["trial"] == index
            and sample.get("contained") is True
            and sample.get("accepted") is True
            and 0 < bo <= bc <= ba <= bx <= ko <= kx <= ao <= ac <= aa <= ax
            and bx - bo < 10_000_000
            and ax - ao < 10_000_000
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
    summary = records[-1]
    return summary == {
        "type": "identity_summary",
        "requested": LAUNCH_CONTROL_SAMPLES,
        "attempted": LAUNCH_CONTROL_SAMPLES,
        "accepted": LAUNCH_CONTROL_SAMPLES,
        "rejected": 0,
        "containment_failures": 0,
        "raw_regressions": 0,
        "ptimer_regressions": 0,
        "cuda_errors": 0,
        "setup_complete": True,
        "cleanup_complete": True,
        "gate_passed": True,
    }


def prepare_clock_controls(output_dir: Path) -> Path:
    root = output_dir / "clock_control_build"
    control_dir = root / "launch-clock-recovery"
    rm_dir = root / "nvbit_adapters" / "observability"
    control_dir.mkdir(parents=True, exist_ok=False)
    rm_dir.mkdir(parents=True, exist_ok=False)
    for name in ("Makefile", "rm_ptimer_correlation_sanity.c",
                 "rm_globaltimer_identity.cu"):
        shutil.copy2(CLOCK_CONTROL_SOURCE_DIR / name, control_dir / name)
    for name in ("rm_ptimer_575.c", "rm_ptimer_575.h"):
        shutil.copy2(NVBIT_SOURCE_DIR / name, rm_dir / name)
    core.run_cmd(["make", "-j4"], cwd=control_dir,
                 log_path=output_dir / "build_clock_controls.log")
    return control_dir


def run_clock_control(command: list[str], run_dir: Path,
                      validator) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    with cell_safety(run_dir) as safety:
        completed = run_cli_separate(
            command, cwd=Path(command[0]).parent, env=os.environ.copy(),
            timeout=300, log_path=run_dir / "process.log",
        )
    stdout_path = run_dir / "stdout.jsonl"
    stderr_path = run_dir / "stderr.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    valid = False
    error = None
    try:
        valid = completed.returncode == 0 and validator(read_strict_jsonl(stdout_path))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "safety": str(run_dir / "gpu-safety.json"),
        "valid": valid,
        "error": error,
    }


def run_launch_clock_controls(output_dir: Path, admission: dict[str, Any]) -> dict[str, Any]:
    control_dir = prepare_clock_controls(output_dir)
    endpoint = run_clock_control(
        [str(control_dir / "rm_ptimer_correlation_sanity"), "--samples",
         str(LAUNCH_CONTROL_SAMPLES), "--control-transport", "direct",
         "--correlation-command", "endpoints-v1"],
        output_dir / "clock_controls" / "endpoint_precision",
        endpoint_control_valid,
    )
    if not endpoint["valid"]:
        result = {"role": "calibration_only", "boot_id": admission["boot_id"],
                  "driver": admission["driver"], "endpoint_precision": endpoint,
                  "globaltimer_identity": None, "passed": False}
    else:
        identity = run_clock_control(
            [str(control_dir / "rm_globaltimer_identity"), "--samples",
             str(LAUNCH_CONTROL_SAMPLES)],
            output_dir / "clock_controls" / "globaltimer_identity",
            identity_control_valid,
        )
        result = {"role": "calibration_only", "boot_id": admission["boot_id"],
                  "driver": admission["driver"], "endpoint_precision": endpoint,
                  "globaltimer_identity": identity,
                  "passed": bool(endpoint["valid"] and identity["valid"])}
    (output_dir / "clock-controls.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if not result["passed"]:
        raise RuntimeError("launch-clock calibration controls failed; timing is forbidden")
    return result


def replay_launch_clock_controls(output_dir: Path,
                                 admission: dict[str, Any]) -> dict[str, Any]:
    record_path = output_dir / "clock-controls.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (record.get("role") != "calibration_only" or record.get("passed") is not True or
            record.get("boot_id") != admission["boot_id"] or
            record.get("driver") != admission["driver"]):
        raise RuntimeError("recorded launch-clock controls do not match this stack")
    for name, validator in (
        ("endpoint_precision", endpoint_control_valid),
        ("globaltimer_identity", identity_control_valid),
    ):
        item = record.get(name)
        if (
            not isinstance(item, dict)
            or item.get("valid") is not True
            or item.get("returncode") != 0
        ):
            raise RuntimeError(f"recorded {name} control is not complete")
        raw = Path(item.get("stdout", "")).resolve()
        safety_path = Path(item.get("safety", "")).resolve()
        if not raw.is_relative_to(output_dir) or not safety_path.is_relative_to(output_dir):
            raise RuntimeError(f"recorded {name} evidence escapes the campaign directory")
        safety = json.loads(safety_path.read_text(encoding="utf-8"))
        if safety.get("passed") is not True or safety.get("boot_id") != admission["boot_id"]:
            raise RuntimeError(f"recorded {name} safety gate is invalid")
        if not validator(read_strict_jsonl(raw)):
            raise RuntimeError(f"raw replay rejected {name}")
    return record


def parse_nvbit(tool: str, text: str) -> dict[str, Any]:
    def last(pattern: str, default: int = 0, flags: int = 0) -> int:
        values = [int(value) for value in re.findall(pattern, text, flags)]
        return values[-1] if values else default

    selected = last(
        r"^NVBIT selected_launches=(\d+)$", flags=re.MULTILINE
    )
    if tool == "kernelretsnoop":
        events = last(r"NVBIT kernelretsnoop events=(\d+)")
        nonzero = last(r"NVBIT kernelretsnoop events=\d+ nonzero_timestamps=(\d+)")
        def field(name: str, default: int = -1) -> int:
            return last(rf"^NVBIT kernelretsnoop {name}=(\d+)(?:\s|$)",
                        default, re.MULTILINE)
        return {
            "sample_count": events,
            "nonzero_timestamps": nonzero,
            "selected_launches": selected,
            "record_bytes": field("record_bytes"),
            "bad_size_bytes": field("bad_size_bytes"),
            "cartesian_launches": field("cartesian_launches"),
            "cartesian_coordinates": field("cartesian_coordinates"),
            "cartesian_complete": field("cartesian_complete"),
            "extent_x": field("extent_x"),
            "extent_y": last(r"^NVBIT kernelretsnoop extent_x=\d+ extent_y=(\d+)",
                             -1, re.MULTILINE),
            "extent_z": last(r"^NVBIT kernelretsnoop extent_x=\d+ extent_y=\d+ extent_z=(\d+)$",
                             -1, re.MULTILINE),
            "multiplicity_220": field("multiplicity_220"),
            "multiplicity_44": last(r"^NVBIT kernelretsnoop multiplicity_220=\d+ multiplicity_44=(\d+)",
                                    -1, re.MULTILINE),
            "multiplicity_22": last(r"^NVBIT kernelretsnoop multiplicity_220=\d+ multiplicity_44=\d+ multiplicity_22=(\d+)",
                                    -1, re.MULTILINE),
            "other_multiplicity": last(r"^NVBIT kernelretsnoop multiplicity_220=\d+ multiplicity_44=\d+ multiplicity_22=\d+ multiplicity_other=(\d+)$",
                                       -1, re.MULTILINE),
            "segment_mismatches": field("segment_mismatches"),
            "invalid_launch_coordinates": last(
                r"^NVBIT kernelretsnoop segment_mismatches=\d+ invalid_coordinates=(\d+)",
                -1, re.MULTILINE),
            "unique_coordinates": last(
                r"^NVBIT kernelretsnoop segment_mismatches=\d+ invalid_coordinates=\d+ unique_coordinates=(\d+)$",
                -1, re.MULTILINE),
            "collector_gate_passed": field("collector_gate_passed"),
            "validation_blocks": len(re.findall(
                r"^NVBIT kernelretsnoop record_bytes=\d+$", text, re.MULTILINE)),
            "process_selected_launches": last(
                r"^NVBIT_OBS process_selected_launches=(\d+)$", -1, re.MULTILINE),
        }
    if tool == "threadhist":
        nonzero = last(r"NVBIT threadhist nonzero_threads=(\d+)")
        total = last(r"NVBIT threadhist nonzero_threads=\d+ total_exit_probes=(\d+)")
        return {
            "sample_count": total,
            "nonzero_threads": nonzero,
            "selected_launches": selected,
        }
    # NVBit increments samples only for classified, single-bin intervals.
    samples = last(
        r"^NVBIT launchlate samples=(\d+) clock_errors=\d+$",
        flags=re.MULTILINE,
    )
    errors = last(
        r"^NVBIT launchlate samples=\d+ clock_errors=(\d+)$",
        -1,
        re.MULTILINE,
    )
    bins = [
        last(
            rf"^NVBIT launchlate bin_{index}=(\d+)$",
            -1,
            re.MULTILINE,
        )
        for index in range(10)
    ]
    result = {
        "sample_count": samples,
        "clock_errors": errors,
        "histogram": bins,
        "histogram_sum": sum(bins),
        "selected_launches": selected,
        "process_selected_launches": last(
            r"^NVBIT_OBS process_selected_launches=(\d+)$",
            -1,
            re.MULTILINE,
        ),
        "result_blocks": len(re.findall(
            r"^NVBIT launchlate samples=\d+ clock_errors=\d+$",
            text,
            re.MULTILINE,
        )),
    }
    integer_fields = {
        "uncertain_samples": r"\d+",
        "pair_capacity": r"\d+",
        "stored_pairs": r"\d+",
        "device_entries": r"\d+",
        "pair_overflows": r"\d+",
        "capture_errors": r"\d+",
        "selected_counter_overflow": r"\d+",
        "accounting_complete": r"\d+",
        "start_clock_offset_lower_ns": r"-?\d+",
        "start_clock_offset_upper_ns": r"-?\d+",
        "start_clock_uncertainty_ns": r"\d+",
        "start_clock_host_anchor_ns": r"\d+",
        "start_clock_calibration_valid": r"\d+",
        "measurement_end_clock_offset_lower_ns": r"-?\d+",
        "measurement_end_clock_offset_upper_ns": r"-?\d+",
        "measurement_end_clock_uncertainty_ns": r"\d+",
        "measurement_end_clock_host_anchor_ns": r"\d+",
        "measurement_end_clock_calibration_valid": r"\d+",
        "validation_end_clock_offset_lower_ns": r"-?\d+",
        "validation_end_clock_offset_upper_ns": r"-?\d+",
        "validation_end_clock_uncertainty_ns": r"\d+",
        "validation_end_clock_host_anchor_ns": r"\d+",
        "validation_end_clock_calibration_valid": r"\d+",
        "clock_offset_change_lower_ns": r"-?\d+",
        "clock_offset_change_upper_ns": r"-?\d+",
        "clock_calibration_elapsed_ns": r"\d+",
        "clock_drift_rate_bound_ppb": r"\d+",
        "clock_drift_limit_ppb": r"\d+",
        "clock_drift_bounded": r"\d+",
        "clock_slope_diagnostic_only": r"\d+",
        "held_out_predicted_lower_ns": r"-?\d+",
        "held_out_predicted_upper_ns": r"-?\d+",
        "held_out_overlap_lower_ns": r"-?\d+",
        "held_out_overlap_upper_ns": r"-?\d+",
        "validation_span_ns": r"\d+",
        "held_out_validation_passed": r"\d+",
        "start_rm_samples_requested": r"\d+",
        "start_rm_samples_accepted": r"\d+",
        "start_rm_samples_rejected": r"\d+",
        "start_rm_outer_before_raw_ns": r"\d+",
        "start_rm_cpu_before_raw_ns": r"\d+",
        "start_rm_gpu_ptimer_ns": r"\d+",
        "start_rm_cpu_after_raw_ns": r"\d+",
        "start_rm_outer_after_raw_ns": r"\d+",
        "start_rm_outer_width_ns": r"\d+",
        "start_rm_selected_gap_ns": r"\d+",
        "start_rm_bracket_width_ns": r"\d+",
        "start_rm_status": r"\d+",
        "start_rm_cleanup_complete": r"\d+",
        **{
            f"{phase}_{suffix}": r"\d+"
            for phase in ("measurement_end", "validation_end")
            for suffix in (
                "rm_samples_requested", "rm_samples_accepted",
                "rm_samples_rejected", "rm_outer_before_raw_ns",
                "rm_cpu_before_raw_ns", "rm_gpu_ptimer_ns",
                "rm_cpu_after_raw_ns", "rm_outer_after_raw_ns",
                "rm_outer_width_ns", "rm_selected_gap_ns",
                "rm_bracket_width_ns", "rm_status",
                "rm_cleanup_complete",
            )
        },
    }
    for key, number in integer_fields.items():
        values = re.findall(
            rf"^NVBIT launchlate {key}=({number})$",
            text,
            re.MULTILINE,
        )
        result[key] = int(values[0]) if len(values) == 1 else None
    methods = re.findall(
        r"^NVBIT launchlate clock_calibration_method=(\S+)$",
        text,
        re.MULTILINE,
    )
    result["clock_calibration_method"] = methods[0] if len(methods) == 1 else ""
    result["calibration_blocks"] = len(methods)
    return result


def launch_clock_model_rate(probe: dict[str, Any]) -> int | None:
    """Recompute the start-to-validation slope diagnostic."""
    fields = (
        "start_clock_offset_lower_ns", "start_clock_offset_upper_ns",
        "start_clock_uncertainty_ns", "start_clock_host_anchor_ns",
        "validation_end_clock_offset_lower_ns",
        "validation_end_clock_offset_upper_ns",
        "validation_end_clock_uncertainty_ns",
        "validation_end_clock_host_anchor_ns",
        "clock_offset_change_lower_ns", "clock_offset_change_upper_ns",
        "clock_calibration_elapsed_ns", "clock_drift_rate_bound_ppb",
        "clock_drift_limit_ppb", "clock_drift_bounded",
        "clock_slope_diagnostic_only",
    )
    if any(type(probe.get(key)) is not int for key in fields):
        return None
    start_low = probe["start_clock_offset_lower_ns"]
    start_high = probe["start_clock_offset_upper_ns"]
    end_low = probe["validation_end_clock_offset_lower_ns"]
    end_high = probe["validation_end_clock_offset_upper_ns"]
    start_anchor = probe["start_clock_host_anchor_ns"]
    end_anchor = probe["validation_end_clock_host_anchor_ns"]
    elapsed = end_anchor - start_anchor
    change_low = end_low - start_high
    change_high = end_high - start_low
    if elapsed <= 0:
        return None
    expected_rate = (
        max(abs(change_low), abs(change_high)) * 1_000_000_000 + elapsed - 1
    ) // elapsed
    consistent = (
        start_low <= start_high
        and end_low <= end_high
        and probe["start_clock_uncertainty_ns"]
        == (start_high - start_low + 1) // 2
        and probe["validation_end_clock_uncertainty_ns"]
        == (end_high - end_low + 1) // 2
        and probe["clock_offset_change_lower_ns"] == change_low
        and probe["clock_offset_change_upper_ns"] == change_high
        and probe["clock_calibration_elapsed_ns"] == elapsed
        and elapsed >= LAUNCH_MIN_CALIBRATION_SPAN_NS
        and probe["clock_drift_rate_bound_ppb"] == expected_rate
        and probe["clock_drift_limit_ppb"] == LAUNCH_CLOCK_DRIFT_LIMIT_PPB
        and probe["clock_drift_bounded"]
        == int(expected_rate <= LAUNCH_CLOCK_DRIFT_LIMIT_PPB)
        and probe["clock_slope_diagnostic_only"] == 1
    )
    return expected_rate if consistent else None


def launch_clock_model_valid(probe: dict[str, Any]) -> bool:
    """Validate three interval anchors; slope magnitude is diagnostic only."""
    if launch_clock_model_rate(probe) is None:
        return False
    fields = (
        "start_clock_offset_lower_ns", "start_clock_offset_upper_ns",
        "start_clock_host_anchor_ns",
        "measurement_end_clock_offset_lower_ns",
        "measurement_end_clock_offset_upper_ns",
        "measurement_end_clock_uncertainty_ns",
        "measurement_end_clock_host_anchor_ns",
        "validation_end_clock_offset_lower_ns",
        "validation_end_clock_offset_upper_ns",
        "validation_end_clock_host_anchor_ns",
        "held_out_predicted_lower_ns", "held_out_predicted_upper_ns",
        "held_out_overlap_lower_ns", "held_out_overlap_upper_ns",
        "validation_span_ns", "held_out_validation_passed",
    )
    if any(type(probe.get(key)) is not int for key in fields):
        return False
    start_anchor = probe["start_clock_host_anchor_ns"]
    middle_anchor = probe["measurement_end_clock_host_anchor_ns"]
    validation_anchor = probe["validation_end_clock_host_anchor_ns"]
    elapsed = validation_anchor - start_anchor
    position = middle_anchor - start_anchor
    if not (elapsed > 0 and 0 < position < elapsed):
        return False
    validation_span = validation_anchor - middle_anchor
    if validation_span < LAUNCH_MIN_CALIBRATION_SPAN_NS:
        return False
    middle_low = probe["measurement_end_clock_offset_lower_ns"]
    middle_high = probe["measurement_end_clock_offset_upper_ns"]
    if middle_low > middle_high:
        return False
    if probe["measurement_end_clock_uncertainty_ns"] != (
        middle_high - middle_low + 1
    ) // 2:
        return False
    start_low = probe["start_clock_offset_lower_ns"]
    start_high = probe["start_clock_offset_upper_ns"]
    validation_low = probe["validation_end_clock_offset_lower_ns"]
    validation_high = probe["validation_end_clock_offset_upper_ns"]
    predicted_low = start_low + (
        (validation_low - start_low) * position // elapsed
    )
    numerator = (validation_high - start_high) * position
    predicted_high = start_high + (-(-numerator // elapsed))
    overlap_low = max(predicted_low, middle_low)
    overlap_high = min(predicted_high, middle_high)
    return (
        overlap_low <= overlap_high
        and probe["held_out_predicted_lower_ns"] == predicted_low
        and probe["held_out_predicted_upper_ns"] == predicted_high
        and probe["held_out_overlap_lower_ns"] == overlap_low
        and probe["held_out_overlap_upper_ns"] == overlap_high
        and probe["validation_span_ns"] == validation_span
        and probe["held_out_validation_passed"] == 1
    )


def parse_legacy_gpubpf_launch_clock(text: str) -> dict[str, int]:
    """Parse only the frozen two-anchor fields used by retained attempt07."""
    labels = {
        "start_clock_offset_lower_ns": "Start clock offset lower",
        "start_clock_offset_upper_ns": "Start clock offset upper",
        "start_clock_uncertainty_ns": "Start clock uncertainty",
        "start_clock_host_anchor_ns": "Start clock host anchor",
        "end_clock_offset_lower_ns": "End clock offset lower",
        "end_clock_offset_upper_ns": "End clock offset upper",
        "end_clock_uncertainty_ns": "End clock uncertainty",
        "end_clock_host_anchor_ns": "End clock host anchor",
        "clock_offset_change_lower_ns": "Clock offset change lower",
        "clock_offset_change_upper_ns": "Clock offset change upper",
        "clock_calibration_elapsed_ns": "Clock calibration elapsed",
        "clock_drift_rate_bound_ppb": "Clock drift rate bound",
        "clock_drift_limit_ppb": "Clock drift limit",
        "clock_drift_bounded": "Clock drift bounded",
    }
    signed = {key for key in labels if "offset" in key}
    result: dict[str, int] = {}
    for key, label in labels.items():
        unit = (r"\s+ppb" if key.endswith("_ppb") else
                r"\s+ns" if key.endswith("_ns") else "")
        number = r"(-?\d+)" if key in signed else r"(\d+)"
        values = re.findall(
            rf"^{re.escape(label)}:\s*{number}{unit}$", text, re.MULTILINE
        )
        if len(values) != 1:
            return {}
        result[key] = int(values[0])
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
    for phase in ("start", "end"):
        display = phase.title()
        for suffix, label in rm_labels.items():
            unit = r"\s+ns" if suffix.endswith("_ns") else ""
            values = re.findall(
                rf"^{display} {re.escape(label)}:\s*(\d+){unit}$",
                text,
                re.MULTILINE,
            )
            if len(values) != 1:
                return {}
            result[f"{phase}_{suffix}"] = int(values[0])
        statuses = re.findall(
            rf"^{display} RM status:\s*0x([0-9a-fA-F]+)$",
            text,
            re.MULTILINE,
        )
        if len(statuses) != 1:
            return {}
        result[f"{phase}_rm_status"] = int(statuses[0], 16)
    return result


def legacy_launch_clock_model_rate(probe: dict[str, Any]) -> int | None:
    """Recompute the superseded attempt07 two-anchor drift gate."""
    try:
        start_low = probe["start_clock_offset_lower_ns"]
        start_high = probe["start_clock_offset_upper_ns"]
        end_low = probe["end_clock_offset_lower_ns"]
        end_high = probe["end_clock_offset_upper_ns"]
        elapsed = probe["end_clock_host_anchor_ns"] - probe["start_clock_host_anchor_ns"]
        if elapsed < LAUNCH_MIN_CALIBRATION_SPAN_NS:
            return None
        change_low = end_low - start_high
        change_high = end_high - start_low
        rate = (max(abs(change_low), abs(change_high)) * 1_000_000_000 + elapsed - 1) // elapsed
        if not (
            start_low <= start_high and end_low <= end_high
            and probe["start_clock_uncertainty_ns"] == (start_high - start_low + 1) // 2
            and probe["end_clock_uncertainty_ns"] == (end_high - end_low + 1) // 2
            and probe["clock_offset_change_lower_ns"] == change_low
            and probe["clock_offset_change_upper_ns"] == change_high
            and probe["clock_calibration_elapsed_ns"] == elapsed
            and probe["clock_drift_rate_bound_ppb"] == rate
            and probe["clock_drift_limit_ppb"] == LAUNCH_CLOCK_DRIFT_LIMIT_PPB
        ):
            return None
        return rate
    except (KeyError, TypeError):
        return None


def launchlate_unbounded_drift_exit(tool: str, returncode: int, text: str) -> bool:
    """Recognize a fully emitted, cleanup-complete drift-gate failure only."""
    if tool != "launchlate" or returncode != errno.ERANGE:
        return False
    if text.count("Clock calibration drift exceeds its bound") != 1:
        return False
    probe = parse_gpubpf(tool, text)
    probe.update(parse_legacy_gpubpf_launch_clock(text))
    expected_rate = legacy_launch_clock_model_rate(probe)
    count_names = ("sample_count", "matched_samples", "classified_samples",
                   "uncertain_samples")
    if any(type(probe.get(name)) is not int for name in count_names):
        return False
    samples = probe["sample_count"]
    matched = probe["matched_samples"]
    classified = probe["classified_samples"]
    uncertain = probe["uncertain_samples"]
    return (
        expected_rate is not None
        and expected_rate > LAUNCH_CLOCK_DRIFT_LIMIT_PPB
        and probe.get("clock_drift_bounded") == 0
        and launch_legacy_rm_anchors_valid(probe)
        and probe.get("probes_detached_before_readback") == 1
        and probe.get("start_rm_cleanup_complete") == 1
        and probe.get("end_rm_cleanup_complete") == 1
        and probe.get("clock_errors") == 0
        and probe.get("queue_underflows") == 0
        and probe.get("queue_overflows") == 0
        and probe.get("queue_update_errors") == 0
        and probe.get("online_accounting_complete") == 1
        and probe.get("accounting_complete") == 1
        and probe.get("pairing_complete") == 1
        and samples > 0
        and probe.get("host_launches") == probe.get("host_enqueued")
        == probe.get("device_entries") == matched == samples
        and probe.get("histogram_samples") == classified
        and launch_uncertainty_valid(classified, uncertain, matched)
    )


def launch_rm_anchors_valid(probe: dict[str, Any]) -> bool:
    """Recompute all three endpoint-v1 RAW/PTIMER intervals."""
    for phase in ("start", "measurement_end", "validation_end"):
        names = (
            "rm_samples_requested", "rm_samples_accepted",
            "rm_samples_rejected", "rm_outer_before_raw_ns",
            "rm_cpu_before_raw_ns", "rm_gpu_ptimer_ns",
            "rm_cpu_after_raw_ns", "rm_outer_after_raw_ns",
            "rm_outer_width_ns", "rm_selected_gap_ns",
            "rm_bracket_width_ns", "rm_status", "rm_cleanup_complete",
        )
        values = {name: probe.get(f"{phase}_{name}") for name in names}
        if any(type(value) is not int for value in values.values()):
            return False
        outer_before = values["rm_outer_before_raw_ns"]
        cpu_before = values["rm_cpu_before_raw_ns"]
        gpu = values["rm_gpu_ptimer_ns"]
        cpu_after = values["rm_cpu_after_raw_ns"]
        outer_after = values["rm_outer_after_raw_ns"]
        selected_gap = cpu_after - cpu_before
        bracket = selected_gap + 2 * 32
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
            and probe.get(f"{phase}_clock_offset_lower_ns")
                == gpu - cpu_after - 32
            and probe.get(f"{phase}_clock_offset_upper_ns")
                == gpu - cpu_before + 32
            and probe.get(f"{phase}_clock_uncertainty_ns")
                == (bracket + 1) // 2
            and probe.get(f"{phase}_clock_host_anchor_ns")
                == cpu_before + selected_gap // 2
        ):
            return False
    return True


def launch_legacy_rm_anchors_valid(probe: dict[str, Any]) -> bool:
    """Recompute only the superseded start/end intervals from attempt07."""
    for phase in ("start", "end"):
        names = (
            "rm_samples_requested", "rm_samples_accepted",
            "rm_samples_rejected", "rm_outer_before_raw_ns",
            "rm_cpu_before_raw_ns", "rm_gpu_ptimer_ns",
            "rm_cpu_after_raw_ns", "rm_outer_after_raw_ns",
            "rm_outer_width_ns", "rm_selected_gap_ns",
            "rm_bracket_width_ns", "rm_status", "rm_cleanup_complete",
        )
        values = {name: probe.get(f"{phase}_{name}") for name in names}
        if any(type(value) is not int for value in values.values()):
            return False
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
            and probe.get(f"{phase}_clock_offset_lower_ns") == gpu - cpu_after - 32
            and probe.get(f"{phase}_clock_offset_upper_ns") == gpu - cpu_before + 32
            and probe.get(f"{phase}_clock_uncertainty_ns") == (bracket + 1) // 2
            and probe.get(f"{phase}_clock_host_anchor_ns") == cpu_before + selected_gap // 2
        ):
            return False
    return True


def launch_uncertainty_valid(classified: int, uncertain: int, total: int) -> bool:
    return (
        classified >= 0
        and uncertain >= 0
        and total > 0
        and classified + uncertain == total
        and uncertain * 100 <= total * LAUNCH_UNCERTAIN_PERCENT_LIMIT
    )


def nvbit_probe_valid(tool: str, probe: dict[str, Any], *,
                      expected_exit_events: int | None = None,
                      expected_exit_launches: int | None = None,
                      expected_exit_coordinates: int | None = None,
                      exact_exit_oracle: bool = False) -> bool:
    if type(probe.get("sample_count")) is not int or type(
        probe.get("selected_launches")
    ) is not int:
        return False
    samples = probe["sample_count"]
    selected = probe["selected_launches"]
    if samples <= 0 or selected <= 0:
        return False
    if tool == "kernelretsnoop":
        if expected_exit_coordinates is None or expected_exit_coordinates % 256:
            return False
        expected_multiplicities = (
            (CORRECTNESS_MULTIPLICITY_220, CORRECTNESS_MULTIPLICITY_44,
             CORRECTNESS_MULTIPLICITY_22, 0)
            if exact_exit_oracle else (0, expected_exit_coordinates, 0, 0)
        )
        return (
            int(probe.get("nonzero_timestamps", 0)) == samples
            and int(probe.get("record_bytes", -1)) == EXIT_RECORD_BYTES
            and int(probe.get("bad_size_bytes", -1)) == 0
            and int(probe.get("cartesian_launches", -1)) == selected
            and int(probe.get("cartesian_coordinates", -1)) == expected_exit_coordinates
            and int(probe.get("cartesian_complete", -1)) == 1
            and int(probe.get("extent_x", -1)) == expected_exit_coordinates // 256
            and int(probe.get("extent_y", -1)) == 256
            and int(probe.get("extent_z", -1)) == 1
            and tuple(int(probe.get(key, -1)) for key in (
                "multiplicity_220", "multiplicity_44", "multiplicity_22",
                "other_multiplicity")) == expected_multiplicities
            and int(probe.get("segment_mismatches", -1)) == 0
            and int(probe.get("invalid_launch_coordinates", -1)) == 0
            and int(probe.get("unique_coordinates", -1)) == expected_exit_coordinates
            and int(probe.get("collector_gate_passed", -1)) == 1
            and int(probe.get("validation_blocks", -1)) == 1
            and int(probe.get("process_selected_launches", -1)) == selected
            and (expected_exit_events is None or samples == expected_exit_events)
            and (expected_exit_launches is None or selected == expected_exit_launches)
        )
    if tool == "threadhist":
        return int(probe.get("nonzero_threads", 0)) > 0

    calibration_fields = (
        "uncertain_samples", "start_clock_calibration_valid",
        "measurement_end_clock_calibration_valid",
        "validation_end_clock_calibration_valid",
        "pair_capacity", "stored_pairs",
        "device_entries", "pair_overflows", "capture_errors",
        "selected_counter_overflow", "accounting_complete",
        "process_selected_launches", "result_blocks", "calibration_blocks",
    )
    if any(type(probe.get(key)) is not int for key in calibration_fields):
        return False
    histogram = probe.get("histogram", ())
    uncertain = int(probe.get("uncertain_samples", -1))
    return (
        probe.get("clock_calibration_method")
        == NVBIT_LAUNCH_CLOCK_METHOD
        and int(probe.get("start_clock_calibration_valid", -1)) == 1
        and int(probe.get("measurement_end_clock_calibration_valid", -1)) == 1
        and int(probe.get("validation_end_clock_calibration_valid", -1)) == 1
        and launch_rm_anchors_valid(probe)
        and launch_clock_model_valid(probe)
        and int(probe.get("clock_errors", -1)) == 0
        and isinstance(histogram, list)
        and len(histogram) == 10
        and all(isinstance(count, int) and count >= 0 for count in histogram)
        and sum(histogram) == samples
        and int(probe.get("histogram_sum", -1)) == samples
        and int(probe.get("pair_capacity", -1)) >= selected
        and int(probe.get("stored_pairs", -1)) == selected
        and int(probe.get("device_entries", -1)) == selected
        and int(probe.get("pair_overflows", -1)) == 0
        and int(probe.get("capture_errors", -1)) == 0
        and int(probe.get("selected_counter_overflow", -1)) == 0
        and int(probe.get("accounting_complete", -1)) == 1
        and int(probe.get("process_selected_launches", -1)) == selected
        and int(probe.get("result_blocks", -1)) == 1
        and int(probe.get("calibration_blocks", -1)) == 1
        and selected == samples + uncertain + int(probe.get("clock_errors", -1))
        and launch_uncertainty_valid(samples, uncertain, selected)
        and (expected_exit_launches is None or selected == expected_exit_launches)
    )


def run_nvbit_once(
    tool: str,
    run_id: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    exit_layout = kernelretsnoop_layout(args.pp, correctness=False)
    label = f"nvbit_{tool}"
    result = run_bench(
        label,
        run_id,
        args,
        output_dir,
        env_extra={
            "LD_PRELOAD": str(args.nvbit_tool),
            "NOBANNER": "1",
            "OBS_MODE": tool,
            "OBS_TARGET_SYMBOL": args.target_symbol,
            "OBS_GPU_THREAD_COUNT": str(
                args.threadhist_gpu_thread_count
                if tool == "threadhist"
                else exit_layout["thread_slots"]
            ),
        },
    )
    log_path = output_dir / result["log"]
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    result["probe"] = parse_nvbit(tool, text)
    result["valid"] = bool(result.get("valid")) and nvbit_probe_valid(
        tool, result["probe"],
        expected_exit_events=(exit_layout["events"] if tool == "kernelretsnoop" else None),
        expected_exit_launches=(exit_layout["launches"] if tool == "kernelretsnoop" else None),
        expected_exit_coordinates=(
            exit_layout["coordinates"] if tool == "kernelretsnoop" else None
        ),
        exact_exit_oracle=False,
    )
    return result


def gpubpf_probe_valid(tool: str, probe: dict[str, Any], *,
                      expected_thread_count: int | None = None,
                      expected_ring_entries: int | None = None,
                      expected_exit_events: int | None = None,
                      expected_exit_launches: int | None = None,
                      expected_exit_coordinates: int | None = None,
                      exact_exit_oracle: bool = False) -> bool:
    if type(probe.get("sample_count")) is not int:
        return False
    samples = probe["sample_count"]
    if samples <= 0:
        return False
    if tool == "kernelretsnoop":
        requested = int(probe.get("requested_thread_slots", -1))
        launches = int(probe.get("cartesian_launches", -1))
        coordinates = int(probe.get("cartesian_coordinates", -1))
        unique_coordinates = int(probe.get("unique_coordinates", -1))
        multiplicity_coordinates = sum(int(probe.get(key, -1)) for key in (
            "multiplicity_220", "multiplicity_44", "multiplicity_22",
            "other_multiplicity",
        ))
        generic_valid = (
            expected_thread_count is not None
            and requested == expected_thread_count
            and int(probe.get("allocated_thread_slots", -1)) == requested
            and expected_ring_entries is not None
            and int(probe.get("requested_entries_per_thread", -1)) == expected_ring_entries
            and int(probe.get("entries_per_thread", -1)) == expected_ring_entries
            and int(probe.get("record_bytes", -1)) == EXIT_RECORD_BYTES
            and int(probe.get("committed_events", -1))
            == int(probe.get("runtime_collected_events", -2))
            == int(probe.get("nonzero_timestamps", -3))
            == samples
            and all(int(probe.get(key, -1)) == 0 for key in (
                "oob_drops", "full_drops", "bad_size_drops", "other_drops",
                "dirty_slots", "pending_events", "second_drain_events",
                "invalid_launch_coordinates",
            ))
            and int(probe.get("final_drain_events", -1)) >= 0
            and int(probe.get("final_drain_events", -1)) <= samples
            and int(probe.get("cartesian_complete", -1)) == 1
            and int(probe.get("collector_gate_passed", -1)) == 1
            and launches > 0
            and coordinates > 0
            and expected_exit_coordinates is not None
            and expected_exit_coordinates % 256 == 0
            and int(probe.get("extent_x", -1)) == expected_exit_coordinates // 256
            and int(probe.get("extent_y", -1)) == 256
            and int(probe.get("extent_z", -1)) == 1
            and coordinates == unique_coordinates == multiplicity_coordinates
            and int(probe.get("segment_mismatches", -1)) == 0
            and int(probe.get("oracle_enabled", -1)) == int(exact_exit_oracle)
            and int(probe.get("oracle_total_events", -1)) == samples
            and int(probe.get("oracle_passed", -1)) == int(exact_exit_oracle)
            and (expected_exit_events is None or samples == expected_exit_events)
            and (expected_exit_launches is None or launches == expected_exit_launches)
            and (expected_exit_coordinates is None or coordinates == expected_exit_coordinates)
            and (exact_exit_oracle or (
                int(probe.get("multiplicity_220", -1)) == 0
                and int(probe.get("multiplicity_44", -1)) == coordinates
                and int(probe.get("multiplicity_22", -1)) == 0
                and int(probe.get("other_multiplicity", -1)) == 0
            ))
        )
        if not generic_valid or not exact_exit_oracle:
            return generic_valid
        return (
            int(probe.get("multiplicity_220", -1)) == CORRECTNESS_MULTIPLICITY_220
            and int(probe.get("multiplicity_44", -1)) == CORRECTNESS_MULTIPLICITY_44
            and int(probe.get("multiplicity_22", -1)) == CORRECTNESS_MULTIPLICITY_22
            and int(probe.get("other_multiplicity", -1)) == 0
            and int(probe.get("segment_mismatches", -1)) == 0
        )
    if tool == "threadhist":
        return (int(probe.get("nonzero_threads", 0)) > 0
                and expected_thread_count is not None and expected_thread_count > 0
                and probe.get("configured_entries") == expected_thread_count
                and probe.get("readback_entries") == expected_thread_count
                and probe.get("readback_bytes") == expected_thread_count * 8
                and probe.get("readback_complete") == 1)

    launch_fields = (
        "histogram_samples", "host_launches", "host_enqueued",
        "device_entries", "matched_samples", "queue_underflows",
        "queue_overflows", "queue_update_errors", "classified_samples",
        "uncertain_samples", "clock_errors", "online_accounting_complete",
        "accounting_complete", "pairing_complete",
        "probes_detached_before_readback",
    )
    if any(type(probe.get(key)) is not int for key in launch_fields):
        return False
    calibration_valid = (
        probe.get("clock_calibration_method")
        == GPUBPF_LAUNCH_CLOCK_METHOD
        and launch_clock_model_valid(probe)
        and launch_rm_anchors_valid(probe)
    )
    matched = int(probe.get("matched_samples", -1))
    classified = int(probe.get("classified_samples", -1))
    uncertain = int(probe.get("uncertain_samples", -1))
    return (
        calibration_valid
        and int(probe.get("probes_detached_before_readback", -1)) == 1
        and int(probe.get("clock_errors", -1)) == 0
        and int(probe.get("queue_underflows", -1)) == 0
        and int(probe.get("queue_overflows", -1)) == 0
        and int(probe.get("queue_update_errors", -1)) == 0
        and int(probe.get("online_accounting_complete", -1)) == 1
        and int(probe.get("accounting_complete", -1)) == 1
        and int(probe.get("pairing_complete", -1)) == 1
        and launch_uncertainty_valid(classified, uncertain, matched)
        and int(probe.get("host_launches", -1))
        == int(probe.get("host_enqueued", -2))
        == int(probe.get("device_entries", -2))
        == matched
        == samples
        and int(probe.get("histogram_samples", -5)) == classified
        and (expected_exit_launches is None or matched == expected_exit_launches)
    )


def normalized_output(stdout: str) -> str:
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", stdout)
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def llama_cli_cmd(args: argparse.Namespace) -> list[str]:
    return [
        str(args.llama_cli),
        "-m",
        str(args.model),
        "-p",
        "Write one sentence explaining why deterministic tests matter.",
        "-n",
        "8",
        "-c",
        "512",
        "-ngl",
        str(args.n_gpu_layers),
        "--seed",
        str(SCHEDULE_SEED),
        "--temp",
        "0",
        "--no-display-prompt",
        "--simple-io",
    ]


def run_cli_separate(
    cmd: list[str], *, cwd: Path, env: dict[str, str], timeout: int, log_path: Path
) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd, env = target_launch(cmd, env)
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdin=subprocess.DEVNULL,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    execution = {"command": cmd, "identity": process_identity(process), "cleanup_passed": False}
    timed_out = False
    try:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            stop_owned(process, "CUDA client", execution["identity"])
            execution["cleanup_passed"] = True
        if timed_out:
            stdout, stderr = process.communicate(timeout=5)
        returncode = -1 if timed_out else process.returncode
        log_path.write_text(
            f"$ {' '.join(cmd)}\n# cwd: {cwd}\n\n## stdout\n{stdout}"
            f"\n## stderr\n{stderr}\n# exit: {returncode}\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)
    except BaseException as error:
        execution["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, OwnedCleanupError):
            execution["cleanup_failure"] = error.details
        raise
    finally:
        execution.update(returncode=process.returncode, timed_out=timed_out)
        log_path.with_suffix(".execution.json").write_text(json.dumps(execution, indent=2) + "\n")


def correctness_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    if args.uvm:
        env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"
    return env


def run_bench(label: str, run_id: int, args: argparse.Namespace, output_dir: Path,
              env_extra: dict[str, str] | None = None) -> dict[str, Any]:
    # Use the same owned-process primitive as correctness. The older helper
    # can leave its child running on interruption, before private SHM teardown.
    log_path = output_dir / f"{label}_run_{run_id:02d}" / "llama_bench.log"
    completed = run_cli_separate(core.make_llama_cmd(args), cwd=core.WORKLOAD_DIR,
                                 env={**correctness_env(args), **(env_extra or {})},
                                 timeout=args.timeout_s, log_path=log_path)
    result = {"run": run_id, "log": str(log_path.relative_to(output_dir)),
              "returncode": completed.returncode, "valid": False}
    if completed.returncode != 0:
        result["error"] = f"llama-bench failed or timed out: {completed.returncode}"
        return result
    try:
        result.update(core.parse_llama_bench(completed.stdout + "\n" + completed.stderr))
        metrics = result["metrics"]
        result["valid"] = (metrics.get("pp_tokens") == args.pp
                           and math.isfinite(float(metrics.get("pp_tok_s", 0)))
                           and float(metrics.get("pp_tok_s", 0)) > 0)
    except (ValueError, KeyError, TypeError) as error:
        result["error"] = f"parse failed: {error}"
    return result


def run_correctness_cell(
    config: str,
    attempt: int,
    args: argparse.Namespace,
    output_dir: Path,
    tool_dirs: dict[str, Path],
    *, diagnostic_log_level: str | None = None,
) -> dict[str, Any]:
    run_dir = output_dir / "correctness" / config / f"attempt_{attempt:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    idle_gpu_or_error(core.nvidia_smi_snapshot())
    env = correctness_env(args)
    probe_context = nullcontext({})
    tool = None
    if config != "baseline":
        system, tool = config.split("_", 1)
        if system == "gpubpf":
            probe_context = private_probe(tool, args, tool_dirs[tool], run_dir,
                                          diagnostic_log_level=diagnostic_log_level,
                                          exact_exit_oracle=tool == "kernelretsnoop")
        else:
            env.update(
                {
                    "LD_PRELOAD": str(args.nvbit_tool),
                    "NOBANNER": "1",
                    "OBS_MODE": tool,
                    "OBS_TARGET_SYMBOL": args.target_symbol,
                    "OBS_GPU_THREAD_COUNT": str(
                        args.threadhist_gpu_thread_count
                        if tool == "threadhist"
                        else args.gpu_thread_count
                    ),
                }
            )
    with cell_safety(run_dir) as safety_record, probe_context as probe_env:
        env.update(probe_env)
        completed = run_cli_separate(
            llama_cli_cmd(args),
            cwd=core.WORKLOAD_DIR,
            env=env,
            timeout=args.timeout_s,
            log_path=run_dir / "llama_cli.log",
        )

    output = normalized_output(completed.stdout)
    result: dict[str, Any] = {
        "attempt": attempt,
        "returncode": completed.returncode,
        "normalized_stdout": output,
        "stdout_bytes": len(output.encode()),
        "log": str((run_dir / "llama_cli.log").relative_to(output_dir)),
        "valid": completed.returncode == 0 and output == EXPECTED_NORMALIZED_STDOUT,
        "safety": safety_record,
    }
    if tool is not None:
        if config.startswith("gpubpf_"):
            probe_log = run_dir / "probe.log"
            probe_text = probe_log.read_text(errors="replace") if probe_log.exists() else ""
            result["probe"] = parse_gpubpf(tool, probe_text)
            result["verifier"] = verifier_evidence(
                args, run_dir, tool, correctness=True
            )
            result["valid"] = bool(result["valid"]) and result["verifier"]["passed"]
            result["valid"] = bool(result["valid"]) and gpubpf_probe_valid(
                tool,
                result["probe"],
                expected_thread_count=(
                    args.gpu_thread_count if tool == "kernelretsnoop"
                    else args.threadhist_gpu_thread_count
                ),
                expected_ring_entries=(
                    CORRECTNESS_RING_ENTRIES_PER_THREAD if tool == "kernelretsnoop" else None
                ),
                expected_exit_events=(
                    CORRECTNESS_EXIT_EVENTS if tool == "kernelretsnoop" else None
                ),
                expected_exit_launches=(
                    CORRECTNESS_EXIT_LAUNCHES
                    if tool in ("kernelretsnoop", "launchlate") else None
                ),
                expected_exit_coordinates=(
                    CORRECTNESS_EXIT_COORDINATES if tool == "kernelretsnoop" else None
                ),
                exact_exit_oracle=tool == "kernelretsnoop",
            )
        else:
            result["probe"] = parse_nvbit(tool, completed.stderr)
            result["valid"] = bool(result["valid"]) and nvbit_probe_valid(
                tool,
                result["probe"],
                expected_exit_events=(
                    CORRECTNESS_EXIT_EVENTS if tool == "kernelretsnoop" else None
                ),
                expected_exit_launches=(
                    CORRECTNESS_EXIT_LAUNCHES
                    if tool in ("kernelretsnoop", "launchlate") else None
                ),
                expected_exit_coordinates=(
                    CORRECTNESS_EXIT_COORDINATES if tool == "kernelretsnoop" else None
                ),
                exact_exit_oracle=tool == "kernelretsnoop",
            )
    return result


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    low = math.floor(location)
    high = math.ceil(location)
    if low == high:
        return ordered[low]
    fraction = location - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def bootstrap_mean_ci(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    rng = random.Random(SCHEDULE_SEED)
    boot = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [values[rng.randrange(len(values))] for _ in values]
        boot.append(sum(sample) / len(sample))
    return {
        "mean": sum(values) / len(values),
        "ci95_low": quantile(boot, 0.025),
        "ci95_high": quantile(boot, 0.975),
    }


def valid_run_for_block(state: dict[str, Any], config: str, block: int) -> dict[str, Any] | None:
    for run in reversed(state["configs"][config]["runs"]):
        if run.get("block") == block and run.get("valid"):
            return run
    return None


def valid_correctness(state: dict[str, Any], config: str) -> dict[str, Any] | None:
    baseline_attempts = state["correctness"]["baseline"]["attempts"]
    baseline = next(
        (attempt for attempt in reversed(baseline_attempts) if attempt.get("valid")),
        None,
    )
    if baseline is None:
        return None
    expected = baseline["normalized_stdout"]
    for attempt in reversed(state["correctness"][config]["attempts"]):
        if attempt.get("valid") and attempt.get("normalized_stdout") == expected:
            return attempt
    return None


def pp_throughput(run: dict[str, Any]) -> float:
    return float(run["metrics"]["pp_tok_s"])


def summarize(state: dict[str, Any]) -> dict[str, Any]:
    selection = argparse.Namespace(tools=state["params"].get("tools", TASKS))
    tools = selected_tools(selection)
    configs = selected_configs(selection)
    if set(state["configs"]) != set(configs):
        raise ValueError("timing state differs from the selected-tool matrix")
    config_rows: list[dict[str, Any]] = []
    for config in configs:
        valid = [run for run in state["configs"][config]["runs"] if run.get("valid")]
        by_block = {int(run["block"]): run for run in valid}
        values = [pp_throughput(by_block[block]) for block in sorted(by_block)]
        config_rows.append(
            {
                "config": config,
                "valid_blocks": len(values),
                "attempts": len(state["configs"][config]["runs"]),
                "pp_tok_s_geomean": core.geomean(values),
            }
        )

    comparisons = []
    for task in tools:
        effects = []
        paired_rows = []
        for block in range(1, int(state["params"]["runs"]) + 1):
            baseline = valid_run_for_block(state, "baseline", block)
            gpubpf = valid_run_for_block(state, f"gpubpf_{task}", block)
            nvbit = valid_run_for_block(state, f"nvbit_{task}", block)
            if not (baseline and gpubpf and nvbit):
                continue
            base_t = pp_throughput(baseline)
            gpubpf_overhead = (base_t - pp_throughput(gpubpf)) / base_t * 100.0
            nvbit_overhead = (base_t - pp_throughput(nvbit)) / base_t * 100.0
            effect = nvbit_overhead - gpubpf_overhead
            effects.append(effect)
            paired_rows.append(
                {
                    "block": block,
                    "baseline_pp_tok_s": base_t,
                    "gpubpf_overhead_pct": gpubpf_overhead,
                    "nvbit_overhead_pct": nvbit_overhead,
                    "effect_pct_points": effect,
                }
            )
        comparisons.append(
            {
                "task": task,
                "paired_blocks": len(effects),
                "effect_definition": "NVBit overhead - gpubpf overhead (percentage points)",
                "paired": paired_rows,
                "bootstrap": bootstrap_mean_ci(effects),
            }
        )
    return {"configs": config_rows, "comparisons": comparisons}


def write_state(output_dir: Path, state: dict[str, Any]) -> None:
    state["summary"] = summarize(state)
    (output_dir / "result.json").write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["config", "valid_blocks", "attempts", "pp_tok_s_geomean"],
        )
        writer.writeheader()
        writer.writerows(state["summary"]["configs"])

    lines = [
        "# RQ4 matched observability experiment",
        "",
        f"- Phase: `{state['phase']}`",
        f"- Driver: `{state['provenance']['driver']}`",
        f"- Target: `{state['params']['target_symbol']}`",
        f"- Blocks requested: `{state['params']['runs']}`",
        "",
        "| Config | Valid blocks | Attempts | Prefill tok/s geomean |",
        "|---|---:|---:|---:|",
    ]
    for row in state["summary"]["configs"]:
        gm = row["pp_tok_s_geomean"]
        lines.append(
            f"| {row['config']} | {row['valid_blocks']} | {row['attempts']} | "
            f"{gm:.2f} |" if gm is not None else
            f"| {row['config']} | {row['valid_blocks']} | {row['attempts']} | n/a |"
        )
    lines.extend(["", "## Paired effects", ""])
    for comparison in state["summary"]["comparisons"]:
        ci = comparison["bootstrap"]
        if ci:
            result = (
                f"mean {ci['mean']:.2f} pp, 95% CI "
                f"[{ci['ci95_low']:.2f}, {ci['ci95_high']:.2f}]"
            )
        else:
            result = "incomplete"
        lines.append(
            f"- {comparison['task']}: {comparison['paired_blocks']} paired blocks; {result}."
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def new_state(args: argparse.Namespace, timestamp: str, snapshot: dict[str, Any],
              supported_clock_pairs: list[list[int]] | None = None) -> dict[str, Any]:
    configs = selected_configs(args)
    artifact = HERE / "deps/nvbit-Linux-x86_64-1.8.tar.bz2"
    return {
        "timestamp": timestamp,
        "phase": args.phase,
        "params": defining_params(args),
        "provenance": {
            "gpu_ext_git": core.git_rev(core.GPU_EXT_ROOT),
            "bpftime_git": core.git_rev(args.bpftime_root),
            "nvidia_smi": snapshot,
            "driver": parse_driver(snapshot),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "nvbit_driver_supported": nvbit_driver_supported(parse_driver(snapshot)),
            "supported_clock_pairs_mhz": supported_clock_pairs,
            "cuda_ptx": core.cuda_ptx_snapshot(args.llama_bench),
            "model_file": file_metadata(args.model),
            "llama_bench_file": file_metadata(args.llama_bench),
            "llama_cli_file": file_metadata(args.llama_cli),
            "libggml_cuda_file": file_metadata(args.llama_bench.parent / "libggml-cuda.so"),
            "bpftime_agent_file": file_metadata(
                args.bpftime_build_dir / "runtime/agent/libbpftime-agent.so"
            ),
            "bpftime_syscall_server_file": file_metadata(
                args.bpftime_build_dir
                / "runtime/syscall-server/libbpftime-syscall-server.so"
            ),
            "nvbit_artifact_file": file_metadata(artifact),
            "source_manifest": source_manifest(args),
        },
        "schedule": fixed_schedule(args),
        "correctness": {config: {"attempts": []} for config in configs},
        "artifacts": {},
        "configs": {config: {"runs": []} for config in configs},
    }


def record_artifacts(
    state: dict[str, Any], args: argparse.Namespace, tool_dirs: dict[str, Path]
) -> None:
    paths = {"nvbit_tool": args.nvbit_tool}
    for tool, directory in tool_dirs.items():
        paths[f"gpubpf_{tool}"] = directory / tool
    state["artifacts"] = {
        name: file_metadata(path)
        for name, path in paths.items()
    }


def verify_resume(
    state: dict[str, Any], args: argparse.Namespace, snapshot: dict[str, Any]
) -> dict[str, Path]:
    recorded_params = dict(state.get("params", {}))
    # Historical full-width campaigns predate the explicit selector. They are
    # equivalent only to the unchanged default three-tool selection.
    recorded_params.setdefault("tools", list(TASKS))
    recorded_params.setdefault("preflight_campaign", None)
    recorded_params.setdefault("verifier_level", "DEFAULT")
    recorded_params.setdefault(
        "verifier_runtime_configuration",
        verifier_runtime_configuration(args.bpftime_build_dir),
    )
    if recorded_params != defining_params(args):
        raise RuntimeError("resume parameters differ from the recorded experiment")
    expected_configs = set(selected_configs(args))
    if set(state.get("correctness", {})) != expected_configs:
        raise RuntimeError("resume correctness matrix differs from the selected tools")
    if set(state.get("configs", {})) != expected_configs:
        raise RuntimeError("resume timing matrix differs from the selected tools")
    if state.get("schedule") != fixed_schedule(args):
        raise RuntimeError("resume schedule differs from the fixed selected-tool matrix")
    if state.get("provenance", {}).get("driver") != parse_driver(snapshot):
        raise RuntimeError("resume driver differs from the recorded experiment")
    checks = {
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "bpftime_git": core.git_rev(args.bpftime_root),
        "model_file": file_metadata(args.model),
        "llama_bench_file": file_metadata(args.llama_bench),
        "llama_cli_file": file_metadata(args.llama_cli),
        "libggml_cuda_file": file_metadata(args.llama_bench.parent / "libggml-cuda.so"),
        "bpftime_agent_file": file_metadata(
            args.bpftime_build_dir / "runtime/agent/libbpftime-agent.so"
        ),
        "bpftime_syscall_server_file": file_metadata(
            args.bpftime_build_dir
            / "runtime/syscall-server/libbpftime-syscall-server.so"
        ),
        "source_manifest": source_manifest(args),
    }
    provenance = state.get("provenance", {})
    for key, actual in checks.items():
        if provenance.get(key) != actual:
            raise RuntimeError(f"resume provenance mismatch: {key}")

    tool_dirs: dict[str, Path] = {}
    for name, artifact in state.get("artifacts", {}).items():
        path = Path(artifact["path"])
        if file_metadata(path) != artifact:
            raise RuntimeError(f"resume artifact mismatch: {name}")
        if name == "nvbit_tool":
            args.nvbit_tool = path
        elif name.startswith("gpubpf_"):
            tool_dirs[name.removeprefix("gpubpf_")] = path.parent
    if set(tool_dirs) != set(selected_tools(args)) or not hasattr(args, "nvbit_tool"):
        raise RuntimeError("resume artifact manifest is incomplete")
    return tool_dirs


def run_cell(
    config: str,
    run_id: int,
    args: argparse.Namespace,
    output_dir: Path,
    tool_dirs: dict[str, Path],
) -> dict[str, Any]:
    with cell_safety(output_dir / f"{config}_run_{run_id:02d}") as safety_record:
        result = run_instrumented_cell(config, run_id, args, output_dir, tool_dirs)
    result["safety"] = safety_record
    return result


def run_instrumented_cell(config: str, run_id: int, args: argparse.Namespace,
                          output_dir: Path, tool_dirs: dict[str, Path]) -> dict[str, Any]:
    idle_gpu_or_error(core.nvidia_smi_snapshot())
    if config == "baseline":
        return run_bench("baseline", run_id, args, output_dir)
    system, tool = config.split("_", 1)
    if system == "gpubpf":
        exit_layout = kernelretsnoop_layout(args.pp, correctness=False)
        # Keep the benchmark, private-probe, telemetry, and safety records in
        # the same config-specific directory.  The independent analyzer uses
        # the benchmark log's parent as the raw cell boundary.
        run_dir = output_dir / f"{config}_run_{run_id:02d}"
        with private_probe(
            tool, args, tool_dirs[tool], run_dir, exact_exit_oracle=False
        ) as env:
            result = run_bench(config, run_id, args, output_dir, env_extra=env)
        result["probe"] = parse_gpubpf(tool, (run_dir / "probe.log").read_text(errors="replace"))
        result["probe_log"] = str((run_dir / "probe.log").relative_to(output_dir))
        result["verifier"] = verifier_evidence(
            args, run_dir, tool, correctness=False
        )
        result["valid"] = bool(result.get("valid")) and gpubpf_probe_valid(
            tool,
            result["probe"],
            expected_thread_count=(
                exit_layout["thread_slots"] if tool == "kernelretsnoop"
                else args.threadhist_gpu_thread_count
            ),
            expected_ring_entries=(
                exit_layout["entries_per_thread"] if tool == "kernelretsnoop" else None
            ),
            expected_exit_events=(exit_layout["events"] if tool == "kernelretsnoop" else None),
            expected_exit_launches=(exit_layout["launches"] if tool == "kernelretsnoop" else None),
            expected_exit_coordinates=(exit_layout["coordinates"] if tool == "kernelretsnoop" else None),
            exact_exit_oracle=False,
        ) and result["verifier"]["passed"]
        return result
    return run_nvbit_once(tool, run_id, args, output_dir)


def reconcile_kernelret_block(state: dict[str, Any], block: int) -> None:
    """Reject a timed pair unless both collectors observed the same exits."""
    gpubpf = valid_run_for_block(state, "gpubpf_kernelretsnoop", block)
    nvbit = valid_run_for_block(state, "nvbit_kernelretsnoop", block)
    if not (gpubpf and nvbit):
        return
    gpubpf_probe = gpubpf.get("probe", {})
    nvbit_probe = nvbit.get("probe", {})
    matched_fields = (
        "sample_count", "nonzero_timestamps", "record_bytes",
        "cartesian_launches", "cartesian_coordinates", "cartesian_complete",
        "extent_x", "extent_y", "extent_z", "multiplicity_220",
        "multiplicity_44", "multiplicity_22", "other_multiplicity",
        "segment_mismatches", "invalid_launch_coordinates",
        "unique_coordinates", "collector_gate_passed",
    )
    matched = all(gpubpf_probe.get(key) == nvbit_probe.get(key)
                  for key in matched_fields)
    matched = matched and (
        gpubpf_probe.get("cartesian_launches")
        == nvbit_probe.get("selected_launches")
    )
    comparison = {
        "matched": matched,
        "matched_fields": list(matched_fields),
        "gpubpf_events": gpubpf_probe.get("sample_count"),
        "nvbit_events": nvbit_probe.get("sample_count"),
        "gpubpf_launches": gpubpf_probe.get("cartesian_launches"),
        "nvbit_launches": nvbit_probe.get("selected_launches"),
    }
    for run in (gpubpf, nvbit):
        run["kernelret_pair"] = comparison
        if not matched:
            run["valid"] = False
            run["pairing_error"] = "gpubpf/NVBit exit events or selected launches differ"


def reconcile_launchlate_clock_block(state: dict[str, Any], block: int) -> None:
    """Fail closed unless all three timing arms saw the same enumerated state set."""
    names = ("baseline", "gpubpf_launchlate", "nvbit_launchlate")
    runs = [valid_run_for_block(state, name, block) for name in names]
    if not all(runs):
        return
    inventory_raw = state.get("provenance", {}).get("supported_clock_pairs_mhz")
    inventory = {
        tuple(pair) for pair in inventory_raw
        if isinstance(pair, list) and len(pair) == 2
        and all(type(value) is int for value in pair)
    } if isinstance(inventory_raw, list) else set()
    states: list[set[tuple[int, int]]] = []
    arms: dict[str, Any] = {}
    passed = bool(inventory)
    for name, run in zip(names, runs):
        safety = run.get("safety", {})
        telemetry = safety.get("telemetry", {}) if isinstance(safety, dict) else {}
        raw_pairs = telemetry.get("clock_pairs_mhz")
        pairs = {
            tuple(pair) for pair in raw_pairs
            if isinstance(pair, (list, tuple)) and len(pair) == 2
            and all(type(value) is int for value in pair)
        } if isinstance(raw_pairs, list) else set()
        snapshots_clear = all(
            isinstance(safety.get(point), dict)
            and safety[point].get("gpu", {}).get("compute_apps") == []
            for point in ("before", "after")
        )
        arm_passed = (
            safety.get("passed") is True
            and telemetry.get("throttled") is False
            and telemetry.get("pstates") == ["P0"]
            and bool(pairs) and pairs <= inventory and snapshots_clear
        )
        passed = passed and arm_passed
        states.append(pairs)
        arms[name] = {
            "pstates": telemetry.get("pstates"),
            "clock_pairs_mhz": [list(pair) for pair in sorted(pairs)],
            "passed": arm_passed,
        }
    matched = len({frozenset(value) for value in states}) == 1
    passed = passed and matched
    comparison = {
        "passed": passed, "exact_sets_matched": matched,
        "supported_inventory_nonempty": bool(inventory), "arms": arms,
    }
    for run in runs:
        run["launchlate_clock_fairness"] = comparison
        if not passed:
            run["valid"] = False
            run["clock_fairness_error"] = (
                "launchlate timing arms did not share one nonempty exact P0 supported-clock set"
            )


def validate_plan(args: argparse.Namespace) -> None:
    tools = selected_tools(args)
    if ("kernelretsnoop" in tools
            and args.gpu_thread_count != EXPECTED_GPU_THREAD_SLOTS):
        raise ValueError(
            f"kernelretsnoop is fixed to {EXPECTED_GPU_THREAD_SLOTS} GPU thread slots"
        )
    if len(EXPECTED_NORMALIZED_STDOUT.encode()) != EXPECTED_NORMALIZED_STDOUT_BYTES:
        raise RuntimeError("the deterministic correctness oracle is not 47 bytes")
    if args.phase == "preflight" and (args.runs != 1 or args.pp != 32):
        raise ValueError("preflight is fixed to --runs 1 --pp 32")
    if args.phase == "full" and (args.runs != 10 or args.pp != 512):
        raise ValueError("paper-facing full run is fixed to --runs 10 --pp 512")
    if "kernelretsnoop" in tools:
        layout = kernelretsnoop_layout(args.pp, correctness=False)
        if layout["shared_bytes"] > KERNELRETSNOOP_SHM_MEMORY_MB * 1024 * 1024:
            raise ValueError("kernelretsnoop timing ring exceeds its frozen shared-memory budget")
    preflight_dir = getattr(args, "preflight_dir", None)
    if args.phase == "preflight" and preflight_dir is not None:
        raise ValueError("--preflight-dir is only valid for a subset full run")
    if args.phase == "full" and tools == TASKS and preflight_dir is not None:
        raise ValueError("--preflight-dir is only valid for a subset full run")
    if args.phase == "full" and tools != TASKS:
        if preflight_dir is None:
            raise ValueError("subset full requires --preflight-dir from the same selected tools")
        if args.output_dir is None:
            raise ValueError("subset full requires an explicit --output-dir")
        require_disjoint_campaign_paths(preflight_dir, args.output_dir)


def require_disjoint_campaign_paths(first: Path, second: Path) -> None:
    first = first.resolve()
    second = second.resolve()
    if first == second or first in second.parents or second in first.parents:
        raise ValueError("preflight and full campaign paths must be distinct and mutually non-nested")


def validate_subset_preflight(args: argparse.Namespace) -> None:
    tools = selected_tools(args)
    if args.phase != "full" or tools == TASKS:
        return
    preflight_dir = args.preflight_dir.resolve()
    require_disjoint_campaign_paths(preflight_dir, args.output_dir)
    import analyze_revision_rq4 as independent
    result = independent.analyze(preflight_dir)
    if (result.get("phase") != "preflight"
            or tuple(result.get("tools", ())) != tools
            or result.get("configs") != list(selected_configs(args))
            or result.get("complete") is not True):
        raise RuntimeError(
            "subset full requires an independently complete preflight with the same tools"
        )


def validate(args: argparse.Namespace) -> None:
    validate_plan(args)
    validate_subset_preflight(args)
    core.validate(args)
    require_explicit_verifier_build(args)
    if not args.llama_cli.exists():
        raise FileNotFoundError(args.llama_cli)
    if not NVBIT_ROOT.exists():
        raise FileNotFoundError(NVBIT_ROOT)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "full"), default="preflight")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--preflight-dir",
        type=Path,
        help="required independently passing campaign for a subset full run",
    )
    parser.add_argument("--model", type=Path, default=core.DEFAULT_MODEL)
    parser.add_argument("--llama-bench", type=Path, default=core.DEFAULT_LLAMA_BENCH)
    parser.add_argument("--llama-cli", type=Path)
    parser.add_argument("--bpftime-root", type=Path, required=True)
    parser.add_argument(
        "--bpftime-build-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--verifier-level",
        choices=VERIFIER_LEVELS,
        default="DEFAULT",
        help=(
            "device-policy admission treatment; STRICT and NO_VERIFY require the same "
            "verifier-enabled runtime, while DEFAULT preserves historical behavior"
        ),
    )
    parser.add_argument("--target-symbol", default=core.DEFAULT_TARGET_SYMBOL)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--pp", type=int)
    parser.add_argument("--tg", type=int, default=0)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--probe-startup-s", type=float, default=3.0)
    parser.add_argument("--gpu-thread-count", type=int, required=True)
    parser.add_argument("--threadhist-gpu-thread-count", type=int, default=1048576)
    parser.add_argument("--uprobe-binary", type=Path, default=core.DEFAULT_LAUNCH_STUB_LIBRARY)
    parser.add_argument("--uprobe-symbol-hint", default=core.DEFAULT_TARGET_SYMBOL)
    parser.add_argument("--uvm", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--tools",
        nargs="+",
        choices=TASKS,
        default=list(TASKS),
        metavar="TOOL",
        help="predeclare one or more tools; default: all three",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the fixed cell matrix and exact gates without touching GPU state",
    )
    parser.add_argument(
        "--inherited-lease-fds", nargs=2, type=int,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)

    args.runs = args.runs if args.runs is not None else (1 if args.phase == "preflight" else 10)
    args.pp = args.pp if args.pp is not None else (32 if args.phase == "preflight" else 512)
    args.llama_cli = args.llama_cli or (args.llama_bench.parent / "llama-cli")
    for field in ("model", "llama_bench", "llama_cli", "bpftime_root", "bpftime_build_dir", "uprobe_binary"):
        setattr(args, field, getattr(args, field).resolve())
    if args.preflight_dir is not None:
        args.preflight_dir = args.preflight_dir.resolve()
    args.tools = list(selected_tools(args))
    return args


def main() -> int:
    args = parse_args()
    if args.dry_run:
        validate_plan(args)
        print(json.dumps(dry_run_plan(args), indent=2), flush=True)
        return 0
    reject_ambient_injection()
    validate(args)

    inherited = (tuple(args.inherited_lease_fds)
                 if args.inherited_lease_fds is not None else None)
    lease = None if inherited is not None else ReadOnlyLeases()
    if inherited is not None:
        validate_inherited_lease_fds(inherited)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    previous_run_cmd = core.run_cmd
    try:
        # Only this runner process uses owned CPU/build helpers; the shared,
        # potentially dirty source file and other coordinators stay untouched.
        core.run_cmd = run_cmd_owned
        return run_campaign(args)
    finally:
        core.run_cmd = previous_run_cmd
        signal.signal(signal.SIGTERM, previous_handler)
        if lease is not None:
            lease.close()


def run_campaign(args: argparse.Namespace) -> int:
    tools = selected_tools(args)
    configs = selected_configs(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir or HERE / "raw" / f"{args.phase}-{timestamp}").resolve()
    if args.resume:
        if not (output_dir / "result.json").exists():
            raise RuntimeError("--resume requires an existing result.json")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("refusing to reuse a nonempty output directory without --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = core.nvidia_smi_snapshot()
    supported_clock_pairs = None

    admission = {
        "timestamp": timestamp,
        "phase": args.phase,
        "nvidia_smi": snapshot,
        "driver": parse_driver(snapshot),
        "expected_driver": EXPECTED_DRIVER,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "nvbit_driver_supported": nvbit_driver_supported(parse_driver(snapshot)),
        "supported_clock_pairs_mhz": supported_clock_pairs,
    }
    (output_dir / f"admission-{timestamp}.json").write_text(
        json.dumps(admission, indent=2) + "\n", encoding="utf-8"
    )
    if admission["driver"] != EXPECTED_DRIVER:
        raise RuntimeError(
            f"This campaign requires driver {EXPECTED_DRIVER}; found {admission['driver']}. "
            "Historical admissions retain their original driver identity."
        )
    idle_gpu_or_error(snapshot)

    result_path = output_dir / "result.json"
    if args.resume:
        state = json.loads(result_path.read_text(encoding="utf-8"))
        supported_clock_pairs = state.get("provenance", {}).get(
            "supported_clock_pairs_mhz"
        ) if "launchlate" in tools else None
        tool_dirs = verify_resume(state, args, snapshot)
    else:
        supported_clock_pairs = (
            query_supported_clock_pairs() if "launchlate" in tools else None
        )
        # Freeze the parameters and randomized schedule before any calibration,
        # correctness, or performance execution starts.
        state = new_state(args, timestamp, snapshot, supported_clock_pairs)
        write_state(output_dir, state)
    admission["supported_clock_pairs_mhz"] = supported_clock_pairs
    (output_dir / f"admission-{timestamp}.json").write_text(
        json.dumps(admission, indent=2) + "\n", encoding="utf-8"
    )

    clock_controls = None
    if "launchlate" in tools:
        clock_controls = (
            replay_launch_clock_controls(output_dir, admission)
            if args.resume else run_launch_clock_controls(output_dir, admission)
        )
        if args.resume:
            if state.get("clock_controls") != clock_controls:
                raise RuntimeError("result.json launch-clock controls differ from raw replay")
        else:
            state["clock_controls"] = clock_controls
            write_state(output_dir, state)

    if not args.resume:
        nvbit_build_dir = output_dir / "nvbit_tool_build"
        shutil.copytree(
            NVBIT_SOURCE_DIR,
            nvbit_build_dir,
            ignore=shutil.ignore_patterns("*.o", "*.so", "*.fatbin", "flush_channel.c"),
        )
        if "kernelretsnoop" in tools:
            validate_nvbit_kernelretsnoop_source_schema(nvbit_build_dir)
        if "launchlate" in tools:
            validate_nvbit_launchlate_source_schema(nvbit_build_dir)
        args.nvbit_tool = build_nvbit(nvbit_build_dir, output_dir)

        build_root = output_dir / "gpubpf_tool_build"
        build_root.mkdir(exist_ok=True)
        tool_dirs = {}
        for tool in tools:
            directory = prepare_tool_source(
                core.TOOLS[tool],
                bpftime_root=args.bpftime_root,
                build_root=build_root,
                target_symbol=args.target_symbol,
            )
            if tool == "launchlate":
                validate_launchlate_source_schema(directory)
            if tool == "kernelretsnoop":
                validate_kernelretsnoop_source_schema(directory)
            core.build_tool(core.TOOLS[tool], directory)
            tool_dirs[tool] = directory

        record_artifacts(state, args, tool_dirs)
        write_state(output_dir, state)

    correctness_order = ["baseline"] + [
        config for config in state["schedule"]["1"] if config != "baseline"
    ]
    for config in correctness_order:
        if valid_correctness(state, config):
            continue
        attempt = len(state["correctness"][config]["attempts"]) + 1
        print(f"correctness config={config} attempt={attempt}", flush=True)
        try:
            check = run_correctness_cell(config, attempt, args, output_dir, tool_dirs)
        except OwnedCleanupError as exc:
            check = {"attempt": attempt, "returncode": -1, "valid": False, "error": str(exc),
                     "fatal_cleanup": exc.details}
            state["correctness"][config]["attempts"].append(check)
            state["fatal_cleanup"] = exc.details
            write_state(output_dir, state)
            raise
        except Exception as exc:  # noqa: BLE001
            check = {"attempt": attempt, "returncode": -1, "valid": False, "error": str(exc)}
        if config != "baseline":
            baseline = valid_correctness(state, "baseline")
            check["matches_baseline"] = bool(
                baseline
                and check.get("normalized_stdout") == baseline.get("normalized_stdout")
            )
            check["valid"] = bool(check.get("valid")) and check["matches_baseline"]
        state["correctness"][config]["attempts"].append(check)
        write_state(output_dir, state)
        if config == "baseline" and valid_correctness(state, "baseline") is None:
            break

    if any(valid_correctness(state, config) is None for config in configs):
        print("Correctness gate incomplete; performance cells were not started.", flush=True)
        return 2

    for block in range(1, args.runs + 1):
        for config in state["schedule"][str(block)]:
            if valid_run_for_block(state, config, block):
                continue
            attempts = [
                run for run in state["configs"][config]["runs"]
                if run.get("block") == block
            ]
            attempt = len(attempts) + 1
            run_id = block * 100 + attempt
            print(f"block={block} config={config} attempt={attempt}", flush=True)
            try:
                run = run_cell(config, run_id, args, output_dir, tool_dirs)
            except OwnedCleanupError as exc:
                run = {"returncode": -1, "valid": False, "error": str(exc),
                       "block": block, "attempt": attempt, "fatal_cleanup": exc.details}
                state["configs"][config]["runs"].append(run)
                state["fatal_cleanup"] = exc.details
                write_state(output_dir, state)
                raise
            except Exception as exc:  # noqa: BLE001
                run = {"returncode": -1, "valid": False, "error": str(exc)}
            run["block"] = block
            run["attempt"] = attempt
            state["configs"][config]["runs"].append(run)
            if config in ("gpubpf_kernelretsnoop", "nvbit_kernelretsnoop"):
                reconcile_kernelret_block(state, block)
            if "launchlate" in tools:
                reconcile_launchlate_clock_block(state, block)
            write_state(output_dir, state)

    write_state(output_dir, state)
    print((output_dir / "summary.md").read_text(encoding="utf-8"), flush=True)
    if any(
        valid_run_for_block(state, config, block) is None
        for block in range(1, args.runs + 1)
        for config in configs
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
