#!/usr/bin/env python3
"""Run the fail-closed RTX 5090 device-trampoline scaling experiment."""

from __future__ import annotations

import argparse
import csv
import fcntl
import importlib.util
import json
import math
import os
import random
import re
import signal
import stat
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
GPU_EXT = HERE.parents[1]
WORKSPACE = GPU_EXT.parent
DEFAULT_BPFTIME_ROOT = WORKSPACE / "bpftime-table1-575"
DEFAULT_BPFTIME_BUILD = DEFAULT_BPFTIME_ROOT / "build-table1-575"
EXPECTED_DRIVER = "575.57.08"
EXPECTED_GPU = "NVIDIA GeForce RTX 5090"
EXPERIMENT_KIND = "RTX 5090 device-trampoline scaling"
SUMMARY_TITLE = "RTX 5090 device-trampoline scaling"
MAX_THREADS = 1_048_576
MAX_THREADS_PER_BLOCK = 256
COUNTER_KEYS = 5
SEED = 1797
POST_RUN_SETTLE_TIMEOUT_SECONDS = 60
LEASE_PATHS = (
    Path("/tmp/gpubpf-revision-gpu0.lock"),
    Path("/tmp/gpubpf-revision-struct-ops.lock"),
)
SHM_ROOT = Path("/dev/shm")
ARMS = ("baseline", "noop", "counter")
CELLS = (
    {"id": 0, "blocks": 256, "threads_per_block": 256,
     "active_threads": 65_536, "counter_key": 0},
    {"id": 1, "blocks": 512, "threads_per_block": 256,
     "active_threads": 65_536, "counter_key": 1},
    {"id": 2, "blocks": 1024, "threads_per_block": 256,
     "active_threads": 65_536, "counter_key": 2},
    {"id": 3, "blocks": 2048, "threads_per_block": 256,
     "active_threads": 65_536, "counter_key": 3},
    {"id": 4, "blocks": 4096, "threads_per_block": 256,
     "active_threads": 65_536, "counter_key": 4},
    {"id": 5, "blocks": 4096, "threads_per_block": 256,
     "active_threads": 131_072, "counter_key": 4},
    {"id": 6, "blocks": 4096, "threads_per_block": 256,
     "active_threads": 262_144, "counter_key": 4},
    {"id": 7, "blocks": 4096, "threads_per_block": 256,
     "active_threads": 524_288, "counter_key": 4},
    {"id": 8, "blocks": 4096, "threads_per_block": 256,
     "active_threads": 1_048_576, "counter_key": 4},
)
PREFLIGHT_CELL_IDS = (0,)
FULL_CELL_IDS = tuple(range(9))
PREFLIGHT_BLOCKS = 1
FULL_BLOCKS = 10
PREFLIGHT_WARMUP = 1
FULL_WARMUP = 2
PREFLIGHT_LAUNCHES = 2
FULL_LAUNCHES = 8
PREFLIGHT_HOOK_REPEATS = 1
FULL_HOOK_REPEATS = 2
RANDOMIZE_CELL_ORDER = False
BALANCE_ARM_ORDER = False
WRITE_INDEPENDENT_RAW_EVIDENCE = False
RAW_EVIDENCE_SCHEMA = 1
EXTRA_SOURCE_PATHS: tuple[Path, ...] = ()
MATRIX_HEADER = HERE / "matrix.h"
APPLICATION_BINARY = HERE / ".output/scaling"
COMPILED_PTX = HERE / ".output/scaling.ptx"
LOADER_BINARY = HERE / ".output/probe"
BPF_OBJECT_PREFIX = "probe"

SAFETY_SPEC = importlib.util.spec_from_file_location(
    "trampoline_scaling_safety",
    GPU_EXT / "workloads/moe-infinity/run_moe_head_to_head.py",
)
if not SAFETY_SPEC or not SAFETY_SPEC.loader:
    raise RuntimeError("cannot load the shared GPU safety implementation")
safety = importlib.util.module_from_spec(SAFETY_SPEC)
sys.modules[SAFETY_SPEC.name] = safety
SAFETY_SPEC.loader.exec_module(safety)


class ReadOnlyLeases:
    """Lock exact pre-created regular files without creating or writing them."""

    def __init__(self, paths: tuple[Path, ...] = LEASE_PATHS):
        self._streams: list[Any] = []
        try:
            for path in paths:
                before = path.lstat()
                if not stat.S_ISREG(before.st_mode):
                    raise RuntimeError(f"lease is not a regular file: {path}")
                stream = path.open("r")
                try:
                    opened = os.fstat(stream.fileno())
                    current = path.lstat()
                    identity = (before.st_dev, before.st_ino)
                    if (
                        (opened.st_dev, opened.st_ino) != identity
                        or (current.st_dev, current.st_ino) != identity
                        or not stat.S_ISREG(opened.st_mode)
                        or not stat.S_ISREG(current.st_mode)
                    ):
                        raise RuntimeError(f"lease inode changed while opening: {path}")
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self._streams.append(stream)
                except BaseException:
                    stream.close()
                    raise
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        for stream in reversed(self._streams):
            stream.close()
        self._streams.clear()

    def __enter__(self) -> "ReadOnlyLeases":
        return self

    def __exit__(self, _kind: Any, _value: Any, _traceback: Any) -> None:
        self.close()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def file_metadata(path: Path) -> dict[str, Any]:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"source manifest entry is not a regular file: {path}")
    return {
        "path": str(path.resolve()), "bytes": info.st_size,
        "device": info.st_dev, "inode": info.st_ino,
        "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns,
    }


def source_manifest(bpftime_root: Path) -> list[dict[str, Any]]:
    paths = (
        MATRIX_HEADER, HERE / "scaling.cu", HERE / "probe.bpf.c",
        HERE / "probe.c", HERE / "Makefile", HERE / "run_scaling.py",
        bpftime_root / "attach/nv_attach_impl/pass/ptxpass_kprobe_entry/main.cpp",
        *EXTRA_SOURCE_PATHS,
    )
    return [file_metadata(path) for path in paths]


def reject_ambient_injection(environment: dict[str, str] | None = None) -> None:
    source = os.environ if environment is None else environment
    forbidden = sorted(
        key for key in source
        if key.startswith("BPFTIME_") or key in {
            "LD_PRELOAD", "LD_AUDIT", "CUDA_INJECTION64_PATH",
            "CUDA_INJECTION32_PATH",
        }
    )
    visible = source.get("CUDA_VISIBLE_DEVICES")
    if forbidden or visible not in (None, "0"):
        raise RuntimeError(
            f"use an uninjected GPU-0 environment; conflicting keys={forbidden}, "
            f"CUDA_VISIBLE_DEVICES={visible!r}"
        )


def base_environment() -> dict[str, str]:
    return {
        "PATH": "/usr/local/cuda-12.9/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "CUDA_VISIBLE_DEVICES": "0",
        "LD_LIBRARY_PATH": "/usr/local/cuda-12.9/lib64",
    }


def runtime_configuration(build: Path) -> dict[str, str]:
    cache = build / "CMakeCache.txt"
    if not cache.is_file():
        raise RuntimeError(f"missing runtime configuration: {cache}")
    parsed: dict[str, str] = {}
    for line in cache.read_text(errors="replace").splitlines():
        left, separator, value = line.partition("=")
        if separator and ":" in left:
            parsed[left.partition(":")[0]] = value
    required = {
        "BPFTIME_ENABLE_CUDA_ATTACH": "ON",
        "BPFTIME_LLVM_JIT": "ON",
        "ENABLE_EBPF_VERIFIER": "OFF",
    }
    wrong = {key: parsed.get(key, "missing") for key, value in required.items()
             if parsed.get(key, "").upper() != value}
    if wrong:
        raise RuntimeError(f"selected runtime feature mismatch: {wrong}")
    return {key: parsed[key] for key in (*required, "CMAKE_HOME_DIRECTORY")}


def extract_ptx_body(text: str, function: str) -> str:
    match = re.search(
        rf"(?:\.visible\s+)?(?:\.entry|\.func(?:\s+\([^)]*\))?)\s+"
        rf"{re.escape(function)}\b",
        text,
    )
    if not match:
        raise RuntimeError(f"PTX function not found: {function}")
    opening = text.find("{", match.end())
    if opening < 0:
        raise RuntimeError(f"PTX function has no body: {function}")
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1:index]
    raise RuntimeError(f"unterminated PTX function: {function}")


def validate_compiled_hook_site(ptx_path: Path) -> dict[str, Any]:
    text = ptx_path.read_text(errors="replace")
    target = extract_ptx_body(text, "trampoline_scale_kernel")
    marker = extract_ptx_body(text, "trampoline_marker_kernel")
    stub = extract_ptx_body(text, "__bpftime_cuda__kernel_trace")
    call = "call.uni __bpftime_cuda__kernel_trace, ();"
    if target.count(call) != 1:
        raise RuntimeError("target PTX must contain exactly one explicit stub call site")
    if "__bpftime_cuda__kernel_trace" in marker:
        raise RuntimeError("marker PTX must use fallback entry instrumentation")
    if stub.strip() == "":
        raise RuntimeError("dummy PTX stub unexpectedly has an empty body")
    return {
        "ptx": str(ptx_path.resolve()),
        "target_explicit_stub_calls": 1,
        "marker_explicit_stub_calls": 0,
        "stub_defined": True,
    }


def audit_runtime_source(root: Path) -> dict[str, Any]:
    source_path = root / "attach/nv_attach_impl/pass/ptxpass_kprobe_entry/main.cpp"
    text = source_path.read_text(errors="replace")
    required = (
        'params.save_strategy = "minimal"',
        'bool add_register_guard = params.save_strategy == "full"',
        '"call " + stub_name',
        '"call.uni " + stub_name',
        'log_transform_stats("kprobe_entry_stub"',
        'log_transform_stats("kprobe_entry"',
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(f"runtime entry pass differs from audited source: {missing}")
    stub_search = text.find("while ((pos = out.find(pat, pos))")
    fallback_search = text.find("find_kernel_body(ptx, kernel)")
    if stub_search < 0 or fallback_search < 0 or stub_search >= fallback_search:
        raise RuntimeError("runtime no longer resolves the module-wide stub before fallback entry")
    return {
        "source": str(source_path.resolve()),
        "default_save_strategy": "minimal",
        "register_guard_for_default": False,
        "stub_replacement_instruction": "ordinary PTX call/call.uni",
        "explicit_stub_resolution_scope": "whole PTX module before named-kernel fallback",
        "required_attach_order": ["cuda__scale_target", "cuda__scale_marker"],
        "interpretation": (
            "source does not establish one scalar handler execution per warp; "
            "the experiment measures observed scaling"
        ),
    }


def audit_loader_source() -> dict[str, Any]:
    source_path = HERE / "probe.c"
    text = source_path.read_text(errors="replace")
    required = (
        'bpf_object__find_program_by_name(object, "cuda__scale_target")',
        'bpf_object__find_program_by_name(object, "cuda__scale_marker")',
        "struct bpf_program *attach_order[] = {target_program, marker_program};",
        '"kprobe/trampoline_scale_kernel"',
        '"kprobe/trampoline_marker_kernel"',
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(f"loader routing differs from audited source: {missing}")
    return {
        "source": str(source_path.resolve()),
        "attach_order": ["cuda__scale_target", "cuda__scale_marker"],
        "reason": "the first link owns the runtime's module-wide explicit stub",
    }


def json_events(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and isinstance(value.get("event"), str):
            records.append(value)
    return records


def selected_cells(cell_ids: tuple[int, ...]) -> tuple[dict[str, int], ...]:
    by_id = {cell["id"]: cell for cell in CELLS}
    if len(set(cell_ids)) != len(cell_ids) or any(index not in by_id for index in cell_ids):
        raise RuntimeError(f"invalid or repeated cell selection: {cell_ids}")
    return tuple(by_id[index] for index in cell_ids)


def validate_application_events(
    records: list[dict[str, Any]], cell_ids: tuple[int, ...], warmup: int,
    launches: int, hook_repeats: int, run_id: int,
) -> list[dict[str, Any]]:
    cells = selected_cells(cell_ids)
    devices = [item for item in records if item["event"] == "device"]
    markers = [item for item in records if item["event"] == "marker"]
    measurements = [item for item in records if item["event"] == "measurement"]
    completes = [item for item in records if item["event"] == "complete"]
    if len(devices) != 1 or (
        devices[0].get("name") != EXPECTED_GPU
        or devices[0].get("major") != 12
        or devices[0].get("minor") != 0
        or devices[0].get("warp_size") != 32
        or devices[0].get("max_threads_per_block", 0) < MAX_THREADS_PER_BLOCK
        or devices[0].get("max_grid_x", 0) < max(cell["blocks"] for cell in CELLS)
    ):
        raise RuntimeError("application device gate failed")
    if markers != [{"event": "marker", "threads": 32, "mismatches": 0}]:
        raise RuntimeError("application marker correctness gate failed")
    if len(measurements) != len(cells):
        raise RuntimeError("application measurement count mismatch")
    for actual, cell in zip(measurements, cells):
        expected = {
            "cell": cell["id"],
            "blocks": cell["blocks"],
            "threads_per_block": cell["threads_per_block"],
            "launched_threads": cell["blocks"] * cell["threads_per_block"],
            "active_threads": cell["active_threads"],
            "active_warps": cell["active_threads"] // 32,
            "counter_key": cell["counter_key"],
            "warmup": warmup,
            "launches": launches,
            "hook_repeats": hook_repeats,
            "checked_values": MAX_THREADS,
            "mismatches": 0,
        }
        if any(actual.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"measurement schema/correctness mismatch for cell {cell['id']}")
        elapsed = actual.get("elapsed_ms")
        if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError(f"invalid CUDA event time for cell {cell['id']}: {elapsed!r}")
    if completes != [{"event": "complete", "cells": len(cells), "run_id": run_id}]:
        raise RuntimeError("application completion gate failed")
    if len(records) != 3 + len(cells):
        raise RuntimeError("unknown or duplicate application JSON events")
    return measurements


def compress_values(values: list[int]) -> list[dict[str, int]]:
    segments: list[dict[str, int]] = []
    begin = 0
    while begin < len(values):
        end = begin + 1
        while end < len(values) and values[end] == values[begin]:
            end += 1
        segments.append({"begin": begin, "end": end, "value": values[begin]})
        begin = end
    return segments


def expected_counter_segments(
    cell_ids: tuple[int, ...], warmup: int, launches: int, hook_repeats: int,
) -> dict[tuple[str, int], list[dict[str, int]]]:
    result: dict[tuple[str, int], list[dict[str, int]]] = {
        ("marker_count", 0): [
            {"begin": 0, "end": 32, "value": 1},
            {"begin": 32, "end": MAX_THREADS, "value": 0},
        ]
    }
    values = [[0] * MAX_THREADS for _ in range(COUNTER_KEYS)]
    increment = (warmup + launches) * hook_repeats
    for cell in selected_cells(cell_ids):
        slots = values[cell["counter_key"]]
        for index in range(cell["active_threads"]):
            slots[index] += increment
    for key, slots in enumerate(values):
        result[("target_count", key)] = compress_values(slots)
    return result


def validate_loader_events(
    records: list[dict[str, Any]], mode: str, cell_ids: tuple[int, ...],
    warmup: int, launches: int, hook_repeats: int,
) -> dict[str, Any]:
    ready = [item for item in records if item["event"] == "ready"]
    detached = [item for item in records if item["event"] == "detached"]
    if ready != [{
        "event": "ready", "mode": mode, "programs": 2,
        "gpu_threads": MAX_THREADS, "target_map": mode == "counter",
        "attach_order": ["cuda__scale_target", "cuda__scale_marker"],
    }]:
        raise RuntimeError("loader readiness gate failed")
    if detached != [{"event": "detached", "links": 2}]:
        raise RuntimeError("loader clean-detach gate failed")
    actual: dict[tuple[str, int], list[dict[str, int]]] = {}
    for item in records:
        if item["event"] != "counter_segment":
            continue
        fields = (item.get("map"), item.get("key"), item.get("begin"),
                  item.get("end"), item.get("value"))
        if not isinstance(fields[0], str) or any(type(value) is not int for value in fields[1:]):
            raise RuntimeError("malformed counter segment")
        actual.setdefault((fields[0], fields[1]), []).append({
            "begin": fields[2], "end": fields[3], "value": fields[4],
        })
    expected = expected_counter_segments(cell_ids, warmup, launches, hook_repeats)
    if mode == "noop":
        expected = {key: value for key, value in expected.items() if key[0] == "marker_count"}
    if actual != expected:
        raise RuntimeError("complete map readback differs from the independent exact oracle")
    if len(records) != 2 + sum(len(value) for value in expected.values()):
        raise RuntimeError("unknown or duplicate loader JSON events")
    return {
        "marker_callbacks": 32,
        "target_counter_exact": mode == "counter",
        "segment_records": sum(len(value) for value in expected.values()),
        "clean_detach": True,
    }


def validate_agent_log(text: str) -> dict[str, Any]:
    exact_patterns = {
        "marker_recorded": r"Recorded pass .*ptxpass_kprobe_entry.* for func trampoline_marker_kernel(?:\s|$)",
        "target_recorded": r"Recorded pass .*ptxpass_kprobe_entry.* for func trampoline_scale_kernel(?:\s|$)",
        "marker_fallback_transform": r"\[ptxpass\] kprobe_entry: matched=1,",
        "target_stub_transform": r"\[ptxpass\] kprobe_entry_stub: matched=1,",
        "module_loaded": r"Loaded module:",
        "attach_success": r"Attach successfully",
    }
    counts = {name: len(re.findall(pattern, text)) for name, pattern in exact_patterns.items()}
    if counts["marker_recorded"] != 1 or counts["target_recorded"] != 1:
        raise RuntimeError(f"exact target pass recording gate failed: {counts}")
    if counts["marker_fallback_transform"] != 1 or counts["target_stub_transform"] != 1:
        raise RuntimeError(f"exact marker/target PTX transform gate failed: {counts}")
    if counts["module_loaded"] < 1 or counts["attach_success"] < 1:
        raise RuntimeError(f"module load/attach gate failed: {counts}")
    route = (
        text.find(" for func trampoline_scale_kernel"),
        text.find("[ptxpass] kprobe_entry_stub: matched=1,"),
        text.find(" for func trampoline_marker_kernel"),
        text.find("[ptxpass] kprobe_entry: matched=1,"),
    )
    if min(route) < 0 or list(route) != sorted(route):
        raise RuntimeError(f"target-stub/marker-fallback routing order failed: {route}")
    counts["routing_order_valid"] = True
    return counts


def validate_agent_bootstrap_log(text: str, segment: str) -> dict[str, int]:
    patterns = {
        "verifier_warning": r"Verifier mode: WARNING",
        "cuda_registration": r"Registered shared memory with CUDA:",
        "shared_memory_constructed": (
            r"Global shm constructed\. shm_open_type 1 for " + re.escape(segment)
            + r"(?:\s|$)"
        ),
        "shared_memory_initialized": r"Global shm initialized",
    }
    counts = {name: len(re.findall(pattern, text)) for name, pattern in patterns.items()}
    if any(value != 1 for value in counts.values()):
        raise RuntimeError(f"agent bootstrap evidence gate failed: {counts}")
    if re.search(r"\[(?:error|critical)\]", text, re.IGNORECASE):
        raise RuntimeError("agent bootstrap log contains an error or critical record")
    return counts


def group_members(pgid: int) -> list[int]:
    members: list[int] = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if fields[0] != "Z" and int(fields[2]) == pgid and int(fields[3]) == pgid:
                members.append(int(path.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    return members


def stop_owned(process: subprocess.Popen[Any] | None) -> None:
    if process is None:
        return
    for request, timeout in ((signal.SIGINT, 15), (signal.SIGTERM, 8), (signal.SIGKILL, 5)):
        if not group_members(process.pid):
            process.wait(timeout=1)
            return
        try:
            os.killpg(process.pid, request)
        except ProcessLookupError:
            continue
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not group_members(process.pid):
                process.wait(timeout=1)
                return
            time.sleep(0.1)
    raise RuntimeError(f"owned process group {process.pid} survived bounded cleanup")


def run_owned_to_log(
    command: list[str], *, cwd: Path, environment: dict[str, str],
    log_path: Path, timeout: float,
) -> int:
    process = None
    with log_path.open("x") as stream:
        try:
            process = subprocess.Popen(
                command, cwd=cwd, env=environment, stdout=stream,
                stderr=subprocess.STDOUT, text=True, start_new_session=True,
            )
            try:
                return process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"command exceeded {timeout:g} seconds: {command}") from exc
        finally:
            stop_owned(process)


def segment_identity(path: Path) -> tuple[int, int, int]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise RuntimeError(f"private segment is not an owned regular file: {path}")
    return info.st_dev, info.st_ino, info.st_uid


def unlink_owned_segment(path: Path, identity: tuple[int, int, int] | None) -> bool:
    try:
        actual = segment_identity(path)
    except FileNotFoundError:
        return True
    if identity is None or actual != identity:
        raise RuntimeError(f"refusing to remove unknown or replaced segment: {path}")
    path.unlink()
    return not os.path.lexists(path)


def wait_ready(path: Path, process: subprocess.Popen[Any], timeout: float = 20) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(item.get("event") == "ready" for item in json_events(path)):
            return
        if process.poll() is not None:
            raise RuntimeError("loader exited before readiness")
        time.sleep(0.1)
    raise RuntimeError("loader did not become ready within 20 seconds")


def application_command(
    cell_ids: tuple[int, ...], warmup: int, launches: int,
    hook_repeats: int, run_id: int,
) -> list[str]:
    return [
        str(APPLICATION_BINARY),
        "--cells", ",".join(str(index) for index in cell_ids),
        "--warmup", str(warmup), "--launches", str(launches),
        "--hook-repeats", str(hook_repeats), "--run-id", str(run_id),
    ]


def run_baseline(
    run_dir: Path, cell_ids: tuple[int, ...], warmup: int, launches: int,
    hook_repeats: int, run_id: int,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "application.log"
    command = application_command(cell_ids, warmup, launches, hook_repeats, run_id)
    returncode = run_owned_to_log(
        command, cwd=HERE, environment=base_environment(),
        log_path=log_path, timeout=300,
    )
    if returncode != 0:
        raise RuntimeError(f"native application exited {returncode}")
    measurements = validate_application_events(
        json_events(log_path), cell_ids, warmup, launches, hook_repeats, run_id,
    )
    return {
        "valid": True, "arm": "baseline", "command": command,
        "measurements": measurements, "application_returncode": returncode,
    }


def attached_environment(build: Path, segment: str, agent_log: Path) -> tuple[dict[str, str], dict[str, str]]:
    common = {
        **base_environment(),
        "BPFTIME_GLOBAL_SHM_NAME": segment,
        "BPFTIME_MAP_GPU_THREAD_COUNT": str(MAX_THREADS),
        "BPFTIME_SHM_MEMORY_MB": "256",
        "BPFTIME_MAX_FD_COUNT": "1024",
        "BPFTIME_LOG_OUTPUT": "console",
        "SPDLOG_LEVEL": "info",
        "BPFTIME_SM_ARCH": "sm_120",
        "BPFTIME_VERIFIER_LEVEL": "WARNING",
        "CUDA_HOME": "/usr/local/cuda-12.9",
        "BPFTIME_CUDA_ROOT": "/usr/local/cuda-12.9",
    }
    loader = {
        **common,
        "LD_PRELOAD": str(build / "runtime/syscall-server/libbpftime-syscall-server.so"),
    }
    agent = {
        **common,
        "LD_PRELOAD": str(build / "runtime/agent/libbpftime-agent.so"),
        "BPFTIME_LOG_OUTPUT": str(agent_log),
        "BPFTIME_CUDA_DEFER_PTX_EXTRACTION": "1",
        "BPFTIME_CUDA_TARGETED_LATE_BOOTSTRAP": "1",
    }
    return loader, agent


def run_attached(
    mode: str, run_dir: Path, build: Path, cell_ids: tuple[int, ...],
    warmup: int, launches: int, hook_repeats: int, run_id: int,
) -> dict[str, Any]:
    if mode not in ("noop", "counter"):
        raise ValueError(mode)
    run_dir.mkdir(parents=True, exist_ok=False)
    segment = f"trampoline_scaling_{os.getpid()}_{time.monotonic_ns()}"
    segment_path = SHM_ROOT / segment
    if os.path.lexists(segment_path):
        raise RuntimeError("unique private shared-memory name already exists")
    loader_log = run_dir / "loader.log"
    application_log = run_dir / "application.log"
    agent_log = run_dir / "agent.log"
    loader_env, agent_env = attached_environment(build, segment, agent_log)
    object_path = HERE / ".output" / f"{BPF_OBJECT_PREFIX}-{mode}.bpf.o"
    loader_command = [
        str(LOADER_BINARY), str(object_path), mode,
        str(MAX_THREADS), "300",
    ]
    command = application_command(cell_ids, warmup, launches, hook_repeats, run_id)
    loader = target = None
    loader_stream = target_stream = None
    identity = None
    cleanup_errors: list[str] = []
    record: dict[str, Any] = {
        "valid": False, "arm": mode, "command": command,
        "loader_command": loader_command, "private_segment": segment,
    }
    try:
        loader_stream = loader_log.open("x")
        loader = subprocess.Popen(
            loader_command, cwd=HERE, env=loader_env, stdout=loader_stream,
            stderr=subprocess.STDOUT, text=True, start_new_session=True,
        )
        wait_ready(loader_log, loader)
        identity = segment_identity(segment_path)
        target_stream = application_log.open("x")
        target = subprocess.Popen(
            command, cwd=HERE, env=agent_env, stdout=target_stream,
            stderr=subprocess.STDOUT, text=True, start_new_session=True,
        )
        try:
            target_returncode = target.wait(timeout=300)
        except subprocess.TimeoutExpired as exc:
            stop_owned(target)
            raise RuntimeError("instrumented application exceeded 300 seconds") from exc
        target_stream.close()
        target_stream = None
        if target_returncode != 0:
            raise RuntimeError(f"instrumented application exited {target_returncode}")
        measurements = validate_application_events(
            json_events(application_log), cell_ids, warmup, launches,
            hook_repeats, run_id,
        )
        os.killpg(loader.pid, signal.SIGINT)
        try:
            loader_returncode = loader.wait(timeout=90)
        except subprocess.TimeoutExpired as exc:
            stop_owned(loader)
            raise RuntimeError("loader readback exceeded 90 seconds") from exc
        loader_stream.close()
        loader_stream = None
        if loader_returncode != 0:
            raise RuntimeError(f"loader exited {loader_returncode}")
        engagement = validate_loader_events(
            json_events(loader_log), mode, cell_ids, warmup, launches, hook_repeats,
        )
        agent_gate = validate_agent_log(application_log.read_text(errors="replace"))
        agent_bootstrap_gate = validate_agent_bootstrap_log(
            agent_log.read_text(errors="replace"), segment,
        )
        record.update(
            valid=True, measurements=measurements, engagement=engagement,
            agent_gate=agent_gate, agent_evidence_log=str(application_log),
            agent_bootstrap_gate=agent_bootstrap_gate,
            application_returncode=target_returncode,
            loader_returncode=loader_returncode,
        )
        return record
    finally:
        for process, role in ((target, "application"), (loader, "loader")):
            try:
                stop_owned(process)
            except BaseException as error:
                cleanup_errors.append(f"{role}: {error}")
        if target_stream is not None:
            target_stream.close()
        if loader_stream is not None:
            loader_stream.close()
        survivors = {
            process.pid: group_members(process.pid)
            for process in (target, loader) if process is not None and group_members(process.pid)
        }
        if survivors:
            cleanup_errors.append(f"owned process groups survived: {survivors}")
            record["private_segment_removed"] = False
        else:
            try:
                record["private_segment_removed"] = unlink_owned_segment(segment_path, identity)
            except BaseException as error:
                cleanup_errors.append(f"segment: {error}")
        record["owned_group_survivors"] = survivors
        if cleanup_errors:
            record.update(valid=False, cleanup_errors=cleanup_errors)
            raise RuntimeError("; ".join(cleanup_errors))


def phase_parameters(phase: str) -> dict[str, Any]:
    if phase == "preflight":
        return {"blocks": PREFLIGHT_BLOCKS, "cell_ids": PREFLIGHT_CELL_IDS,
                "warmup": PREFLIGHT_WARMUP, "launches": PREFLIGHT_LAUNCHES,
                "hook_repeats": PREFLIGHT_HOOK_REPEATS}
    if phase == "full":
        return {"blocks": FULL_BLOCKS, "cell_ids": FULL_CELL_IDS,
                "warmup": FULL_WARMUP, "launches": FULL_LAUNCHES,
                "hook_repeats": FULL_HOOK_REPEATS}
    raise ValueError(phase)


def frozen_schedule(phase: str) -> list[dict[str, Any]]:
    params = phase_parameters(phase)
    balanced_orders = None
    if BALANCE_ARM_ORDER and phase == "full":
        if params["blocks"] != 10 or len(ARMS) != 3:
            raise RuntimeError("balanced arm order is frozen for ten blocks and three arms")
        rng = random.Random(SEED + 20_000)
        labels = list(ARMS)
        rng.shuffle(labels)
        forward = [tuple(labels[index:] + labels[:index]) for index in range(3)]
        reverse_labels = [labels[0], labels[2], labels[1]]
        reverse = [
            tuple(reverse_labels[index:] + reverse_labels[:index])
            for index in range(3)
        ]
        final_order = list(labels)
        rng.shuffle(final_order)
        balanced_orders = [*forward, *reverse, *forward, tuple(final_order)]
        rng.shuffle(balanced_orders)
    schedule = []
    for block in range(params["blocks"]):
        if balanced_orders is None:
            arms = list(ARMS)
            random.Random(SEED + block).shuffle(arms)
        else:
            arms = list(balanced_orders[block])
        cell_ids = list(params["cell_ids"])
        if RANDOMIZE_CELL_ORDER:
            random.Random(SEED + 10_000 + block).shuffle(cell_ids)
        for order, arm in enumerate(arms):
            item = {"block": block, "order": order, "arm": arm,
                    "run_id": block}
            if RANDOMIZE_CELL_ORDER:
                item["cell_ids"] = cell_ids
            schedule.append(item)
    return schedule


def attempt_directory(output: Path, item: dict[str, Any]) -> Path:
    stem = f"block-{item['block'] + 1:02d}-order-{item['order'] + 1}-{item['arm']}"
    path = output / stem
    if not path.exists():
        return path
    attempt = 2
    while (output / f"{stem}-attempt-{attempt}").exists():
        attempt += 1
    return output / f"{stem}-attempt-{attempt}"


def defining_parameters(args: argparse.Namespace) -> dict[str, Any]:
    phase = phase_parameters(args.phase)
    return {
        "kind": EXPERIMENT_KIND,
        "phase": args.phase,
        "bpftime_root": str(args.bpftime_root.resolve()),
        "bpftime_build": str(args.bpftime_build.resolve()),
        "blocks": phase["blocks"],
        "cell_ids": list(phase["cell_ids"]),
        "warmup": phase["warmup"],
        "launches": phase["launches"],
        "hook_repeats": phase["hook_repeats"],
        "schedule_seed": SEED,
        "expected_driver": EXPECTED_DRIVER,
        "expected_gpu": EXPECTED_GPU,
        "matrix": [dict(cell) for cell in CELLS],
        "randomize_cell_order": RANDOMIZE_CELL_ORDER,
        "balance_arm_order": BALANCE_ARM_ORDER,
        "independent_raw_evidence": WRITE_INDEPENDENT_RAW_EVIDENCE,
    }


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize(records: list[dict[str, Any]], blocks: int) -> list[dict[str, Any]]:
    lookup: dict[tuple[int, str, int], float] = {}
    for record in records:
        if not record.get("valid"):
            continue
        for measurement in record["measurements"]:
            key = (record["block"], record["arm"], measurement["cell"])
            if key in lookup:
                raise RuntimeError(f"duplicate valid measurement: {key}")
            lookup[key] = float(measurement["elapsed_ms"])
    rows: list[dict[str, Any]] = []
    for cell in CELLS:
        if not any((block, "baseline", cell["id"]) in lookup for block in range(blocks)):
            continue
        for arm in ("noop", "counter"):
            pairs = []
            for block in range(blocks):
                native_key = (block, "baseline", cell["id"])
                attached_key = (block, arm, cell["id"])
                if native_key not in lookup or attached_key not in lookup:
                    raise RuntimeError(f"incomplete pair: block={block}, arm={arm}, cell={cell['id']}")
                native, attached = lookup[native_key], lookup[attached_key]
                pairs.append((attached - native, (attached / native - 1.0) * 100.0))
            rng = random.Random(SEED + cell["id"] * 10 + (0 if arm == "noop" else 1))
            bootstrap = []
            for _ in range(10_000 if len(pairs) > 1 else 1):
                sample = [pairs[rng.randrange(len(pairs))][1] for _ in pairs]
                bootstrap.append(statistics.median(sample))
            rows.append({
                "cell": cell["id"], "blocks": cell["blocks"],
                "threads_per_block": cell["threads_per_block"],
                "active_warps": cell["active_threads"] // 32,
                "arm": arm, "pairs": len(pairs),
                "median_delta_ms": statistics.median(value[0] for value in pairs),
                "median_overhead_pct": statistics.median(value[1] for value in pairs),
                "bootstrap_median_overhead_low_pct": percentile(bootstrap, 0.025),
                "bootstrap_median_overhead_high_pct": percentile(bootstrap, 0.975),
            })
    return rows


def write_summary(output: Path, result: dict[str, Any]) -> None:
    rows = summarize(result["records"], result["params"]["blocks"])
    result["summary"] = rows
    fields = list(rows[0]) if rows else []
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        f"# {SUMMARY_TITLE}", "",
        f"- Phase: `{result['params']['phase']}`",
        f"- Status: `{result['status']}`", "",
        "| Cell | Blocks | Threads/block | Active warps | Arm | Pairs | Median delta (ms) | Median overhead | 95% paired-bootstrap interval |",
        "|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['cell']} | {row['blocks']} | {row['threads_per_block']} | "
            f"{row['active_warps']} | {row['arm']} | "
            f"{row['pairs']} | {row['median_delta_ms']:.6f} | "
            f"{row['median_overhead_pct']:.3f}% | "
            f"[{row['bootstrap_median_overhead_low_pct']:.3f}%, "
            f"{row['bootstrap_median_overhead_high_pct']:.3f}%] |"
        )
    lines.extend([
        "", "Positive overhead means attached execution was slower than its paired native run.",
        "This experiment measures the current runtime; it does not assume once-per-warp dispatch.",
    ])
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def next_numbered_path(directory: Path, stem: str, suffix: str = "") -> Path:
    index = 1
    while (directory / f"{stem}-{index:02d}{suffix}").exists():
        index += 1
    return directory / f"{stem}-{index:02d}{suffix}"


def build_harness(args: argparse.Namespace, output: Path) -> Path:
    command = [
        "make", "-j4", f"BPFTIME_ROOT={args.bpftime_root}",
        f"BPFTIME_BUILD={args.bpftime_build}",
    ]
    log_path = next_numbered_path(output, "build", ".log")
    returncode = run_owned_to_log(
        command, cwd=HERE,
        environment={**base_environment(), "CC": "cc", "CLANG": "clang"},
        log_path=log_path, timeout=300,
    )
    if returncode != 0:
        raise RuntimeError(f"harness build failed ({returncode})")
    return log_path


def write_raw_arm_evidence(
    output: Path, run_dir: Path, item: dict[str, Any], cell_ids: tuple[int, ...],
    record: dict[str, Any], telemetry_path: Path,
    safety_before: dict[str, Any], safety_after: dict[str, Any],
) -> None:
    """Persist independently re-openable raw lifecycle and safety evidence."""
    if record.get("application_returncode") != 0:
        raise RuntimeError("refusing to admit nonzero application return code")
    attached = item["arm"] != "baseline"
    if attached and (
        record.get("loader_returncode") != 0
        or record.get("private_segment_removed") is not True
        or record.get("owned_group_survivors") != {}
    ):
        raise RuntimeError("refusing to admit incomplete attached-arm cleanup")
    try:
        telemetry_path.resolve().relative_to(output.resolve())
    except ValueError as error:
        raise RuntimeError("telemetry evidence escaped the campaign directory") from error
    relative_telemetry = os.path.relpath(telemetry_path, start=run_dir)
    atomic_write_json(run_dir / "safety-before.json", safety_before)
    atomic_write_json(run_dir / "safety-after.json", safety_after)
    lifecycle = {
        "schema": RAW_EVIDENCE_SCHEMA,
        "experiment_kind": EXPERIMENT_KIND,
        "block": item["block"],
        "order": item["order"],
        "arm": item["arm"],
        "run_id": item["run_id"],
        "cell_ids": list(cell_ids),
        "application_command": record["command"],
        "application_returncode": record["application_returncode"],
        "application_log": "application.log",
        "loader_command": record.get("loader_command"),
        "loader_returncode": record.get("loader_returncode"),
        "loader_log": "loader.log" if attached else None,
        "agent_log": "agent.log" if attached else None,
        "private_segment": record.get("private_segment"),
        "private_segment_removed": record.get("private_segment_removed") if attached else None,
        "owned_group_survivors": record.get("owned_group_survivors", {}),
        "telemetry_log": relative_telemetry,
        "safety_before": "safety-before.json",
        "safety_after": "safety-after.json",
    }
    atomic_write_json(run_dir / "lifecycle.json", lifecycle)


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    reject_ambient_injection()
    args.output = args.output.resolve()
    params = defining_parameters(args)
    state_path = args.output / "result.json"
    if args.resume:
        if not state_path.is_file():
            raise RuntimeError("--resume requires an existing result.json")
        result = json.loads(state_path.read_text())
        if result.get("params") != params or result.get("schedule") != frozen_schedule(args.phase):
            raise RuntimeError("resume parameters or frozen schedule changed")
        if result.get("status") == "complete":
            return result
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        result = {
            "kind": EXPERIMENT_KIND,
            "status": "preparing", "params": params,
            "schedule": frozen_schedule(args.phase), "records": [], "failures": [],
        }
        atomic_write_json(state_path, result)

    build_log = build_harness(args, args.output)
    current_evidence = {
        "runtime_configuration": runtime_configuration(args.bpftime_build),
        "compiled_hook_site": validate_compiled_hook_site(COMPILED_PTX),
        "runtime_source_audit": audit_runtime_source(args.bpftime_root),
        "loader_source_audit": audit_loader_source(),
        "source_manifest": source_manifest(args.bpftime_root),
    }
    if args.resume:
        for key, value in current_evidence.items():
            if result.get(key) != value:
                raise RuntimeError(f"resume rejected because {key} changed or is missing")
    else:
        result.update(current_evidence)
    result.setdefault("build_logs", []).append(str(build_log))
    atomic_write_json(state_path, result)

    completed = {
        (record["block"], record["arm"])
        for record in result["records"] if record.get("valid")
    }
    phase = phase_parameters(args.phase)
    before = None
    campaign_error = None
    with ReadOnlyLeases():
        try:
            before = safety.safety_snapshot()
            safety.validate_pre_server_safety(before)
            if before["gpu"]["driver"] != EXPECTED_DRIVER or before["gpu"]["name"] != EXPECTED_GPU:
                raise RuntimeError(f"frozen GPU/driver mismatch: {before['gpu']}")
            result.update(status="running", safety_before=before)
            if WRITE_INDEPENDENT_RAW_EVIDENCE:
                atomic_write_json(args.output / "safety-before.json", before)
            atomic_write_json(state_path, result)
            deadline = time.monotonic() + 3600
            for item in result["schedule"]:
                key = (item["block"], item["arm"])
                if key in completed:
                    continue
                if time.monotonic() >= deadline:
                    raise RuntimeError("one-hour campaign deadline reached")
                run_dir = attempt_directory(args.output, item)
                cell_ids = tuple(item.get("cell_ids", phase["cell_ids"]))
                telemetry_dir = next_numbered_path(args.output, "telemetry")
                telemetry_dir.mkdir()
                telemetry = telemetry_stream = telemetry_path = None
                try:
                    arm_before = before
                    if WRITE_INDEPENDENT_RAW_EVIDENCE:
                        arm_before = safety.safety_snapshot()
                        safety.validate_pre_server_safety(arm_before)
                        if (
                            arm_before["gpu"]["driver"] != EXPECTED_DRIVER
                            or arm_before["gpu"]["name"] != EXPECTED_GPU
                        ):
                            raise RuntimeError(
                                f"frozen GPU/driver mismatch before arm: {arm_before['gpu']}"
                            )
                    if arm_before is None:
                        raise RuntimeError("missing pre-arm safety snapshot")
                    record = telemetry_summary = primary_error = None
                    telemetry_errors = []
                    try:
                        telemetry, telemetry_stream, telemetry_path = (
                            safety.start_gpu_telemetry(telemetry_dir)
                        )
                        if item["arm"] == "baseline":
                            record = run_baseline(
                                run_dir, cell_ids, phase["warmup"], phase["launches"],
                                phase["hook_repeats"], item["run_id"],
                            )
                        else:
                            record = run_attached(
                                item["arm"], run_dir, args.bpftime_build, cell_ids,
                                phase["warmup"], phase["launches"], phase["hook_repeats"],
                                item["run_id"],
                            )
                    except BaseException as error:
                        primary_error = error
                    finally:
                        try:
                            if telemetry is not None:
                                stop_owned(telemetry)
                        except BaseException as error:
                            telemetry_errors.append(
                                f"stop: {type(error).__name__}: {error}"
                            )
                        try:
                            if telemetry_stream is not None:
                                telemetry_stream.close()
                        except BaseException as error:
                            telemetry_errors.append(
                                f"stream close: {type(error).__name__}: {error}"
                            )
                        try:
                            if telemetry_path is not None:
                                telemetry_summary = safety.validate_gpu_telemetry(
                                    telemetry_path, allow_fixed_power_cap=True,
                                )
                        except BaseException as error:
                            telemetry_errors.append(
                                f"validation: {type(error).__name__}: {error}"
                            )
                    if primary_error is not None or telemetry_errors:
                        errors = []
                        if primary_error is not None:
                            errors.append(
                                f"arm: {type(primary_error).__name__}: {primary_error}"
                            )
                        errors.extend(f"telemetry {error}" for error in telemetry_errors)
                        raise RuntimeError("; ".join(errors)) from primary_error
                    if record is None or telemetry_path is None or telemetry_summary is None:
                        raise RuntimeError("arm telemetry lifecycle completed without evidence")
                    record.update(block=item["block"], order=item["order"], directory=str(run_dir))
                    record["telemetry"] = {
                        "path": str(telemetry_path), "summary": telemetry_summary,
                    }
                    record["safety_after"] = safety.wait_for_post_server_safety(
                        arm_before, timeout=POST_RUN_SETTLE_TIMEOUT_SECONDS,
                    )
                    if WRITE_INDEPENDENT_RAW_EVIDENCE:
                        write_raw_arm_evidence(
                            args.output, run_dir, item, cell_ids, record,
                            telemetry_path, arm_before, record["safety_after"],
                        )
                    result["records"].append(record)
                    completed.add(key)
                    result["last_safety"] = record["safety_after"]
                    atomic_write_json(state_path, result)
                except BaseException as error:
                    failure = {
                        **item, "directory": str(run_dir),
                        "telemetry_directory": str(telemetry_dir),
                        "error": f"{type(error).__name__}: {error}",
                    }
                    if telemetry_path is not None:
                        failure["telemetry_path"] = str(telemetry_path)
                    result["failures"].append(failure)
                    result["status"] = "failed"
                    atomic_write_json(state_path, result)
                    raise
            result["status"] = "complete"
        except BaseException as error:
            campaign_error = error
            result.update(status="failed", campaign_error=f"{type(error).__name__}: {error}")
            raise
        finally:
            cleanup_errors = []
            try:
                if before is not None:
                    result["safety_after"] = safety.wait_for_post_server_safety(
                        before, timeout=POST_RUN_SETTLE_TIMEOUT_SECONDS,
                    )
                    if WRITE_INDEPENDENT_RAW_EVIDENCE:
                        atomic_write_json(
                            args.output / "safety-final.json", result["safety_after"],
                        )
            except BaseException as error:
                cleanup_errors.append(f"final safety: {type(error).__name__}: {error}")
            if cleanup_errors:
                result.update(status="failed", cleanup_errors=cleanup_errors)
            if result.get("status") == "complete":
                try:
                    write_summary(args.output, result)
                except BaseException as error:
                    result.update(status="failed", summary_error=f"{type(error).__name__}: {error}")
                    cleanup_errors.append(f"summary: {error}")
            atomic_write_json(state_path, result)
            if cleanup_errors:
                message = "; ".join(cleanup_errors)
                if campaign_error is not None:
                    message = (
                        f"campaign: {type(campaign_error).__name__}: {campaign_error}; "
                        f"finalization: {message}"
                    )
                raise RuntimeError(message) from campaign_error
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=EXPERIMENT_KIND)
    parser.add_argument("--phase", choices=("preflight", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bpftime-root", type=Path, default=DEFAULT_BPFTIME_ROOT)
    parser.add_argument("--bpftime-build", type=Path, default=DEFAULT_BPFTIME_BUILD)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_campaign(args)
    print(json.dumps({"status": result["status"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
