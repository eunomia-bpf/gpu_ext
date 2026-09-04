#!/usr/bin/env python3
"""Run the frozen CPU-only device-verifier scaling experiment."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any


RESULT_SCHEMA = "device-verifier-scaling-run-v1"
PROBE_SCHEMA = "device-verifier-scaling-probe-v1"
SEED = 1797
CPU = 23
BLOCKS = 20
TIMEOUT_SECONDS = 60
SIZES = (16, 64, 256, 1024, 4096)
FAMILIES = ("linear", "diamonds")
ARMS = tuple((family, size) for size in SIZES for family in FAMILIES)
SECTION = "cuda__verifier_scaling"


class RunFailure(RuntimeError):
    """A fail-closed experiment error."""


def exact_json_equal(actual: Any, expected: Any) -> bool:
    """Compare JSON-like values without Python's bool/int equivalence."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            exact_json_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            exact_json_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def expected_shape(family: str, size: int, mode: str) -> dict[str, Any]:
    if family not in FAMILIES or size not in SIZES:
        raise ValueError("arm is outside the frozen matrix")
    branches = (size - 4) // 2 if family == "diamonds" else 0
    return {
        "schema": PROBE_SCHEMA,
        "mode": mode,
        "family": family,
        "requested_instructions": size,
        "instruction_count": size,
        "conditional_branches": branches,
        "helper_calls": 1,
        "exits": 1,
        "minimum_branch_offset": 1 if family == "diamonds" else None,
        "maximum_branch_offset": 1 if family == "diamonds" else None,
        "section": SECTION,
    }


def _next_prng(state: int) -> int:
    return (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)


def frozen_schedule(blocks: int = BLOCKS) -> list[dict[str, Any]]:
    if blocks != BLOCKS:
        raise ValueError(f"full schedule requires exactly {BLOCKS} blocks")
    state = SEED
    schedule: list[dict[str, Any]] = []
    sequence = 0
    for block in range(1, blocks + 1):
        ordered = list(ARMS)
        for index in range(len(ordered) - 1, 0, -1):
            state = _next_prng(state)
            swap = state % (index + 1)
            ordered[index], ordered[swap] = ordered[swap], ordered[index]
        for position, (family, size) in enumerate(ordered, start=1):
            sequence += 1
            schedule.append(
                {
                    "sequence": sequence,
                    "block": block,
                    "position": position,
                    "family": family,
                    "instructions": size,
                }
            )
    return schedule


def preflight_schedule() -> list[dict[str, Any]]:
    return [
        {
            "sequence": 1,
            "block": 1,
            "position": 1,
            "family": "linear",
            "instructions": 16,
        },
        {
            "sequence": 2,
            "block": 1,
            "position": 2,
            "family": "diamonds",
            "instructions": 4096,
        },
    ]


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_one_json_line(text: str) -> dict[str, Any]:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RunFailure(f"probe emitted {len(lines)} non-empty stdout lines")
    try:
        value = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise RunFailure(f"probe stdout is not JSON: {error}") from error
    if not isinstance(value, dict):
        raise RunFailure("probe JSON is not an object")
    return value


def validate_probe_record(
    record: dict[str, Any], family: str, size: int, mode: str
) -> None:
    expected = expected_shape(family, size, mode)
    for key, value in expected.items():
        if not exact_json_equal(record.get(key), value):
            raise RunFailure(f"probe {key} mismatch: {record.get(key)!r} != {value!r}")
    if record.get("build_type") != "Release":
        raise RunFailure("probe was not built in Release mode")
    revision = record.get("bpftime_source_revision")
    if not isinstance(revision, str) or not revision:
        raise RunFailure("probe lacks its bpftime source Git revision")

    if mode == "describe":
        if record.get("accepted") is not None or record.get("error") != "":
            raise RunFailure("describe record contains an admission decision")
        timed_keys = (
            "elapsed_ns",
            "process_cpu_ns",
            "cpu_before",
            "cpu_after",
            "minor_faults",
            "major_faults",
            "voluntary_context_switches",
            "involuntary_context_switches",
        )
        if any(record.get(key) is not None for key in timed_keys):
            raise RunFailure("describe record contains timing diagnostics")
        return

    if record.get("accepted") is not True or record.get("error") != "":
        raise RunFailure("safe program was not accepted")
    if mode == "accept_only":
        timed_keys = (
            "elapsed_ns",
            "process_cpu_ns",
            "cpu_before",
            "cpu_after",
            "minor_faults",
            "major_faults",
            "voluntary_context_switches",
            "involuntary_context_switches",
        )
        if any(record.get(key) is not None for key in timed_keys):
            raise RunFailure("warmup unexpectedly emitted timing")
        return

    integer_keys = (
        "elapsed_ns",
        "process_cpu_ns",
        "minor_faults",
        "major_faults",
        "voluntary_context_switches",
        "involuntary_context_switches",
    )
    for key in integer_keys:
        value = record.get(key)
        if type(value) is not int or value < 0:
            raise RunFailure(f"timed record has invalid {key}")
    if record["elapsed_ns"] <= 0 or record["process_cpu_ns"] <= 0:
        raise RunFailure("timed record contains a non-positive clock interval")
    if record.get("cpu_before") != CPU or record.get("cpu_after") != CPU:
        raise RunFailure("probe did not stay on frozen CPU")


def _read_first(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _git_revision(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    if not revision:
        raise RunFailure("empty bpftime Git revision")
    return revision


def _git_verifier_status(root: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--", "bpftime-verifier"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in completed.stdout.splitlines() if line]


def _cmake_cache(build_dir: Path) -> dict[str, str]:
    cache_path = build_dir / "CMakeCache.txt"
    if not cache_path.is_file():
        raise RunFailure(f"missing isolated CMake cache: {cache_path}")
    values: dict[str, str] = {}
    for line in cache_path.read_text().splitlines():
        if line.startswith("//") or line.startswith("#") or "=" not in line:
            continue
        left, value = line.split("=", 1)
        key = left.split(":", 1)[0]
        values[key] = value
    return values


def _command_first_line(argv: list[str]) -> str:
    completed = subprocess.run(argv, check=True, capture_output=True, text=True)
    lines = completed.stdout.splitlines()
    if not lines:
        raise RunFailure(f"command emitted no version line: {argv}")
    return lines[0]


def collect_environment(probe: Path, bpftime_root: Path) -> dict[str, Any]:
    affinity = sorted(os.sched_getaffinity(0))
    if affinity != [CPU]:
        raise RunFailure(f"runner affinity must be exactly CPU {CPU}: {affinity}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RunFailure("CUDA_VISIBLE_DEVICES must be set to the empty string")
    if os.environ.get("LD_PRELOAD"):
        raise RunFailure("LD_PRELOAD must be unset")
    if not probe.is_file() or not os.access(probe, os.X_OK):
        raise RunFailure(f"probe is missing or not executable: {probe}")
    if not (bpftime_root / "bpftime-verifier/src/gpu/gpu_verifier.cpp").is_file():
        raise RunFailure("bpftime root lacks the GPU verifier source")

    build_dir = probe.parent
    cache = _cmake_cache(build_dir)
    if cache.get("CMAKE_BUILD_TYPE") != "Release":
        raise RunFailure("isolated build is not Release")
    if Path(cache.get("BPFTIME_ROOT", "")).resolve() != bpftime_root.resolve():
        raise RunFailure("isolated build points at a different bpftime root")

    cpu_model = None
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break
    cpu_base = Path(f"/sys/devices/system/cpu/cpu{CPU}/cpufreq")
    stat = probe.stat()
    compiler = cache.get("CMAKE_CXX_COMPILER")
    if not compiler:
        raise RunFailure("isolated build lacks CMAKE_CXX_COMPILER")
    return {
        "captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "python": platform.python_version(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "runner_affinity": affinity,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ld_preload": os.environ.get("LD_PRELOAD"),
        "cpufreq": {
            "driver": _read_first(cpu_base / "scaling_driver"),
            "governor": _read_first(cpu_base / "scaling_governor"),
            "energy_performance_preference": _read_first(
                cpu_base / "energy_performance_preference"
            ),
        },
        "probe": {
            "path": str(probe),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "cmake_build_type": cache.get("CMAKE_BUILD_TYPE"),
            "cmake_bpftime_root": cache.get("BPFTIME_ROOT"),
        },
        "bpftime_root": str(bpftime_root),
        "bpftime_current_revision": _git_revision(bpftime_root),
        "bpftime_verifier_status": _git_verifier_status(bpftime_root),
        "compiler": {
            "path": compiler,
            "version_line": _command_first_line([compiler, "--version"]),
        },
        "cmake_version_line": _command_first_line(["cmake", "--version"]),
    }


def collect_end_environment(probe: Path, bpftime_root: Path) -> dict[str, Any]:
    cpu_base = Path(f"/sys/devices/system/cpu/cpu{CPU}/cpufreq")
    stat = probe.stat()
    return {
        "captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runner_affinity": sorted(os.sched_getaffinity(0)),
        "cpufreq": {
            "driver": _read_first(cpu_base / "scaling_driver"),
            "governor": _read_first(cpu_base / "scaling_governor"),
            "energy_performance_preference": _read_first(
                cpu_base / "energy_performance_preference"
            ),
        },
        "probe": {
            "path": str(probe),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "bpftime_current_revision": _git_revision(bpftime_root),
        "bpftime_verifier_status": _git_verifier_status(bpftime_root),
    }


def execute_probe(
    probe: Path,
    output_dir: Path,
    family: str,
    size: int,
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if mode == "describe":
        argv = [
            str(probe),
            "--describe",
            "--family",
            family,
            "--instructions",
            str(size),
        ]
    elif mode == "accept_only":
        argv = [
            str(probe),
            "--accept-only",
            "--family",
            family,
            "--instructions",
            str(size),
            "--require-cpu",
            str(CPU),
        ]
    elif mode == "timed":
        argv = [
            str(probe),
            "--family",
            family,
            "--instructions",
            str(size),
            "--require-cpu",
            str(CPU),
        ]
    else:
        raise ValueError(f"unknown probe mode: {mode}")

    output_dir.mkdir(parents=True, exist_ok=False)
    child_environment = dict(os.environ)
    child_environment["CUDA_VISIBLE_DEVICES"] = ""
    child_environment.pop("LD_PRELOAD", None)
    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    started_ns = time.monotonic_ns()
    timed_out = False
    try:
        completed = subprocess.run(
            argv,
            cwd=output_dir,
            env=child_environment,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
        returncode: int | None = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        returncode = None
        stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or "")
    duration_ns = time.monotonic_ns() - started_ns
    (output_dir / "stdout.log").write_text(stdout)
    (output_dir / "stderr.log").write_text(stderr)
    execution = {
        "argv": argv,
        "cwd": str(output_dir),
        "started_utc": started_utc,
        "duration_ns": duration_ns,
        "timeout_seconds": TIMEOUT_SECONDS,
        "timed_out": timed_out,
        "returncode": returncode,
        "environment": {
            "CUDA_VISIBLE_DEVICES": child_environment.get("CUDA_VISIBLE_DEVICES"),
            "LD_PRELOAD": child_environment.get("LD_PRELOAD"),
        },
    }
    write_json(output_dir / "execution.json", execution)
    if timed_out:
        raise RunFailure(f"probe timed out for {family}/{size}")
    if returncode != 0:
        raise RunFailure(f"probe returned {returncode} for {family}/{size}")
    if stderr:
        raise RunFailure(f"probe emitted stderr for {family}/{size}")
    record = read_one_json_line(stdout)
    validate_probe_record(record, family, size, mode)
    return execution, record


def run(args: argparse.Namespace) -> int:
    if args.dry_run:
        print(json.dumps(frozen_schedule(), indent=2))
        return 0

    probe = args.probe.resolve()
    bpftime_root = args.bpftime_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RunFailure(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    mode = "preflight" if args.preflight else "full"
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "running",
        "mode": mode,
        "seed": SEED,
        "cpu": CPU,
        "blocks": 1 if args.preflight else BLOCKS,
        "sizes": list(SIZES),
        "families": list(FAMILIES),
        "timeout_seconds": TIMEOUT_SECONDS,
        "environment": None,
        "end_environment": None,
        "descriptions": [],
        "warmups": [],
        "cells": [],
        "error": None,
    }
    write_json(output_dir / "result.json", result)

    try:
        result["environment"] = collect_environment(probe, bpftime_root)
        describe_arms = (
            (("linear", 16), ("diamonds", 4096))
            if args.preflight
            else ARMS
        )
        probe_revisions: set[str] = set()
        for family, size in describe_arms:
            relative = Path("descriptions") / f"{family}-{size}"
            _, record = execute_probe(
                probe, output_dir / relative, family, size, "describe"
            )
            probe_revisions.add(record["bpftime_source_revision"])
            result["descriptions"].append(
                {"family": family, "instructions": size, "directory": str(relative)}
            )
            write_json(output_dir / "result.json", result)
        if len(probe_revisions) != 1:
            raise RunFailure("probe descriptions report inconsistent source revisions")
        result["probe_source_revision"] = next(iter(probe_revisions))
        if result["probe_source_revision"] != result["environment"]["bpftime_current_revision"]:
            raise RunFailure("probe was not built from the current bpftime revision")
        if result["environment"]["bpftime_verifier_status"]:
            raise RunFailure("bpftime-verifier has tracked working-tree changes")

        if not args.preflight:
            for family, size in ARMS:
                relative = Path("warmups") / f"{family}-{size}"
                _, record = execute_probe(
                    probe, output_dir / relative, family, size, "accept_only"
                )
                if record["bpftime_source_revision"] != result["probe_source_revision"]:
                    raise RunFailure("warmup probe source revision changed")
                result["warmups"].append(
                    {"family": family, "instructions": size, "directory": str(relative)}
                )
                write_json(output_dir / "result.json", result)

        schedule = preflight_schedule() if args.preflight else frozen_schedule()
        for item in schedule:
            family = item["family"]
            size = item["instructions"]
            relative = Path("cells") / (
                f"seq-{item['sequence']:03d}-block-{item['block']:02d}-"
                f"pos-{item['position']:02d}-{family}-{size}"
            )
            cell = dict(item)
            cell["directory"] = str(relative)
            cell["valid"] = False
            result["cells"].append(cell)
            write_json(output_dir / "result.json", result)
            _, record = execute_probe(
                probe, output_dir / relative, family, size, "timed"
            )
            if record["bpftime_source_revision"] != result["probe_source_revision"]:
                raise RunFailure("timed probe source revision changed")
            cell["valid"] = True
            write_json(output_dir / "result.json", result)

        result["end_environment"] = collect_end_environment(probe, bpftime_root)
        if result["end_environment"]["runner_affinity"] != [CPU]:
            raise RunFailure("runner affinity changed during run")
        if result["end_environment"]["cpufreq"] != result["environment"]["cpufreq"]:
            raise RunFailure("CPU frequency policy changed during run")
        if (
            result["end_environment"]["bpftime_current_revision"]
            != result["probe_source_revision"]
        ):
            raise RunFailure("bpftime revision changed during run")
        if result["end_environment"]["bpftime_verifier_status"]:
            raise RunFailure("bpftime-verifier changed during run")
        for key in ("path", "size", "mtime_ns"):
            if (
                result["end_environment"]["probe"][key]
                != result["environment"]["probe"][key]
            ):
                raise RunFailure(f"probe {key} changed during run")
        result["status"] = "complete"
        write_json(output_dir / "result.json", result)
        return 0
    except Exception as error:  # preserve every fail-closed partial result
        result["status"] = "invalid"
        result["error"] = f"{type(error).__name__}: {error}"
        write_json(output_dir / "result.json", result)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--bpftime-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.dry_run and any(
        value is None for value in (args.probe, args.bpftime_root, args.output_dir)
    ):
        parser.error("--probe, --bpftime-root, and --output-dir are required")
    if args.dry_run and args.preflight:
        parser.error("--dry-run and --preflight are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (RunFailure, OSError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
