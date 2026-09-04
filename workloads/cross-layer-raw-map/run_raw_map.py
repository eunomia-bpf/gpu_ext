#!/usr/bin/env python3
"""Run a fail-closed RTX 5090 raw-record/aggregate cross-layer study."""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any

import protocol

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_RUNTIME = ROOT.parent / "bpftime-table1-575/build-table1-575"
LEASE_PATHS = (
    Path("/tmp/gpubpf-revision-gpu0.lock"),
    Path("/tmp/gpubpf-revision-struct-ops.lock"),
)

SAFETY_SPEC = importlib.util.spec_from_file_location(
    "raw_map_safety", ROOT / "workloads/moe-infinity/run_moe_head_to_head.py"
)
assert SAFETY_SPEC and SAFETY_SPEC.loader
safety = importlib.util.module_from_spec(SAFETY_SPEC)
sys.modules[SAFETY_SPEC.name] = safety
SAFETY_SPEC.loader.exec_module(safety)


class ReadOnlyLeases:
    """Lock exact pre-created regular files without creating or writing them."""

    def __init__(self, paths=LEASE_PATHS):
        self.streams = []
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
                    if ((opened.st_dev, opened.st_ino) != identity
                            or (current.st_dev, current.st_ino) != identity
                            or not stat.S_ISREG(opened.st_mode)
                            or not stat.S_ISREG(current.st_mode)):
                        raise RuntimeError(f"lease identity changed while opening: {path}")
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.streams.append(stream)
                except BaseException:
                    stream.close()
                    raise
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        for stream in reversed(self.streams):
            stream.close()
        self.streams.clear()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def runtime_configuration(build: Path) -> dict[str, str]:
    cache = build / "CMakeCache.txt"
    if not cache.is_file():
        raise RuntimeError(f"runtime cache is missing: {cache}")
    values: dict[str, str] = {}
    for line in cache.read_text().splitlines():
        if "=" in line and ":" in line.partition("=")[0]:
            key, _, value = line.partition("=")
            values[key.partition(":")[0]] = value
    required = {
        "BPFTIME_ENABLE_CUDA_ATTACH": "ON",
        "BPFTIME_LLVM_JIT": "ON",
    }
    for key, expected in required.items():
        if values.get(key, "").upper() != expected:
            raise RuntimeError(f"runtime requires {key}={expected}, got {values.get(key)}")
    artifacts = (
        build / "runtime/syscall-server/libbpftime-syscall-server.so",
        build / "runtime/agent/libbpftime-agent.so",
        HERE / ".output/raw_map_probe",
        HERE / ".output/raw_map.bpf.o",
        HERE / ".output/raw_map_target",
    )
    missing = [str(path) for path in artifacts if not path.is_file()]
    if missing:
        raise RuntimeError(f"required built artifacts are missing: {missing}")
    return {
        "BPFTIME_ENABLE_CUDA_ATTACH": values["BPFTIME_ENABLE_CUDA_ATTACH"],
        "BPFTIME_LLVM_JIT": values["BPFTIME_LLVM_JIT"],
        "ENABLE_EBPF_VERIFIER": values.get("ENABLE_EBPF_VERIFIER", "unknown"),
        "CMAKE_HOME_DIRECTORY": values.get("CMAKE_HOME_DIRECTORY", "unknown"),
    }


def group_members(pgid: int) -> list[int]:
    members = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if fields[0] != "Z" and int(fields[2]) == pgid and int(fields[3]) == pgid:
                members.append(int(path.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    return members


def stop_owned(process: subprocess.Popen[Any]) -> None:
    for sig, seconds in ((signal.SIGINT, 5), (signal.SIGTERM, 5), (signal.SIGKILL, 3)):
        process.poll()
        if not group_members(process.pid):
            process.wait(timeout=1)
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            continue
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            process.poll()
            if not group_members(process.pid):
                process.wait(timeout=1)
                return
            time.sleep(0.05)
    raise RuntimeError(f"owned process group {process.pid} survived cleanup")


def segment_identity(path: Path) -> tuple[int, int, int]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise RuntimeError(f"shared segment is not an owned regular file: {path}")
    return info.st_dev, info.st_ino, info.st_uid


def process_holds_segment(process: subprocess.Popen[Any],
                          identity: tuple[int, int, int]) -> bool:
    """Confirm the live child has the exact segment open or mapped."""
    device, inode, _uid = identity
    proc = Path("/proc") / str(process.pid)
    for descriptor in (proc / "fd").glob("*"):
        try:
            info = descriptor.stat()
        except OSError:
            continue
        if (info.st_dev, info.st_ino) == (device, inode):
            return True
    try:
        mappings = (proc / "maps").read_text()
    except OSError:
        return False
    expected_device = f"{os.major(device):02x}:{os.minor(device):02x}"
    for line in mappings.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) >= 5 and fields[3] == expected_device:
            try:
                if int(fields[4]) == inode:
                    return True
            except ValueError:
                continue
    return False


def wait_owned_segment(process: subprocess.Popen[Any], path: Path,
                       timeout: float = 20) -> tuple[int, int, int]:
    """Capture identity only while the exact segment is held by our child."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            identity = segment_identity(path)
        except FileNotFoundError:
            identity = None
        if identity is not None and process_holds_segment(process, identity):
            return identity
        if process.poll() is not None:
            raise RuntimeError(
                "probe exited before shared-segment ownership was confirmed"
            )
        time.sleep(0.01)
    raise RuntimeError("shared-segment ownership confirmation timed out")


def unlink_owned_segment(path: Path, identity: tuple[int, int, int] | None) -> None:
    try:
        actual = segment_identity(path)
    except FileNotFoundError:
        return
    if identity is None or actual != identity:
        raise RuntimeError(f"retaining unknown or replaced shared segment: {path}")
    path.unlink()


def wait_ready(process: subprocess.Popen[Any], log: Path, stderr_log: Path | None = None,
               timeout: float = 20) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        events = protocol.json_events(log) if log.exists() else []
        ready = [event for event in events if event.get("event") == "ready"]
        if ready:
            if len(ready) != 1:
                raise RuntimeError("probe emitted multiple ready records")
            return ready[0]
        if process.poll() is not None:
            tail = log.read_text(errors="replace")[-4000:] if log.exists() else ""
            diagnostic = (stderr_log.read_text(errors="replace")[-4000:]
                          if stderr_log is not None and stderr_log.exists() else "")
            raise RuntimeError(
                f"probe exited before ready; stdout tail: {tail}; stderr tail: {diagnostic}"
            )
        time.sleep(0.05)
    raise RuntimeError("probe readiness timed out")


def controlled_environment() -> dict[str, str]:
    return {
        "PATH": "/usr/local/cuda-12.9/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "CUDA_VISIBLE_DEVICES": "0",
        "LD_LIBRARY_PATH": "/usr/local/cuda-12.9/lib64",
    }


def run_process(log: Path, argv: list[str], env: dict[str, str],
                processes: list[subprocess.Popen[Any]], streams: list[Any],
                stderr_log: Path | None = None) -> subprocess.Popen[Any]:
    stream = log.open("x")
    streams.append(stream)
    error_stream = stderr_log.open("x") if stderr_log is not None else stream
    if error_stream is not stream:
        streams.append(error_stream)
    process = subprocess.Popen(
        argv, cwd=log.parent, env=env, stdout=stream, stderr=error_stream,
        start_new_session=True,
    )
    processes.append(process)
    return process


def run_cell(directory: Path, arm: protocol.Arm, runtime_build: Path,
             safety_before: dict[str, Any]) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "schema": protocol.SCHEMA,
        "protocol": protocol.PROTOCOL,
        "status": "running",
        "arm": arm.name,
        "threads": arm.threads,
        "launches": arm.launches,
        "expect_drop_rejection": arm.expect_drop_rejection,
        "started_ns": time.time_ns(),
    }
    processes: list[subprocess.Popen[Any]] = []
    streams: list[Any] = []
    segment = f"gpubpf_raw_map_{os.getpid()}_{time.time_ns()}"
    segment_path = Path("/dev/shm") / segment
    identity = None
    if os.path.lexists(segment_path):
        raise RuntimeError("unique private shared-memory segment already exists")

    env = controlled_environment()
    target = str(HERE / ".output/raw_map_target")
    target_args = [target, str(arm.threads), str(arm.launches)]
    server_so = str(runtime_build / "runtime/syscall-server/libbpftime-syscall-server.so")
    agent_so = str(runtime_build / "runtime/agent/libbpftime-agent.so")
    common = {
        **env,
        "BPFTIME_GLOBAL_SHM_NAME": segment,
        "BPFTIME_MAP_GPU_THREAD_COUNT": str(arm.threads),
        "BPFTIME_SHM_MEMORY_MB": "128",
        "BPFTIME_MAX_FD_COUNT": "1024",
        "BPFTIME_LOG_OUTPUT": "console",
        "SPDLOG_LEVEL": "info",
        "BPFTIME_SM_ARCH": "sm_120",
        "CUDA_HOME": "/usr/local/cuda-12.9",
        "BPFTIME_CUDA_ROOT": "/usr/local/cuda-12.9",
    }
    try:
        native = run_process(
            directory / "native.log", target_args, env, processes, streams,
            stderr_log=directory / "native.stderr.log",
        )
        if native.wait(timeout=30) != 0:
            raise RuntimeError("native CUDA truth process failed")

        probe_args = [
            str(HERE / ".output/raw_map_probe"),
            str(HERE / ".output/raw_map.bpf.o"),
            str(arm.threads),
            str(protocol.BLOCK_DIM),
            str(arm.launches),
        ]
        probe_stderr = directory / "probe.stderr.log"
        probe = run_process(
            directory / "probe.log", probe_args, {**common, "LD_PRELOAD": server_so},
            processes, streams, stderr_log=probe_stderr,
        )
        identity = wait_owned_segment(probe, segment_path)
        ready = wait_ready(probe, directory / "probe.log", probe_stderr)
        protocol.require_fields(ready, {
            "thread_slots": arm.threads,
            "threads_per_block": protocol.BLOCK_DIM,
            "launches": arm.launches,
            "ring_capacity_per_thread": protocol.RING_CAPACITY,
        }, "live.ready")
        instrumented = run_process(
            directory / "instrumented.log", target_args,
            {**common, "LD_PRELOAD": agent_so}, processes, streams,
            stderr_log=directory / "instrumented.stderr.log",
        )
        if instrumented.wait(timeout=60) != 0:
            raise RuntimeError("instrumented CUDA truth process failed")
        probe.send_signal(signal.SIGUSR1)
        if probe.wait(timeout=30) != 0:
            raise RuntimeError("raw-record probe failed its internal accounting")

        validation = protocol.validate_cell_logs(
            directory / "native.log", directory / "instrumented.log",
            directory / "probe.log", arm,
        )
        result.update(status="passed", validation=validation, **validation)
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        cleanup_errors = []
        for process in reversed(processes):
            try:
                stop_owned(process)
            except BaseException as exc:
                cleanup_errors.append(str(exc))
        for stream in streams:
            stream.close()
        survivors = {process.pid: members for process in processes
                     if (members := group_members(process.pid))}
        result["owned_group_survivors"] = survivors
        try:
            if survivors:
                raise RuntimeError(f"owned process groups survived: {survivors}")
            unlink_owned_segment(segment_path, identity)
            result["private_segment_removed"] = not os.path.lexists(segment_path)
            result["safety_after"] = safety.wait_for_post_server_safety(safety_before)
        except BaseException as exc:
            cleanup_errors.append(str(exc))
        result["cleanup_errors"] = cleanup_errors
        result["finished_ns"] = time.time_ns()
        if cleanup_errors:
            result["status"] = "failed"
        atomic_json(directory / "cell.json", result)
        if cleanup_errors:
            raise RuntimeError("; ".join(cleanup_errors))
    return result


def run_campaign(mode: str, output: Path, runtime_build: Path,
                 preflight: Path | None) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"output directory already exists: {output}")
    if mode == "full":
        if preflight is None:
            raise RuntimeError("formal run requires --preflight")
        protocol.validate_preflight_manifest(preflight)
    config = runtime_configuration(runtime_build)
    leases = ReadOnlyLeases()
    manifest: dict[str, Any] = {
        "schema": protocol.SCHEMA,
        "protocol": protocol.PROTOCOL,
        "mode": mode,
        "status": "running",
        "seed": protocol.SEED,
        "runtime_build": str(runtime_build),
        "runtime_configuration": config,
        "preflight": str(preflight) if preflight else None,
        "cells": [],
        "started_ns": time.time_ns(),
    }
    try:
        output.mkdir(parents=True, exist_ok=False)
        before = safety.safety_snapshot()
        safety.validate_pre_server_safety(before)
        gpu = before["gpu"]
        if gpu["driver"] != "575.57.08" or "RTX 5090" not in gpu["name"]:
            raise RuntimeError(f"campaign is fixed to RTX 5090 / 575.57.08, got {gpu}")
        manifest["safety_before"] = before
        for item in protocol.campaign_order(mode):
            arm = protocol.ARM_BY_NAME[item["name"]]
            cell_name = (
                f"block-{item['block']:02d}-order-{item['order']:02d}-{arm.name}"
            )
            cell = run_cell(output / cell_name, arm, runtime_build, before)
            manifest["cells"].append({
                "block": item["block"],
                "order": item["order"],
                "directory": cell_name,
                **cell["validation"],
            })
            atomic_json(output / "manifest.json", manifest)
        manifest.update(
            status="passed",
            cell_count=len(manifest["cells"]),
            positive_cells=sum(
                cell["evidence_disposition"] == "accepted_complete_raw_stream"
                for cell in manifest["cells"]
            ),
            negative_drop_gates=sum(
                cell["evidence_disposition"] == "rejected_incomplete_raw_stream"
                for cell in manifest["cells"]
            ),
            safety_after=safety.wait_for_post_server_safety(before),
            finished_ns=time.time_ns(),
            claim_boundary=(
                "Functional raw-record expressibility and exact host readback only; no "
                "latency, bandwidth, shared-memory-shard, automatic-placement, strict-"
                "verifier, or arbitrary-data-structure claim."
            ),
        )
        expected_cells = protocol.blocks_for(mode) * len(protocol.ARMS)
        if (manifest["cell_count"] != expected_cells
                or manifest["positive_cells"] != protocol.blocks_for(mode) * 2
                or manifest["negative_drop_gates"] != protocol.blocks_for(mode)):
            raise RuntimeError("campaign completion counts are inconsistent")
        atomic_json(output / "manifest.json", manifest)
        return manifest
    except BaseException as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                        finished_ns=time.time_ns())
        if output.exists():
            atomic_json(output / "manifest.json", manifest)
        raise
    finally:
        leases.close()


def main() -> int:
    def interrupted(signum, _frame):
        raise InterruptedError(f"signal {signum}; clean owned raw-map processes")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("dry-run", "preflight", "full"))
    parser.add_argument("--plan-mode", choices=("preflight", "full"), default="full",
                        help="matrix printed by dry-run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-build", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--preflight", type=Path)
    args = parser.parse_args()
    if args.phase == "dry-run":
        plan = protocol.dry_run_plan(
            args.plan_mode, args.output, args.runtime_build, args.preflight
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    result = run_campaign(
        args.phase, args.output.absolute(), args.runtime_build.absolute(),
        args.preflight.absolute() if args.preflight else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
