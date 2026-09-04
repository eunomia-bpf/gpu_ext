#!/usr/bin/env python3
"""Run the matched RTX 5090 gpubpf/NVBit observability experiment."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import csv
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
SCHEDULE_SEED = 1797
BOOTSTRAP_SAMPLES = 10000
EXPECTED_DRIVER = "575.57.08"
SHM_ROOT = Path("/dev/shm")
CLIENT_CPUS = "8-15"
EXPECTED_GPU_THREAD_SLOTS = 22528
MIN_RING_ENTRIES_PER_THREAD = 256
EXIT_RECORD_BYTES = 80
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
EXPECTED_NORMALIZED_STDOUT = "Deterministic tests are essential\n> EOF by user"
EXPECTED_NORMALIZED_STDOUT_BYTES = 47
LEASE_PATHS = (
    Path("/tmp/gpubpf-revision-gpu0.lock"),
    Path("/tmp/gpubpf-revision-struct-ops.lock"),
)
RELATIVE_RUNTIME_INCLUDE = "../../../runtime/include"
RELATIVE_RUNTIME_INCLUDE_PATTERN = re.compile(r"(?:\.\./)+runtime/include")


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
        process, stream, path = shared.safety.start_gpu_telemetry(directory)
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
    if tool == "kernelretsnoop":
        exit_environment = {
            "BPFTIME_MAP_GPU_THREAD_COUNT": str(args.gpu_thread_count),
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
                        raise RuntimeError("private probe did not exit cleanly")
            except OwnedCleanupError as error:
                record["cleanup_error"] = str(error)
                raise
            finally:
                (run_dir / "probe-execution.json").write_text(json.dumps(record, indent=2) + "\n")


def validate_launchlate_source_schema(directory: Path) -> None:
    """Reject a stale launchlate copy instead of rewriting it at run time."""
    required = {
        "launchlate.bpf.c": (
            "BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP",
            "LAUNCHLATE_TARGET_SYMBOL",
            "MATCHED_SAMPLES",
            "UNCERTAIN_SAMPLES",
            "gpu_entry_ns",
        ),
        "launchlate.c": (
            "affine CLOCK_MONOTONIC interpolation",
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


def validate_nvbit_launchlate_source_schema(directory: Path) -> None:
    """Require the fail-closed raw-pair affine ABI in the copied NVBit tool."""
    required = {
        "Makefile": (
            "observability.o: observability.cu common.h clock_domain.h",
            "inject_funcs.o: inject_funcs.cu common.h clock_domain.h",
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
            "minimum_end_calibration_deadline",
            "affine_clock_offset_interval",
            "classify_affine_launch_latency",
            "if (latency_high_ns < 0)",
        ),
        "common.h": (
            "LAUNCH_PAIR_CAPACITY",
            "struct launch_pair_t",
            "uint64_t host_mono_ns;",
            "uint64_t gpu_entry_ns;",
            "uint64_t sequence;",
        ),
        "inject_funcs.cu": (
            "pair_ptr",
            "device_entry_count_ptr",
            "capture_error_count_ptr",
            "pair->gpu_entry_ns = gpu_ns",
            "__threadfence_system()",
        ),
        "observability.cu": (
            "bracketed_globaltimer_endpoints_against_CLOCK_MONOTONIC_",
            "with_affine_interpolation_and_drift_bound",
            "cudaMallocManaged(\n            &state->launch_pairs",
            "nvbit_set_at_launch(ctx, func, pair_ptr)",
            "state->launch_pair_overflows++",
            "print_launchlate_results(state, end_calibration)",
            "classify_affine_launch_latency(",
            "accounting_complete=",
            "calibrate_gpu_clock(ctx, state, state->start_calibration)",
            "calibrate_gpu_clock(ctx, state, &end_calibration)",
            "clock_calibration_drift(",
            "wait_for_minimum_clock_span(",
            'print_clock_calibration("start"',
            'print_clock_calibration("end"',
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
            "pair_capacity=",
            "stored_pairs=",
            "device_entries=",
            "pair_overflows=",
            "capture_errors=",
            "selected_counter_overflow=",
            "uncertain_samples=",
            "samples=%llu clock_errors=%llu",
        ),
        "tool_func/flush_channel.cu": (
            "sample_globaltimer",
            "%globaltimer",
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
    """Require the per-launch block-dimension record ABI used by the oracle."""
    required = {
        "kernelretsnoop.bpf.c": (
            "u64 block_dim_x, block_dim_y, block_dim_z;",
            "bpf_get_block_dim(&data.block_dim_x",
            "sizeof(struct data)",
        ),
        "kernelretsnoop.c": (
            "uint64_t block_dim_x, block_dim_y, block_dim_z;",
            "event_coordinate(&state->events[i]",
            "Invalid launch coordinates:",
            "sizeof(struct data)",
        ),
    }
    for name, markers in required.items():
        text = (directory / name).read_text()
        missing = [marker for marker in markers if marker not in text]
        if missing:
            raise RuntimeError(
                f"kernelretsnoop source lacks the block-dimension record ABI in {name}: "
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
        signed = {
            "start_clock_offset_lower_ns", "start_clock_offset_upper_ns",
            "end_clock_offset_lower_ns", "end_clock_offset_upper_ns",
            "clock_offset_change_lower_ns", "clock_offset_change_upper_ns",
        }
        for key, label in labels.items():
            unit = (r"\s+ppb" if key.endswith("_ppb") else
                    r"\s+ns" if key.endswith("_ns") else "")
            number = r"(-?\d+)" if key in signed else r"(\d+)"
            values = re.findall(
                rf"^{re.escape(label)}:\s*{number}{unit}$", text, re.MULTILINE
            )
            result[key] = int(values[-1]) if values else -1
        methods = re.findall(
            r"^Clock calibration method:\s*(.+)$", text, re.MULTILINE
        )
        result["clock_calibration_method"] = methods[-1].strip() if methods else ""
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
        OBS_ROOT / "run_observability_overhead.py",
        Path(shared.__file__),
        Path(shared.run_smoke.__file__),
        Path(shared.safety.__file__),
        NVBIT_SOURCE_DIR / "Makefile",
        NVBIT_SOURCE_DIR / "clock_domain.h",
        NVBIT_SOURCE_DIR / "common.h",
        NVBIT_SOURCE_DIR / "inject_funcs.cu",
        NVBIT_SOURCE_DIR / "observability.cu",
        NVBIT_SOURCE_DIR / "tool_func/flush_channel.cu",
        args.bpftime_root / "runtime/include/bpf_attach_ctx.hpp",
        args.bpftime_root / "runtime/include/bpftime_gpu_ringbuf.h",
        args.bpftime_root / "runtime/src/bpf_map/gpu/nv_gpu_ringbuf_map.cpp",
        args.bpftime_root / "runtime/src/bpf_map/gpu/nv_gpu_ringbuf_map.hpp",
        args.bpftime_root / "runtime/syscall-server/syscall_server_main.cpp",
        args.bpftime_root / "attach/nv_attach_impl/trampoline/default_trampoline.cu",
    ]
    for tool in TASKS:
        spec = core.TOOLS[tool]
        paths.extend(
            [
                args.bpftime_root / spec.example_dir / "Makefile",
                args.bpftime_root / spec.example_dir / spec.bpf_file,
                args.bpftime_root / spec.example_dir / spec.user_file,
            ]
        )
    return {str(path): file_metadata(path) for path in paths}


def defining_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "phase": args.phase,
        "model": str(args.model),
        "llama_bench": str(args.llama_bench),
        "llama_cli": str(args.llama_cli),
        "bpftime_root": str(args.bpftime_root),
        "bpftime_build_dir": str(args.bpftime_build_dir),
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
        return {
            "sample_count": events,
            "nonzero_timestamps": nonzero,
            "selected_launches": selected,
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
        "end_clock_offset_lower_ns": r"-?\d+",
        "end_clock_offset_upper_ns": r"-?\d+",
        "end_clock_uncertainty_ns": r"\d+",
        "end_clock_host_anchor_ns": r"\d+",
        "end_clock_calibration_valid": r"\d+",
        "clock_offset_change_lower_ns": r"-?\d+",
        "clock_offset_change_upper_ns": r"-?\d+",
        "clock_calibration_elapsed_ns": r"\d+",
        "clock_drift_rate_bound_ppb": r"\d+",
        "clock_drift_limit_ppb": r"\d+",
        "clock_drift_bounded": r"\d+",
    }
    for key, number in integer_fields.items():
        values = re.findall(
            rf"^NVBIT launchlate {key}=({number})$",
            text,
            re.MULTILINE,
        )
        result[key] = int(values[-1]) if values else None
    methods = re.findall(
        r"^NVBIT launchlate clock_calibration_method=(\S+)$",
        text,
        re.MULTILINE,
    )
    result["clock_calibration_method"] = methods[-1] if methods else ""
    result["calibration_blocks"] = len(methods)
    return result


def launch_clock_model_valid(probe: dict[str, Any]) -> bool:
    fields = (
        "start_clock_offset_lower_ns", "start_clock_offset_upper_ns",
        "start_clock_uncertainty_ns", "start_clock_host_anchor_ns",
        "end_clock_offset_lower_ns", "end_clock_offset_upper_ns",
        "end_clock_uncertainty_ns", "end_clock_host_anchor_ns",
        "clock_offset_change_lower_ns", "clock_offset_change_upper_ns",
        "clock_calibration_elapsed_ns", "clock_drift_rate_bound_ppb",
        "clock_drift_limit_ppb", "clock_drift_bounded",
    )
    if any(type(probe.get(key)) is not int for key in fields):
        return False
    start_low = probe["start_clock_offset_lower_ns"]
    start_high = probe["start_clock_offset_upper_ns"]
    end_low = probe["end_clock_offset_lower_ns"]
    end_high = probe["end_clock_offset_upper_ns"]
    start_anchor = probe["start_clock_host_anchor_ns"]
    end_anchor = probe["end_clock_host_anchor_ns"]
    elapsed = end_anchor - start_anchor
    change_low = end_low - start_high
    change_high = end_high - start_low
    if elapsed <= 0:
        return False
    expected_rate = (
        max(abs(change_low), abs(change_high)) * 1_000_000_000 + elapsed - 1
    ) // elapsed
    return (
        start_low <= start_high
        and end_low <= end_high
        and probe["start_clock_uncertainty_ns"]
        == (start_high - start_low + 1) // 2
        and probe["end_clock_uncertainty_ns"]
        == (end_high - end_low + 1) // 2
        and probe["clock_offset_change_lower_ns"] == change_low
        and probe["clock_offset_change_upper_ns"] == change_high
        and probe["clock_calibration_elapsed_ns"] == elapsed
        and elapsed >= LAUNCH_MIN_CALIBRATION_SPAN_NS
        and probe["clock_drift_rate_bound_ppb"] == expected_rate
        and probe["clock_drift_limit_ppb"] == LAUNCH_CLOCK_DRIFT_LIMIT_PPB
        and expected_rate <= LAUNCH_CLOCK_DRIFT_LIMIT_PPB
        and probe["clock_drift_bounded"] == 1
    )


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
                      expected_exit_launches: int | None = None) -> bool:
    samples = int(probe.get("sample_count", 0))
    selected = int(probe.get("selected_launches", 0))
    if samples <= 0 or selected <= 0:
        return False
    if tool == "kernelretsnoop":
        return (
            int(probe.get("nonzero_timestamps", 0)) == samples
            and (expected_exit_events is None or samples == expected_exit_events)
            and (expected_exit_launches is None or selected == expected_exit_launches)
        )
    if tool == "threadhist":
        return int(probe.get("nonzero_threads", 0)) > 0

    calibration_fields = (
        "uncertain_samples", "start_clock_calibration_valid",
        "end_clock_calibration_valid", "pair_capacity", "stored_pairs",
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
        == "bracketed_globaltimer_endpoints_against_CLOCK_MONOTONIC_with_affine_interpolation_and_drift_bound"
        and int(probe.get("start_clock_calibration_valid", -1)) == 1
        and int(probe.get("end_clock_calibration_valid", -1)) == 1
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
    )


def run_nvbit_once(
    tool: str,
    run_id: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
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
                else args.gpu_thread_count
            ),
        },
    )
    log_path = output_dir / result["log"]
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    result["probe"] = parse_nvbit(tool, text)
    result["valid"] = bool(result.get("valid")) and nvbit_probe_valid(
        tool, result["probe"]
    )
    return result


def gpubpf_probe_valid(tool: str, probe: dict[str, Any], *,
                      expected_thread_count: int | None = None,
                      expected_exit_events: int | None = None,
                      expected_exit_launches: int | None = None,
                      expected_exit_coordinates: int | None = None,
                      exact_exit_oracle: bool = False) -> bool:
    samples = int(probe.get("sample_count", 0))
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
            and int(probe.get("entries_per_thread", -1)) >= MIN_RING_ENTRIES_PER_THREAD
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
            and coordinates == unique_coordinates == multiplicity_coordinates
            and int(probe.get("oracle_enabled", -1)) == int(exact_exit_oracle)
            and int(probe.get("oracle_total_events", -1)) == samples
            and int(probe.get("oracle_passed", -1)) == int(exact_exit_oracle)
            and (expected_exit_events is None or samples == expected_exit_events)
            and (expected_exit_launches is None or launches == expected_exit_launches)
            and (expected_exit_coordinates is None or coordinates == expected_exit_coordinates)
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

    calibration_valid = (
        probe.get("clock_calibration_method")
        == "bracketed %globaltimer endpoint intervals with affine CLOCK_MONOTONIC interpolation"
        and launch_clock_model_valid(probe)
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
            result["valid"] = bool(result["valid"]) and gpubpf_probe_valid(
                tool,
                result["probe"],
                expected_thread_count=(
                    args.gpu_thread_count if tool == "kernelretsnoop"
                    else args.threadhist_gpu_thread_count
                ),
                expected_exit_events=(
                    CORRECTNESS_EXIT_EVENTS if tool == "kernelretsnoop" else None
                ),
                expected_exit_launches=(
                    CORRECTNESS_EXIT_LAUNCHES if tool == "kernelretsnoop" else None
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
                    CORRECTNESS_EXIT_LAUNCHES if tool == "kernelretsnoop" else None
                ),
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
    config_rows: list[dict[str, Any]] = []
    for config in CONFIGS:
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
    for task in TASKS:
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


def new_state(args: argparse.Namespace, timestamp: str, snapshot: dict[str, Any]) -> dict[str, Any]:
    schedules = {}
    for block in range(1, args.runs + 1):
        order = list(CONFIGS)
        random.Random(SCHEDULE_SEED + block).shuffle(order)
        schedules[str(block)] = order
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
        "schedule": schedules,
        "correctness": {config: {"attempts": []} for config in CONFIGS},
        "artifacts": {},
        "configs": {config: {"runs": []} for config in CONFIGS},
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
    if state.get("params") != defining_params(args):
        raise RuntimeError("resume parameters differ from the recorded experiment")
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
    if set(tool_dirs) != set(TASKS) or not hasattr(args, "nvbit_tool"):
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
        run_dir = output_dir / f"{tool}_run_{run_id:02d}"
        with private_probe(
            tool, args, tool_dirs[tool], run_dir, exact_exit_oracle=False
        ) as env:
            result = run_bench(tool, run_id, args, output_dir, env_extra=env)
        result["probe"] = parse_gpubpf(tool, (run_dir / "probe.log").read_text(errors="replace"))
        result["probe_log"] = str((run_dir / "probe.log").relative_to(output_dir))
        result["valid"] = bool(result.get("valid")) and gpubpf_probe_valid(
            tool,
            result["probe"],
            expected_thread_count=(
                args.gpu_thread_count if tool == "kernelretsnoop"
                else args.threadhist_gpu_thread_count
            ),
            exact_exit_oracle=False,
        )
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
    matched = (
        gpubpf_probe.get("sample_count") == nvbit_probe.get("sample_count")
        and gpubpf_probe.get("cartesian_launches")
        == nvbit_probe.get("selected_launches")
    )
    comparison = {
        "matched": matched,
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


def validate(args: argparse.Namespace) -> None:
    core.validate(args)
    if not args.llama_cli.exists():
        raise FileNotFoundError(args.llama_cli)
    if not NVBIT_ROOT.exists():
        raise FileNotFoundError(NVBIT_ROOT)
    if ("kernelretsnoop" in args.tools
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "full"), default="preflight")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", type=Path, default=core.DEFAULT_MODEL)
    parser.add_argument("--llama-bench", type=Path, default=core.DEFAULT_LLAMA_BENCH)
    parser.add_argument("--llama-cli", type=Path)
    parser.add_argument("--bpftime-root", type=Path, required=True)
    parser.add_argument(
        "--bpftime-build-dir",
        type=Path,
        required=True,
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
    args = parser.parse_args()

    args.runs = args.runs if args.runs is not None else (1 if args.phase == "preflight" else 10)
    args.pp = args.pp if args.pp is not None else (32 if args.phase == "preflight" else 512)
    args.llama_cli = args.llama_cli or (args.llama_bench.parent / "llama-cli")
    for field in ("model", "llama_bench", "llama_cli", "bpftime_root", "bpftime_build_dir", "uprobe_binary"):
        setattr(args, field, getattr(args, field).resolve())
    args.tools = list(TASKS)
    reject_ambient_injection()
    validate(args)

    lease = ReadOnlyLeases()
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
        lease.close()


def run_campaign(args: argparse.Namespace) -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir or HERE / "raw" / f"{args.phase}-{timestamp}").resolve()
    if args.resume:
        if not (output_dir / "result.json").exists():
            raise RuntimeError("--resume requires an existing result.json")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("refusing to reuse a nonempty output directory without --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = core.nvidia_smi_snapshot()

    admission = {
        "timestamp": timestamp,
        "phase": args.phase,
        "nvidia_smi": snapshot,
        "driver": parse_driver(snapshot),
        "expected_driver": EXPECTED_DRIVER,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "nvbit_driver_supported": nvbit_driver_supported(parse_driver(snapshot)),
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
        tool_dirs = verify_resume(state, args, snapshot)
    else:
        nvbit_build_dir = output_dir / "nvbit_tool_build"
        shutil.copytree(
            NVBIT_SOURCE_DIR,
            nvbit_build_dir,
            ignore=shutil.ignore_patterns("*.o", "*.so", "*.fatbin", "flush_channel.c"),
        )
        validate_nvbit_launchlate_source_schema(nvbit_build_dir)
        args.nvbit_tool = build_nvbit(nvbit_build_dir, output_dir)

        build_root = output_dir / "gpubpf_tool_build"
        build_root.mkdir(exist_ok=True)
        tool_dirs = {}
        for tool in TASKS:
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

        state = new_state(args, timestamp, snapshot)
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

    if any(valid_correctness(state, config) is None for config in CONFIGS):
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
            write_state(output_dir, state)

    write_state(output_dir, state)
    print((output_dir / "summary.md").read_text(encoding="utf-8"), flush=True)
    if any(
        valid_run_for_block(state, config, block) is None
        for block in range(1, args.runs + 1)
        for config in CONFIGS
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
