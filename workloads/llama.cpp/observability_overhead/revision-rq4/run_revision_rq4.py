#!/usr/bin/env python3
"""Run the matched RTX 5090 gpubpf/NVBit observability experiment."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import csv
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


class OwnedCleanupError(RuntimeError):
    """Unsafe to continue the campaign; an owned resource may still be live."""

    def __init__(self, message: str, details: dict[str, Any]):
        super().__init__(message)
        self.details = details


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
                  *, diagnostic_log_level: str | None = None):
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


def patch_launchlate_clock(directory: Path) -> None:
    """Patch only the per-run copy; do not change the shared bpftime checkout."""
    replacements = {
        "launchlate.bpf.c": [
            ("__uint(max_entries, 4);", "__uint(max_entries, 5);"),
            ("u64 delta_ns = gpu_ts > *launch_ts ? gpu_ts - *launch_ts : 0;",
             "if (gpu_ts < *launch_ts) {\n\t\tu32 error_key = 4;\n"
             "\t\tu64 *errors = bpf_map_lookup_elem(&queue_state, &error_key);\n"
             "\t\tif (errors)\n\t\t\t__sync_fetch_and_add(errors, 1);\n\t\treturn 0;\n\t}\n"
             "\tu64 delta_ns = gpu_ts - *launch_ts;"),
        ],
        "launchlate.c": [
            ("uint64_t queue_values[4] = {0};", "uint64_t queue_values[5] = {0};"),
            ("for (i = 0; i < 4; i++) {", "for (i = 0; i < 5; i++) {"),
            ('printf("Queue overflows: %" PRIu64 "\\n", queue_values[3]);',
             'printf("Queue overflows: %" PRIu64 "\\n", queue_values[3]);\n'
             '\tprintf("Clock errors: %" PRIu64 "\\n", queue_values[4]);'),
        ],
    }
    updated = {}
    for name, edits in replacements.items():
        text = (directory / name).read_text()
        for before, after in edits:
            if text.count(before) != 1:
                raise RuntimeError(f"private launchlate clock patch does not match {name}")
            text = text.replace(before, after)
        updated[name] = text
    for name, text in updated.items():
        (directory / name).write_text(text)


def parse_gpubpf(tool: str, text: str) -> dict[str, Any]:
    result = core.parse_probe_samples(tool, text)
    if tool == "threadhist":
        for key, label in (("configured_entries", "Configured thread entries"),
                           ("readback_entries", "Readback entries"),
                           ("readback_bytes", "Readback bytes"),
                           ("readback_complete", "Readback complete")):
            values = re.findall(rf"^{label}:\s*(\d+)$", text, re.MULTILINE)
            result[key] = int(values[-1]) if values else -1
    if tool == "launchlate":
        for key, label in (("clock_errors", "Clock errors"), ("queue_underflows", "Queue underflows"),
                           ("queue_overflows", "Queue overflows")):
            values = re.findall(rf"^{label}:\s*(\d+)$", text, re.MULTILINE)
            result[key] = int(values[-1]) if values else -1
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
        NVBIT_SOURCE_DIR / "common.h",
        NVBIT_SOURCE_DIR / "inject_funcs.cu",
        NVBIT_SOURCE_DIR / "observability.cu",
        NVBIT_SOURCE_DIR / "tool_func/flush_channel.cu",
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
    def last(pattern: str, default: int = 0) -> int:
        values = [int(value) for value in re.findall(pattern, text)]
        return values[-1] if values else default

    selected = last(r"NVBIT selected_launches=(\d+)")
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
    samples = last(r"NVBIT launchlate samples=(\d+)")
    errors = last(r"NVBIT launchlate samples=\d+ clock_errors=(\d+)", -1)
    bins = [last(rf"NVBIT launchlate bin_{index}=(\d+)") for index in range(10)]
    return {
        "sample_count": samples,
        "clock_errors": errors,
        "histogram": bins,
        "histogram_sum": sum(bins),
        "selected_launches": selected,
    }


def nvbit_probe_valid(tool: str, probe: dict[str, Any]) -> bool:
    samples = int(probe.get("sample_count", 0))
    selected = int(probe.get("selected_launches", 0))
    if samples <= 0 or selected <= 0:
        return False
    if tool == "kernelretsnoop":
        return int(probe.get("nonzero_timestamps", 0)) == samples
    if tool == "threadhist":
        return int(probe.get("nonzero_threads", 0)) > 0
    return (
        int(probe.get("clock_errors", -1)) == 0
        and int(probe.get("histogram_sum", -1)) == samples
        and selected == samples
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
                      expected_thread_count: int | None = None) -> bool:
    samples = int(probe.get("sample_count", 0))
    if samples <= 0:
        return False
    if tool == "kernelretsnoop":
        return int(probe.get("nonzero_timestamps", 0)) == samples
    if tool == "threadhist":
        return (int(probe.get("nonzero_threads", 0)) > 0
                and expected_thread_count is not None and expected_thread_count > 0
                and probe.get("configured_entries") == expected_thread_count
                and probe.get("readback_entries") == expected_thread_count
                and probe.get("readback_bytes") == expected_thread_count * 8
                and probe.get("readback_complete") == 1)
    return (
        int(probe.get("clock_errors", -1)) == 0
        and int(probe.get("queue_underflows", -1)) == 0
        and int(probe.get("queue_overflows", -1)) == 0
        and int(probe.get("host_launches", -1))
        == int(probe.get("device_entries", -2))
        == samples
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
                                          diagnostic_log_level=diagnostic_log_level)
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
        "valid": completed.returncode == 0 and bool(output),
        "safety": safety_record,
    }
    if tool is not None:
        if config.startswith("gpubpf_"):
            probe_log = run_dir / "probe.log"
            probe_text = probe_log.read_text(errors="replace") if probe_log.exists() else ""
            result["probe"] = parse_gpubpf(tool, probe_text)
            result["valid"] = bool(result["valid"]) and gpubpf_probe_valid(
                tool, result["probe"], expected_thread_count=args.threadhist_gpu_thread_count
            )
        else:
            result["probe"] = parse_nvbit(tool, completed.stderr)
            result["valid"] = bool(result["valid"]) and nvbit_probe_valid(
                tool, result["probe"]
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
        with private_probe(tool, args, tool_dirs[tool], run_dir) as env:
            result = run_bench(tool, run_id, args, output_dir, env_extra=env)
        result["probe"] = parse_gpubpf(tool, (run_dir / "probe.log").read_text(errors="replace"))
        result["probe_log"] = str((run_dir / "probe.log").relative_to(output_dir))
        result["valid"] = bool(result.get("valid")) and gpubpf_probe_valid(
            tool, result["probe"], expected_thread_count=args.threadhist_gpu_thread_count)
        return result
    return run_nvbit_once(tool, run_id, args, output_dir)


def validate(args: argparse.Namespace) -> None:
    core.validate(args)
    if not args.llama_cli.exists():
        raise FileNotFoundError(args.llama_cli)
    if not NVBIT_ROOT.exists():
        raise FileNotFoundError(NVBIT_ROOT)
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
    parser.add_argument("--bpftime-root", type=Path, default=core.DEFAULT_BPFTIME_ROOT)
    parser.add_argument(
        "--bpftime-build-dir",
        type=Path,
        default=Path("/home/yunwei37/workspace/gpu/bpftime/build-cuda-pr503"),
    )
    parser.add_argument("--target-symbol", default=core.DEFAULT_TARGET_SYMBOL)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--pp", type=int)
    parser.add_argument("--tg", type=int, default=0)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--probe-startup-s", type=float, default=3.0)
    parser.add_argument("--gpu-thread-count", type=int, default=8192)
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

    lease = shared.Leases()
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
        args.nvbit_tool = build_nvbit(nvbit_build_dir, output_dir)

        build_root = output_dir / "gpubpf_tool_build"
        build_root.mkdir(exist_ok=True)
        tool_dirs = {}
        for tool in TASKS:
            directory = core.prepare_tool_source(
                core.TOOLS[tool],
                bpftime_root=args.bpftime_root,
                build_root=build_root,
                target_symbol=args.target_symbol,
            )
            if tool == "launchlate":
                patch_launchlate_clock(directory)
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
