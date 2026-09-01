#!/usr/bin/env python3
"""Run the approved four-cell expert-policy correctness/lifecycle gate."""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import os
import re
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
GPU_EXT = HERE.parents[1]
LLAMA_ROOT = GPU_EXT / "workloads/llama.cpp"
LLAMA_SERVER = LLAMA_ROOT / "build/bin/llama-server"
LIBGGML_BASE = LLAMA_ROOT / "build/bin/libggml-base.so"
MOE = GPU_EXT / "workloads/moe-infinity"
MODEL = (
    MOE
    / "deps/hf-cache/hub/models--ggml-org--gpt-oss-120b-GGUF/snapshots"
    / "238abdd290bb874b90a5da1b4549881b7d05c091"
    / "gpt-oss-120b-MXFP4.gguf"
)
PROMPTS = MOE / "prompts.json"
SCHEDULE = HERE / "correctness-schedule.json"
TRACE = GPU_EXT / "extension/expert_buffering_trace"
POLICY = GPU_EXT / "extension/expert_buffering_policy"
COMPILE_LAYOUT = HERE / "compile_layout.py"
COMPILE_HOT_SET = HERE / "compile_hot_set.py"
HOT_SET = HERE / "calibration-hot-set.txt"
EVICTION_MONITOR = HERE / "expert_eviction_monitor"
CUSTOM_UVM = GPU_EXT.parent / "gpu_ext-kernel-610/kernel-open/nvidia-uvm.ko"
RAW = HERE / "raw/correctness"
CONFIGS = (
    "plain_uvm",
    "gpubpf_observe",
    "gpubpf_profile_protect",
    "llama_ncmoe32",
)
POLICY_STAT_KEYS = (
    "activate", "mapped", "hot_tail", "cold_head", "shared_tail",
    "default", "setter_failure", "access", "cold_native",
    "hot_access_tail", "shared_access_tail", "observe_activate",
    "observe_access",
)
EVICTION_KEYS = (
    "evictions", "evicted_bytes", "default_bytes", "cold_bytes",
    "hot_bytes", "shared_bytes", "dropped_evictions",
)


class GateError(RuntimeError):
    pass


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def run_checked(argv: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode:
        raise GateError(
            f"command failed ({result.returncode}): {argv!r}\n{result.stderr[-4000:]}"
        )
    return result.stdout.strip()


def json_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("event"), str):
            events.append(value)
    return events


def wait_event(process: subprocess.Popen[Any], path: Path, event: str,
               timeout: float = 30) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for value in reversed(json_events(path)):
            if value.get("event") == event:
                return value
        if process.poll() is not None:
            tail = path.read_text(errors="replace")[-4000:] if path.exists() else ""
            raise GateError(f"process exited before {event}: {tail}")
        time.sleep(0.1)
    raise GateError(f"timed out waiting for {event}")


def latest_event(path: Path, event: str) -> dict[str, Any]:
    for value in reversed(json_events(path)):
        if value.get("event") == event:
            return value
    raise GateError(f"no {event} record in {path}")


def counter_delta(before: dict[str, Any], after: dict[str, Any],
                  keys: tuple[str, ...]) -> dict[str, int]:
    result = {}
    for key in keys:
        start = int(before[key])
        end = int(after[key])
        if end < start:
            raise GateError(f"counter {key} decreased: {start} -> {end}")
        result[key] = end - start
    return result


def stop_group(process: subprocess.Popen[Any] | None, timeout: float = 30) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def module_refcount() -> int:
    for line in Path("/proc/modules").read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == "nvidia_uvm":
            return int(fields[2])
    return -1


def struct_ops_maps() -> list[dict[str, Any]]:
    raw = run_checked(["sudo", "-n", "bpftool", "map", "show", "-j"])
    return [item for item in json.loads(raw or "[]") if item.get("type") == "struct_ops"]


def gpu_state() -> dict[str, Any]:
    row = run_checked([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]).splitlines()
    if len(row) != 1:
        raise GateError(f"expected one GPU, found {len(row)}")
    fields = [item.strip() for item in row[0].split(",")]
    apps_raw = run_checked([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ])
    return {
        "name": fields[0], "driver": fields[1],
        "memory_used_mib": int(fields[2]), "utilization_percent": int(fields[3]),
        "compute_apps": apps_raw.splitlines() if apps_raw else [],
    }


def require_idle() -> dict[str, Any]:
    state = gpu_state()
    errors = []
    if state["name"] != "NVIDIA GeForce RTX 5090":
        errors.append(f"unexpected GPU: {state['name']}")
    if state["driver"] != "610.43.02":
        errors.append(f"unexpected driver: {state['driver']}")
    if state["compute_apps"]:
        errors.append(f"foreign compute apps: {state['compute_apps']}")
    if state["memory_used_mib"] > 256:
        errors.append(f"residual GPU memory: {state['memory_used_mib']} MiB")
    if struct_ops_maps():
        errors.append("pre-existing struct_ops map")
    if module_refcount() != 0:
        errors.append(f"nvidia_uvm refcount is {module_refcount()}")
    if errors:
        raise GateError("admission refused: " + "; ".join(errors))
    return state


def load_custom_uvm() -> None:
    require_idle()
    run_checked(["sudo", "-n", "rmmod", "nvidia_uvm"])
    run_checked(["sudo", "-n", "insmod", str(CUSTOM_UVM)])
    if not Path("/sys/kernel/btf/nvidia_uvm").is_file():
        raise GateError("custom nvidia_uvm did not expose module BTF")


def restore_stock_uvm() -> None:
    if module_refcount() != 0:
        raise GateError(f"cannot restore stock UVM with refcount {module_refcount()}")
    if struct_ops_maps():
        raise GateError("cannot restore stock UVM with struct_ops registered")
    run_checked(["sudo", "-n", "rmmod", "nvidia_uvm"])
    run_checked(["sudo", "-n", "modprobe", "nvidia_uvm"])
    if Path("/sys/kernel/btf/nvidia_uvm").exists():
        raise GateError("distribution nvidia_uvm restoration was not proven")


def controlled_environment(config: str) -> dict[str, str]:
    env = os.environ.copy()
    for key in ("PYTHONPATH", "LD_PRELOAD", "GGML_CUDA_ENABLE_UNIFIED_MEMORY",
                "GPUBPF_EXPERT_LAYOUT_TRACE", "GPUBPF_EXPERT_ROUTE_TRACE"):
        env.pop(key, None)
    env.update(
        CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        GPUBPF_EXPERT_LAYOUT_TRACE="1",
    )
    if config != "llama_ncmoe32":
        env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"
    else:
        env["GPUBPF_EXPERT_ROUTE_TRACE"] = "1"
    return env


def server_command(config: str, port: int) -> list[str]:
    command = [
        "taskset", "-c", "0-7", str(LLAMA_SERVER), "--model", str(MODEL),
        "--alias", "gpt-oss-120b", "--host", "127.0.0.1", "--port", str(port),
        "--n-gpu-layers", "99", "--parallel", "1", "--ctx-size", "4096",
        "--threads", "8", "--threads-batch", "8", "--cache-ram", "0",
        "--flash-attn", "on", "--no-warmup", "--timeout", "600",
    ]
    if config == "llama_ncmoe32":
        command.extend(["--n-cpu-moe", "32"])
    return command


def wait_ready(process: subprocess.Popen[Any], port: int, log_path: Path) -> None:
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise GateError(
                f"server exited before ready: {log_path.read_text(errors='replace')[-6000:]}"
            )
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise GateError("server readiness timeout")


def completion(port: int, token_ids: list[int], output: Path) -> dict[str, Any]:
    payload = {
        "model": "gpt-oss-120b", "prompt": token_ids, "max_tokens": 64,
        "temperature": 0.0, "top_p": 1.0, "stop": [], "stream": False,
        "cache_prompt": False, "return_tokens": True,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    start_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            value = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise GateError(f"completion HTTP {exc.code}: {exc.read()[-2000:]!r}") from exc
    end_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    choices = value.get("choices")
    usage = value.get("usage")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(usage, dict):
        raise GateError(f"malformed completion response: {value}")
    choice = choices[0]
    text = choice.get("text")
    if (choice.get("finish_reason") != "length" or not isinstance(text, str) or
            int(usage.get("prompt_tokens", -1)) != 512 or
            int(usage.get("completion_tokens", -1)) != 64):
        raise GateError(f"completion correctness gate failed: {value}")
    text.encode("utf-8", errors="strict")
    atomic_write_json(output, value)
    return {"text": text, "start_ns": start_ns, "end_ns": end_ns,
            "e2e_ms": (end_ns - start_ns) / 1e6}


def descendants(root_pid: int) -> list[int]:
    found = {root_pid}
    changed = True
    while changed:
        changed = False
        for stat in Path("/proc").glob("[0-9]*/stat"):
            try:
                text = stat.read_text()
                fields = text[text.rfind(")") + 2:].split()
                pid, ppid = int(stat.parent.name), int(fields[1])
            except (OSError, ValueError, IndexError):
                continue
            if ppid in found and pid not in found:
                found.add(pid)
                changed = True
    return sorted(found)


def find_uvm_fd(root_pid: int) -> tuple[int, int]:
    matches = []
    for pid in descendants(root_pid):
        for fd_path in (Path("/proc") / str(pid) / "fd").glob("[0-9]*"):
            try:
                if os.readlink(fd_path) == "/dev/nvidia-uvm":
                    matches.append((pid, int(fd_path.name)))
            except OSError:
                pass
    if len(matches) != 1:
        raise GateError(f"expected one owned UVM fd, found {matches}")
    return matches[0]


def duplicate_fd(pid: int, target_fd: int) -> int:
    pidfd = os.pidfd_open(pid, 0)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        result = libc.syscall(438, pidfd, target_fd, 0)
        if result < 0:
            error = ctypes.get_errno()
            raise GateError(f"pidfd_getfd failed: {os.strerror(error)}")
        return int(result)
    finally:
        os.close(pidfd)


def start_eviction_monitor(server_pid: int, class_table: Path,
                           run_dir: Path) -> tuple[subprocess.Popen[Any], Any, Path]:
    pid, target_fd = find_uvm_fd(server_pid)
    inherited_fd = duplicate_fd(pid, target_fd)
    path = run_dir / "evictions.jsonl"
    log = path.open("x", buffering=1)
    try:
        process = subprocess.Popen(
            [str(EVICTION_MONITOR), "--uvm-fd", str(inherited_fd), str(class_table)],
            stdout=log, stderr=subprocess.STDOUT, text=True,
            pass_fds=(inherited_fd,), start_new_session=True,
        )
    finally:
        os.close(inherited_fd)
    try:
        wait_event(process, path, "ready")
        return process, log, path
    except Exception:
        stop_group(process)
        log.close()
        raise


def start_policy(mode: str, class_table: Path,
                 run_dir: Path) -> tuple[subprocess.Popen[Any], Any, Path]:
    path = run_dir / "policy.jsonl"
    log = path.open("x", buffering=1)
    process = subprocess.Popen(
        ["sudo", "-n", str(POLICY), mode, str(class_table)],
        stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True,
    )
    try:
        wait_event(process, path, "policy_ready")
        return process, log, path
    except Exception:
        stop_group(process)
        log.close()
        raise


def compile_layout(trace_path: Path, class_table: Path, report: Path) -> None:
    output = run_checked([
        "python3", str(COMPILE_LAYOUT), "--input", str(trace_path),
        "--hot-set", str(HOT_SET), "--output", str(class_table),
        "--strict", "--expected-layouts", "216",
    ])
    report.write_text(output + "\n", encoding="utf-8")


def validate_server_log(path: Path) -> None:
    text = path.read_text(errors="replace")
    patterns = (r"CUDA error", r"illegal memory access", r"out of memory",
                r"failed to load", r"Traceback")
    found = [pattern for pattern in patterns if re.search(pattern, text, re.I)]
    if found:
        raise GateError(f"fatal patterns in server log: {found}")


def run_cell(config: str, run_dir: Path, port: int,
             prompts: dict[str, Any], prompt_order: list[int]) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    trace = server = policy = monitor = None
    trace_log = server_log = policy_log = monitor_log = None
    trace_path = run_dir / "trace.jsonl"
    server_path = run_dir / "server.log"
    class_table = run_dir / "class-table.txt"
    try:
        trace_log = trace_path.open("x", buffering=1)
        trace = subprocess.Popen(
            ["sudo", "-n", str(TRACE), str(LIBGGML_BASE), "0", "7200"],
            stdout=trace_log, stderr=subprocess.STDOUT, text=True,
            start_new_session=True,
        )
        wait_event(trace, trace_path, "ready")
        server_log = server_path.open("x", buffering=1)
        command = server_command(config, port)
        atomic_write_json(run_dir / "launch.json", {
            "argv": command, "cwd": str(LLAMA_ROOT),
            "environment_overrides": {
                key: value for key, value in controlled_environment(config).items()
                if key.startswith(("CUDA_", "GGML_", "GPUBPF_", "HF_HUB_", "TRANSFORMERS_"))
            },
        })
        server = subprocess.Popen(
            command, cwd=LLAMA_ROOT, env=controlled_environment(config),
            stdout=server_log, stderr=subprocess.STDOUT, text=True,
            start_new_session=True,
        )
        wait_ready(server, port, server_path)
        compile_layout(trace_path, class_table, run_dir / "layout-report.json")
        if config != "llama_ncmoe32":
            stop_group(trace)
            trace = None
            trace_log.close()
            trace_log = None
        if config == "gpubpf_observe":
            policy, policy_log, policy_path = start_policy("observe", class_table, run_dir)
        elif config == "gpubpf_profile_protect":
            policy, policy_log, policy_path = start_policy("protect", class_table, run_dir)
        else:
            policy_path = None
        monitor, monitor_log, eviction_path = start_eviction_monitor(
            server.pid, class_table, run_dir
        )

        warmup = completion(port, prompts["records"][0]["prompt_token_ids"],
                            run_dir / "warmup.json")
        time.sleep(1.1)
        policy_before = latest_event(policy_path, "policy_stats") if policy_path else None
        eviction_before = latest_event(eviction_path, "eviction_stats")
        passes = []
        for pass_number in (1, 2):
            current = []
            for sequence, prompt_number in enumerate(prompt_order, start=1):
                current.append(completion(
                    port, prompts["records"][prompt_number]["prompt_token_ids"],
                    run_dir / f"pass-{pass_number}-request-{sequence:02d}-prompt-{prompt_number}.json",
                ))
            passes.append(current)
        for sequence, (first, second) in enumerate(zip(*passes), start=1):
            if first["text"] != second["text"]:
                raise GateError(f"non-deterministic output at sequence {sequence}")
        time.sleep(1.1)
        eviction_after = latest_event(eviction_path, "eviction_stats")
        eviction_delta = counter_delta(eviction_before, eviction_after, EVICTION_KEYS)
        if eviction_delta["evictions"] <= 0 or eviction_delta["evicted_bytes"] <= 0:
            raise GateError(f"no completed UVM evictions: {eviction_delta}")
        if eviction_delta["dropped_evictions"] != 0:
            raise GateError(f"dropped UVM eviction events: {eviction_delta}")
        policy_delta = None
        if policy_path:
            policy_after = latest_event(policy_path, "policy_stats")
            policy_delta = counter_delta(policy_before, policy_after, POLICY_STAT_KEYS)
            if config == "gpubpf_observe":
                if policy_delta["observe_activate"] <= 0 or policy_delta["observe_access"] <= 0:
                    raise GateError(f"observe engagement failed: {policy_delta}")
                reordered = sum(policy_delta[key] for key in (
                    "hot_tail", "cold_head", "shared_tail", "hot_access_tail",
                    "shared_access_tail", "setter_failure",
                ))
                if reordered != 0:
                    raise GateError(f"observe mode reordered pages: {policy_delta}")
            else:
                required = ("mapped", "hot_tail", "cold_native", "hot_access_tail")
                if any(policy_delta[key] <= 0 for key in required):
                    raise GateError(f"protect engagement failed: {policy_delta}")
                if policy_delta["cold_head"] != 0 or policy_delta["setter_failure"] != 0:
                    raise GateError(f"protect safety gate failed: {policy_delta}")

        route_diagnostic = None
        if config == "llama_ncmoe32":
            stop_group(trace)
            trace = None
            trace_log.close()
            trace_log = None
            final_trace = latest_event(trace_path, "final")
            if (int(final_trace["layouts"]) != 216 or
                    int(final_trace["routes"]) <= 0 or
                    int(final_trace["dropped"]) != 0):
                raise GateError(f"framework route trace gate failed: {final_trace}")
            summary_text = run_checked([
                "python3", str(COMPILE_HOT_SET), "--input", str(trace_path),
                "--output", str(run_dir / "route-diagnostic-hot-set.txt"),
                "--report", str(run_dir / "route-diagnostic-report.json"),
                "--expected-layers", "36", "--expected-experts", "128",
                "--top-k", "10",
            ])
            route_diagnostic = json.loads(summary_text)
            if int(route_diagnostic["layers"]) != 36:
                raise GateError(f"framework layer coverage failed: {route_diagnostic}")

        result = {
            "config": config, "warmup": warmup,
            "outputs": [[item["text"] for item in current] for current in passes],
            "requests": passes, "policy_delta": policy_delta,
            "eviction_delta": eviction_delta, "route_diagnostic": route_diagnostic,
        }
        atomic_write_json(run_dir / "result.json", result)
        return result
    finally:
        stop_group(policy)
        stop_group(monitor)
        stop_group(trace)
        stop_group(server, 120)
        for stream in (policy_log, monitor_log, trace_log, server_log):
            if stream is not None:
                stream.close()
        if server_path.exists():
            validate_server_log(server_path)
        if struct_ops_maps():
            raise GateError("owned policy did not detach cleanly")


class Lease:
    def __init__(self) -> None:
        self.files = []

    def __enter__(self) -> "Lease":
        for path in (Path("/tmp/gpubpf-revision-gpu0.lock"),
                     Path("/tmp/gpubpf-revision-struct-ops.lock")):
            stream = path.open("a+")
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise GateError(f"experiment lease is busy: {path}") from exc
            self.files.append(stream)
        return self

    def __exit__(self, *_: Any) -> None:
        for stream in reversed(self.files):
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()


def run_correctness(attempt: int, port: int) -> dict[str, Any]:
    output = RAW / f"attempt-{attempt:02d}"
    if output.exists():
        raise GateError(f"attempt output already exists: {output}")
    required = (LLAMA_SERVER, LIBGGML_BASE, MODEL, PROMPTS, SCHEDULE, TRACE,
                POLICY, COMPILE_LAYOUT, COMPILE_HOT_SET, HOT_SET,
                EVICTION_MONITOR, CUSTOM_UVM)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise GateError(f"missing runtime files: {missing}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise GateError(f"port {port} is in use")
    prompts = json.loads(PROMPTS.read_text())
    schedule = json.loads(SCHEDULE.read_text())
    if tuple(schedule["configuration_order"]) != CONFIGS:
        raise GateError("configuration order differs from the frozen schedule")
    prompt_order = [int(value) for value in schedule["prompt_order"]]
    if sorted(prompt_order) != list(range(1, 9)):
        raise GateError("prompt order is not a permutation of 1..8")

    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output / "status.json", {
        "status": "running", "attempt": attempt, "started_ns": time.time_ns(),
    })
    results = []
    custom_loaded = False
    try:
        admitted = require_idle()
        atomic_write_json(output / "admission.json", admitted)
        load_custom_uvm()
        custom_loaded = True
        for config in CONFIGS:
            results.append(run_cell(config, output / config, port, prompts, prompt_order))
        reference = results[0]["outputs"][0]
        for result in results:
            if result["outputs"][0] != reference or result["outputs"][1] != reference:
                raise GateError(f"cross-configuration output mismatch: {result['config']}")
        final = {
            "status": "passed", "attempt": attempt, "prompt_order": prompt_order,
            "configuration_order": list(CONFIGS),
            "completed_ns": time.time_ns(),
            "cells": [str((output / config / "result.json").resolve()) for config in CONFIGS],
        }
        atomic_write_json(output / "status.json", final)
        return final
    except Exception as exc:
        atomic_write_json(output / "status.json", {
            "status": "failed", "attempt": attempt, "error_type": type(exc).__name__,
            "error": str(exc), "completed_ns": time.time_ns(),
        })
        raise
    finally:
        if custom_loaded:
            restore_stock_uvm()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("admit", "correctness"))
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--port", type=int, default=18082)
    args = parser.parse_args()
    try:
        with Lease():
            if args.action == "admit":
                print(json.dumps(require_idle(), sort_keys=True))
            else:
                print(json.dumps(run_correctness(args.attempt, args.port), sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__,
                          "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
