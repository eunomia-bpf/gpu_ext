#!/usr/bin/env python3
"""Admission-gated RQ3 runner for native, XSched Level-1, and gpubpf."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import queue
import random
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from typing import Any

HERE = Path(__file__).resolve().parent
GPU_EXT = HERE.parents[1]
XSCHED = HERE / "deps" / "xsched"
XSCHED_OUTPUT = XSCHED / "output"
EXTENSION = GPU_EXT / "extension"
RAW = HERE / "raw"
sys.dont_write_bytecode = True
sys.path.insert(0, str(GPU_EXT / "workloads/moe-infinity"))
import run_moe_head_to_head as shared

EXPECTED_DRIVER = "575.57.08"
XSCHED_COMMIT = "f49289f0220931df78de948ed841ecbaf960a919"
SEED = 1797
CONFIGS = ("native", "xsched", "gpubpf")
GPUBPF_CONFIGS = ("gpubpf", "gpubpf_nocooldown", "gpubpf_interleave")
SUPPORTED_CONFIGS = ("native", "xsched") + GPUBPF_CONFIGS
REPETITIONS = 10


def run(command: list[str], *, check: bool = True, text: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=text, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def file_metadata(path: Path) -> dict[str, Any]:
    """Record ordinary identity metadata without reading file contents."""
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


def patch_body(text: str) -> str:
    """Compare the small source edit, never Git's blob-index metadata."""
    return "\n".join(line for line in text.splitlines() if not line.startswith("index ")) + "\n"


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("empty sample")
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def driver_version() -> str:
    result = run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    versions = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    if len(versions) != 1:
        raise RuntimeError(f"expected one driver version, found {sorted(versions)}")
    return versions.pop()


def gpu_processes() -> list[str]:
    result = run([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ], check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stdout.strip())
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def sudo_prefix() -> list[str]:
    if os.geteuid() == 0:
        return []
    probe = run(["sudo", "-n", "true"], check=False)
    if probe.returncode != 0:
        raise RuntimeError("gpubpf requires root; configure non-interactive sudo or run as root")
    return ["sudo", "-n"]


def admission(require_runtime: bool) -> dict[str, Any]:
    workload = HERE / "build" / "priority_workload"
    expected_xsched_diff = patch_body((HERE / "xsched-engagement.patch").read_text())
    actual_xsched_diff = patch_body(run(["git", "-C", str(XSCHED), "diff", "--", "preempt"]).stdout)
    source_paths = [
        EXTENSION / "uprobe_preempt_multi.c", EXTENSION / "uprobe_preempt_multi.bpf.c",
        EXTENSION / "gpu_sched_set_timeslices.c", EXTENSION / "gpu_sched_set_timeslices.bpf.c",
        EXTENSION / "gpu_sched_set_timeslices.h", EXTENSION / "Makefile",
    ]
    checks: dict[str, Any] = {
        "kernel": os.uname().release,
        "driver": driver_version(),
        "gpu_processes": gpu_processes(),
        "xsched_commit": run(["git", "-C", str(XSCHED), "rev-parse", "HEAD"]).stdout.strip(),
        "xsched_diff": actual_xsched_diff,
        "xsched_diff_matches_reviewed_patch": actual_xsched_diff == expected_xsched_diff,
        "workload_file": file_metadata(workload),
        "reviewed_patch_file": file_metadata(HERE / "xsched-engagement.patch"),
        "gpubpf_binaries": {
            "uprobe_preempt_multi": file_metadata(EXTENSION / "uprobe_preempt_multi"),
            "gpu_sched_set_timeslices": file_metadata(EXTENSION / "gpu_sched_set_timeslices"),
        },
        "gpubpf_source_files": {
            str(path.relative_to(GPU_EXT)): file_metadata(path) for path in source_paths
        },
        "xsched_runtime_files": {
            name: file_metadata(XSCHED_OUTPUT / "lib" / name)
            for name in ("libpreempt.so", "libhalcuda.so", "libshimcuda.so")
        },
    }
    errors: list[str] = []
    if checks["xsched_commit"] != XSCHED_COMMIT:
        errors.append(f"XSched commit is {checks['xsched_commit']}, expected {XSCHED_COMMIT}")
    if not checks["xsched_diff_matches_reviewed_patch"]:
        errors.append("XSched checkout diff does not exactly equal the reviewed engagement patch")
    if not workload.exists():
        errors.append("workload is not built; run the build phase")
    for path in (XSCHED_OUTPUT / "bin" / "xserver", XSCHED_OUTPUT / "lib" / "libcuda.so",
                 EXTENSION / "uprobe_preempt_multi", EXTENSION / "gpu_sched_set_timeslices"):
        if not path.exists():
            errors.append(f"missing required binary: {path}")
    if require_runtime:
        if checks["driver"] != EXPECTED_DRIVER:
            errors.append(f"driver {checks['driver']} is not the frozen gpubpf stack {EXPECTED_DRIVER}")
        if checks["gpu_processes"]:
            errors.append("GPU is not idle: " + "; ".join(checks["gpu_processes"]))
        try:
            sudo = sudo_prefix()
            # C output omits standalone FUNC records; use raw BTF for kfuncs.
            btf = run(sudo + ["bpftool", "btf", "dump", "file", "/sys/kernel/btf/nvidia", "format", "raw"], check=False)
            checks["nvidia_btf_probe_rc"] = btf.returncode
            checks["has_nv_gpu_sched_ops"] = "STRUCT 'nv_gpu_sched_ops'" in btf.stdout
            checks["has_preempt_kfunc"] = "FUNC 'bpf_nv_gpu_preempt_tsg'" in btf.stdout
            if not checks["has_nv_gpu_sched_ops"] or not checks["has_preempt_kfunc"]:
                errors.append("NVIDIA BTF lacks nv_gpu_sched_ops or bpf_nv_gpu_preempt_tsg")
            checks["safety"] = shared.safety_snapshot()
            shared.validate_pre_server_safety(checks["safety"])
        except Exception as exc:  # admission must preserve the concrete failure
            errors.append(f"BTF probe failed: {exc}")
    checks["errors"] = errors
    checks["admitted"] = not errors
    return checks


class ManagedProcess:
    def __init__(self, name: str, command: list[str], env: dict[str, str], cpu: int | list[int]):
        self.name = name
        self.stdout_lines: list[str] = []
        self.stderr_lines: list[str] = []
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        cpu_mask = str(cpu) if isinstance(cpu, int) else ",".join(map(str, cpu))
        command = ["taskset", "-c", cpu_mask] + command
        self.proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=env, start_new_session=True,
        )
        self.threads = [
            threading.Thread(target=self._reader, args=(self.proc.stdout, self.stdout_lines, True), daemon=True),
            threading.Thread(target=self._reader, args=(self.proc.stderr, self.stderr_lines, False), daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    def _reader(self, pipe: Any, target: list[str], parse_json: bool) -> None:
        for line in iter(pipe.readline, ""):
            line = line.rstrip("\n")
            target.append(line)
            if parse_json and line.startswith("{"):
                try:
                    self.events.put(json.loads(line))
                except json.JSONDecodeError:
                    pass

    def send(self, command: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def wait_event(self, expected: str, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None and self.events.empty():
                raise RuntimeError(f"{self.name} exited {self.proc.returncode}: {'; '.join(self.stderr_lines[-8:])}")
            try:
                event = self.events.get(timeout=min(0.2, max(0.01, deadline - time.monotonic())))
            except queue.Empty:
                continue
            if event.get("event") == expected:
                return event
        raise TimeoutError(f"timeout waiting for {expected} from {self.name}")

    def stop(self) -> None:
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGINT)
            try:
                self.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGTERM)
                try:
                    self.proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                    self.proc.wait(timeout=8)
        for thread in self.threads:
            thread.join(timeout=1)

    def collect(self, timeout: float) -> int:
        rc = self.proc.wait(timeout=timeout)
        for thread in self.threads:
            thread.join(timeout=1)
        return rc


def allowed_cpus(required: int) -> list[int]:
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < required:
        raise RuntimeError(f"need {required} allowed CPUs for fixed affinity, only {len(cpus)} available")
    return cpus[:required]


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("XSCHED_"):
            env.pop(key)
    for key in ("LD_PRELOAD", "LD_LIBRARY_PATH"):
        env.pop(key, None)
    return env


def parse_configs(text: str) -> tuple[str, ...]:
    configs = tuple(item.strip() for item in text.split(",") if item.strip())
    if not configs or len(configs) != len(set(configs)):
        raise argparse.ArgumentTypeError("configuration list must be nonempty and unique")
    unknown = set(configs) - set(SUPPORTED_CONFIGS)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown configurations: {sorted(unknown)}")
    return configs


def gpubpf_policy_commands(config: str) -> tuple[list[str], list[str]]:
    if config not in GPUBPF_CONFIGS:
        raise ValueError(f"not a gpubpf configuration: {config}")
    timeslice = [str(EXTENSION / "gpu_sched_set_timeslices"),
                 "-p", "bench_lc:1000000", "-p", "bench_be:200"]
    if config == "gpubpf_interleave":
        timeslice += ["-i", "bench_lc:2", "-i", "bench_be:0"]
    preempt = [str(EXTENSION / "uprobe_preempt_multi"),
               "--be-name", "bench_be", "--lc-name", "bench_lc",
               "--cooldown-us", "0" if config == "gpubpf_nocooldown" else "100"]
    return timeslice, preempt


def start_policy(config: str, cpus: list[int], run_dir: Path) -> list[ManagedProcess]:
    policies: list[ManagedProcess] = []
    env = base_env()
    try:
        if config == "xsched":
            policies.append(ManagedProcess(
                "xserver", [str(XSCHED_OUTPUT / "bin" / "xserver"), "HPF", "50000"], env, cpus[0]))
            time.sleep(0.5)
            if policies[0].proc.poll() is not None:
                raise RuntimeError("xserver failed to stay running")
        elif config in GPUBPF_CONFIGS:
            sudo = sudo_prefix()
            timeslice_command, preempt_command = gpubpf_policy_commands(config)
            policies.append(ManagedProcess(
                "timeslice", sudo + timeslice_command, env, cpus[0]))
            policies.append(ManagedProcess(
                "preempt", sudo + preempt_command, env, cpus[1]))
            time.sleep(1.0)
            if any(policy.proc.poll() is not None for policy in policies):
                raise RuntimeError("gpubpf policy failed to stay running")
            inventory = shared.struct_ops_inventory()
            # Some installed bpftool versions do not name new struct-ops
            # link types. The map's kernel-reported owner PID is decisive.
            owners = {int(owner["pid"]) for item in inventory["maps"]
                      for owner in item.get("pids", ())}
            owned_pids = set(shared.descendants(policies[0].proc.pid))
            if (len(inventory["maps"]) != 1 or len(inventory["links"]) > 1
                    or not owners or not owners.issubset(owned_pids)):
                raise RuntimeError(f"expected one owned scheduling map: {inventory}")
            inventory["owned_loader_pids"] = sorted(owned_pids)
            (run_dir / "policy-attachment.json").write_text(json.dumps(inventory, indent=2) + "\n")
    except BaseException:
        for policy in reversed(policies):
            policy.stop()
            write_process_log(run_dir / f"{policy.name}.json", policy)
        raise
    return policies


def workload_env(config: str, role: str) -> dict[str, str]:
    env = base_env()
    if config != "xsched":
        return env
    env.update({
        "XSCHED_SCHEDULER": "GLB",
        "XSCHED_AUTO_XQUEUE": "ON",
        "XSCHED_AUTO_XQUEUE_LEVEL": "1",
        "XSCHED_AUTO_XQUEUE_PRIORITY": "1" if role == "lc" else "0",
        "XSCHED_AUTO_XQUEUE_THRESHOLD": "16" if role == "lc" else "4",
        "XSCHED_AUTO_XQUEUE_BATCH_SIZE": "8" if role == "lc" else "2",
        "LD_LIBRARY_PATH": str(XSCHED_OUTPUT / "lib") + ":" + env.get("LD_LIBRARY_PATH", ""),
    })
    env.pop("XSCHED_CUDA_LV3_IMPL", None)
    return env


def clock_calibration() -> dict[str, int]:
    completed = subprocess.run(
        [str(HERE / "build" / "priority_workload"), "--clock-probe"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=base_env(),
        timeout=30,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"clock probe emitted unexpected output: {completed.stdout!r}")
    record = json.loads(lines[0])
    if record.get("event") != "clock_probe":
        raise RuntimeError(f"clock probe emitted the wrong event: {record}")
    required = ("offset_ns", "offset_low_ns", "offset_high_ns", "uncertainty_ns")
    if any(not isinstance(record.get(name), int) for name in required):
        raise RuntimeError(f"clock probe fields are invalid: {record}")
    if not record["offset_low_ns"] <= record["offset_ns"] <= record["offset_high_ns"]:
        raise RuntimeError(f"clock probe offset is outside its bracket: {record}")
    if record["uncertainty_ns"] > 1_000_000:
        raise RuntimeError(f"clock probe uncertainty exceeds 1 ms: {record}")
    return record


def validate_clock_pair(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    """Bound offset movement instead of assuming two clock epochs are equal."""
    drift = abs(after["offset_ns"] - before["offset_ns"])
    bound = drift + before["uncertainty_ns"] + after["uncertainty_ns"]
    if bound > 1_000_000:
        raise RuntimeError(f"clock offset uncertainty/drift exceeds 1 ms: {bound} ns")
    return {"offset_drift_ns": drift, "conservative_error_bound_ns": bound}


@contextmanager
def safe_cell(run_dir: Path):
    """Save real before/after state even when a worker or policy fails."""
    before = shared.safety_snapshot()
    shared.validate_pre_server_safety(before)
    (run_dir / "safety-before.json").write_text(json.dumps(before, indent=2) + "\n")
    try:
        yield
    except BaseException as exc:
        (run_dir / "failure.json").write_text(json.dumps({
            "error_type": type(exc).__name__, "error": str(exc),
        }, indent=2) + "\n")
        raise
    finally:
        after = shared.wait_for_post_server_safety(before)
        (run_dir / "safety-after.json").write_text(json.dumps(after, indent=2) + "\n")


def safe_run(function):
    def wrapped(*args, **kwargs):
        # All three cell functions have their output directory at index 1,
        # except calibration, which accepts it at index 0.
        run_dir = args[0] if function.__name__ == "calibrate_reps" else args[1]
        run_dir.mkdir(parents=True, exist_ok=False)
        with safe_cell(run_dir):
            return function(*args, **kwargs)
    return wrapped


def write_process_log(path: Path, process: ManagedProcess) -> None:
    path.write_text(json.dumps({
        "name": process.name, "command": process.proc.args, "returncode": process.proc.returncode,
        "stdout": process.stdout_lines, "stderr": process.stderr_lines,
    }, indent=2) + "\n")


def parse_final_counters(text: str, names: tuple[str, ...]) -> dict[str, int]:
    result = {}
    for name in names:
        matches = re.findall(rf"(?m)^\s*{re.escape(name)}:\s+(\d+)\s*$", text)
        if not matches:
            raise RuntimeError(f"policy final counter is missing: {name}")
        result[name] = int(matches[-1])
    return result


def validate_gpubpf_engagement(config: str, timeslice_text: str,
                               preempt_text: str, tasks: int) -> dict[str, int]:
    result = parse_final_counters(timeslice_text, ("timeslice_mod",))
    result.update(parse_final_counters(preempt_text, (
        "uprobe_hit", "preempt_ok", "preempt_err", "skipped", "cooldown_skip",
        "targets_hit", "tsg_captured", "active_targets",
    )))
    if result["timeslice_mod"] < 6:
        raise RuntimeError("gpubpf did not modify timeslices for all six processes")
    if result["preempt_ok"] == 0 or result["preempt_err"] != 0:
        raise RuntimeError("gpubpf preemption engagement gate failed")
    if result["active_targets"] != 4 or result["tsg_captured"] != 4:
        raise RuntimeError("gpubpf did not capture exactly four BE GR targets")
    if result["uprobe_hit"] != 6 * 4 * tasks or result["skipped"] != 4 * 4 * tasks:
        raise RuntimeError("gpubpf launch/filter counts differ from the fixed workload")
    if config == "gpubpf_nocooldown":
        expected = 2 * 4 * tasks * 4
        if (result["preempt_ok"] != expected or result["targets_hit"] != expected
                or result["cooldown_skip"] != 0):
            raise RuntimeError(f"no-cooldown requires exactly {expected} successful target preemptions")
        result["expected_preemptions"] = expected
    if config == "gpubpf_interleave":
        result.update(parse_final_counters(timeslice_text, (
            "interleave_mod", "interleave_observed", "interleave_mismatch", "setter_error",
        )))
        if (result["interleave_mod"] < 6 or result["interleave_observed"] < 6
                or result["interleave_mismatch"] != 0 or result["setter_error"] != 0):
            raise RuntimeError("interleave requests were not confirmed at channel-group bind")
    return result


@safe_run
def execute_round(config: str, run_dir: Path, reps: int, tasks: int, blocks: int,
                  threads: int, timeout: float) -> dict[str, Any]:
    cpus = allowed_cpus(10)
    clock = clock_calibration()
    policies = start_policy(config, cpus, run_dir)
    workers: list[ManagedProcess] = []
    try:
        deadline = time.monotonic() + timeout
        shapes = [("be", i + 1) for i in range(4)] + [("lc", i + 1) for i in range(2)]
        for index, (role, process_id) in enumerate(shapes):
            binary = HERE / "build" / ("bench_lc" if role == "lc" else "bench_be")
            command = [str(binary), role, str(process_id), "4", str(tasks), str(reps),
                       str(blocks), str(threads), "1" if role == "be" else "0",
                       str(clock["offset_ns"])]
            workers.append(ManagedProcess(f"{role}{process_id}", command,
                                          workload_env(config, role), cpus[2:]))
        for worker in workers:
            worker.wait_event("ready", 30)

        be_workers = workers[:4]
        lc_workers = workers[4:]
        be_release_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        for worker in be_workers:
            worker.send("GO")
        running_events = [worker.wait_event("running", max(0.0, deadline - time.monotonic()))
                          for worker in be_workers]
        all_be_running_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        fixed_delay_ns = 5_000_000
        release_deadline = all_be_running_ns + fixed_delay_ns
        while time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) < release_deadline:
            pass
        lc_release_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        for worker in lc_workers:
            worker.send("GO")

        results = [worker.wait_event("result", max(0.0, deadline - time.monotonic()))
                   for worker in workers]
        for worker in workers:
            if worker.collect(10) != 0:
                raise RuntimeError(f"{worker.name} exited nonzero")
    finally:
        for worker in workers:
            if worker.proc.poll() is None:
                worker.stop()
        for policy in reversed(policies):
            policy.stop()
        for process in workers + policies:
            write_process_log(run_dir / f"{process.name}.json", process)

    clock_after = clock_calibration()
    clock_error = validate_clock_pair(clock, clock_after)

    expected_values = 4 * tasks * blocks * threads
    if any(result.get("outputs_validated") != expected_values for result in results):
        raise RuntimeError(
            f"workload did not semantically validate all {expected_values} output values per process"
        )

    lc_samples = [
        (sample["entry_ns"] - sample["submit_ns"]) / 1000.0
        for result in results if result["role"] == "lc" for sample in result["samples"]
    ]
    lc_completion = [
        (sample["exit_ns"] - sample["submit_ns"]) / 1000.0
        for result in results if result["role"] == "lc" for sample in result["samples"]
    ]
    be_results = [result for result in results if result["role"] == "be"]
    be_end_ns = max(result["completion_host_ns"] for result in be_results)
    be_count = sum(len(result["samples"]) for result in be_results)
    if len(lc_samples) != 2 * 4 * tasks or be_count != 4 * 4 * tasks:
        raise RuntimeError("sample count mismatch")

    engagement: dict[str, Any] = {}
    if config == "xsched":
        audit = []
        pattern = re.compile(
            r"XSCHED_AUDIT pid=(\d+) xqueue=(\d+) level=(\d+) threshold=(\d+) batch=(\d+) "
            r"suspend_ok=(\d+) resume_ok=(\d+)"
        )
        unique_handles = set()
        for worker in workers:
            matches = [pattern.search(line) for line in worker.stderr_lines]
            parsed = [tuple(map(int, match.groups())) for match in matches if match]
            expected_threshold, expected_batch = ((16, 8) if worker.name.startswith("lc") else (4, 2))
            if len(parsed) != 4 or any(
                level != 1 or threshold != expected_threshold or batch != expected_batch
                for _, _, level, threshold, batch, _, _ in parsed
            ):
                raise RuntimeError(f"invalid XSched audit for {worker.name}: {parsed}")
            unique_handles.update((pid, handle) for pid, handle, *_ in parsed)
            audit.append({"worker": worker.name, "queues": parsed})
        if len(unique_handles) != 24:
            raise RuntimeError(f"expected 24 unique XQueues, found {len(unique_handles)}")
        be_transitions = sum(suspend for item in audit if item["worker"].startswith("be")
                             for *_, suspend, _ in item["queues"])
        be_resumes = sum(resume for item in audit if item["worker"].startswith("be")
                         for *_, resume in item["queues"])
        server_text = "\n".join(policies[0].stdout_lines + policies[0].stderr_lines)
        if be_transitions == 0 or be_resumes == 0:
            raise RuntimeError("XSched did not record a successful BE suspend and resume")
        if server_text.count("set priority 1") < 8 or server_text.count("set priority 0") < 16:
            raise RuntimeError("xserver log did not establish both priority classes on 24 XQueues")
        engagement = {"audit": audit, "be_suspend_ok": be_transitions, "be_resume_ok": be_resumes}
    elif config in GPUBPF_CONFIGS:
        timeslice_text = "\n".join(policies[0].stdout_lines + policies[0].stderr_lines)
        preempt_text = "\n".join(policies[1].stdout_lines + policies[1].stderr_lines)
        engagement = validate_gpubpf_engagement(config, timeslice_text, preempt_text, tasks)

    result = {
        "config": config, "reps": reps, "tasks_per_stream": tasks, "blocks": blocks,
        "threads": threads, "be_release_ns": be_release_ns,
        "all_be_running_ns": all_be_running_ns, "lc_release_ns": lc_release_ns,
        "be_running_events": running_events,
        "lc_samples": len(lc_samples), "be_completed": be_count,
        "lc_mean_us": statistics.mean(lc_samples),
        "lc_p50_us": percentile(lc_samples, 0.50),
        "lc_p95_us": percentile(lc_samples, 0.95),
        "lc_p99_us": percentile(lc_samples, 0.99),
        "lc_p99_is_sample_maximum": math.ceil(0.99 * len(lc_samples)) == len(lc_samples),
        "lc_completion_p99_us": percentile(lc_completion, 0.99),
        "be_throughput_kernels_s": be_count / ((be_end_ns - be_release_ns) / 1e9),
        "outputs_validated_per_process": expected_values,
        "clock_calibration": clock,
        "clock_calibration_after": clock_after,
        "clock_error": clock_error,
        "engagement": engagement,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


@safe_run
def calibrate_reps(root: Path, initial_reps: int, blocks: int, threads: int,
                   timeout: float) -> dict[str, Any]:
    """Tune only fixed compute repetitions; the accepted value is then frozen."""
    reps = initial_reps
    attempts = []
    cpu = allowed_cpus(1)[0]
    clock = clock_calibration()
    for attempt in range(5):
        worker = ManagedProcess(
            f"calibration-{attempt}",
            [str(HERE / "build" / "priority_workload"), "lc", "1", "1", "1",
             str(reps), str(blocks), str(threads), "0", str(clock["offset_ns"])],
            base_env(), cpu,
        )
        try:
            worker.wait_event("ready", 30)
            worker.send("GO")
            result = worker.wait_event("result", timeout)
            if worker.collect(10) != 0:
                raise RuntimeError("calibration worker failed")
        finally:
            if worker.proc.poll() is None:
                worker.stop()
            write_process_log(root / f"attempt-{attempt}.json", worker)
        sample = result["samples"][0]
        duration_ms = (sample["exit_ns"] - sample["entry_ns"]) / 1e6
        attempts.append({"attempt": attempt, "reps": reps, "duration_ms": duration_ms})
        if 76.0 <= duration_ms <= 84.0:
            break
        if duration_ms <= 0:
            raise RuntimeError("nonpositive calibration duration")
        reps = max(1, round(reps * 80.0 / duration_ms))
    if not 76.0 <= attempts[-1]["duration_ms"] <= 84.0:
        raise RuntimeError(f"could not tune a kernel to 80 ms: {attempts}")
    record = {"target_ms": 80.0, "tolerance_ms": 4.0, "frozen_reps": reps,
              "clock_calibration": clock,
              "blocks": blocks, "threads": threads, "attempts": attempts}
    (root / "calibration.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


@safe_run
def execute_isolated(role: str, run_dir: Path, reps: int, tasks: int, blocks: int,
                     threads: int, timeout: float) -> dict[str, Any]:
    binary = HERE / "build" / ("bench_lc" if role == "lc" else "bench_be")
    clock = clock_calibration()
    worker = ManagedProcess(
        f"isolated-{role}",
        [str(binary), role, "1", "4", str(tasks), str(reps), str(blocks), str(threads),
         "0", str(clock["offset_ns"])],
        base_env(), allowed_cpus(1)[0],
    )
    try:
        worker.wait_event("ready", 30)
        release_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        worker.send("GO")
        result = worker.wait_event("result", timeout)
        if worker.collect(10) != 0:
            raise RuntimeError(f"isolated {role} worker failed")
    finally:
        if worker.proc.poll() is None:
            worker.stop()
        write_process_log(run_dir / f"isolated-{role}.json", worker)
    latencies = [(item["exit_ns"] - item["submit_ns"]) / 1000.0 for item in result["samples"]]
    clock_after = clock_calibration()
    record = {
        "control": f"isolated-{role}", "samples": len(latencies),
        "outputs_validated": result["outputs_validated"],
        "completion_host_ns": result["completion_host_ns"],
        "clock_calibration": clock,
        "clock_calibration_after": clock_after,
        "clock_error": validate_clock_pair(clock, clock_after),
        "throughput_kernels_s": len(latencies) / ((result["completion_host_ns"] - release_ns) / 1e9),
        "p99_us": percentile(latencies, 0.99),
    }
    (run_dir / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def bootstrap_ci(values: list[float], seed: int, draws: int = 10000) -> list[float]:
    rng = random.Random(seed)
    means = [statistics.mean(rng.choice(values) for _ in values) for _ in range(draws)]
    means.sort()
    return [means[int(0.025 * draws)], means[int(0.975 * draws)]]


def classify_comparison(latency_ci: list[float], throughput_ci: list[float]) -> str:
    latency_better = latency_ci[1] < 0
    latency_not_better = latency_ci[0] >= 0
    throughput_noninferior = throughput_ci[0] >= -0.05
    throughput_inferior = throughput_ci[1] < -0.05
    if latency_better and throughput_noninferior:
        return "positive"
    if latency_better and throughput_inferior:
        return "mixed"
    if latency_not_better or (throughput_inferior and not latency_better):
        return "negative"
    return "inconclusive"


def analyze(root: Path) -> dict[str, Any]:
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {
        "phase": "full", "repetitions": REPETITIONS, "tasks_per_stream": 50,
    }
    repetitions = protocol["repetitions"]
    configs = tuple(protocol.get("configs", CONFIGS))
    records = []
    for path in sorted(root.glob("**/result.json")):
        record = json.loads(path.read_text())
        if "block" in record:
            records.append(record)
    by_block: dict[int, dict[str, dict[str, Any]]] = {}
    for record in records:
        if record["config"] in by_block.get(record["block"], {}):
            raise RuntimeError("duplicate cell for one block/config")
        if record["tasks_per_stream"] != protocol["tasks_per_stream"]:
            raise RuntimeError("cell task count differs from the frozen protocol")
        by_block.setdefault(record["block"], {})[record["config"]] = record
    complete = {block: values for block, values in by_block.items() if set(values) == set(configs)}
    if len(complete) != repetitions or set(complete) != set(range(repetitions)):
        raise RuntimeError(f"analysis requires {repetitions} complete randomized blocks, found {len(complete)}")
    summary: dict[str, Any] = {
        "protocol": protocol, "complete_blocks": len(complete), "configs": {}, "paired": {},
        "xsched_scope": "upstream Level-1 HPF on sm_120; not XSched paper Level-3 reproduction",
    }
    for config in configs:
        lc = [complete[block][config]["lc_p99_us"] for block in sorted(complete)]
        be = [complete[block][config]["be_throughput_kernels_s"] for block in sorted(complete)]
        summary["configs"][config] = {
            "lc_p99_median_us": statistics.median(lc), "lc_p99_mean_ci95": bootstrap_ci(lc, SEED),
            "be_throughput_median": statistics.median(be), "be_throughput_mean_ci95": bootstrap_ci(be, SEED + 1),
            "lc_samples_per_cell": complete[0][config]["lc_samples"],
            "be_kernels_per_cell": complete[0][config]["be_completed"],
            "lc_p99_is_sample_maximum": complete[0][config].get("lc_p99_is_sample_maximum", False),
        }
        for metric in ("lc_mean_us", "lc_p50_us", "lc_p95_us"):
            if all(metric in complete[b][config] for b in complete):
                summary["configs"][config][f"{metric}_median"] = statistics.median(
                    complete[b][config][metric] for b in complete)
    for candidate in GPUBPF_CONFIGS:
        if candidate not in configs:
            continue
        for baseline in ("native", "xsched", "gpubpf"):
            if baseline not in configs or baseline == candidate:
                continue
            lc_delta = [complete[b][candidate]["lc_p99_us"] - complete[b][baseline]["lc_p99_us"] for b in sorted(complete)]
            be_ratio = [complete[b][candidate]["be_throughput_kernels_s"] /
                        complete[b][baseline]["be_throughput_kernels_s"] - 1.0 for b in sorted(complete)]
            summary["paired"][f"{candidate}_vs_{baseline}"] = {
                "lc_p99_delta_us_mean_ci95": bootstrap_ci(lc_delta, SEED + 2),
                "be_throughput_relative_mean_ci95": bootstrap_ci(be_ratio, SEED + 3),
            }
    summary["be_noninferiority_margin"] = -0.05
    rules = {
        "positive_rule": "LC paired-difference CI upper < 0 and BE relative-throughput CI lower >= -0.05",
        "mixed_rule": "LC CI upper < 0 and BE relative-throughput CI upper < -0.05",
        "negative_rule": "LC CI lower >= 0, or BE CI upper < -0.05 without established LC improvement",
        "inconclusive_rule": "all remaining combinations",
    }
    for candidate in GPUBPF_CONFIGS:
        comparison = summary["paired"].get(f"{candidate}_vs_xsched")
        if comparison is None:
            continue
        decision = {"classification": classify_comparison(
            comparison["lc_p99_delta_us_mean_ci95"],
            comparison["be_throughput_relative_mean_ci95"],
        ), **rules}
        if candidate == "gpubpf":
            summary["predeclared_decision"] = decision
        else:
            summary.setdefault("candidate_decisions", {})[candidate] = decision
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("build", "admission", "calibrate", "preflight", "pilot", "full", "analyze"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reps", type=int)
    parser.add_argument("--tasks", type=int)
    parser.add_argument("--blocks", type=int, default=340)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--configs", type=parse_configs, default=CONFIGS,
                        help="comma-separated independent configurations; default preserves the original three")
    args = parser.parse_args()
    if args.tasks is None:
        args.tasks = {"pilot": 5, "preflight": 2}.get(args.phase, 50)
    if args.tasks < 1 or args.blocks < 1 or args.threads < 1 or args.timeout <= 0:
        parser.error("workload dimensions and timeout must be positive")
    if args.reps is not None and args.reps < 1:
        parser.error("--reps must be positive")
    if args.phase == "build":
        subprocess.run(["make", "-C", str(HERE)], check=True)
        current_diff = run(["git", "-C", str(XSCHED), "diff", "--", "preempt"]).stdout
        reviewed_patch = patch_body((HERE / "xsched-engagement.patch").read_text())
        if not current_diff:
            subprocess.run(["git", "-C", str(XSCHED), "apply", str(HERE / "xsched-engagement.patch")], check=True)
        elif patch_body(current_diff) != reviewed_patch:
            raise RuntimeError("refusing to build unexpected XSched source diff")
        build_env = {**os.environ, "CC": "/usr/bin/gcc", "CXX": "/usr/bin/g++"}
        subprocess.run(["make", "-C", str(XSCHED), "cuda"], check=True, env=build_env)
        subprocess.run(["make", "-B", "-C", str(EXTENSION), "uprobe_preempt_multi",
                        "gpu_sched_set_timeslices"], check=True, env=build_env)
        post = admission(require_runtime=False)
        if not post["admitted"]:
            raise RuntimeError("post-build admission failed: " + "; ".join(post["errors"]))
        print(json.dumps(post, indent=2))
        return 0
    if args.phase == "analyze":
        if not args.output:
            parser.error("--output is required for analyze")
        print(json.dumps(analyze(args.output), indent=2))
        return 0

    if args.phase == "admission":
        checks = admission(require_runtime=True)
        print(json.dumps(checks, indent=2))
        return 0 if checks["admitted"] else 2
    if args.phase != "calibrate" and not args.reps:
        parser.error("--reps is required and must be frozen by an isolated ~80 ms calibration")
    lease = shared.LeaseSet.acquire()
    try:
        checks = admission(require_runtime=True)
        print(json.dumps(checks, indent=2), flush=True)
        if not checks["admitted"]:
            return 2
        return execute_phase(args, checks)
    finally:
        lease.close()


def execute_phase(args: argparse.Namespace, checks: dict[str, Any]) -> int:
    if args.phase == "calibrate":
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        root = args.output or RAW / f"calibration-{timestamp}"
        root.mkdir(parents=True, exist_ok=False)
        (root / "admission.json").write_text(json.dumps(checks, indent=2) + "\n")
        # calibrate_reps owns its attempt directory, not the already-created root.
        record = calibrate_reps(root / "attempts", args.reps or 1_000_000,
                                args.blocks, args.threads, args.timeout)
        (root / "calibration.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))
        return 0
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    root = args.output or RAW / f"{args.phase}-{timestamp}"
    root.mkdir(parents=True, exist_ok=False)
    (root / "admission.json").write_text(json.dumps(checks, indent=2) + "\n")
    repetitions = {"pilot": 5, "preflight": 1}.get(args.phase, REPETITIONS)
    protocol = {
        "phase": args.phase, "repetitions": repetitions, "seed": SEED,
        "tasks_per_stream": min(args.tasks, 2) if args.phase == "preflight" else args.tasks,
        "streams_per_process": 4, "lc_processes": 2, "be_processes": 4,
        "configs": args.configs, "reps": args.reps, "blocks": args.blocks,
        "threads": args.threads, "timeout_seconds_per_cell": args.timeout,
        "full_50_kernel_10_block_protocol": args.phase == "full" and args.tasks == 50,
        "short_budget_difference": (
            "5 complete blocks and 5 kernels/stream instead of 10 blocks and 50 kernels/stream; "
            "40 LC samples/cell means nearest-rank P99 is the sample maximum"
        ) if args.phase == "pilot" else None,
        "kernel_target_ms": 80, "xsched_level": 1,
        "candidate_policy_differences": {
            "gpubpf_nocooldown": "old timeslices, every LC cuLaunchKernel preempts all four BE GR targets; cooldown=0",
            "gpubpf_interleave": "old timeslices and 100 us preemption cooldown; LC interleave=2, BE interleave=0, verified at bind",
        } if any(config not in CONFIGS for config in args.configs) else {},
    }
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    if args.phase == "preflight":
        results = []
        for config in args.configs:
            result = execute_round(config, root / config, args.reps, min(args.tasks, 2),
                                   args.blocks, args.threads, args.timeout)
            result["block"] = 0
            (root / config / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            results.append(result)
            print(json.dumps(result), flush=True)
    else:
        rng = random.Random(SEED)
        if args.phase == "full":
            for role in ("lc", "be"):
                for repetition in range(3):
                    execute_isolated(role, root / f"control-{role}-{repetition}", args.reps,
                                     args.tasks, args.blocks, args.threads, args.timeout)
        for block in range(repetitions):
            order = list(args.configs)
            rng.shuffle(order)
            for order_index, config in enumerate(order):
                result = execute_round(config, root / f"block-{block:02d}-{order_index}-{config}",
                                       args.reps, args.tasks, args.blocks, args.threads, args.timeout)
                result.update({"block": block, "order_index": order_index})
                result_path = root / f"block-{block:02d}-{order_index}-{config}" / "result.json"
                result_path.write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(result), flush=True)
        print(json.dumps(analyze(root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
