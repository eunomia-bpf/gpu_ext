#!/usr/bin/env python3
"""Admission-gated RQ3 runner for native, XSched Level-1, and gpubpf."""

from __future__ import annotations

import argparse
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
EXPECTED_DRIVER = "575.57.08"
XSCHED_COMMIT = "f49289f0220931df78de948ed841ecbaf960a919"
SEED = 1797
CONFIGS = ("native", "xsched", "gpubpf")


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
    expected_xsched_diff = (HERE / "xsched-engagement.patch").read_text()
    actual_xsched_diff = run(["git", "-C", str(XSCHED), "diff", "--", "preempt"]).stdout
    source_paths = [
        EXTENSION / "uprobe_preempt_multi.c", EXTENSION / "uprobe_preempt_multi.bpf.c",
        EXTENSION / "gpu_sched_set_timeslices.c", EXTENSION / "gpu_sched_set_timeslices.bpf.c",
        EXTENSION / "gpu_sched_set_timeslices.h", EXTENSION / "Makefile",
    ]
    checks: dict[str, Any] = {
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
            btf = run(sudo + ["bpftool", "btf", "dump", "file", "/sys/kernel/btf/nvidia", "format", "c"], check=False)
            checks["nvidia_btf_probe_rc"] = btf.returncode
            checks["has_nv_gpu_sched_ops"] = "struct nv_gpu_sched_ops" in btf.stdout
            checks["has_preempt_kfunc"] = "bpf_nv_gpu_preempt_tsg" in btf.stdout
            if not checks["has_nv_gpu_sched_ops"] or not checks["has_preempt_kfunc"]:
                errors.append("NVIDIA BTF lacks nv_gpu_sched_ops or bpf_nv_gpu_preempt_tsg")
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
                self.proc.wait(timeout=8)

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


def start_policy(config: str, cpus: list[int], run_dir: Path) -> list[ManagedProcess]:
    policies: list[ManagedProcess] = []
    env = base_env()
    if config == "xsched":
        policies.append(ManagedProcess(
            "xserver", [str(XSCHED_OUTPUT / "bin" / "xserver"), "HPF", "50000"], env, cpus[0]))
        time.sleep(0.5)
        if policies[0].proc.poll() is not None:
            raise RuntimeError("xserver failed to stay running")
    elif config == "gpubpf":
        sudo = sudo_prefix()
        policies.append(ManagedProcess(
            "timeslice", sudo + [str(EXTENSION / "gpu_sched_set_timeslices"),
                                  "-p", "bench_lc:1000000", "-p", "bench_be:200"], env, cpus[0]))
        policies.append(ManagedProcess(
            "preempt", sudo + [str(EXTENSION / "uprobe_preempt_multi"),
                                "--be-name", "bench_be", "--lc-name", "bench_lc",
                                "--cooldown-us", "100"], env, cpus[1]))
        time.sleep(1.0)
        if any(policy.proc.poll() is not None for policy in policies):
            raise RuntimeError("gpubpf policy failed to stay running")
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


def write_process_log(path: Path, process: ManagedProcess) -> None:
    path.write_text(json.dumps({
        "name": process.name, "command": process.proc.args, "returncode": process.proc.returncode,
        "stdout": process.stdout_lines, "stderr": process.stderr_lines,
    }, indent=2) + "\n")


def execute_round(config: str, run_dir: Path, reps: int, tasks: int, blocks: int,
                  threads: int, timeout: float) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    cpus = allowed_cpus(10)
    clock = clock_calibration()
    policies = start_policy(config, cpus, run_dir)
    workers: list[ManagedProcess] = []
    try:
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
        running_events = [worker.wait_event("running", timeout) for worker in be_workers]
        all_be_running_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        fixed_delay_ns = 5_000_000
        deadline = all_be_running_ns + fixed_delay_ns
        while time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) < deadline:
            pass
        lc_release_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        for worker in lc_workers:
            worker.send("GO")

        results = [worker.wait_event("result", timeout) for worker in workers]
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
    elif config == "gpubpf":
        timeslice_text = "\n".join(policies[0].stdout_lines + policies[0].stderr_lines)
        preempt_text = "\n".join(policies[1].stdout_lines + policies[1].stderr_lines)
        modified = re.findall(r"timeslice_mod:\s+(\d+)", timeslice_text)
        preempt_ok = re.findall(r"preempt_ok:\s+(\d+)", preempt_text)
        preempt_err = re.findall(r"preempt_err:\s+(\d+)", preempt_text)
        if not modified or int(modified[-1]) < 6:
            raise RuntimeError("gpubpf did not modify timeslices for all six processes")
        if not preempt_ok or int(preempt_ok[-1]) == 0 or (preempt_err and int(preempt_err[-1]) != 0):
            raise RuntimeError("gpubpf preemption engagement gate failed")
        engagement = {"timeslice_mod": int(modified[-1]), "preempt_ok": int(preempt_ok[-1])}

    result = {
        "config": config, "reps": reps, "tasks_per_stream": tasks, "blocks": blocks,
        "threads": threads, "be_release_ns": be_release_ns,
        "all_be_running_ns": all_be_running_ns, "lc_release_ns": lc_release_ns,
        "be_running_events": running_events,
        "lc_samples": len(lc_samples), "be_completed": be_count,
        "lc_p99_us": percentile(lc_samples, 0.99),
        "lc_completion_p99_us": percentile(lc_completion, 0.99),
        "be_throughput_kernels_s": be_count / ((be_end_ns - be_release_ns) / 1e9),
        "outputs_validated_per_process": expected_values,
        "clock_calibration": clock,
        "engagement": engagement,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def calibrate_reps(root: Path, initial_reps: int, blocks: int, threads: int,
                   timeout: float) -> dict[str, Any]:
    """Tune only fixed compute repetitions; the accepted value is then frozen."""
    root.mkdir(parents=True, exist_ok=False)
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


def execute_isolated(role: str, run_dir: Path, reps: int, tasks: int, blocks: int,
                     threads: int, timeout: float) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
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
    record = {
        "control": f"isolated-{role}", "samples": len(latencies),
        "outputs_validated": result["outputs_validated"],
        "completion_host_ns": result["completion_host_ns"],
        "clock_calibration": clock,
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


def analyze(root: Path) -> dict[str, Any]:
    records = []
    for path in sorted(root.glob("**/result.json")):
        record = json.loads(path.read_text())
        if "block" in record:
            records.append(record)
    by_block: dict[int, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_block.setdefault(record["block"], {})[record["config"]] = record
    complete = {block: values for block, values in by_block.items() if set(values) == set(CONFIGS)}
    if len(complete) != 10:
        raise RuntimeError(f"analysis requires 10 complete randomized blocks, found {len(complete)}")
    summary: dict[str, Any] = {"complete_blocks": len(complete), "configs": {}, "paired": {}}
    for config in CONFIGS:
        lc = [complete[block][config]["lc_p99_us"] for block in sorted(complete)]
        be = [complete[block][config]["be_throughput_kernels_s"] for block in sorted(complete)]
        summary["configs"][config] = {
            "lc_p99_median_us": statistics.median(lc), "lc_p99_mean_ci95": bootstrap_ci(lc, SEED),
            "be_throughput_median": statistics.median(be), "be_throughput_mean_ci95": bootstrap_ci(be, SEED + 1),
        }
    for baseline in ("native", "xsched"):
        lc_delta = [complete[b]["gpubpf"]["lc_p99_us"] - complete[b][baseline]["lc_p99_us"] for b in sorted(complete)]
        be_ratio = [complete[b]["gpubpf"]["be_throughput_kernels_s"] /
                    complete[b][baseline]["be_throughput_kernels_s"] - 1.0 for b in sorted(complete)]
        summary["paired"][f"gpubpf_vs_{baseline}"] = {
            "lc_p99_delta_us_mean_ci95": bootstrap_ci(lc_delta, SEED + 2),
            "be_throughput_relative_mean_ci95": bootstrap_ci(be_ratio, SEED + 3),
        }
    summary["be_noninferiority_margin"] = -0.05
    xsched_comparison = summary["paired"]["gpubpf_vs_xsched"]
    latency_ci = xsched_comparison["lc_p99_delta_us_mean_ci95"]
    throughput_ci = xsched_comparison["be_throughput_relative_mean_ci95"]
    if latency_ci[1] < 0 and throughput_ci[0] >= -0.05:
        decision = "positive"
    elif latency_ci[0] >= 0 or throughput_ci[1] < -0.05:
        decision = "negative"
    elif (latency_ci[1] < 0) != (throughput_ci[0] >= -0.05):
        decision = "mixed"
    else:
        decision = "inconclusive"
    summary["predeclared_decision"] = {
        "classification": decision,
        "positive_rule": "LC paired-difference CI upper < 0 and BE relative-throughput CI lower >= -0.05",
        "negative_rule": "LC CI lower >= 0 or BE relative-throughput CI upper < -0.05",
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("build", "admission", "calibrate", "preflight", "full", "analyze"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reps", type=int)
    parser.add_argument("--tasks", type=int, default=50)
    parser.add_argument("--blocks", type=int, default=340)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()
    if args.phase == "build":
        subprocess.run(["make", "-C", str(HERE)], check=True)
        current_diff = run(["git", "-C", str(XSCHED), "diff", "--", "preempt"]).stdout
        reviewed_patch = (HERE / "xsched-engagement.patch").read_text()
        if not current_diff:
            subprocess.run(["git", "-C", str(XSCHED), "apply", str(HERE / "xsched-engagement.patch")], check=True)
        elif current_diff != reviewed_patch:
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

    checks = admission(require_runtime=True)
    print(json.dumps(checks, indent=2))
    if not checks["admitted"]:
        return 2
    if args.phase == "admission":
        return 0
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
    if not args.reps:
        parser.error("--reps is required and must be frozen by an isolated ~80 ms calibration")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    root = args.output or RAW / f"{args.phase}-{timestamp}"
    root.mkdir(parents=True, exist_ok=False)
    (root / "admission.json").write_text(json.dumps(checks, indent=2) + "\n")

    if args.phase == "preflight":
        results = []
        for config in CONFIGS:
            result = execute_round(config, root / config, args.reps, min(args.tasks, 2),
                                   args.blocks, args.threads, args.timeout)
            result["block"] = 0
            (root / config / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            results.append(result)
    else:
        rng = random.Random(SEED)
        for role in ("lc", "be"):
            for repetition in range(3):
                execute_isolated(role, root / f"control-{role}-{repetition}", args.reps,
                                 args.tasks, args.blocks, args.threads, args.timeout)
        for block in range(10):
            order = list(CONFIGS)
            rng.shuffle(order)
            for order_index, config in enumerate(order):
                result = execute_round(config, root / f"block-{block:02d}-{order_index}-{config}",
                                       args.reps, args.tasks, args.blocks, args.threads, args.timeout)
                result.update({"block": block, "order_index": order_index})
                result_path = root / f"block-{block:02d}-{order_index}-{config}" / "result.json"
                result_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(analyze(root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
