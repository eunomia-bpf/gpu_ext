#!/usr/bin/env python3
"""XSched Level-2 sm_120 native-C/BPF tool-actuator pair runner (bring-up).

Runs the existing bounded priority scheduling workload (2 LC + 4 BE
processes, 4 streams each, the existing ready/GO/running protocol) in
three matched configurations:

  baseline     plain CUDA; no XSched, no NVBit tool
  native_port  deps/xsched/output/bin/xserver HPF 50000 (native HPF) +
               Level-2 XQueues + the preloaded NVBit tool actuator,
               XG_DECISION=native (device decision compiled as native C)
  bpf_port     workloads/xsched/build/xserver-bpftime HPF 50000 with
               GPUBPF_HPF_CODE=workloads/xsched/build/bpftime_hpf.bin +
               the same Level-2 queues and tool actuator,
               XG_DECISION=bpf (device decision compiled as eBPF)

The two port arms share the same trusted NVBit actuator; host HPF and device
decision implementations differ. This is a port bring-up of the Level-2 tool-actuator
path on sm_120, not an upstream cuXtra Level-3 reproduction.

Workers run in the opt-in service-only output mode (XG_SERVICE_ONLY=1):
per-kernel GPU exit-entry service and host start->completion elapsed on
one host clock. No submit/queue/cross-clock metric is read or emitted.
The default legacy workload behavior is unchanged.

The runner owns only its own child processes (xserver + six workers),
keeps per-process launch counts below the 8192-slot non-retiring tool
context pool, and saves raw outputs plus service metrics. It takes no GPU
lock, runs no preflight/audit framework, and builds nothing: root wraps
the GPU lock and supplies the rebuilt workload binary and the exact
compute_task target symbol.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import statistics
import subprocess
import sys
import threading
import time

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
XSCHED_DIR = HERE.parent
BUILD_DIR = XSCHED_DIR / "build"
DEPS_XSCHED = XSCHED_DIR / "deps" / "xsched"
XSERVER_NATIVE = DEPS_XSCHED / "output" / "bin" / "xserver"
XSERVER_BPFTIME = BUILD_DIR / "xserver-bpftime"
HPF_BIN = BUILD_DIR / "bpftime_hpf.bin"
GUARD_TOOL = XSCHED_DIR / "level2-build" / ".output" / "xsched_guard_tool.so"
HAL_LIB_DIR = XSCHED_DIR / "level2-build" / ".output" / "hal-tool-install-20260908" / "lib"
DEFAULT_WORKLOAD = BUILD_DIR / "priority_workload"

CONFIGS = ("baseline", "native_port", "bpf_port")
STREAMS = 4
LC_PROCESSES = 2
BE_PROCESSES = 4
HPF_QUANTUM = "50000"
FIXED_DELAY_NS = 5_000_000
TOOL_CTX_SLOTS = 8192
METRIC_SCOPE = "gpu_service_and_host_elapsed"


def resolve_component_paths(args) -> dict:
    def choose(value, default):
        return (Path(value).expanduser().resolve()
                if value else Path(default).expanduser().resolve())

    return {
        "xserver_native": choose(args.xserver_native, XSERVER_NATIVE),
        "xserver_bpftime": choose(args.xserver_bpftime, XSERVER_BPFTIME),
        "hpf_bin": choose(args.hpf_bin, HPF_BIN),
        "guard_tool": choose(args.guard_tool, GUARD_TOOL),
        "hal_lib_dir": choose(args.hal_lib_dir, HAL_LIB_DIR),
    }

XG_LOADED = re.compile(r"(?m)^XG tool loaded decision=(\S+) target=(.*)$")
XG_ENTRY = re.compile(
    r"(?m)^XG instrumented_entry function_idx=(\d+) entry_offset=(\d+) decision_mode=(\d+)$")
XG_DONE = re.compile(r"(?m)^XG done functions=(\d+) launches=(\d+)(?: [^\n]*)?$")


def monotonic_raw_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def percentile(values: list, fraction: float) -> int:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def clean_env() -> dict:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("XSCHED_") or key.startswith("XG_"):
            env.pop(key)
    for key in ("LD_PRELOAD", "LD_LIBRARY_PATH", "GPUBPF_HPF_CODE"):
        env.pop(key, None)
    return env


def runtime_environment(env: dict) -> dict:
    return {key: value for key, value in env.items()
            if key in ("LD_PRELOAD", "LD_LIBRARY_PATH", "GPUBPF_HPF_CODE")
            or key.startswith(("XSCHED_", "XG_", "CUDA_", "CUPTI_"))}


def allowed_cpus(required: int) -> list:
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < required:
        raise RuntimeError(f"need {required} allowed CPUs for fixed affinity, "
                           f"only {len(cpus)} available")
    return cpus[:required]


def server_spec(config: str, paths: dict):
    if config == "baseline":
        return None
    env = clean_env()
    if config == "native_port":
        return [str(paths["xserver_native"]), "HPF", HPF_QUANTUM], env
    if config == "bpf_port":
        env["GPUBPF_HPF_CODE"] = str(paths["hpf_bin"])
        return [str(paths["xserver_bpftime"]), "HPF", HPF_QUANTUM], env
    raise ValueError(f"unknown config: {config}")


def worker_env(config: str, role: str, target_symbol: str, paths: dict) -> dict:
    env = clean_env()
    if config != "baseline":
        env.update({
            "XSCHED_SCHEDULER": "GLB",
            "XSCHED_AUTO_XQUEUE": "ON",
            "XSCHED_AUTO_XQUEUE_LEVEL": "2",
            "XSCHED_AUTO_XQUEUE_PRIORITY": "1" if role == "lc" else "0",
            "XSCHED_AUTO_XQUEUE_THRESHOLD": "16" if role == "lc" else "4",
            "XSCHED_AUTO_XQUEUE_BATCH_SIZE": "8" if role == "lc" else "2",
            "XSCHED_CUDA_LV2_PORT_120": "1",
            "XSCHED_LEVEL2_TOOL_ACTUATOR": "1",
            "XG_DECISION": "native" if config == "native_port" else "bpf",
            "XG_TARGET_SYMBOL": target_symbol,
            "LD_PRELOAD": str(paths["guard_tool"]),
            "LD_LIBRARY_PATH": str(paths["hal_lib_dir"]),
        })
    env["XG_SERVICE_ONLY"] = "1"
    return env


class ManagedProcess:
    """One owned child process with raw line capture and JSON events."""

    def __init__(self, name: str, command: list, env: dict, cpus):
        self.name = name
        self.runtime_environment = runtime_environment(env)
        self.stdout_lines: list = []
        self.stderr_lines: list = []
        self.events = queue.Queue()
        cpu_mask = str(cpus) if isinstance(cpus, int) else ",".join(map(str, cpus))
        # Load instrumentation in the workload, not in taskset before exec.
        launcher_env = env.copy()
        preload = launcher_env.pop("LD_PRELOAD", None)
        delayed_preload = ["/usr/bin/env", f"LD_PRELOAD={preload}"] if preload else []
        full = ["taskset", "-c", cpu_mask] + delayed_preload + list(command)
        self.proc = subprocess.Popen(
            full, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, encoding="utf-8", errors="backslashreplace",
            env=launcher_env, start_new_session=True,
        )
        self.threads = [
            threading.Thread(target=self._reader, args=(self.proc.stdout, self.stdout_lines, True),
                             daemon=True),
            threading.Thread(target=self._reader, args=(self.proc.stderr, self.stderr_lines, False),
                             daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    def _reader(self, pipe, target: list, parse_json: bool) -> None:
        for line in iter(pipe.readline, ""):
            line = line.rstrip("\n")
            target.append(line)
            if parse_json and line.startswith("{"):
                try:
                    self.events.put(json.loads(line))
                except json.JSONDecodeError:
                    pass

    def send(self, command: str) -> None:
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    @staticmethod
    def is_service_result(event: dict) -> bool:
        return (event.get("service_only") is True
                or event.get("metric_scope") == METRIC_SCOPE)

    def wait_event(self, expected: str) -> dict:
        while True:
            if self.proc.poll() is not None:
                time.sleep(0.5)
                if self.events.empty():
                    raise RuntimeError(
                        f"{self.name} exited {self.proc.returncode} before {expected}: "
                        + "; ".join(self.stderr_lines[-8:]))
            try:
                event = self.events.get(timeout=0.2)
            except queue.Empty:
                continue
            if event.get("event") == expected:
                return event

    def wait_service_result(self) -> dict:
        while True:
            if self.proc.poll() is not None:
                time.sleep(0.5)
                if self.events.empty():
                    raise RuntimeError(
                        f"{self.name} exited {self.proc.returncode} before the service result: "
                        + "; ".join(self.stderr_lines[-8:]))
            try:
                event = self.events.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.is_service_result(event):
                return event

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

    def collect(self) -> int:
        rc = self.proc.wait()
        for thread in self.threads:
            thread.join(timeout=1)
        return rc


def write_process_log(path: Path, process: ManagedProcess) -> None:
    path.write_text(json.dumps({
        "name": process.name,
        "command": process.proc.args,
        "returncode": process.proc.returncode,
        "stdout": process.stdout_lines,
        "stderr": process.stderr_lines,
        "runtime_environment": process.runtime_environment,
    }, indent=2) + "\n")


def validate_service_result(event: dict, name: str, role: str, process_id: int,
                            streams: int, tasks: int, blocks: int, threads: int) -> None:
    samples = event.get("samples")
    if not isinstance(samples, list) or len(samples) != streams * tasks:
        raise RuntimeError(f"{name}: expected {streams * tasks} samples, "
                           f"got {len(samples) if isinstance(samples, list) else samples!r}")
    for sample in samples:
        if (not all(isinstance(sample.get(field), int)
                    for field in ("entry_ns", "exit_ns", "service_ns"))
                or sample["service_ns"] < 0 or sample["exit_ns"] < sample["entry_ns"]):
            raise RuntimeError(f"{name}: invalid sample: {sample}")
    if event.get("role") not in (None, role) or event.get("process_id") not in (None, process_id):
        raise RuntimeError(f"{name}: event identity mismatch: "
                           f"{event.get('role')}/{event.get('process_id')}")
    expected_values = streams * tasks * blocks * threads
    if "outputs_validated" in event and event["outputs_validated"] != expected_values:
        raise RuntimeError(f"{name}: outputs_validated is {event['outputs_validated']}, "
                           f"expected {expected_values}")


def parse_worker_xg(stderr_lines: list, decision: str, target_symbol: str,
                    expected_launches: int) -> dict:
    text = "\n".join(stderr_lines)
    loaded = XG_LOADED.findall(text)
    entries = XG_ENTRY.findall(text)
    done = XG_DONE.findall(text)
    if len(loaded) != 1:
        raise RuntimeError(f"expected exactly one 'XG tool loaded' line: {loaded}")
    if loaded[0][0] != decision:
        raise RuntimeError(f"XG decision is {loaded[0][0]!r}, expected {decision!r}")
    if loaded[0][1] != target_symbol:
        raise RuntimeError(f"XG target is {loaded[0][1]!r}, expected {target_symbol!r}")
    if not entries:
        raise RuntimeError("the NVBit tool did not report an instrumented entry")
    if len(done) != 1:
        raise RuntimeError(f"expected exactly one 'XG done' line: {done}")
    functions, launches = (int(value) for value in done[0])
    if functions != 1:
        raise RuntimeError(f"XG done functions={functions} launches={launches}; "
                           "expected functions=1")
    # Filtered callbacks include replay launches and are not a task count.
    return {"decision": decision, "target_symbol": target_symbol,
            "instrumented_entries": len(entries), "functions": functions,
            "launches": launches, "submitted_tasks": expected_launches}


def run_cell(config: str, block: int, block_type: str, run_dir: Path, workload: Path,
             streams: int, tasks: int, reps: int, blocks: int, threads: int,
             target_symbol: str, paths: dict) -> dict:
    cpus = allowed_cpus(10)
    server = None
    workers = []
    try:
        spec = server_spec(config, paths)
        if spec is not None:
            command, env = spec
            server = ManagedProcess("xserver", command, env, cpus[0])
            time.sleep(0.5)
            if server.proc.poll() is not None:
                raise RuntimeError("xserver failed to stay running: "
                                   + "; ".join(server.stderr_lines[-8:]))

        shapes = [("be", pid) for pid in range(1, BE_PROCESSES + 1)] \
            + [("lc", pid) for pid in range(1, LC_PROCESSES + 1)]
        for role, pid in shapes:
            command = [str(workload), role, str(pid), str(streams), str(tasks),
                       str(reps), str(blocks), str(threads),
                       "1" if role == "be" else "0", "0"]
            workers.append(ManagedProcess(f"{role}{pid}", command,
                                          worker_env(config, role, target_symbol,
                                           paths), cpus[2:]))
        for worker in workers:
            worker.wait_event("ready")

        be_workers = workers[:BE_PROCESSES]
        lc_workers = workers[BE_PROCESSES:]
        be_go_ns = monotonic_raw_ns()
        for worker in be_workers:
            worker.send("GO")
        running_events = [worker.wait_event("running") for worker in be_workers]
        all_be_running_ns = monotonic_raw_ns()
        while monotonic_raw_ns() < all_be_running_ns + FIXED_DELAY_NS:
            pass
        lc_go_ns = monotonic_raw_ns()
        for worker in lc_workers:
            worker.send("GO")

        results = []
        for worker in workers:
            event = worker.wait_service_result()
            # Preserve observations; diagnostics do not reject timing rows.
            try:
                validate_service_result(event, worker.name, worker.name[:2],
                                        int(worker.name[2:]), streams, tasks, blocks, threads)
            except RuntimeError as exc:
                event["sample_diagnostic"] = str(exc)
            results.append(event)
        for worker in workers:
            if worker.collect() != 0:
                raise RuntimeError(f"{worker.name} exited {worker.proc.returncode}")
    finally:
        for worker in workers:
            if worker.proc.poll() is None:
                worker.stop()
        if server is not None:
            server.stop()
        for process in workers + ([server] if server is not None else []):
            write_process_log(run_dir / f"{process.name}.json", process)

    engagement = {}
    if config != "baseline":
        decision = "native" if config == "native_port" else "bpf"
        for worker in workers:
            try:
                engagement[worker.name] = parse_worker_xg(worker.stderr_lines, decision,
                                                          target_symbol, streams * tasks)
            except RuntimeError as exc:
                # Replay can add launches. Raw counters remain in worker logs.
                engagement[worker.name] = {"diagnostic": str(exc)}
        text = "\n".join(server.stdout_lines + server.stderr_lines)
        if text.count("set priority 1") < LC_PROCESSES * streams \
                or text.count("set priority 0") < BE_PROCESSES * streams:
            engagement["priority_log_diagnostic"] = "expected priority messages not all observed"

    def stats_for(prefix: str) -> dict:
        values = [sample["service_ns"]
                  for worker, event in zip(workers, results)
                  if worker.name.startswith(prefix)
                  for sample in event["samples"]]
        return {"kernels": len(values),
                "mean_us": statistics.mean(values) / 1000.0,
                "p50_us": percentile(values, 0.50) / 1000.0,
                "p95_us": percentile(values, 0.95) / 1000.0,
                "p99_us": percentile(values, 0.99) / 1000.0}

    lc = stats_for("lc")
    be = stats_for("be")
    lc_end_ns = max(event["completion_host_ns"]
                    for worker, event in zip(workers, results) if worker.name.startswith("lc"))
    be_end_ns = max(event["completion_host_ns"]
                    for worker, event in zip(workers, results) if worker.name.startswith("be"))

    record = {
        "block": block, "block_type": block_type, "config": config,
        "metric_scope": METRIC_SCOPE,
        "decision": None if config == "baseline" else ("native" if config == "native_port" else "bpf"),
        "target_symbol": target_symbol if config != "baseline" else None,
        "workload": str(workload),
        "streams": streams, "lc_processes": LC_PROCESSES, "be_processes": BE_PROCESSES,
        "tasks_per_stream": tasks, "reps": reps, "blocks": blocks, "threads": threads,
        "be_go_ns": be_go_ns, "all_be_running_ns": all_be_running_ns, "lc_go_ns": lc_go_ns,
        "be_running_events": running_events,
        "lc_service": lc,
        "lc_host_elapsed_ns": lc_end_ns - lc_go_ns,
        "be_service": be,
        "be_host_elapsed_ns": be_end_ns - be_go_ns,
        "be_kernels_per_s": be["kernels"] / ((be_end_ns - be_go_ns) / 1e9),
        "engagement": engagement,
        "sample_diagnostics": [event["sample_diagnostic"] for event in results
                               if "sample_diagnostic" in event],
    }
    (run_dir / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def required_files(configs, workload: Path, paths: dict) -> list:
    required = [workload]
    if any(config != "baseline" for config in configs):
        required += [paths["guard_tool"],
                     paths["hal_lib_dir"] / "libshimcuda.so",
                     paths["hal_lib_dir"] / "libhalcuda.so",
                     paths["hal_lib_dir"] / "libpreempt.so",
                     paths["hal_lib_dir"] / "libcuda.so.1"]
    if "native_port" in configs:
        required.append(paths["xserver_native"])
    if "bpf_port" in configs:
        required += [paths["xserver_bpftime"], paths["hpf_bin"]]
    return [str(path) for path in required if not path.is_file()]


def build_protocol(args, workload: Path, paths: dict) -> dict:
    return {
        "phase": "level2-tool-actuator-pair",
        "scope": ("sm_120 Level-2 tool-actuator port bring-up: native-C vs device-BPF "
                  "decisions on the same trusted NVBit actuator; not an upstream cuXtra "
                  "reproduction; service-only metrics, no queue or cross-clock metric"),
        "initial_run": not args.no_initial,
        "pair_blocks": args.repetitions,
        "config_order": list(args.configs),
        "streams": STREAMS, "lc_processes": LC_PROCESSES, "be_processes": BE_PROCESSES,
        "tasks_per_stream": args.tasks, "reps": args.reps,
        "blocks": args.blocks, "threads": args.threads,
        "fixed_delay_ns": FIXED_DELAY_NS,
        "tool_ctx_slots": TOOL_CTX_SLOTS,
        "target_symbol": args.target_symbol,
        "clock_offset_ns": 0,
        "metric_scope": METRIC_SCOPE,
        "worker_environments": {
            config: {role: runtime_environment(
                worker_env(config, role, args.target_symbol, paths))
                for role in ("lc", "be")} for config in CONFIGS
        },
        "paths": {
            "workload": str(workload),
            "xserver_native": str(paths["xserver_native"]),
            "xserver_bpftime": str(paths["xserver_bpftime"]),
            "bpftime_hpf_code": str(paths["hpf_bin"]),
            "guard_tool": str(paths["guard_tool"]),
            "hal_lib_dir": str(paths["hal_lib_dir"]),
        },
    }


def summarize(out_root: Path, protocol: dict) -> dict:
    records = [json.loads(path.read_text())
               for path in sorted(out_root.glob("block-*/result.json"))]
    blocks = sorted({record["block"] for record in records
                     if record["block_type"] == "paired"})
    by = {(record["block"], record["config"]): record for record in records}
    summary = {
        "metric_scope": METRIC_SCOPE,
        "complete_pair_blocks": [block for block in blocks
                                 if all((block, config) in by
                                        for config in protocol["config_order"])],
        "configs": {},
        "paired": {},
        "scope": protocol["scope"],
    }
    for config in protocol["config_order"]:
        rows = [by[(block, config)] for block in blocks if (block, config) in by]
        if not rows:
            continue
        summary["configs"][config] = {
            "pair_blocks": len(rows),
            "lc_service_p99_median_us": statistics.median(row["lc_service"]["p99_us"]
                                                          for row in rows),
            "lc_service_mean_median_us": statistics.median(row["lc_service"]["mean_us"]
                                                           for row in rows),
            "be_service_p99_median_us": statistics.median(row["be_service"]["p99_us"]
                                                          for row in rows),
            "be_kernels_per_s_median": statistics.median(row["be_kernels_per_s"]
                                                         for row in rows),
            "lc_host_elapsed_median_ns": statistics.median(row["lc_host_elapsed_ns"]
                                                           for row in rows),
            "be_host_elapsed_median_ns": statistics.median(row["be_host_elapsed_ns"]
                                                           for row in rows),
        }
    if "native_port" in summary["configs"] and "bpf_port" in summary["configs"]:
        deltas = [by[(block, "bpf_port")]["lc_service"]["p99_us"]
                  - by[(block, "native_port")]["lc_service"]["p99_us"]
                  for block in blocks
                  if (block, "native_port") in by and (block, "bpf_port") in by]
        if deltas:
            summary["paired"]["bpf_port_vs_native_port"] = {
                "metric": "lc_service_p99_us_delta",
                "per_block_us": deltas,
                "mean_us": statistics.mean(deltas),
            }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_configs(text: str):
    configs = tuple(item.strip() for item in text.split(",") if item.strip())
    if not configs:
        raise argparse.ArgumentTypeError("configuration list must be nonempty")
    if len(configs) != len(set(configs)):
        raise argparse.ArgumentTypeError("duplicate configurations")
    unknown = set(configs) - set(CONFIGS)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown configurations: {sorted(unknown)}")
    return configs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="XSched Level-2 sm_120 tool-actuator pair runner (bring-up)")
    parser.add_argument("phase", choices=("run",))
    parser.add_argument("--output", type=Path, default=None,
                        help="new output directory "
                             "(default workloads/xsched/raw/level2-tool-pair-<timestamp>)")
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD,
                        help="root-built priority_workload binary")
    parser.add_argument("--xserver-native", type=Path, default=None,
                         help="override the native xserver binary")
    parser.add_argument("--xserver-bpftime", type=Path, default=None,
                         help="override the bpftime xserver binary")
    parser.add_argument("--hpf-bin", type=Path, default=None,
                         help="override the bpftime HPF binary")
    parser.add_argument("--guard-tool", type=Path, default=None,
                         help="override the NVBit guard tool shared library")
    parser.add_argument("--hal-lib-dir", type=Path, default=None,
                         help="override the isolated HAL library directory")
    parser.add_argument("--reps", type=int, required=True,
                        help="kernel recurrence repetitions; set directly, no calibration")
    parser.add_argument("--tasks", type=int, default=5,
                        help="kernels per stream (default 5)")
    parser.add_argument("--blocks", type=int, default=340,
                        help="grid blocks per kernel (default 340)")
    parser.add_argument("--threads", type=int, default=256,
                        help="threads per block (default 256)")
    parser.add_argument("--repetitions", type=int, default=5,
                        help="paired blocks after the one initial run (default 5)")
    parser.add_argument("--no-initial", action="store_true",
                        help="run only the paired blocks")
    parser.add_argument("--configs", type=parse_configs, default=CONFIGS,
                        help="comma-separated subset of baseline,native_port,bpf_port "
                             "in run order")
    parser.add_argument("--target-symbol", required=True,
                        help="exact mangled or demangled compute_task symbol "
                             "for XG_TARGET_SYMBOL")
    args = parser.parse_args()

    if args.reps < 1 or args.tasks < 1 or args.blocks < 1 or args.threads < 1:
        parser.error("reps/tasks/blocks/threads must be positive")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if STREAMS * args.tasks >= TOOL_CTX_SLOTS:
        parser.error(f"streams*tasks ({STREAMS * args.tasks}) must stay below the "
                     f"{TOOL_CTX_SLOTS}-slot non-retiring tool context pool")

    paths = resolve_component_paths(args)
    missing = required_files(args.configs, args.workload, paths)
    if missing:
        print(json.dumps({"error": "missing required components", "missing": missing},
                         indent=2))
        return 2

    out_root = args.output or (XSCHED_DIR / "raw"
                               / f"level2-tool-pair-{time.strftime('%Y%m%d_%H%M%S')}")
    out_root.mkdir(parents=True, exist_ok=False)
    protocol = build_protocol(args, args.workload, paths)
    (out_root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(protocol, indent=2), flush=True)

    schedule = []
    if not args.no_initial:
        schedule.append((0, "initial"))
    for block in range(1, args.repetitions + 1):
        schedule.append((block, "paired"))

    for block, block_type in schedule:
        first = max(0, block - 1) % len(args.configs)
        for config in args.configs[first:] + args.configs[:first]:
            cell_dir = out_root / f"block-{block:02d}-{config}"
            cell_dir.mkdir(parents=True, exist_ok=False)
            try:
                record = run_cell(config, block, block_type, cell_dir, args.workload,
                                  STREAMS, args.tasks, args.reps, args.blocks, args.threads,
                                  args.target_symbol, paths)
                print(json.dumps(record), flush=True)
            except BaseException as exc:
                (cell_dir / "failure.json").write_text(json.dumps(
                    {"error_type": type(exc).__name__, "error": str(exc)}, indent=2) + "\n")
                raise

    print(json.dumps(summarize(out_root, protocol), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
