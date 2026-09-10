#!/usr/bin/env python3
"""XSched Level-2 sm_120 native-cuXtra-blob arm runner.

Single-arm bring-up (2026-09-09) and matched multi-arm route comparison
(2026-09-10): the original-actuator Level-2 route (captured SASS guardian,
LDC-extracted resume blob, type-2 resume relaunches) against a same-source
Level-1 control, the native-driver baseline, and the same cuXtra actuator
driven by BPF HPF host decisions.

Arms (all share one workload binary, one HPF policy setting, 2 LC + 4 BE
processes, 4 streams each):

  baseline     no XSched: no shim, no server (plain CUDA scheduling).
  l1_native    same HAL install, upstream Level-1 queue actuation
               (XSCHED_AUTO_XQUEUE_LEVEL=1), native xserver HPF; no SASS
               guardian, no resume blob, no type-2 relaunch.
  l2_cuxtra    the original cuXtra/SASS route: XSCHED_CUDA_LV2_PORT_120=1,
               tool actuator off, the bring-up opt-in forwarding
               (XG_NATIVE_META_EXTEND=1, XG_NATIVE_META_KPARAM=1, no
               XG_NATIVE_ORIGINAL_ENTRY_CONTROL).
  l2_bpfhost   same cuXtra actuator; host HPF decisions by xserver-bpftime
               with GPUBPF_HPF_CODE (device side unchanged).

The completed single bring-up cell (workloads/xsched/raw/
level2-native-retabs-20260909.oJDCbU) remains the historical record for the
bring-up; this file gained the extra arms and the rotation for the matched
campaign. Service-only metrics; no queue or cross-clock metric.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_tool_pair as rtp  # noqa: E402

XSCHED_DIR = HERE.parent.parent
LEVEL2_BUILD_OUT = XSCHED_DIR / "level2-build" / ".output"
NATIVE_HAL_INSTALL = LEVEL2_BUILD_OUT / "hal-native-20260908.i8na51" / "install"
NATIVE_HAL_LIB_DIR = NATIVE_HAL_INSTALL / "lib"
DEFAULT_WORKLOAD = LEVEL2_BUILD_OUT / "service-mismatch-20260908.sHSYtE" / "priority_workload"
ORIGINAL_MEASURED_WORKLOAD = LEVEL2_BUILD_OUT / "service-output-20260908.66ppYl" / "priority_workload"
BUILD_DIR = XSCHED_DIR / "build"
XSERVER_BPFTIME_DEFAULT = BUILD_DIR / "xserver-bpftime"
HPF_BIN_DEFAULT = BUILD_DIR / "bpftime_hpf.bin"

CONFIGS = ("baseline", "l1_native", "l2_cuxtra", "l2_bpfhost")
NOT_APPLICABLE = ("not_applicable: the native cuXtra blob route runs without the "
                  "NVBit tool, so no 'XG tool loaded/instrumented entry/done' "
                  "lines exist by design")

HAL_LIB_NAMES = ("libshimcuda.so", "libhalcuda.so", "libpreempt.so", "libcuda.so.1")


def default_paths() -> dict:
    """Component paths exactly as the runner hardcoded them before the CLI overrides."""
    return {
        "xserver_native": rtp.XSERVER_NATIVE,
        "native_hal_install_dir": NATIVE_HAL_INSTALL,
        "hal_lib_dir": NATIVE_HAL_LIB_DIR,
        "xserver_bpftime": XSERVER_BPFTIME_DEFAULT,
        "hpf_bin": HPF_BIN_DEFAULT,
    }


def resolve_paths(args) -> dict:
    """Resolve the CLI overrides (or the original defaults) once per invocation."""
    def choose(value, default):
        return (Path(value).expanduser().resolve()
                if value else Path(default).expanduser().resolve())

    hal_install = choose(args.hal_install_dir, NATIVE_HAL_INSTALL)
    # prefer the xserver shipped with the selected install; fall back to the
    # historical deps checkout binary when the install has no bin/ entry
    xserver_default = hal_install / "bin" / "xserver"
    if not xserver_default.is_file():
        xserver_default = rtp.XSERVER_NATIVE
    return {
        "xserver_native": choose(args.xserver_native, xserver_default),
        "native_hal_install_dir": hal_install,
        "hal_lib_dir": hal_install / "lib",
        "xserver_bpftime": choose(args.xserver_bpftime, XSERVER_BPFTIME_DEFAULT),
        "hpf_bin": choose(args.hpf_bin, HPF_BIN_DEFAULT),
    }


def _xsched_env(role: str, level: int) -> dict:
    env = {
        "XSCHED_SCHEDULER": "GLB",
        "XSCHED_AUTO_XQUEUE": "ON",
        "XSCHED_AUTO_XQUEUE_LEVEL": str(level),
        "XSCHED_AUTO_XQUEUE_PRIORITY": "1" if role == "lc" else "0",
        "XSCHED_AUTO_XQUEUE_THRESHOLD": "16" if role == "lc" else "4",
        "XSCHED_AUTO_XQUEUE_BATCH_SIZE": "8" if role == "lc" else "2",
    }
    if level == 2:
        # opt-in sm_120 Level-2 queue port + original cuXtra blob route:
        # tool actuator explicitly OFF
        env["XSCHED_CUDA_LV2_PORT_120"] = "1"
        env["XSCHED_LEVEL2_TOOL_ACTUATOR"] = "0"
        # bring-up opt-in forwarding (presence-based at the shim)
        env["XG_NATIVE_META_EXTEND"] = "1"
        env["XG_NATIVE_META_KPARAM"] = "1"
    return env


def worker_env(config: str, role: str, paths: dict = None) -> dict:
    """Worker environment for one arm (no target symbol, no tool preload)."""
    if paths is None:
        paths = default_paths()
    env = rtp.clean_env()
    if config != "baseline":
        env.update(_xsched_env(role, 1 if config == "l1_native" else 2))
        # shim isolation via LD_LIBRARY_PATH only (no NVBit LD_PRELOAD)
        env["LD_LIBRARY_PATH"] = str(paths["hal_lib_dir"])
    env["XG_SERVICE_ONLY"] = "1"
    return env


def server_spec(config: str, paths: dict = None):
    """(command, env) for the arm's HPF server; None for the baseline arm."""
    if paths is None:
        paths = default_paths()
    if config == "baseline":
        return None
    if config == "l2_bpfhost":
        env = rtp.clean_env()
        env["GPUBPF_HPF_CODE"] = str(paths["hpf_bin"])
        return [str(paths["xserver_bpftime"]), "HPF", rtp.HPF_QUANTUM], env
    return [str(paths["xserver_native"]), "HPF", rtp.HPF_QUANTUM], rtp.clean_env()


def required_files(workload: Path, configs: tuple, paths: dict = None) -> list:
    if paths is None:
        paths = default_paths()
    required = [workload]
    if any(config != "baseline" for config in configs):
        required = required + [paths["xserver_native"]] \
            + [paths["hal_lib_dir"] / name for name in HAL_LIB_NAMES]
    if "l2_bpfhost" in configs:
        required += [paths["xserver_bpftime"], paths["hpf_bin"]]
    return [str(path) for path in required if not path.is_file()]


def parse_audit(line: str):
    if not line.startswith("XSCHED_AUDIT"):
        return None
    fields = {}
    for part in line.split():
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    try:
        return {"level": int(fields["level"]),
                "threshold": int(fields["threshold"]),
                "batch": int(fields["batch"]),
                "suspend_ok": int(fields["suspend_ok"]),
                "resume_ok": int(fields["resume_ok"])}
    except (KeyError, ValueError):
        return None


def engagement_for(config: str, workers: list, server) -> dict:
    engagement = {"nvbit_entry_accounting": NOT_APPLICABLE}
    if server is not None:
        text = "\n".join(server.stdout_lines + server.stderr_lines)
        if text.count("set priority 1") < rtp.LC_PROCESSES * rtp.STREAMS \
                or text.count("set priority 0") < rtp.BE_PROCESSES * rtp.STREAMS:
            engagement["priority_log_diagnostic"] = \
                "expected priority messages not all observed"
    gates = {}
    per_worker = {}
    for worker in workers:
        stderr = "\n".join(worker.stderr_lines)
        audits = [audit for audit in (parse_audit(line)
                                      for line in worker.stderr_lines)
                  if audit is not None]
        type2 = stderr.count("probe 2: type=2")
        probes = stderr.count("native-700 probe")
        entry = {"queues_audited": len(audits),
                 "audit_levels": [audit["level"] for audit in audits],
                 "suspend_ok_sum": sum(audit["suspend_ok"] for audit in audits),
                 "resume_ok_sum": sum(audit["resume_ok"] for audit in audits),
                 "type2_resume_lines": type2,
                 "probe_lines": probes}
        per_worker[worker.name] = entry
        if config == "l2_cuxtra" or config == "l2_bpfhost":
            entry["expected"] = "level-2 queues audited; type-2 resume relaunch in BE logs"
            entry["gate"] = (len(audits) == rtp.STREAMS
                             and all(level == 2 for level in entry["audit_levels"]))
            if worker.name.startswith("be"):
                entry["gate"] = entry["gate"] and type2 >= 1
        elif config == "l1_native":
            entry["expected"] = "level-1 queues audited; no level-2 probe or type-2 lines"
            entry["gate"] = (len(audits) == rtp.STREAMS
                             and all(level == 1 for level in entry["audit_levels"])
                             and probes == 0 and type2 == 0)
        else:
            entry["expected"] = "no XSched audit or probe lines"
            entry["gate"] = len(audits) == 0 and probes == 0
        gates[worker.name] = bool(entry["gate"])
    if server is not None:
        gates["xserver_priority_log"] = \
            "priority_log_diagnostic" not in engagement
    engagement["workers"] = per_worker
    engagement["gates"] = gates
    return engagement


def run_cell(config: str, block: int, block_type: str, run_dir: Path,
             workload: Path, streams: int, tasks: int, reps: int, blocks: int,
             threads: int, paths: dict = None) -> dict:
    if paths is None:
        paths = default_paths()
    cpus = rtp.allowed_cpus(10)
    server = None
    workers = []
    try:
        spec = server_spec(config, paths)
        if spec is not None:
            command, env = spec
            server = rtp.ManagedProcess("xserver", command, env, cpus[0])
            time.sleep(0.5)
            if server.proc.poll() is not None:
                raise RuntimeError("xserver failed to stay running: "
                                   + "; ".join(server.stderr_lines[-8:]))

        shapes = [("be", pid) for pid in range(1, rtp.BE_PROCESSES + 1)] \
            + [("lc", pid) for pid in range(1, rtp.LC_PROCESSES + 1)]
        for role, pid in shapes:
            command = [str(workload), role, str(pid), str(streams), str(tasks),
                       str(reps), str(blocks), str(threads),
                       "1" if role == "be" else "0", "0"]
            workers.append(rtp.ManagedProcess(f"{role}{pid}", command,
                                              worker_env(config, role, paths),
                                              cpus[2:]))
        for worker in workers:
            worker.wait_event("ready")

        be_workers = workers[:rtp.BE_PROCESSES]
        lc_workers = workers[rtp.BE_PROCESSES:]
        be_go_ns = rtp.monotonic_raw_ns()
        for worker in be_workers:
            worker.send("GO")
        running_events = [worker.wait_event("running") for worker in be_workers]
        all_be_running_ns = rtp.monotonic_raw_ns()
        while rtp.monotonic_raw_ns() < all_be_running_ns + rtp.FIXED_DELAY_NS:
            pass
        lc_go_ns = rtp.monotonic_raw_ns()
        for worker in lc_workers:
            worker.send("GO")

        results = []
        for worker in workers:
            event = worker.wait_service_result()
            try:
                rtp.validate_service_result(event, worker.name, worker.name[:2],
                                             int(worker.name[2:]), streams, tasks,
                                             blocks, threads)
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
            rtp.write_process_log(run_dir / f"{process.name}.json", process)

    actuator = {"baseline": "none",
                "l1_native": "level1-shmqueue",
                "l2_cuxtra": "native-cuxtra-blob",
                "l2_bpfhost": "native-cuxtra-blob"}[config]
    decision = {"l2_bpfhost": "bpf"}.get(config, "native")

    def stats_for(prefix: str) -> dict:
        values = [sample["service_ns"]
                  for worker, event in zip(workers, results)
                  if worker.name.startswith(prefix)
                  for sample in event["samples"]]
        return {"kernels": len(values),
                "mean_us": statistics.mean(values) / 1000.0,
                "p50_us": rtp.percentile(values, 0.50) / 1000.0,
                "p95_us": rtp.percentile(values, 0.95) / 1000.0,
                "p99_us": rtp.percentile(values, 0.99) / 1000.0}

    lc = stats_for("lc")
    be = stats_for("be")
    lc_end_ns = max(event["completion_host_ns"]
                    for worker, event in zip(workers, results) if worker.name.startswith("lc"))
    be_end_ns = max(event["completion_host_ns"]
                    for worker, event in zip(workers, results) if worker.name.startswith("be"))

    record = {
        "block": block, "block_type": block_type, "config": config,
        "metric_scope": rtp.METRIC_SCOPE,
        "actuator": actuator,
        "host_policy": decision,
        "workload": str(workload),
        "streams": streams, "lc_processes": rtp.LC_PROCESSES,
        "be_processes": rtp.BE_PROCESSES,
        "tasks_per_stream": tasks, "reps": reps, "blocks": blocks, "threads": threads,
        "be_go_ns": be_go_ns, "all_be_running_ns": all_be_running_ns, "lc_go_ns": lc_go_ns,
        "be_running_events": running_events,
        "lc_service": lc,
        "lc_host_elapsed_ns": lc_end_ns - lc_go_ns,
        "be_service": be,
        "be_host_elapsed_ns": be_end_ns - be_go_ns,
        "be_kernels_per_s": be["kernels"] / ((be_end_ns - be_go_ns) / 1e9),
        "engagement": engagement_for(config, workers, server),
        "sample_diagnostics": [event["sample_diagnostic"] for event in results
                               if "sample_diagnostic" in event],
    }
    (run_dir / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def build_protocol(args, workload: Path, paths: dict = None) -> dict:
    if paths is None:
        paths = default_paths()
    return {
        "phase": "level2-native-route-comparison",
        "scope": ("sm_120 matched route comparison on the isolated native HAL "
                  "install: original cuXtra/SASS Level-2 actuator (l2_cuxtra, "
                  "l2_bpfhost with BPF host HPF) against the same-source "
                  "Level-1 queue actuator (l1_native) and the no-policy native "
                  "baseline; service-only metrics; no NVBit tool, no tool-entry "
                  "accounting"),
        "config_order": list(args.configs),
        "pair_blocks": args.repetitions,
        "preflight": args.tasks == 2 and args.repetitions == 1,
        "streams": rtp.STREAMS, "lc_processes": rtp.LC_PROCESSES,
        "be_processes": rtp.BE_PROCESSES,
        "tasks_per_stream": args.tasks, "reps": args.reps,
        "blocks": args.blocks, "threads": args.threads,
        "fixed_delay_ns": rtp.FIXED_DELAY_NS,
        "clock_offset_ns": 0,
        "metric_scope": rtp.METRIC_SCOPE,
        "worker_environments": {
            config: {role: rtp.runtime_environment(worker_env(config, role))
                     for role in ("lc", "be")}
            for config in args.configs
        },
        "paths": {
            "workload": str(workload),
            "workload_original_measured": str(ORIGINAL_MEASURED_WORKLOAD),
            "xserver_native": str(paths["xserver_native"]),
            "native_hal_install_dir": str(paths["native_hal_install_dir"]),
            "hal_lib_dir": str(paths["hal_lib_dir"]),
            "xserver_bpftime": str(paths["xserver_bpftime"]),
            "hpf_bin": str(paths["hpf_bin"]),
        },
    }


def summarize(out_root: Path, protocol: dict) -> dict:
    records = [json.loads(path.read_text())
               for path in sorted(out_root.glob("block-*/result.json"))]
    blocks = sorted({record["block"] for record in records
                     if record["block_type"] in ("paired", "preflight")})
    by = {(record["block"], record["config"]): record for record in records}
    stats = {"lc_p99": "lc_service/p99_us",
             "lc_mean": "lc_service/mean_us",
             "be_p99": "be_service/p99_us",
             "be_rate": "be_kernels_per_s"}

    def value(record, dotted):
        key = stats.get(dotted, dotted)
        node = record
        for part in key.split("/"):
            node = node[part]
        return node

    def passes_gates(record):
        gates = record["engagement"].get("gates", {})
        return all(value for value in gates.values())

    summary = {
        "metric_scope": rtp.METRIC_SCOPE,
        "scope": protocol["scope"],
        "config_order": protocol["config_order"],
        "cells": {},
        "configs": {},
        "paired": {},
    }
    for record in records:
        key = f"block-{record['block']:02d}-{record['config']}"
        summary["cells"][key] = {
            "block_type": record["block_type"],
            "gates_passed": all(value for value in
                                record["engagement"].get("gates", {}).values()),
            "gate_failures": [key for key, value in
                              record["engagement"].get("gates", {}).items()
                              if not value],
            "sample_diagnostics": record.get("sample_diagnostics", []),
        }
    for config in protocol["config_order"]:
        rows = [by[(block, config)] for block in blocks
                if (block, config) in by and passes_gates(by[(block, config)])]
        excluded = [f"block-{block:02d}-{config}" for block in blocks
                    if (block, config) in by and not passes_gates(by[(block, config)])]
        if not rows:
            summary["configs"][config] = {"pair_blocks": 0, "cells_excluded": excluded}
            continue
        entry = {"pair_blocks": len(rows), "cells_excluded": excluded}
        entry["lc_service_p99_median_us"] = statistics.median(
            value(row, "lc_p99") for row in rows)
        entry["lc_service_mean_median_us"] = statistics.median(
            value(row, "lc_mean") for row in rows)
        entry["be_service_p99_median_us"] = statistics.median(
            value(row, "be_p99") for row in rows)
        entry["be_kernels_per_s_median"] = statistics.median(
            value(row, "be_rate") for row in rows)
        entry["lc_service_p99_per_block_us"] = [value(row, "lc_p99") for row in rows]
        entry["be_kernels_per_s_per_block"] = [value(row, "be_rate") for row in rows]
        summary["configs"][config] = entry

    def paired_delta(left, right, metric):
        pairs = []
        for block in blocks:
            if (block, left) in by and (block, right) in by:
                a, b = by[(block, left)], by[(block, right)]
                if passes_gates(a) and passes_gates(b) and value(b, metric) != 0:
                    pairs.append(round((value(a, metric) - value(b, metric))
                                       / value(b, metric) * 100.0, 6))
        return pairs

    for left, right in (("l2_cuxtra", "l1_native"),
                        ("l1_native", "baseline"),
                        ("l2_cuxtra", "baseline"),
                        ("l2_bpfhost", "l2_cuxtra")):
        if left in summary.get("configs", {}) and right in summary.get("configs", {}) \
                and summary["configs"][left].get("pair_blocks") \
                and summary["configs"][right].get("pair_blocks"):
            summary["paired"][f"{left}_vs_{right}"] = {
                "within_block_pct": {metric: paired_delta(left, right, metric)
                                     for metric in ("lc_p99", "lc_mean", "be_rate")},
                "estimator": ("median of within-block pct(left - right)/right; "
                              "negative LC pct = lower latency, negative BE pct "
                              "= lower throughput"),
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
        description="XSched Level-2 sm_120 native-cuXtra-blob arm runner")
    parser.add_argument("phase", choices=("run",))
    parser.add_argument("--output", type=Path, default=None,
                        help="new output directory (default "
                             "workloads/xsched/raw/level2-native-route-<timestamp>)")
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD,
                        help="service-only priority_workload binary (default: the "
                             "mismatch-diagnostic build; the original measured "
                             "worker is preserved unchanged at "
                             "service-output-20260908.66ppYl)")
    parser.add_argument("--hal-install-dir", type=Path, default=None,
                        help="override the isolated native HAL install directory")
    parser.add_argument("--xserver-native", type=Path, default=None,
                        help="override the native xserver binary (default: the "
                             "selected install's bin/xserver, else the deps "
                             "checkout binary)")
    parser.add_argument("--xserver-bpftime", type=Path, default=None,
                        help="override the BPF HPF xserver binary (l2_bpfhost arm)")
    parser.add_argument("--hpf-bin", type=Path, default=None,
                        help="override the BPF HPF program binary (l2_bpfhost arm)")
    parser.add_argument("--configs", type=parse_configs, default=CONFIGS,
                        help="comma-separated subset of "
                             "baseline,l1_native,l2_cuxtra,l2_bpfhost in run order")
    parser.add_argument("--reps", type=int, required=True,
                        help="kernel recurrence repetitions; set directly (the "
                             "campaign used 9511106)")
    parser.add_argument("--tasks", type=int, default=50,
                        help="kernels per stream (default 50; 2 marks a preflight)")
    parser.add_argument("--blocks", type=int, default=340,
                        help="grid blocks per kernel (default 340)")
    parser.add_argument("--threads", type=int, default=256,
                        help="threads per block (default 256)")
    parser.add_argument("--repetitions", type=int, default=5,
                        help="paired route-comparison blocks (default 5)")
    args = parser.parse_args()

    if args.reps < 1 or args.tasks < 1 or args.blocks < 1 or args.threads < 1:
        parser.error("reps/tasks/blocks/threads must be positive")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    paths = resolve_paths(args)
    missing = required_files(args.workload, args.configs, paths)
    if missing:
        print(json.dumps({"error": "missing required components", "missing": missing},
                         indent=2))
        return 2

    out_root = args.output or (XSCHED_DIR / "raw"
                               / f"level2-native-route-{time.strftime('%Y%m%d.%H%M%S')}")
    out_root.mkdir(parents=True, exist_ok=False)
    protocol = build_protocol(args, args.workload, paths)
    (out_root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(protocol, indent=2), flush=True)

    for block in range(1, args.repetitions + 1):
        block_type = "paired" if not protocol["preflight"] else "preflight"
        first = max(0, block - 1) % len(args.configs)
        for config in args.configs[first:] + args.configs[:first]:
            cell_dir = out_root / f"block-{block:02d}-{config}"
            cell_dir.mkdir(parents=True, exist_ok=False)
            try:
                record = run_cell(config, block, block_type, cell_dir,
                                  args.workload, rtp.STREAMS, args.tasks,
                                  args.reps, args.blocks, args.threads, paths)
                print(json.dumps(record), flush=True)
                time.sleep(5)
            except BaseException as exc:
                (cell_dir / "failure.json").write_text(json.dumps(
                    {"error_type": type(exc).__name__, "error": str(exc)},
                    indent=2) + "\n")
                raise

    print(json.dumps(summarize(out_root, protocol), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
