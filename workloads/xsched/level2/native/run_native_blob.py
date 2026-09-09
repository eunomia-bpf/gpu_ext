#!/usr/bin/env python3
"""XSched Level-2 sm_120 native-cuXtra-blob arm runner (bring-up).

First native-blob execution on the isolated native HAL install
(hal-native-20260908.i8na51). The isolated native HAL was configured with
XG_SM120_GENERATED_HEADER pointing at the array header emitted by
level2/native/ldc_patcher.cpp, so GuardianSM120 serves the real captured
SASS guardian (0x280 B) and resume (0x150 B) streams; instrument.cpp copies
them into instruction memory with cuXtra binary surgery. This differs from
the tool-actuator pair runner (../run_tool_pair.py, owned separately):

  - no NVBit tool and no LD_PRELOAD: the Level-2 isolation comes from
    LD_LIBRARY_PATH resolving libcuda.so.1 -> libshimcuda.so in the native
    install directory;
  - XSCHED_LEVEL2_TOOL_ACTUATOR is explicitly 0 (kToolActuator false),
    so InstrumentContext takes the cuXtra path: GuardianSM120 +
    InstrMemAllocator + cuXtraSetDebuggerParams / cuXtraSetEntryPoint
    launch surgery; no xg_host_publish, no context pool;
  - NVBit-entry accounting is not applicable: no tool logs exist by
    design, so they are never expected or parsed;
  - only the native-blob arm runs. The completed block-1 baseline of
    raw/level2-tool-native-bpf-20260908.sjyzim is not rerun, and no BPF
    raw-blob arm is invented.

The worker shape is the existing campaign shape: 2 LC + 4 BE processes,
4 streams each (24 XQueues), 50 kernels per stream, recurrence reps set
directly, 340 blocks and 256 threads. Workers run the service-only output
branch (XG_SERVICE_ONLY=1): per-kernel GPU exit-entry service and host
elapsed on one clock; no submit/queue/cross-clock metric is emitted.

Process machinery (ManagedProcess, ready/GO/running bar, affinity, log
capture) is reused unchanged from the shared runner; that module owns all
process cleanup semantics. This wrapper owns only the native environment
and the single native-blob arm. It takes no GPU lock: root wraps the GPU
lease around the invocation.
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

NATIVE_BLOB = "native_blob"
CONFIGS = (NATIVE_BLOB,)
NOT_APPLICABLE = ("not_applicable: the native cuXtra blob route runs without the "
                  "NVBit tool, so no 'XG tool loaded/instrumented entry/done' "
                  "lines exist by design")

HAL_LIB_NAMES = ("libshimcuda.so", "libhalcuda.so", "libpreempt.so", "libcuda.so.1")


def native_worker_env(role: str) -> dict:
    """Native-blob worker environment (no target symbol, no tool preload)."""
    env = rtp.clean_env()
    env.update({
        "XSCHED_SCHEDULER": "GLB",
        "XSCHED_AUTO_XQUEUE": "ON",
        "XSCHED_AUTO_XQUEUE_LEVEL": "2",
        "XSCHED_AUTO_XQUEUE_PRIORITY": "1" if role == "lc" else "0",
        "XSCHED_AUTO_XQUEUE_THRESHOLD": "16" if role == "lc" else "4",
        "XSCHED_AUTO_XQUEUE_BATCH_SIZE": "8" if role == "lc" else "2",
        # opt-in sm_120 Level-2 queue port
        "XSCHED_CUDA_LV2_PORT_120": "1",
        # native cuXtra blob route: tool actuator explicitly OFF
        "XSCHED_LEVEL2_TOOL_ACTUATOR": "0",
        # shim isolation via LD_LIBRARY_PATH only (no NVBit LD_PRELOAD)
        "LD_LIBRARY_PATH": str(NATIVE_HAL_LIB_DIR),
    })
    env["XG_SERVICE_ONLY"] = "1"
    if "XG_NATIVE_ORIGINAL_ENTRY_CONTROL" in os.environ:
        env["XG_NATIVE_ORIGINAL_ENTRY_CONTROL"] = os.environ["XG_NATIVE_ORIGINAL_ENTRY_CONTROL"]
    # opt-in fused-window metadata extension is propagated only when the
    # user asks for it (presence-based at the shim: XG_NATIVE_META_EXTEND)
    if "XG_NATIVE_META_EXTEND" in os.environ:
        env["XG_NATIVE_META_EXTEND"] = os.environ["XG_NATIVE_META_EXTEND"]
    # second opt-in: KPARAM_INFO ordinal growth (XG_NATIVE_META_KPARAM),
    # presence-based at the shim, same forwarding pattern
    if "XG_NATIVE_META_KPARAM" in os.environ:
        env["XG_NATIVE_META_KPARAM"] = os.environ["XG_NATIVE_META_KPARAM"]
    return env


def native_server_env() -> dict:
    return rtp.clean_env()


def required_files(workload: Path) -> list:
    required = [workload, rtp.XSERVER_NATIVE]
    required += [NATIVE_HAL_LIB_DIR / name for name in HAL_LIB_NAMES]
    return [str(path) for path in required if not path.is_file()]


def run_native_cell(block: int, block_type: str, run_dir: Path, workload: Path,
                    streams: int, tasks: int, reps: int, blocks: int,
                    threads: int) -> dict:
    cpus = rtp.allowed_cpus(10)
    server = None
    workers = []
    try:
        server = rtp.ManagedProcess("xserver",
                                    [str(rtp.XSERVER_NATIVE), "HPF", rtp.HPF_QUANTUM],
                                    native_server_env(), cpus[0])
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
                                              native_worker_env(role), cpus[2:]))
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

    # Native route: no NVBit tool, so tool-entry accounting is not
    # applicable and never expected. Only diagnosable observations are the
    # xserver priority assignment log and per-sample validity (above).
    engagement = {"nvbit_entry_accounting": NOT_APPLICABLE}
    text = "\n".join(server.stdout_lines + server.stderr_lines)
    if text.count("set priority 1") < rtp.LC_PROCESSES * streams \
            or text.count("set priority 0") < rtp.BE_PROCESSES * streams:
        engagement["priority_log_diagnostic"] = \
            "expected priority messages not all observed"

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
        "block": block, "block_type": block_type, "config": NATIVE_BLOB,
        "metric_scope": rtp.METRIC_SCOPE,
        "actuator": "native-cuxtra-blob",
        "decision": None,
        "target_symbol": None,
        "tool_actuator": False,
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
        "engagement": engagement,
        "sample_diagnostics": [event["sample_diagnostic"] for event in results
                               if "sample_diagnostic" in event],
    }
    (run_dir / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def build_protocol(args, workload: Path) -> dict:
    return {
        "phase": "level2-native-blob-run",
        "scope": ("sm_120 Level-2 native cuXtra blob route bring-up on the isolated "
                  "native HAL install: GuardianSM120 serves the captured SASS "
                  "arrays; no NVBit tool, no tool-entry accounting (not "
                  "applicable); service-only metrics; the completed historical "
                  "baseline is not rerun and no BPF raw-blob arm is run"),
        "config_order": [NATIVE_BLOB],
        "pair_blocks": args.repetitions,
        "initial_run": False,
        "streams": rtp.STREAMS, "lc_processes": rtp.LC_PROCESSES,
        "be_processes": rtp.BE_PROCESSES,
        "tasks_per_stream": args.tasks, "reps": args.reps,
        "blocks": args.blocks, "threads": args.threads,
        "fixed_delay_ns": rtp.FIXED_DELAY_NS,
        "clock_offset_ns": 0,
        "metric_scope": rtp.METRIC_SCOPE,
        "worker_environments": {
            role: rtp.runtime_environment(native_worker_env(role))
            for role in ("lc", "be")
        },
        "paths": {
            "workload": str(workload),
            "workload_original_measured": str(ORIGINAL_MEASURED_WORKLOAD),
            "xserver_native": str(rtp.XSERVER_NATIVE),
            "native_hal_install_dir": str(NATIVE_HAL_INSTALL),
            "hal_lib_dir": str(NATIVE_HAL_LIB_DIR),
            "guard_tool": None,
        },
    }


def summarize(out_root: Path, protocol: dict) -> dict:
    records = [json.loads(path.read_text())
               for path in sorted(out_root.glob("block-*/result.json"))]
    blocks = sorted({record["block"] for record in records})
    by = {(record["block"], record["config"]): record for record in records}
    rows = [by[(block, NATIVE_BLOB)] for block in blocks
            if (block, NATIVE_BLOB) in by]
    summary = {
        "metric_scope": rtp.METRIC_SCOPE,
        "scope": protocol["scope"],
        "configs": {},
    }
    if rows:
        summary["configs"][NATIVE_BLOB] = {
            "pair_blocks": len(rows),
            "lc_service_p99_median_us": statistics.median(
                row["lc_service"]["p99_us"] for row in rows),
            "lc_service_mean_median_us": statistics.median(
                row["lc_service"]["mean_us"] for row in rows),
            "be_service_p99_median_us": statistics.median(
                row["be_service"]["p99_us"] for row in rows),
            "be_kernels_per_s_median": statistics.median(
                row["be_kernels_per_s"] for row in rows),
            "lc_host_elapsed_median_ns": statistics.median(
                row["lc_host_elapsed_ns"] for row in rows),
            "be_host_elapsed_median_ns": statistics.median(
                row["be_host_elapsed_ns"] for row in rows),
        }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="XSched Level-2 sm_120 native-cuXtra-blob arm runner (bring-up)")
    parser.add_argument("phase", choices=("run",))
    parser.add_argument("--output", type=Path, default=None,
                        help="new output directory (default "
                             "workloads/xsched/raw/level2-native-blob-<timestamp>)")
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD,
                        help="service-only priority_workload binary (default: the "
                             "mismatch-diagnostic build; the original measured "
                             "worker is preserved unchanged at "
                             "service-output-20260908.66ppYl)")
    parser.add_argument("--reps", type=int, required=True,
                        help="kernel recurrence repetitions; set directly (the "
                             "campaign used 9511106)")
    parser.add_argument("--tasks", type=int, default=50,
                        help="kernels per stream (default 50)")
    parser.add_argument("--blocks", type=int, default=340,
                        help="grid blocks per kernel (default 340)")
    parser.add_argument("--threads", type=int, default=256,
                        help="threads per block (default 256)")
    parser.add_argument("--repetitions", type=int, default=5,
                        help="paired native-blob blocks (default 5)")
    args = parser.parse_args()

    if args.reps < 1 or args.tasks < 1 or args.blocks < 1 or args.threads < 1:
        parser.error("reps/tasks/blocks/threads must be positive")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    missing = required_files(args.workload)
    if missing:
        print(json.dumps({"error": "missing required components", "missing": missing},
                         indent=2))
        return 2

    out_root = args.output or (XSCHED_DIR / "raw"
                               / f"level2-native-blob-{time.strftime('%Y%m%d.%H%M%S')}")
    out_root.mkdir(parents=True, exist_ok=False)
    protocol = build_protocol(args, args.workload)
    (out_root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(protocol, indent=2), flush=True)

    for block in range(1, args.repetitions + 1):
        cell_dir = out_root / f"block-{block:02d}-{NATIVE_BLOB}"
        cell_dir.mkdir(parents=True, exist_ok=False)
        try:
            record = run_native_cell(block, "paired", cell_dir, args.workload,
                                     rtp.STREAMS, args.tasks, args.reps,
                                     args.blocks, args.threads)
            print(json.dumps(record), flush=True)
        except BaseException as exc:
            (cell_dir / "failure.json").write_text(json.dumps(
                {"error_type": type(exc).__name__, "error": str(exc)}, indent=2) + "\n")
            raise

    print(json.dumps(summarize(out_root, protocol), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
