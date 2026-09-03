#!/usr/bin/env python3
"""Finite bpftime device-return engagement, not a scheduling performance test."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[1]
SPEC = importlib.util.spec_from_file_location("moe_safety", ROOT / "workloads/moe-infinity/run_moe_head_to_head.py")
assert SPEC and SPEC.loader
safety = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = safety
SPEC.loader.exec_module(safety)


def events(path: Path) -> list[dict]:
    result = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and "event" in item:
            result.append(item)
    return result


def runtime_configuration(build: Path, strict: bool) -> dict[str, str]:
    cache = build / "CMakeCache.txt"
    config = {}
    if cache.is_file():
        for line in cache.read_text().splitlines():
            if "=" in line and ":" in line.partition("=")[0]:
                key, _, value = line.partition("=")
                config[key.partition(":")[0]] = value
    keys = ("ENABLE_EBPF_VERIFIER", "BPFTIME_ENABLE_CUDA_ATTACH", "BPFTIME_LLVM_JIT")
    if strict and any(config.get(key, "").upper() not in {"ON", "YES", "TRUE", "1"} for key in keys):
        raise RuntimeError("strict smoke requires a verifier-enabled CUDA/LLVM runtime build")
    return {key: config.get(key, "unknown") for key in (*keys, "CMAKE_HOME_DIRECTORY")}


def require_strict_verdict(log: str, negative: bool) -> None:
    accepted = "GPU eBPF verification accepted: mode=STRICT program=cuda__count_return"
    rejected = "GPU eBPF verification failed for cuda__count_return:"
    if "Skipping GPU eBPF verification" in log or "; continuing" in log:
        raise RuntimeError("verification bypass is not strict evidence")
    if negative:
        required = (rejected, "branch predicate is lane-varying", "(mode=STRICT, hook_created=0)",
                    "GPU verifier rejected handler ", "Failed to initialize attach context, exiting..")
        if any(marker not in log for marker in required):
            raise RuntimeError("missing explicit SIMT rejection or fail-closed propagation")
        if "GPU eBPF verification accepted:" in log or "Recorded pass " in log:
            raise RuntimeError("rejected object was admitted or attached")
    else:
        admission = re.escape(accepted + " attach=kretprobe/_Z9vectorAddPKfS0_Pfi instructions=") + r"[1-9][0-9]*(?=\s|$)"
        map_record = (r"GPU eBPF verified map: program=cuda__count_return fd=[0-9]+ "
                      r"type=1502 key_size=4 value_size=8 max_entries=1(?=\s|$)")
        if rejected in log or not re.search(admission, log) or not re.search(map_record, log):
            raise RuntimeError("missing strict admission of the actual return counter and map")


def require_zero_counters(snapshots: list[dict]) -> None:
    fields = ("device_thread_returns", "nonzero_threads", "threads_with_eight_returns", "maximum_returns")
    if not snapshots or any(type(item.get(field)) is not int or item[field] != 0
                            for item in snapshots for field in fields):
        raise RuntimeError("negative case requires a fresh all-zero counter snapshot after rejection")


def group_members(pgid: int) -> list[int]:
    # Reuse GPreempt's owned-session rule locally: importing its run_three_way
    # would resolve its unqualified `import run_smoke` to this module instead.
    members = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if fields[0] != "Z" and int(fields[2]) == pgid and int(fields[3]) == pgid:
                members.append(int(path.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    return members


def stop_owned(process: subprocess.Popen) -> None:
    # start_new_session=True records ownership by PGID/SID, even if the leader exits.
    for sig, seconds in ((signal.SIGINT, 8), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
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
            time.sleep(0.1)
    raise RuntimeError(f"owned process group {process.pid} survived bounded cleanup")


def segment_identity(path: Path) -> tuple[int, int, int]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise RuntimeError(f"shared memory is not an owned regular file: {path}")
    return info.st_dev, info.st_ino, info.st_uid


def unlink_owned_segment(path: Path, identity: tuple[int, int, int] | None) -> None:
    try:
        actual = segment_identity(path)
    except FileNotFoundError:
        return
    if identity is None or actual != identity:
        raise RuntimeError(f"retaining unknown or replaced shared memory: {path}")
    path.unlink()


def run(output: Path, build: Path, *, strict: bool = False, negative: bool = False) -> dict:
    if negative and not strict:
        raise RuntimeError("the negative object must never run without strict verification")
    config = runtime_configuration(build, strict)
    output = output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    lease = safety.LeaseSet.acquire()
    processes = []
    streams = []
    segment = "bpftime_device_smoke_" + str(os.getpid()) + "_" + str(time.time_ns())
    segment_path = Path("/dev/shm") / segment
    owned_segment_identity = None
    if os.path.lexists(segment_path):
        lease.close()
        raise RuntimeError("unique shared-memory name already exists")
    before = None
    result = {"kind": "bpftime device-return engagement only", "status": "running",
              "private_shared_memory": segment, "runtime_build": str(build.absolute()),
              "runtime_configuration": config, "verifier_mode": "STRICT" if strict else "WARNING",
              "case": "negative_lane_branch" if negative else "positive_counter"}

    def start(name: str, argv: list[str], env: dict[str, str]) -> subprocess.Popen:
        stream = (output / (name + ".log")).open("x")
        streams.append(stream)
        process = subprocess.Popen(argv, env=env, cwd=output, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(process)
        return process

    try:
        before = safety.safety_snapshot()
        safety.validate_pre_server_safety(before)
        if before["gpu"]["driver"] != "575.57.08":
            raise RuntimeError("this smoke is fixed to driver 575.57.08")
        env = {"PATH": "/usr/local/cuda-12.9/bin:/usr/bin:/bin",
               "LANG": "C.UTF-8", "CUDA_VISIBLE_DEVICES": "0",
               "LD_LIBRARY_PATH": "/usr/local/cuda-12.9/lib64"}
        vector = str(HERE / ".output/vector")
        baseline = start("baseline", [vector], env)
        if baseline.wait(timeout=30) != 0:
            raise RuntimeError("finite native CUDA baseline failed")
        expected = {"event": "correctness", "launches": 8, "checked_values": 32768, "mismatches": 0}
        if events(output / "baseline.log") != [expected]:
            raise RuntimeError("native correctness evidence mismatch")
        result["native_correctness"] = expected
        safety.wait_for_post_server_safety(before)
        common = {**env, "BPFTIME_GLOBAL_SHM_NAME": segment,
                  "BPFTIME_MAP_GPU_THREAD_COUNT": "4096", "BPFTIME_SHM_MEMORY_MB": "64",
                  "BPFTIME_MAX_FD_COUNT": "1024", "BPFTIME_LOG_OUTPUT": "console",
                  "SPDLOG_LEVEL": "debug" if strict else "info", "BPFTIME_SM_ARCH": "sm_120",
                  "BPFTIME_VERIFIER_LEVEL": "STRICT" if strict else "WARNING",
                  "CUDA_HOME": "/usr/local/cuda-12.9", "BPFTIME_CUDA_ROOT": "/usr/local/cuda-12.9"}
        object_path = HERE / ".output" / ("probe-negative.bpf.o" if negative else "probe.bpf.o")
        result["bpf_object"] = str(object_path)
        probe = start("probe", [str(HERE / ".output/probe"), str(object_path)],
                      {**common, "LD_PRELOAD": str(build / "runtime/syscall-server/libbpftime-syscall-server.so")})
        deadline = time.monotonic() + 15
        while not any(item["event"] == "ready" for item in events(output / "probe.log")):
            if probe.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("private bpftime probe did not become ready within 15 seconds")
            time.sleep(0.1)
        owned_segment_identity = segment_identity(segment_path)
        result["shared_memory_identity"] = owned_segment_identity
        target = start("instrumented", [vector],
                       {**common, "LD_PRELOAD": str(build / "runtime/agent/libbpftime-agent.so")})
        if target.wait(timeout=75) != 0:
            raise RuntimeError("instrumented finite CUDA target failed")
        if expected not in events(output / "instrumented.log"):
            raise RuntimeError("instrumented correctness evidence mismatch")
        result["instrumented_correctness"] = expected
        if strict:
            target_log = (output / "instrumented.log").read_text(errors="replace")
            require_strict_verdict(target_log, negative)
            result["verifier_records"] = [line for line in target_log.splitlines()
                                           if "GPU eBPF verif" in line or "GPU verifier rejected" in line
                                           or "Failed to initialize attach context" in line]
        if negative:
            # The app may execute natively after rejection; that alone proves nothing.
            # A later observer report is needed, not a snapshot taken before the target.
            snapshot_count = len([item for item in events(output / "probe.log")
                                  if item["event"] == "counter_snapshot"])
            deadline = time.monotonic() + 5
            while True:
                snapshots = [item for item in events(output / "probe.log")
                             if item["event"] == "counter_snapshot"]
                if len(snapshots) > snapshot_count:
                    require_zero_counters(snapshots[snapshot_count:])
                    break
                if probe.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("no fresh counter observation after explicit strict rejection")
                time.sleep(0.1)
            result.update(status="passed", rejection={"diagnostic": "branch predicate is lane-varying",
                          "hook_created": False, "post_rejection_snapshots": snapshots[snapshot_count:]})
            return result
        if probe.wait(timeout=10) != 0:
            raise RuntimeError("device-return counter did not reach the exact expected count")
        engagement = [item for item in events(output / "probe.log") if item["event"] == "engagement"]
        if engagement != [{"event": "engagement", "device_thread_returns": 32768,
                           "threads_with_eight_returns": 4096}]:
            raise RuntimeError("expected every GPU thread to return exactly eight times")
        result.update(status="passed", native_correctness=expected,
                      instrumented_correctness=expected, engagement=engagement[0])
    except BaseException as exc:
        result.update(status="failed", error=str(exc))
        raise
    finally:
        cleanup_errors = []
        try:
            for process in reversed(processes):
                try:
                    stop_owned(process)
                except BaseException as cleanup_error:
                    cleanup_errors.append(str(cleanup_error))
            for stream in streams:
                stream.close()
            probe_log = output / "probe.log"
            if probe_log.exists():
                result["counter_snapshots"] = [item for item in events(probe_log)
                                               if item["event"] == "counter_snapshot"]
            survivors = {process.pid: members for process in processes
                         if (members := group_members(process.pid))}
            result["owned_group_survivors"] = survivors
            result["private_shared_memory_removed"] = not os.path.lexists(segment_path)
            if survivors:
                raise RuntimeError(f"owned smoke groups survived cleanup: {survivors}")
            # Never touch the default segment or a prefix/glob of other users' segments.
            unlink_owned_segment(segment_path, owned_segment_identity)
            result["private_shared_memory_removed"] = not os.path.lexists(segment_path)
            if before is not None:
                result["safety_after"] = safety.wait_for_post_server_safety(before)
            result["safety_before"] = before
        except BaseException as cleanup_error:
            cleanup_errors.append(str(cleanup_error))
        finally:
            if cleanup_errors:
                result.update(status="failed", cleanup_errors=cleanup_errors)
            try:
                safety.atomic_write_json(output / "result.json", result)
            finally:
                lease.close()
        if cleanup_errors:
            raise RuntimeError("; ".join(cleanup_errors))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-build", type=Path)
    parser.add_argument("--strict", action="store_true",
                        help="require strict admission and run a positive-then-negative pair")
    args = parser.parse_args()
    def interrupted(signum, _frame):
        raise InterruptedError(f"signal {signum}; cleanup owned smoke processes")
    signal.signal(signal.SIGTERM, interrupted)
    build = args.runtime_build or (ROOT.parent / ("bpftime-r5/build-r5-strict-device" if args.strict
                                                  else "bpftime/build-cuda-pr503"))
    if args.strict:
        if args.output.exists():
            raise RuntimeError("strict pair output directory must be new")
        positive = run(args.output / "positive", build.absolute(), strict=True)
        # Never interpret a broken instrumentation path as successful rejection.
        negative = run(args.output / "negative", build.absolute(), strict=True, negative=True)
        result = {"positive": positive, "negative": negative}
    else:
        result = run(args.output, build.absolute())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
