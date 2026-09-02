#!/usr/bin/env python3
"""Finite bpftime device-return engagement, not a scheduling performance test."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
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


def run(output: Path, build: Path) -> dict:
    output = output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    lease = safety.LeaseSet.acquire()
    processes = []
    streams = []
    segment = "bpftime_device_smoke_" + str(os.getpid()) + "_" + str(time.time_ns())
    segment_path = Path("/dev/shm") / segment
    if segment_path.exists():
        lease.close()
        raise RuntimeError("unique shared-memory name already exists")
    before = None
    result = {"kind": "bpftime device-return engagement only", "status": "running",
              "private_shared_memory": segment, "runtime_build": str(build.absolute()),
              "runtime_source_caveat": "Existing development CUDA build; not the r5 verifier-linked runtime."}

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
                  "SPDLOG_LEVEL": "info", "BPFTIME_SM_ARCH": "sm_120",
                  "CUDA_HOME": "/usr/local/cuda-12.9", "BPFTIME_CUDA_ROOT": "/usr/local/cuda-12.9"}
        probe = start("probe", [str(HERE / ".output/probe"), str(HERE / ".output/probe.bpf.o")],
                      {**common, "LD_PRELOAD": str(build / "runtime/syscall-server/libbpftime-syscall-server.so")})
        deadline = time.monotonic() + 15
        while not any(item["event"] == "ready" for item in events(output / "probe.log")):
            if probe.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("private bpftime probe did not become ready within 15 seconds")
            time.sleep(0.1)
        target = start("instrumented", [vector],
                       {**common, "LD_PRELOAD": str(build / "runtime/agent/libbpftime-agent.so")})
        if target.wait(timeout=75) != 0:
            raise RuntimeError("instrumented finite CUDA target failed")
        if expected not in events(output / "instrumented.log"):
            raise RuntimeError("instrumented correctness evidence mismatch")
        result["instrumented_correctness"] = expected
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
        try:
            for process in reversed(processes):
                safety.stop_owned_process_group(process)
            for stream in streams:
                stream.close()
            probe_log = output / "probe.log"
            if probe_log.exists():
                result["counter_snapshots"] = [item for item in events(probe_log)
                                               if item["event"] == "counter_snapshot"]
            if any(process.poll() is None for process in processes):
                raise RuntimeError("owned smoke process survived cleanup")
            # Never touch the default segment or a prefix/glob of other users' segments.
            if segment_path.exists():
                segment_path.unlink()
            result["private_shared_memory_removed"] = not segment_path.exists()
            if before is not None:
                result["safety_after"] = safety.wait_for_post_server_safety(before)
            result["safety_before"] = before
        except BaseException as cleanup_error:
            result.update(status="failed", cleanup_error=str(cleanup_error))
            raise
        finally:
            safety.atomic_write_json(output / "result.json", result)
            lease.close()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-build", type=Path,
                        default=ROOT.parent / "bpftime/build-cuda-pr503")
    args = parser.parse_args()
    def interrupted(signum, _frame):
        raise InterruptedError(f"signal {signum}; cleanup owned smoke processes")
    signal.signal(signal.SIGTERM, interrupted)
    print(json.dumps(run(args.output, args.runtime_build.absolute()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
