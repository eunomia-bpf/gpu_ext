#!/usr/bin/env python3
"""Run the frozen RTX 5090 STRICT-admitted uniform-map experiment."""

from __future__ import annotations

import argparse
import csv
import fcntl
import os
import random
import re
import signal
import stat
import subprocess
import time
from pathlib import Path
from typing import IO


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
WORKSPACE = HERE.parents[3]
DEFAULT_BPFTIME_ROOT = WORKSPACE / "bpftime-table1-575"
DEFAULT_BPFTIME_BUILD = DEFAULT_BPFTIME_ROOT / "build-table1-575-strict"
APPLICATION = PARENT / ".output/map-bench"
PTX = PARENT / ".output/map-bench.ptx"
LOADER = HERE / ".output/uniform-probe"
BPF_OBJECT = HERE / ".output/uniform-probe.bpf.o"
GPU_NAME = "NVIDIA GeForce RTX 5090"
DRIVER = "575.57.08"
SEED = 1797
ARMS = (
    "native", "noop", "device_update", "host_update",
    "device_lookup", "host_lookup",
)
ATTACHED_ARMS = ARMS[1:]
PROGRAMS = {
    "noop": "cuda__noop",
    "device_update": "cuda__dev_up",
    "host_update": "cuda__host_up",
    "device_lookup": "cuda__dev_look",
    "host_lookup": "cuda__host_look",
}
LEASES = (
    Path("/tmp/gpubpf-revision-gpu0.lock"),
    Path("/tmp/gpubpf-revision-struct-ops.lock"),
)
UPDATE_MAGIC = 0x51A7CAFE00000001
LOOKUP_MAGIC = 0x10C4CAFE00000001


class ReadOnlyLeases:
    def __init__(self) -> None:
        self.streams: list[IO[str]] = []

    def __enter__(self) -> "ReadOnlyLeases":
        try:
            for path in LEASES:
                before = path.lstat()
                if not stat.S_ISREG(before.st_mode):
                    raise RuntimeError(f"lease is not a regular file: {path}")
                stream = path.open("r", encoding="utf-8")
                opened = os.fstat(stream.fileno())
                if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                    stream.close()
                    raise RuntimeError(f"lease changed while opening: {path}")
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.streams.append(stream)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, _kind: object, _value: object, _trace: object) -> None:
        for stream in reversed(self.streams):
            stream.close()
        self.streams.clear()


def phase_parameters(phase: str) -> tuple[int, int, int]:
    if phase == "preflight":
        return 1, 1, 2
    if phase == "full":
        return 12, 8, 64
    raise ValueError(phase)


def arm_base_order() -> tuple[str, ...]:
    values = list(ARMS)
    random.Random(SEED).shuffle(values)
    return tuple(values)


def frozen_schedule(phase: str) -> list[dict[str, int | str]]:
    blocks, _warmup, _launches = phase_parameters(phase)
    base = arm_base_order()
    schedule: list[dict[str, int | str]] = []
    for block in range(blocks):
        cycle = base if block < len(ARMS) else tuple(reversed(base))
        offset = block % len(ARMS)
        order = cycle[offset:] + cycle[:offset]
        for position, arm in enumerate(order):
            schedule.append({
                "block": block + 1,
                "order": position + 1,
                "arm": arm,
                "run_id": block + 1,
            })
    return schedule


def reject_ambient_injection() -> None:
    forbidden = sorted(
        key for key in os.environ
        if key.startswith("BPFTIME_") or key in {
            "LD_PRELOAD", "LD_AUDIT", "CUDA_INJECTION64_PATH",
            "CUDA_INJECTION32_PATH",
        }
    )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if forbidden or visible not in (None, "0"):
        raise RuntimeError(
            f"start from an uninjected GPU-0 shell; keys={forbidden}, "
            f"CUDA_VISIBLE_DEVICES={visible!r}"
        )


def base_environment() -> dict[str, str]:
    return {
        "PATH": "/usr/local/cuda-12.9/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "CUDA_VISIBLE_DEVICES": "0",
        "LD_LIBRARY_PATH": "/usr/local/cuda-12.9/lib64",
    }


def checked_output(command: list[str], cwd: Path = HERE) -> str:
    result = subprocess.run(
        command, cwd=cwd, env=base_environment(), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {command}")
    return result.stdout


def cmake_settings(build: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in (build / "CMakeCache.txt").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        left, separator, value = line.partition("=")
        if separator and ":" in left:
            result[left.partition(":")[0]] = value
    return result


def binary_contains(path: Path, marker: bytes) -> bool:
    overlap = b""
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            window = overlap + chunk
            if marker in window:
                return True
            overlap = window[-(len(marker) - 1):] if len(marker) > 1 else b""
    return False


def validate_environment(root: Path, build: Path) -> str:
    gpu = checked_output([
        "nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader",
    ]).strip().splitlines()
    if gpu != [f"{GPU_NAME}, {DRIVER}"]:
        raise RuntimeError(f"frozen GPU/driver mismatch: {gpu}")
    settings = cmake_settings(build)
    required = {
        "BPFTIME_ENABLE_CUDA_ATTACH": "ON",
        "BPFTIME_LLVM_JIT": "ON",
        "ENABLE_EBPF_VERIFIER": "ON",
    }
    wrong = {
        key: settings.get(key, "missing")
        for key, value in required.items()
        if settings.get(key, "").upper() != value
    }
    if wrong:
        raise RuntimeError(f"strict runtime feature mismatch: {wrong}")
    agent = build / "runtime/agent/libbpftime-agent.so"
    server = build / "runtime/syscall-server/libbpftime-syscall-server.so"
    markers = (
        b"GPU eBPF verification accepted: mode=STRICT",
        b"GPU eBPF verification timing: program=",
        b"GPU eBPF verified map: program=",
    )
    for binary in (agent, server):
        if not binary.is_file() or binary.stat().st_size <= 0:
            raise RuntimeError(f"missing strict runtime binary: {binary}")
        if any(not binary_contains(binary, marker) for marker in markers):
            raise RuntimeError(f"strict marker missing from runtime binary: {binary}")
    revision = checked_output(["git", "rev-parse", "HEAD"], cwd=root).strip()
    status = checked_output(["git", "status", "--short"], cwd=root)
    nvcc = checked_output(["/usr/local/cuda-12.9/bin/nvcc", "--version"])
    return (
        f"gpu\t{GPU_NAME}\n"
        f"driver\t{DRIVER}\n"
        f"bpftime_root\t{root.resolve()}\n"
        f"bpftime_revision\t{revision}\n"
        f"BPFTIME_ENABLE_CUDA_ATTACH\t{settings['BPFTIME_ENABLE_CUDA_ATTACH']}\n"
        f"BPFTIME_LLVM_JIT\t{settings['BPFTIME_LLVM_JIT']}\n"
        f"ENABLE_EBPF_VERIFIER\t{settings['ENABLE_EBPF_VERIFIER']}\n"
        f"agent_bytes\t{agent.stat().st_size}\n"
        f"server_bytes\t{server.stat().st_size}\n"
        "strict_binary_markers\tpresent\n"
        "bpftime_status_begin\n"
        f"{status}"
        "bpftime_status_end\n"
        "nvcc_begin\n"
        f"{nvcc}"
        "nvcc_end\n"
    )


def validate_built_inputs() -> None:
    for path in (APPLICATION, PTX, LOADER, BPF_OBJECT):
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"missing built input: {path}; run make first")
    ptx = PTX.read_text(encoding="utf-8", errors="replace")
    if ptx.count("call.uni __bpftime_cuda__kernel_trace, ();") != 1:
        raise RuntimeError("target PTX does not contain exactly one explicit hook call")
    if "fig15_map_kernel" not in ptx:
        raise RuntimeError("target PTX does not contain fig15_map_kernel")


def write_schedule(path: Path, schedule: list[dict[str, int | str]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("block", "order", "arm", "run_id"), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(schedule)


def process_group_members(pgid: int) -> list[int]:
    members = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text(encoding="utf-8").rsplit(")", 1)[1].split()
            if fields[0] != "Z" and int(fields[2]) == pgid and int(fields[3]) == pgid:
                members.append(int(path.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    return members


def stop_owned(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    for request, timeout in (
        (signal.SIGINT, 10), (signal.SIGTERM, 5), (signal.SIGKILL, 3),
    ):
        if not process_group_members(process.pid):
            process.wait(timeout=1)
            return
        try:
            os.killpg(process.pid, request)
        except ProcessLookupError:
            continue
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not process_group_members(process.pid):
                process.wait(timeout=1)
                return
            time.sleep(0.1)
    raise RuntimeError(f"owned process group survived: {process.pid}")


def owned_segment_identity(path: Path) -> tuple[int, int, int]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise RuntimeError("private shared-memory segment is not an owned file")
    return info.st_dev, info.st_ino, info.st_uid


def wait_for_ready(path: Path, process: subprocess.Popen[str], segment_path: Path,
                   identity: list[tuple[int, int, int]]) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if not identity and os.path.lexists(segment_path):
            identity.append(owned_segment_identity(segment_path))
        if path.exists() and "FIG15_UNIFORM_READY\t" in path.read_text(
            encoding="utf-8", errors="replace"
        ):
            if not identity:
                raise RuntimeError("loader ready without its shared-memory segment")
            return
        if process.poll() is not None:
            raise RuntimeError("loader exited before readiness")
        time.sleep(0.1)
    raise RuntimeError("loader did not become ready")


def application_command(warmup: int, launches: int, run_id: int) -> list[str]:
    return [
        str(APPLICATION), "--warmup", str(warmup), "--launches", str(launches),
        "--run-id", str(run_id),
    ]


def validate_application_log(path: Path, warmup: int, launches: int) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    device = re.findall(r"^FIG15_DEVICE\t(.+)\t12\t0\t32$", text, re.MULTILINE)
    measurement = re.findall(
        r"^FIG15_MEASUREMENT\t(\d+)\t(\d+)\t([0-9.eE+-]+)$", text, re.MULTILINE,
    )
    correct = re.findall(r"^FIG15_CORRECT\t32\t0$", text, re.MULTILINE)
    if device != [GPU_NAME] or len(measurement) != 1 or len(correct) != 1:
        raise RuntimeError("application output/correctness record is incomplete")
    if (int(measurement[0][0]), int(measurement[0][1])) != (warmup, launches):
        raise RuntimeError("application timing parameters changed")
    elapsed = float(measurement[0][2])
    if elapsed <= 0:
        raise RuntimeError("application elapsed time is not positive")
    return elapsed


def expected_map_rows(arm: str) -> dict[tuple[str, int], int]:
    if arm == "noop":
        return {}
    source = "device_values" if arm.startswith("device_") else "host_values"
    if arm.endswith("_update"):
        return {(source, 0): UPDATE_MAGIC}
    return {(source, 0): LOOKUP_MAGIC, ("observed_values", 0): LOOKUP_MAGIC}


def validate_loader_log(path: Path, arm: str) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    prime = list(re.finditer(
        r"^FIG15_UNIFORM_SERVER_PRIMED\t1$", text, re.MULTILINE
    ))
    object_load = list(re.finditer(r"^libbpf: loading object from .+$", text, re.MULTILINE))
    if len(prime) != 1 or len(object_load) != 1 or prime[0].start() >= object_load[0].start():
        raise RuntimeError("loader syscall-server prime record is incomplete")
    if re.findall(r"^FIG15_UNIFORM_READY\t([^\t]+)\t1$", text, re.MULTILINE) != [arm]:
        raise RuntimeError("loader readiness record is incomplete")
    if len(re.findall(r"^FIG15_UNIFORM_DETACHED\t1$", text, re.MULTILINE)) != 1:
        raise RuntimeError("loader detach record is incomplete")
    rows = re.findall(
        r"^FIG15_UNIFORM_MAP\t([^\t]+)\t(\d+)\t(\d+)$", text, re.MULTILINE
    )
    parsed = {(name, int(key)): int(value) for name, key, value in rows}
    if len(parsed) != len(rows) or parsed != expected_map_rows(arm):
        raise RuntimeError("map readback differs from the uniform-operation oracle")


def validate_strict_admission(application_path: Path, target_pid: int,
                              arm: str) -> None:
    text = application_path.read_text(encoding="utf-8", errors="replace")
    prefix = rf"^\[[^\]\r\n]+\]\[[^\]\r\n]+\]\[{target_pid}\] "
    program = re.escape(PROGRAMS[arm])
    accepted = re.findall(
        prefix + rf"GPU eBPF verification accepted: mode=STRICT program={program} "
        + r"attach=kprobe/fig15_map_kernel instructions=([1-9][0-9]*)$",
        text, re.MULTILINE,
    )
    timings = re.findall(
        prefix + rf"GPU eBPF verification timing: program={program} "
        + r"verification_elapsed_ns=([1-9][0-9]*)$",
        text, re.MULTILINE,
    )
    maps = re.findall(
        prefix + rf"GPU eBPF verified map: program={program} fd=([0-9]+) "
        + r"type=([0-9]+) key_size=4 value_size=8 max_entries=1$",
        text, re.MULTILINE,
    )
    fragments = (
        "GPU eBPF verification accepted:", "GPU eBPF verification timing:",
        "GPU eBPF verified map:", "GPU eBPF verification failed",
        "Skipping GPU eBPF verification", "verifier unavailable",
    )
    target_records = [line for line in text.splitlines()
                      if f"][{target_pid}] " in line
                      and any(fragment in line for fragment in fragments)]
    if len(accepted) != 1 or len(timings) != 1:
        raise RuntimeError("target-PID STRICT acceptance/timing marker is incomplete")
    if len(maps) != 3 or sorted(int(map_type) for _fd, map_type in maps) != [1503, 1503, 1513]:
        raise RuntimeError(f"unexpected strict map descriptor records: {maps}")
    if len(target_records) != 5:
        raise RuntimeError(f"unexpected target-PID verifier record count: {target_records}")
    if any(fragment in text for fragment in (
        "GPU eBPF verification failed", "Skipping GPU eBPF verification",
        "verifier unavailable",
    )):
        raise RuntimeError("strict execution contains reject/skip/unavailable record")


def validate_engagement(application_path: Path, agent_path: Path, arm: str,
                        target_pid: int) -> None:
    application = application_path.read_text(encoding="utf-8", errors="replace")
    agent = agent_path.read_text(encoding="utf-8", errors="replace")
    required = {
        "target_transform": r"^\[ptxpass\] kprobe_entry_stub: matched=1, in=\d+, out=\d+$",
        "module_load": r"Loaded module: patched\.map_bench\.sm_120\.ptx",
        "attach": r"Attach successfully",
    }
    counts = {name: len(re.findall(pattern, application, re.MULTILINE))
              for name, pattern in required.items()}
    programs = re.findall(
        r"corresponding program ([A-Za-z0-9_]+) is cuda program", application
    )
    if any(value != 1 for value in counts.values()) or not programs or set(programs) != {PROGRAMS[arm]}:
        raise RuntimeError(
            f"selected program/transform/module/attach evidence is incomplete: "
            f"arm={arm}, programs={programs}, counts={counts}"
        )
    bootstrap = {
        "verifier_mode": r"Verifier mode: STRICT",
        "cuda_shm": r"Registered shared memory with CUDA:",
        "global_shm": r"Global shm constructed\. shm_open_type 1 for fig15_uniform_",
        "global_shm_ready": r"Global shm initialized",
    }
    bootstrap_counts = {name: len(re.findall(pattern, agent))
                        for name, pattern in bootstrap.items()}
    if any(value != 1 for value in bootstrap_counts.values()):
        raise RuntimeError(f"agent strict bootstrap evidence is incomplete: {bootstrap_counts}")
    validate_strict_admission(application_path, target_pid, arm)
    if re.search(r"\[(?:error|critical)\]", application + "\n" + agent, re.IGNORECASE):
        raise RuntimeError("application/agent log contains runtime error/critical record")


def run_native(directory: Path, warmup: int, launches: int, run_id: int) -> None:
    directory.mkdir(parents=True)
    path = directory / "application.log"
    with path.open("x", encoding="utf-8") as stream:
        result = subprocess.run(
            application_command(warmup, launches, run_id), cwd=HERE,
            env=base_environment(), stdout=stream, stderr=subprocess.STDOUT,
            text=True, timeout=120, check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"native application exited {result.returncode}")
    validate_application_log(path, warmup, launches)


def attached_environment(build: Path, segment: str,
                         agent_log: Path) -> tuple[dict[str, str], dict[str, str]]:
    common = {
        **base_environment(),
        "BPFTIME_GLOBAL_SHM_NAME": segment,
        "BPFTIME_MAP_GPU_THREAD_COUNT": "32",
        "BPFTIME_SHM_MEMORY_MB": "256",
        "BPFTIME_MAX_FD_COUNT": "1024",
        "BPFTIME_LOG_OUTPUT": "console",
        "SPDLOG_LEVEL": "info",
        "BPFTIME_SM_ARCH": "sm_120",
        "BPFTIME_VERIFIER_LEVEL": "STRICT",
        "CUDA_HOME": "/usr/local/cuda-12.9",
        "BPFTIME_CUDA_ROOT": "/usr/local/cuda-12.9",
    }
    loader = {
        **common,
        "LD_PRELOAD": str(build / "runtime/syscall-server/libbpftime-syscall-server.so"),
    }
    agent = {
        **common,
        "LD_PRELOAD": str(build / "runtime/agent/libbpftime-agent.so"),
        "BPFTIME_LOG_OUTPUT": str(agent_log),
        "BPFTIME_CUDA_DEFER_PTX_EXTRACTION": "1",
        "BPFTIME_CUDA_TARGETED_LATE_BOOTSTRAP": "1",
    }
    return loader, agent


def run_attached(arm: str, directory: Path, build: Path, warmup: int,
                 launches: int, run_id: int) -> None:
    if arm not in ATTACHED_ARMS:
        raise ValueError(arm)
    directory.mkdir(parents=True)
    loader_log = directory / "loader.log"
    application_log = directory / "application.log"
    agent_log = directory / "agent.log"
    execution = directory / "execution.tsv"
    segment = f"fig15_uniform_{os.getpid()}_{time.monotonic_ns()}"
    segment_path = Path("/dev/shm") / segment
    if os.path.lexists(segment_path):
        raise RuntimeError("private shared-memory name already exists")
    loader_env, agent_env = attached_environment(build, segment, agent_log)
    loader_process = application_process = None
    loader_stream = application_stream = None
    identity: list[tuple[int, int, int]] = []
    try:
        loader_stream = loader_log.open("x", encoding="utf-8")
        loader_process = subprocess.Popen(
            [str(LOADER), str(BPF_OBJECT), arm, "300"], cwd=HERE,
            env=loader_env, stdout=loader_stream, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,
        )
        wait_for_ready(loader_log, loader_process, segment_path, identity)
        application_stream = application_log.open("x", encoding="utf-8")
        application_process = subprocess.Popen(
            application_command(warmup, launches, run_id), cwd=HERE,
            env=agent_env, stdout=application_stream, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,
        )
        target_pid = application_process.pid
        returncode = application_process.wait(timeout=180)
        execution.write_text(
            "target_pid\treturncode\tverifier_level\n"
            f"{target_pid}\t{returncode}\tSTRICT\n", encoding="utf-8"
        )
        if returncode != 0:
            raise RuntimeError("attached application returned nonzero")
        application_stream.close()
        application_stream = None
        os.killpg(loader_process.pid, signal.SIGINT)
        if loader_process.wait(timeout=90) != 0:
            raise RuntimeError("loader returned nonzero")
        loader_stream.close()
        loader_stream = None
        validate_application_log(application_log, warmup, launches)
        validate_loader_log(loader_log, arm)
        if not agent_log.is_file() or not agent_log.read_text(
            encoding="utf-8", errors="replace"
        ).strip():
            raise RuntimeError("agent bootstrap log is empty")
        validate_engagement(application_log, agent_log, arm, target_pid)
    finally:
        stop_owned(application_process)
        stop_owned(loader_process)
        if application_stream is not None:
            application_stream.close()
        if loader_stream is not None:
            loader_stream.close()
        if os.path.lexists(segment_path):
            info = segment_path.lstat()
            actual = (info.st_dev, info.st_ino, info.st_uid)
            if not identity or actual != identity[0] or not stat.S_ISREG(info.st_mode):
                raise RuntimeError("refusing to remove unknown shared-memory segment")
            segment_path.unlink()
        if os.path.lexists(segment_path):
            raise RuntimeError("private shared-memory segment survived cleanup")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bpftime-root", type=Path, default=DEFAULT_BPFTIME_ROOT)
    parser.add_argument("--bpftime-build", type=Path, default=DEFAULT_BPFTIME_BUILD)
    args = parser.parse_args()
    reject_ambient_injection()
    validate_built_inputs()
    blocks, warmup, launches = phase_parameters(args.phase)
    schedule = frozen_schedule(args.phase)
    if len(schedule) != blocks * len(ARMS):
        raise RuntimeError("frozen schedule size changed")
    args.output.mkdir(parents=True, exist_ok=False)
    write_schedule(args.output / "schedule.tsv", schedule)
    (args.output / "environment.txt").write_text(
        validate_environment(args.bpftime_root, args.bpftime_build), encoding="utf-8"
    )
    deadline = time.monotonic() + 3600
    with ReadOnlyLeases():
        for item in schedule:
            if time.monotonic() >= deadline:
                raise RuntimeError("one-hour campaign deadline reached")
            arm = str(item["arm"])
            directory = args.output / (
                f"block-{int(item['block']):02d}-order-{int(item['order']):02d}-{arm}"
            )
            print(
                f"block={item['block']} order={item['order']} arm={arm}", flush=True
            )
            if arm == "native":
                run_native(directory, warmup, launches, int(item["run_id"]))
            else:
                run_attached(
                    arm, directory, args.bpftime_build, warmup, launches,
                    int(item["run_id"]),
                )
    print(f"completed {len(schedule)} frozen arm processes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
