#!/usr/bin/env python3
"""Run the frozen RTX 5090 operation-matched device-map placement experiment."""

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
WORKSPACE = HERE.parents[2]
DEFAULT_BPFTIME_ROOT = WORKSPACE / "bpftime-table1-575"
DEFAULT_BPFTIME_BUILD = DEFAULT_BPFTIME_ROOT / "build-table1-575"
APPLICATION = HERE / ".output/map-bench"
PTX = HERE / ".output/map-bench.ptx"
LOADER = HERE / ".output/map-probe"
BPF_OBJECT = HERE / ".output/map-probe.bpf.o"
GPU_NAME = "NVIDIA GeForce RTX 5090"
DRIVER = "575.57.08"
SEED = 1797
ARMS = (
    "native",
    "noop",
    "device_update",
    "host_update",
    "rpc_update",
    "device_lookup",
    "host_lookup",
    "rpc_lookup",
)
ATTACHED_ARMS = ARMS[1:]
PROGRAM_PREFIXES = {
    "noop": "cuda__noop",
    "device_update": "cuda__device_up",
    "host_update": "cuda__host_upda",
    "rpc_update": "cuda__rpc_updat",
    "device_lookup": "cuda__device_lo",
    "host_lookup": "cuda__host_look",
    "rpc_lookup": "cuda__rpc_looku",
}
LEASES = (
    Path("/tmp/gpubpf-revision-gpu0.lock"),
    Path("/tmp/gpubpf-revision-struct-ops.lock"),
)
UPDATE_MAGIC = 0x51A7000000000000
LOOKUP_MAGIC = 0x10C4000000000000


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
        return 16, 8, 64
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
            schedule.append(
                {"block": block + 1, "order": position + 1, "arm": arm,
                 "run_id": block + 1}
            )
    return schedule


def reject_ambient_injection() -> None:
    forbidden = sorted(
        key
        for key in os.environ
        if key.startswith("BPFTIME_")
        or key in {"LD_PRELOAD", "LD_AUDIT", "CUDA_INJECTION64_PATH",
                   "CUDA_INJECTION32_PATH"}
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


def checked_output(command: list[str]) -> str:
    result = subprocess.run(
        command, cwd=HERE, env=base_environment(), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {command}")
    return result.stdout


def validate_environment(root: Path, build: Path) -> str:
    gpu = checked_output([
        "nvidia-smi", "--query-gpu=name,driver_version",
        "--format=csv,noheader",
    ]).strip().splitlines()
    if gpu != [f"{GPU_NAME}, {DRIVER}"]:
        raise RuntimeError(f"frozen GPU/driver mismatch: {gpu}")
    cache = build / "CMakeCache.txt"
    settings: dict[str, str] = {}
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        left, separator, value = line.partition("=")
        if separator and ":" in left:
            settings[left.partition(":")[0]] = value
    required = {
        "BPFTIME_ENABLE_CUDA_ATTACH": "ON",
        "BPFTIME_LLVM_JIT": "ON",
        "ENABLE_EBPF_VERIFIER": "OFF",
    }
    wrong = {
        key: settings.get(key, "missing")
        for key, value in required.items()
        if settings.get(key, "").upper() != value
    }
    if wrong:
        raise RuntimeError(f"runtime feature mismatch: {wrong}")
    revision = checked_output(["git", "-C", str(root), "rev-parse", "HEAD"]).strip()
    status = checked_output(["git", "-C", str(root), "status", "--short"])
    nvcc = checked_output(["/usr/local/cuda-12.9/bin/nvcc", "--version"])
    return (
        f"gpu\t{GPU_NAME}\n"
        f"driver\t{DRIVER}\n"
        f"bpftime_root\t{root.resolve()}\n"
        f"bpftime_revision\t{revision}\n"
        f"BPFTIME_ENABLE_CUDA_ATTACH\t{settings['BPFTIME_ENABLE_CUDA_ATTACH']}\n"
        f"BPFTIME_LLVM_JIT\t{settings['BPFTIME_LLVM_JIT']}\n"
        f"ENABLE_EBPF_VERIFIER\t{settings['ENABLE_EBPF_VERIFIER']}\n"
        "bpftime_status_begin\n"
        f"{status}"
        "bpftime_status_end\n"
        "nvcc_begin\n"
        f"{nvcc}"
        "nvcc_end\n"
    )


def validate_built_inputs() -> None:
    for path in (APPLICATION, PTX, LOADER, BPF_OBJECT):
        if not path.is_file():
            raise RuntimeError(f"missing built input: {path}; run make first")
    ptx = PTX.read_text(encoding="utf-8", errors="replace")
    if ptx.count("call.uni __bpftime_cuda__kernel_trace, ();") != 1:
        raise RuntimeError("target PTX does not contain exactly one explicit hook call")
    if "fig15_map_kernel" not in ptx:
        raise RuntimeError("target PTX does not contain fig15_map_kernel")


def write_schedule(path: Path, schedule: list[dict[str, int | str]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("block", "order", "arm", "run_id"),
                                delimiter="\t")
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
    for request, timeout in ((signal.SIGINT, 10), (signal.SIGTERM, 5),
                             (signal.SIGKILL, 3)):
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


def wait_for_ready(path: Path, process: subprocess.Popen[str],
                   segment_path: Path,
                   identity: list[tuple[int, int, int]]) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        # The server creates its segment before opening the BPF object. Record
        # its identity immediately so an early loader failure remains safely
        # reclaimable; keep it in caller-owned state so an exception cannot
        # discard the observation before the finally block runs.
        if not identity and os.path.lexists(segment_path):
            identity.append(owned_segment_identity(segment_path))
        if path.exists() and "FIG15_READY\t" in path.read_text(
            encoding="utf-8", errors="replace"
        ):
            if not identity:
                raise RuntimeError("loader became ready without its shared-memory segment")
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


def validate_application_log(path: Path, warmup: int, launches: int) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    device = re.findall(r"^FIG15_DEVICE\t(.+)\t12\t0\t32$", text, re.MULTILINE)
    measurement = re.findall(
        r"^FIG15_MEASUREMENT\t(\d+)\t(\d+)\t([0-9.eE+-]+)$",
        text, re.MULTILINE,
    )
    correct = re.findall(r"^FIG15_CORRECT\t32\t0$", text, re.MULTILINE)
    if device != [GPU_NAME] or len(measurement) != 1 or len(correct) != 1:
        raise RuntimeError("application output/correctness record is incomplete")
    if (int(measurement[0][0]), int(measurement[0][1])) != (warmup, launches):
        raise RuntimeError("application timing parameters changed")
    if float(measurement[0][2]) <= 0:
        raise RuntimeError("application elapsed time is not positive")
    if re.search(r"\[(?:error|critical)\]", text, re.IGNORECASE):
        raise RuntimeError("application log contains runtime error/critical record")


def validate_engagement_logs(application_path: Path, agent_path: Path,
                             arm: str) -> None:
    """Bind the selected arm to the transformed and loaded application module."""
    application = application_path.read_text(encoding="utf-8", errors="replace")
    agent = agent_path.read_text(encoding="utf-8", errors="replace")
    combined = application + "\n" + agent
    required = {
        "target_transform": (
            r"^\[ptxpass\] kprobe_entry_stub: matched=1, "
            r"in=\d+, out=\d+$"
        ),
        "module_load": r"Loaded module: patched\.map_bench\.sm_120\.ptx",
        "attach": r"Attach successfully",
    }
    counts = {name: len(re.findall(pattern, application, re.MULTILINE))
              for name, pattern in required.items()}
    programs = re.findall(
        r"corresponding program ([A-Za-z0-9_]+) is cuda program", application,
    )
    expected_program = PROGRAM_PREFIXES[arm]
    if any(value != 1 for value in counts.values()) or not programs or \
            set(programs) != {expected_program}:
        raise RuntimeError(
            "selected program/transform/module/attach evidence is incomplete: "
            f"arm={arm}, expected_program={expected_program}, "
            f"programs={programs}, counts={counts}"
        )
    bootstrap = {
        "verifier_mode": r"Verifier mode: WARNING",
        "cuda_shm": r"Registered shared memory with CUDA:",
        "global_shm": r"Global shm constructed\. shm_open_type 1 for fig15_map_",
        "global_shm_ready": r"Global shm initialized",
    }
    bootstrap_counts = {
        name: len(re.findall(pattern, agent))
        for name, pattern in bootstrap.items()
    }
    if any(value != 1 for value in bootstrap_counts.values()):
        raise RuntimeError(
            f"agent bootstrap evidence is incomplete: {bootstrap_counts}"
        )
    if re.search(r"\[(?:error|critical)\]", combined, re.IGNORECASE):
        raise RuntimeError("application/agent log contains runtime error/critical record")


def expected_map(arm: str) -> tuple[str, dict[int, int]] | None:
    if arm == "noop":
        return None
    name = "observed_values" if arm.endswith("_lookup") else {
        "device_update": "device_values",
        "host_update": "host_values",
        "rpc_update": "rpc_values",
    }[arm]
    magic = LOOKUP_MAGIC if arm.endswith("_lookup") else UPDATE_MAGIC
    return name, {key: magic ^ key for key in range(32)}


def validate_loader_log(path: Path, arm: str) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    prime = list(re.finditer(r"^FIG15_SERVER_PRIMED\t1$", text, re.MULTILINE))
    object_load = list(re.finditer(
        r"^libbpf: loading object from .+$", text, re.MULTILINE,
    ))
    if len(prime) != 1 or len(object_load) != 1 or \
            prime[0].start() >= object_load[0].start():
        raise RuntimeError("loader syscall-server prime record is incomplete")
    if re.findall(r"^FIG15_READY\t([^\t]+)\t1$", text, re.MULTILINE) != [arm]:
        raise RuntimeError("loader readiness record is incomplete")
    if len(re.findall(r"^FIG15_DETACHED\t1$", text, re.MULTILINE)) != 1:
        raise RuntimeError("loader detach record is incomplete")
    rows = re.findall(r"^FIG15_MAP\t([^\t]+)\t(\d+)\t(\d+)$", text, re.MULTILINE)
    expectation = expected_map(arm)
    if expectation is None:
        if rows:
            raise RuntimeError("no-op loader unexpectedly emitted map data")
        return
    name, values = expectation
    parsed = {(row_name, int(key)): int(value) for row_name, key, value in rows}
    expected = {(name, key): value for key, value in values.items()}
    if len(parsed) != len(rows) or parsed != expected:
        raise RuntimeError("complete map readback differs from the operation oracle")


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


def attached_environment(build: Path, segment: str, agent_log: Path) -> tuple[dict[str, str], dict[str, str]]:
    common = {
        **base_environment(),
        "BPFTIME_GLOBAL_SHM_NAME": segment,
        "BPFTIME_MAP_GPU_THREAD_COUNT": "32",
        "BPFTIME_SHM_MEMORY_MB": "256",
        "BPFTIME_MAX_FD_COUNT": "1024",
        "BPFTIME_LOG_OUTPUT": "console",
        "SPDLOG_LEVEL": "info",
        "BPFTIME_SM_ARCH": "sm_120",
        "BPFTIME_VERIFIER_LEVEL": "WARNING",
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
    segment = f"fig15_map_{os.getpid()}_{time.monotonic_ns()}"
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
        if application_process.wait(timeout=180) != 0:
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
        validate_engagement_logs(application_log, agent_log, arm)
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
        validate_environment(args.bpftime_root, args.bpftime_build),
        encoding="utf-8",
    )
    deadline = time.monotonic() + 3600
    with ReadOnlyLeases():
        for item in schedule:
            if time.monotonic() >= deadline:
                raise RuntimeError("one-hour campaign deadline reached")
            directory = args.output / (
                f"block-{int(item['block']):02d}-order-{int(item['order']):02d}-"
                f"{item['arm']}"
            )
            print(f"block={item['block']} order={item['order']} arm={item['arm']}",
                  flush=True)
            if item["arm"] == "native":
                run_native(directory, warmup, launches, int(item["run_id"]))
            else:
                run_attached(
                    str(item["arm"]), directory, args.bpftime_build, warmup,
                    launches, int(item["run_id"]),
                )
    print(f"completed {len(schedule)} frozen arm processes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
