#!/usr/bin/env python3
"""Run the excluded seven-cell stale-state preflight on an already loaded bridge.

The destructive module swap is deliberately outside this child. A lifecycle
wrapper must load the reviewed candidate, invoke this runner with both lease
descriptors inherited, and restore the admitted module even if this child fails.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from typing import Any, Sequence

import coordinator
import observer_protocol
import protocol


HERE = Path(__file__).resolve().parent
RAW_ROOT = HERE / "raw"
WORKLOAD = HERE / "stale_state_workload"
UVM_MONITOR = HERE / "uvm_event_monitor"
COMPUTE_MONITOR = HERE / "compute_monitor.py"
LIVE_LOADER = HERE / "driver-bridge-v1/live_loader"
LOADED_UVM_BTF = Path("/sys/kernel/btf/nvidia_uvm")
LOADED_UVM_VERSION = Path("/sys/module/nvidia_uvm/version")
UVM_REFCOUNT = Path("/sys/module/nvidia_uvm/refcnt")
TELEMETRY_CPU = 16
KERNEL_ABNORMAL = re.compile(
    r"NVRM: Xid|BUG: unable to handle|Kernel panic|Oops:|"
    r"GPU has fallen off the bus|RmInitAdapter.*failed|"
    r"NVRM:.*(?:fatal|error)|nvidia-uvm.*(?:fatal|error)",
    re.IGNORECASE,
)


class LiveError(RuntimeError):
    pass


def demand(condition: bool, message: str) -> None:
    if not condition:
        raise LiveError(message)


def run_checked(argv: list[str], timeout: float = 30) -> str:
    completed = subprocess.run(
        argv, text=True, capture_output=True, check=False, timeout=timeout
    )
    if completed.returncode:
        raise LiveError(
            f"command failed ({completed.returncode}): {argv!r}\n"
            f"{completed.stderr[-4000:]}"
        )
    return completed.stdout.strip()


def atomic_json(path: Path, value: Any) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_jsonl_new(path: Path, values: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def json_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return observer_protocol.parse_jsonl(path.read_text(encoding="utf-8", errors="strict"))


def wait_event(process: subprocess.Popen[Any], path: Path, event: str,
               timeout: float = 30) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            values = json_events(path)
        except protocol.ValidationError:
            values = []
        for value in reversed(values):
            if value.get("event") == event:
                return value
        if process.poll() is not None:
            tail = path.read_text(errors="replace")[-4000:] if path.exists() else ""
            raise LiveError(
                f"owned process {process.pid} exited {process.returncode} before {event}: {tail}"
            )
        time.sleep(0.1)
    raise LiveError(f"timed out waiting for {event} from owned PID {process.pid}")


def stop_owned(process: subprocess.Popen[Any], timeout: float = 30) -> None:
    if process.poll() is not None:
        return
    pgid = os.getpgid(process.pid)
    demand(pgid == process.pid, "owned process-group identity changed")
    os.killpg(pgid, signal.SIGINT)
    try:
        process.wait(timeout=min(timeout, 15))
    except subprocess.TimeoutExpired:
        os.killpg(pgid, signal.SIGTERM)
        try:
            process.wait(timeout=min(timeout, 10))
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            process.wait(timeout=5)


class InheritedLeases:
    def __init__(self, descriptors: Sequence[int]):
        self.descriptors = tuple(descriptors)

    def validate(self) -> list[dict[str, Any]]:
        demand(len(self.descriptors) == len(protocol.LEASE_PATHS),
               "both inherited lease descriptors are required")
        result = []
        for descriptor, expected_path in zip(self.descriptors, protocol.LEASE_PATHS):
            info = os.fstat(descriptor)
            expected = Path(expected_path).stat()
            demand(stat.S_ISREG(info.st_mode), "lease descriptor is not regular")
            demand((info.st_dev, info.st_ino) == (expected.st_dev, expected.st_ino),
                   f"lease descriptor does not name {expected_path}")
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result.append({"path": expected_path, "device": info.st_dev,
                           "inode": info.st_ino})
        return result


def struct_ops_inventory() -> dict[str, list[dict[str, Any]]]:
    maps = json.loads(run_checked(["sudo", "-n", "bpftool", "map", "show", "-j"]) or "[]")
    links = json.loads(run_checked(["sudo", "-n", "bpftool", "link", "show", "-j"]) or "[]")
    return {
        "maps": [item for item in maps if item.get("type") == "struct_ops"],
        "links": [item for item in links if item.get("type") == "struct_ops"],
    }


def gpu_state() -> dict[str, Any]:
    rows = run_checked([
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]).splitlines()
    demand(len(rows) == 1, "exactly one GPU is required")
    fields = [field.strip() for field in rows[0].split(",")]
    demand(len(fields) == 5, "unexpected GPU identity output")
    applications = run_checked([
        "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits",
    ])
    pids = sorted({int(line.strip()) for line in applications.splitlines()
                   if line.strip().isdigit()})
    return {"index": int(fields[0]), "name": fields[1], "driver": fields[2],
            "memory_used_mib": int(fields[3]),
            "utilization_gpu_percent": int(fields[4]), "compute_apps": pids}


def filtered_kernel(text: str) -> list[str]:
    return [line for line in text.splitlines() if KERNEL_ABNORMAL.search(line)]


def safety_snapshot() -> dict[str, Any]:
    service = run_checked([
        "systemctl", "show", "nvidia-power-limit.service", "-p", "ActiveState", "--value",
    ])
    power = float(run_checked([
        "nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits",
    ]))
    dmesg = run_checked(["sudo", "-n", "dmesg", "--color=never"])
    journal = run_checked([
        "journalctl", "-k", "-b", "--no-pager", "-o", "short-monotonic",
    ])
    abnormal_dmesg = filtered_kernel(dmesg)
    abnormal_journal = filtered_kernel(journal)
    return {
        "timestamp_ns": time.time_ns(),
        "power_limit_service": service,
        "power_limit_w": power,
        "gpu": gpu_state(),
        "uvm_refcount": int(UVM_REFCOUNT.read_text().strip()),
        "struct_ops": struct_ops_inventory(),
        "dmesg_abnormal": abnormal_dmesg,
        "journal_abnormal": abnormal_journal,
        "xids": [line for line in abnormal_journal if "NVRM: Xid" in line],
    }


def validate_idle(snapshot: dict[str, Any], before: dict[str, Any] | None = None) -> None:
    gpu = snapshot["gpu"]
    demand(snapshot["power_limit_service"] == "active", "power-limit service is inactive")
    demand(abs(snapshot["power_limit_w"] - 400.0) <= 0.01, "power limit differs from 400 W")
    demand(gpu["index"] == 0 and gpu["name"] == protocol.EXPECTED_GPU and
           gpu["driver"] == protocol.EXPECTED_DRIVER, "GPU/driver identity differs")
    demand(not gpu["compute_apps"] and gpu["memory_used_mib"] <= 256 and
           gpu["utilization_gpu_percent"] == 0, "GPU is not idle")
    demand(snapshot["uvm_refcount"] == 0, "UVM reference count is not zero")
    demand(snapshot["struct_ops"] == {"maps": [], "links": []},
           "struct_ops state is not empty")
    demand(not snapshot["dmesg_abnormal"] and not snapshot["journal_abnormal"],
           "kernel safety history is not clean")
    if before is not None:
        for field in ("dmesg_abnormal", "journal_abnormal", "xids"):
            demand(snapshot[field] == before[field], f"kernel safety history changed: {field}")


def duplicate_workload_uvm_fd(pid: int) -> int:
    matches = []
    for fd_path in (Path("/proc") / str(pid) / "fd").glob("[0-9]*"):
        try:
            if os.readlink(fd_path) == "/dev/nvidia-uvm":
                matches.append(int(fd_path.name))
        except OSError:
            continue
    demand(len(matches) == 1, f"owned workload has {len(matches)} UVM fds, expected one")
    pidfd = os.pidfd_open(pid, 0)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        duplicated = libc.syscall(438, pidfd, matches[0], 0)
        if duplicated < 0:
            error = ctypes.get_errno()
            raise LiveError(f"pidfd_getfd failed: {os.strerror(error)}")
        return int(duplicated)
    finally:
        os.close(pidfd)


def start_json_process(argv: list[str], output: Path, error: Path,
                       pass_fds: tuple[int, ...] = ()) -> tuple[subprocess.Popen[Any], Any, Any]:
    stdout = output.open("x", buffering=1)
    stderr = error.open("x", buffering=1)
    process = subprocess.Popen(
        argv, stdout=stdout, stderr=stderr, text=True, pass_fds=pass_fds,
        start_new_session=True,
    )
    return process, stdout, stderr


def start_telemetry(cell_dir: Path) -> tuple[subprocess.Popen[Any], Any]:
    demand(TELEMETRY_CPU in os.sched_getaffinity(0), "telemetry CPU 16 is unavailable")
    output = (cell_dir / "gpu-telemetry.csv").open("x", buffering=1)
    query = ",".join(("timestamp", "power.draw", "memory.used", "utilization.gpu"))
    process = subprocess.Popen(
        ["taskset", "-c", str(TELEMETRY_CPU), "nvidia-smi", f"--query-gpu={query}",
         "--format=csv", "--loop-ms=200"], stdout=output, stderr=subprocess.STDOUT,
        text=True, start_new_session=True,
    )
    time.sleep(0.3)
    demand(process.poll() is None, "GPU telemetry exited before workload")
    return process, output


def wait_compute_boundary(path: Path, *, empty: bool, timeout: float = 15) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            rows = observer_protocol.parse_jsonl(path.read_text())
        except (OSError, protocol.ValidationError):
            rows = []
        if rows and rows[-1].get("error") is None and bool(rows[-1].get("pids")) is not empty:
            return
        time.sleep(0.1)
    raise LiveError(f"compute monitor did not reach {'empty' if empty else 'target'} boundary")


def validate_loaded_bridge() -> dict[str, Any]:
    demand(os.geteuid() == 0, "live runner is root-only")
    demand(LOADED_UVM_VERSION.read_text().strip() == protocol.EXPECTED_DRIVER,
           "loaded UVM version differs")
    demand(coordinator.PROC_PATH.is_file(), "stale-state proc endpoint is absent")
    raw = run_checked(["bpftool", "btf", "dump", "file", str(LOADED_UVM_BTF), "format", "raw"])
    required = (
        "FUNC 'uvm_stale_state_v1_diagnostic'",
        "FUNC 'bpf_gpu_stale_state_v1_request'",
        "'gpu_stale_state_prefetch_v1'",
    )
    missing = [name for name in required if name not in raw]
    demand(not missing, f"loaded bridge BTF lacks: {missing}")
    demand(coordinator.ProcBridge().status().mode == "off", "bridge is not initially off")
    return {"version": protocol.EXPECTED_DRIVER, "required_btf": list(required)}


def validate_paths(output: Path) -> Path:
    output = protocol.lexical_absolute(output)
    demand(output.parent == RAW_ROOT, f"output must be a fresh direct child of {RAW_ROOT}")
    demand(output.name.startswith("stale-state-575-preflight-"), "output namespace differs")
    demand(not output.exists(), "refusing to reuse preflight output")
    for path in (WORKLOAD, UVM_MONITOR, LIVE_LOADER):
        demand(path.is_file() and not path.is_symlink() and os.access(path, os.X_OK),
               f"required executable is absent: {path}")
    demand(COMPUTE_MONITOR.is_file() and not COMPUTE_MONITOR.is_symlink(),
           "compute monitor is absent")
    return output


def dry_run(output: Path, lease_fds: Sequence[int]) -> dict[str, Any]:
    output = protocol.lexical_absolute(output)
    return {
        "mode": "cpu-only-dry-run",
        "experiment_evidence": False,
        "executes_gpu": False,
        "loads_modules": False,
        "output": str(output),
        "lease_fds": list(lease_fds),
        "baseline_policy_artifacts": False,
        "order": [asdict(cell) for cell in protocol.matrix("preflight")],
        "policy_loader": {
            "native": "fentry observer only",
            "bpf": "fentry observer plus one owned struct_ops link",
            "baseline": None,
        },
        "module_boundary": (
            "an outer reviewed lifecycle must load the candidate and restore the "
            "admitted 575 module; this child never changes modules"
        ),
    }


def run_cell(cell: protocol.MatrixCell, cell_dir: Path) -> dict[str, Any]:
    cell_dir.mkdir()
    execution: dict[str, Any] = {
        "protocol": protocol.PROTOCOL, "timeline": protocol.TIMELINE,
        "block": cell.block, "arm": cell.arm,
        "implementation": cell.implementation, "delay_ms": cell.delay_ms,
        "status": "running", "complete": False, "cleanup_errors": [],
        "lease_paths": list(protocol.LEASE_PATHS),
        "lease_mode": "read_only_exclusive",
    }
    atomic_json(cell_dir / "execution.json", execution)
    before = safety_snapshot()
    validate_idle(before)
    atomic_json(cell_dir / "safety-before.json", before)

    owned: list[tuple[subprocess.Popen[Any], Any, Any | None]] = []
    workload: subprocess.Popen[Any] | None = None
    workload_log: Any | None = None
    uvm: subprocess.Popen[Any] | None = None
    observer: subprocess.Popen[Any] | None = None
    truth_read = truth_write = release_read = release_write = -1
    coordinator_result: dict[str, Any] | None = None
    primary: BaseException | None = None
    try:
        telemetry, telemetry_output = start_telemetry(cell_dir)
        owned.append((telemetry, telemetry_output, None))
        compute, compute_output, compute_error = start_json_process(
            [sys.executable, "-B", str(COMPUTE_MONITOR), "--interval-ms", "200"],
            cell_dir / "compute-apps.jsonl", cell_dir / "compute-apps.stderr.log",
        )
        owned.append((compute, compute_output, compute_error))
        wait_compute_boundary(cell_dir / "compute-apps.jsonl", empty=True)
        kernel_output = (cell_dir / "kernel-monitor.log").open("x", buffering=1)
        kernel = subprocess.Popen(
            ["journalctl", "-kf", "-n", "0", "--no-pager", "-o", "short-monotonic"],
            stdout=kernel_output, stderr=subprocess.STDOUT, text=True,
            start_new_session=True,
        )
        owned.append((kernel, kernel_output, None))

        release_read, release_write = os.pipe()
        truth_read, truth_write = os.pipe()
        workload_log = (cell_dir / "workload.stderr.log").open("x", buffering=1)
        workload = subprocess.Popen(
            [str(WORKLOAD), "--result", str(cell_dir / "workload-result.json"),
             "--truth", str(cell_dir / "phase-truth.jsonl"),
             "--release-fd", str(release_read), "--truth-fd", str(truth_write)],
            stdout=workload_log, stderr=subprocess.STDOUT, text=True,
            pass_fds=(release_read, truth_write), start_new_session=True,
        )
        os.close(release_read)
        release_read = -1
        os.close(truth_write)
        truth_write = -1
        execution["target_pid"] = workload.pid

        def before_release(_: dict[str, Any]) -> None:
            nonlocal uvm, observer
            duplicated = duplicate_workload_uvm_fd(workload.pid)
            try:
                uvm, uvm_output, uvm_error = start_json_process(
                    [str(UVM_MONITOR), "--uvm-fd", str(duplicated),
                     "--target-pid", str(workload.pid)],
                    cell_dir / "uvm-events.jsonl", cell_dir / "uvm-events.stderr.log",
                    (duplicated,),
                )
            finally:
                os.close(duplicated)
            owned.append((uvm, uvm_output, uvm_error))
            wait_event(uvm, cell_dir / "uvm-events.jsonl", "ready")
            if cell.implementation is not None:
                observer, observer_output, observer_error = start_json_process(
                    [str(LIVE_LOADER), "--target-pid", str(workload.pid),
                     "--implementation", cell.implementation,
                     "--verifier-log", str(cell_dir / "verifier.log")],
                    cell_dir / "policy-observer.jsonl",
                    cell_dir / "policy-observer.stderr.log",
                )
                owned.append((observer, observer_output, observer_error))
                ready = wait_event(observer, cell_dir / "policy-observer.jsonl", "ready")
                inventory = struct_ops_inventory()
                if cell.implementation == "native":
                    demand(inventory == {"maps": [], "links": []},
                           "native observer unexpectedly owns struct_ops state")
                else:
                    demand({row.get("id") for row in inventory["maps"]} ==
                           {ready["struct_map_id"]}, "BPF struct_ops map ownership differs")
                    demand({row.get("id") for row in inventory["links"]} ==
                           {ready["struct_link_id"]}, "BPF struct_ops link ownership differs")

        def release() -> None:
            nonlocal release_write
            written = os.write(release_write, b"R")
            demand(written == 1, "short workload release write")
            os.close(release_write)
            release_write = -1

        generation = time.time_ns() if cell.implementation is not None else None
        coordinator_result = coordinator.TruthFDCoordinator(
            coordinator.ProcBridge(), truth_timeout_seconds=45
        ).run(
            truth_fd=truth_read, expected_pid=workload.pid, release=release,
            implementation=cell.implementation, generation=generation,
            delay_ms=cell.delay_ms, before_release=before_release,
        )
        os.close(truth_read)
        truth_read = -1
        workload.wait(timeout=30)
        demand(workload.returncode == 0, f"workload exited {workload.returncode}")
        wait_compute_boundary(cell_dir / "compute-apps.jsonl", empty=True)

        if uvm is not None:
            stop_owned(uvm)
            demand(uvm.returncode == 0, f"UVM monitor exited {uvm.returncode}")
        if observer is not None:
            stop_owned(observer)
            demand(observer.returncode == 0, f"observer exited {observer.returncode}")
            observed = observer_protocol.validate_records(
                json_events(cell_dir / "policy-observer.jsonl"),
                expected_pid=workload.pid, implementation=cell.implementation,
            )
            status = coordinator.BridgeStatus(**coordinator_result["final_enabled_status"])
            policy_final = observer_protocol.reconcile_driver(
                observed, status, implementation=cell.implementation
            )
            write_jsonl_new(cell_dir / "snapshot-publications.jsonl",
                            coordinator_result["publications"])
            write_jsonl_new(cell_dir / "policy-decisions.jsonl", observed["decisions"])
            atomic_json(cell_dir / "policy-final.json", policy_final)
        else:
            demand(cell.role == "context_control", "policy row lacked observer")
            demand(not any((cell_dir / name).exists()
                           for name in protocol.POLICY_ARTIFACT_NAMES),
                "baseline created a policy artifact")
    except BaseException as exc:
        primary = exc
    finally:
        cleanup_errors = []
        for descriptor in (truth_read, truth_write, release_read, release_write):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError as exc:
                    cleanup_errors.append(str(exc))
        if workload is not None:
            try:
                stop_owned(workload)
            except BaseException as exc:
                cleanup_errors.append(str(exc))
        if workload_log is not None and not workload_log.closed:
            workload_log.close()
        for process, stdout, stderr in reversed(owned):
            try:
                stop_owned(process)
                if stdout and not stdout.closed:
                    stdout.close()
                if stderr and not stderr.closed:
                    stderr.close()
            except BaseException as exc:
                cleanup_errors.append(str(exc))
        execution["cleanup_errors"] = cleanup_errors

    if primary is not None or execution["cleanup_errors"]:
        execution.update(status="failed", complete=False,
                         failure=None if primary is None else f"{type(primary).__name__}: {primary}")
        atomic_json(cell_dir / "execution.json", execution)
        if primary is not None:
            raise primary
        raise LiveError(f"cell cleanup failed: {execution['cleanup_errors']}")

    after = safety_snapshot()
    validate_idle(after, before)
    atomic_json(cell_dir / "safety-after.json", after)
    telemetry_valid = (cell_dir / "gpu-telemetry.csv").stat().st_size > 0
    monitor_coverage = {"uvm": True, "gpu_telemetry": True,
                        "compute_apps": True, "kernel_log": True,
                        "phase_truth": True}
    if cell.role == "context_control":
        monitor_coverage["policy_artifact_absence"] = True
    else:
        monitor_coverage["policy_diagnostics"] = True
    execution.update(
        status="passed", complete=True,
        monitor_coverage=monitor_coverage,
        cleanup={"workload_reaped": workload is not None and workload.poll() is not None,
                 "monitors_reaped": all(item[0].poll() is not None for item in owned),
                 "policy_detached": struct_ops_inventory() == {"maps": [], "links": []},
                 "leases_retained": True},
        safety={"pre_valid": True, "post_valid": True,
                "gpu_telemetry_valid": telemetry_valid,
                "foreign_compute_pids": [], "new_kernel_anomalies": []},
    )
    atomic_json(cell_dir / "execution.json", execution)
    return protocol.validate_cell(cell_dir, cell)


def execute(output: Path, lease_fds: Sequence[int]) -> dict[str, Any]:
    output = validate_paths(output)
    leases = InheritedLeases(lease_fds).validate()
    bridge = validate_loaded_bridge()
    output.mkdir(parents=False, exist_ok=False)
    cells = protocol.matrix("preflight")
    completed = []
    manifest = {
        "protocol": protocol.PROTOCOL, "timeline": protocol.TIMELINE,
        "stage": "preflight", "seed": protocol.SEED,
        "blocks": protocol.PREFLIGHT_BLOCKS, "complete": False,
        "order": [asdict(cell) for cell in cells], "completed": completed,
        "leases": leases, "loaded_bridge": bridge,
    }
    atomic_json(output / "campaign.json", manifest)
    for cell in cells:
        run_cell(cell, output / f"block-{cell.block:02d}-{cell.arm}")
        completed.append(asdict(cell))
        atomic_json(output / "campaign.json", manifest)
    manifest["complete"] = True
    atomic_json(output / "campaign.json", manifest)
    return protocol.validate_preflight(output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("dry-run", "execute-preflight"):
        command = commands.add_parser(name)
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--inherited-lease-fds", nargs="*", type=int, default=[])
    args = parser.parse_args(argv)
    try:
        result = (dry_run(args.output, args.inherited_lease_fds)
                  if args.command == "dry-run"
                  else execute(args.output, args.inherited_lease_fds))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (LiveError, protocol.ValidationError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
