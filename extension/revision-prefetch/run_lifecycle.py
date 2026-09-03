#!/usr/bin/env python3
"""Run the Q2 controls under one reversible, UVM-only module lifecycle."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Callable

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_safety as cells  # noqa: E402

safety = cells.safety

EXPECTED_KERNEL = "6.15.11-061511-generic"
EXPECTED_DRIVER = "575.57.08"
EXPECTED_VERMAGIC = (
    "6.15.11-061511-generic SMP preempt mod_unload modversions"
)
MODULE_NAME = "nvidia_uvm"
MODULE_FILENAME = "nvidia-uvm.ko"
LOADED_MODULE = Path("/sys/module/nvidia_uvm")
LOADED_UVM_BTF = Path("/sys/kernel/btf/nvidia_uvm")
CORE_MODULE = Path("/sys/module/nvidia")
CORE_BTF = Path("/sys/kernel/btf/nvidia")
PARAMETERS = LOADED_MODULE / "parameters"
BOOT_ID = Path("/proc/sys/kernel/random/boot_id")
STAGE_ROOT = Path("/opt/gpubpf/modules/575.57.08")
KNOWN_RESTORE = (
    STAGE_ROOT / "gpreempt-849ea75d-6.15.11" / MODULE_FILENAME
)
RESULT_ROOT = ROOT / "docs/experiment/revision-safety"
SERVICES = ("gdm.service", "nvidia-persistenced.service")
LEASE_PATHS = (Path("/tmp/gpubpf-revision-gpu0.lock"),
               Path("/tmp/gpubpf-revision-struct-ops.lock"))
MODES = tuple(cells.MODES)
COMMAND_TIMEOUT = 30.0
STATE_TIMEOUT = 30.0


class LifecycleError(RuntimeError):
    pass


class LifecycleLeases:
    """Lock only the two existing inodes through read-only descriptors."""
    def __init__(self, paths: tuple[Path, ...] = LEASE_PATHS):
        self.fds: list[int] = []
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            for path in paths:
                fd = os.open(path, flags)
                try:
                    if not stat.S_ISREG(os.fstat(fd).st_mode):
                        raise LifecycleError(f"lease descriptor is not a regular file: {path}")
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BaseException:
                    os.close(fd)
                    raise
                self.fds.append(fd)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        for fd in reversed(self.fds):
            os.close(fd)
        self.fds.clear()


@dataclass
class RuntimeState:
    destructive_started: bool = False
    old_unloaded: bool = False
    last_insert: str = "initial"
    candidate_loaded: bool = False
    cells_completed: list[str] = field(default_factory=list)
    old_restored: bool = False
    services_restored: bool = False


class Recorder:
    def __init__(self, output: Path, arguments: dict[str, str]):
        self.output = output
        self.value: dict[str, Any] = {
            "complete": False,
            "started_ns": time.time_ns(),
            "arguments": arguments,
            "transitions": [],
        }

    def transition(self, name: str, status: str, **details: Any) -> None:
        self.value["transitions"].append({
            "name": name,
            "status": status,
            "timestamp_ns": time.time_ns(),
            **details,
        })
        self.write()

    def write(self) -> None:
        safety.atomic_write_json(self.output / "lifecycle.json", self.value)


def run_command(argv: list[str], *, timeout: float = COMMAND_TIMEOUT,
                allowed: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(argv, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        try:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate(timeout=2)
        raise LifecycleError(f"command exceeded {timeout} seconds: {argv!r}") from error
    result = subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
    if result.returncode not in allowed:
        raise LifecycleError(
            f"command failed ({result.returncode}): {argv!r}\n{result.stderr[-4000:]}"
        )
    return result


def checked_stdout(argv: list[str], *, timeout: float = COMMAND_TIMEOUT) -> str:
    return run_command(argv, timeout=timeout).stdout.strip()


def file_inventory(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise LifecycleError(f"module path is not a non-symlink regular file: {path}")
    if info.st_size <= 0:
        raise LifecycleError(f"module is empty: {path}")
    return {
        "path": str(resolved),
        "size_bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "device": info.st_dev,
        "inode": info.st_ino,
        "uid": info.st_uid,
        "gid": info.st_gid,
        "mode": stat.S_IMODE(info.st_mode),
    }


def btf_raw(path: Path) -> str:
    return checked_stdout(["bpftool", "btf", "dump", "file", str(path), "format", "raw"])


def generic_uvm_interface(raw: str) -> dict[str, Any]:
    match = re.search(
        r"STRUCT 'gpu_mem_ops' size=48 vlen=6\n(?P<members>(?:\t[^\n]+\n){6})",
        raw,
    )
    expected = (
        "gpu_test_trigger", "gpu_page_prefetch", "gpu_page_prefetch_iter",
        "gpu_block_activate", "gpu_block_access", "gpu_evict_prepare",
    )
    members = tuple(re.findall(r"\t'([^']+)'", match["members"])) if match else ()
    if members != expected:
        raise LifecycleError(f"gpu_mem_ops ABI differs: {members}")
    required = (
        "bpf_gpu_request_reorder", "bpf_gpu_set_prefetch_region",
        "uvm_bpf_call_gpu_page_prefetch",
    )
    missing = [name for name in required if f"FUNC '{name}'" not in raw]
    if missing:
        raise LifecycleError(f"UVM BTF lacks required functions: {missing}")
    cells.validate_layout(raw, "nv_gpu_prefetch_decision_t", 24,
                          [("attempted", 0), ("conflict", 8),
                           ("first", 64), ("outer", 128)])
    cells.validate_layout(raw, "uvm_va_block_region_t", 4,
                          [("first", 0), ("outer", 16)])
    return {
        "gpu_mem_ops_members": list(members),
        "required_functions": list(required),
        "decision_size": 24,
        "region_size": 4,
    }


def module_descriptor(path: Path, *, diagnostic: bool) -> dict[str, Any]:
    inventory = file_inventory(path)
    name = checked_stdout(["modinfo", "-F", "name", str(path)])
    version = checked_stdout(["modinfo", "-F", "version", str(path)])
    vermagic = checked_stdout(["modinfo", "-F", "vermagic", str(path)])
    depends_text = checked_stdout(["modinfo", "-F", "depends", str(path)])
    depends = sorted(item for item in depends_text.split(",") if item)
    parms = checked_stdout(["modinfo", "-F", "parm", str(path)]).splitlines()
    parameter_names = sorted({line.split(":", 1)[0] for line in parms if ":" in line})
    raw = btf_raw(path)
    interface = generic_uvm_interface(raw)
    diagnostic_present = "FUNC 'uvm_bpf_prefetch_diagnostic'" in raw
    if diagnostic:
        cells.validate_diagnostic_interface(raw)
    elif diagnostic_present:
        raise LifecycleError(f"restore module unexpectedly contains the diagnostic: {path}")
    descriptor = {
        "inventory": inventory,
        "name": name,
        "version": version,
        "vermagic": vermagic,
        "depends": depends,
        "parameter_names": parameter_names,
        "interface": interface,
        "diagnostic_present": diagnostic_present,
    }
    validate_module_descriptor(descriptor, diagnostic=diagnostic)
    return descriptor


def validate_module_descriptor(value: dict[str, Any], *, diagnostic: bool) -> None:
    if value.get("name") != MODULE_NAME:
        raise LifecycleError(f"module name differs: {value.get('name')}")
    if value.get("version") != EXPECTED_DRIVER:
        raise LifecycleError(f"module version differs: {value.get('version')}")
    if value.get("vermagic") != EXPECTED_VERMAGIC:
        raise LifecycleError(f"module vermagic differs: {value.get('vermagic')}")
    if value.get("depends") != ["nvidia"]:
        raise LifecycleError(f"module dependencies differ: {value.get('depends')}")
    if value.get("diagnostic_present") is not diagnostic:
        raise LifecycleError("module diagnostic interface differs from its assigned role")
    names = value.get("parameter_names")
    if not isinstance(names, list) or "uvm_perf_prefetch_enable" not in names:
        raise LifecycleError("module parameter inventory lacks uvm_perf_prefetch_enable")


def comparable_descriptor(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "size_bytes": value["inventory"]["size_bytes"],
        "name": value["name"],
        "version": value["version"],
        "vermagic": value["vermagic"],
        "depends": value["depends"],
        "parameter_names": value["parameter_names"],
        "interface": value["interface"],
        "diagnostic_present": value["diagnostic_present"],
    }


def require_unchanged_module(path: Path, expected: dict[str, Any], *,
                             diagnostic: bool) -> dict[str, Any]:
    observed = module_descriptor(path, diagnostic=diagnostic)
    if observed != expected:
        raise LifecycleError(f"module artifact changed during the lifecycle: {path}")
    return observed


def read_parameters(directory: Path = PARAMETERS) -> dict[str, str]:
    if not directory.is_dir():
        raise LifecycleError(f"module parameter directory is absent: {directory}")
    values: dict[str, str] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", path.name):
            raise LifecycleError(f"unexpected parameter entry: {path}")
        value = path.read_text().strip()
        if not value or "\x00" in value or "\n" in value:
            raise LifecycleError(f"unsafe/empty module parameter value: {path.name}")
        values[path.name] = value
    if values.get("uvm_perf_prefetch_enable") != "1":
        raise LifecycleError("the admitted runtime does not have native prefetch enabled")
    return values


def parameter_arguments(parameters: dict[str, str]) -> list[str]:
    arguments = []
    for name, value in sorted(parameters.items()):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise LifecycleError(f"invalid module parameter name: {name!r}")
        if not isinstance(value, str) or not value or "\x00" in value or "\n" in value:
            raise LifecycleError(f"invalid module parameter value for {name}")
        if value == "(null)":
            continue
        arguments.append(f"{name}={value}")
    return arguments


def insmod_command(path: Path, parameters: dict[str, str]) -> list[str]:
    return ["sudo", "-n", "insmod", str(path), *parameter_arguments(parameters)]


def service_state(unit: str, *, allow_transitional: bool = False) -> dict[str, str]:
    raw = checked_stdout([
        "systemctl", "show", unit, "--no-pager",
        "-p", "LoadState", "-p", "ActiveState", "-p", "SubState",
        "-p", "Result", "-p", "UnitFileState",
    ])
    result = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
    if result.get("LoadState") != "loaded":
        raise LifecycleError(f"required service is not loaded: {unit}: {result}")
    active = result.get("ActiveState")
    if active in {"activating", "deactivating", "reloading"} and allow_transitional:
        return result
    if active not in {"active", "inactive"}:
        raise LifecycleError(f"service is in a transitional/failed state: {unit}: {result}")
    return result


def wait_for(description: str, predicate: Callable[[], bool],
             timeout: float = STATE_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while True:
        if predicate():
            return
        if time.monotonic() >= deadline:
            raise LifecycleError(f"timed out waiting for {description}")
        time.sleep(0.2)


def set_service(unit: str, action: str) -> None:
    if action not in {"start", "stop"}:
        raise LifecycleError(f"unsupported service action: {action}")
    target = "active" if action == "start" else "inactive"
    run_command(["sudo", "-n", "systemctl", "--no-block", action, unit])
    def reached_target() -> bool:
        return service_state(unit, allow_transitional=True)["ActiveState"] == target
    wait_for(f"{unit} to become {target}",
             reached_target)
    final = service_state(unit)
    if action == "start" and (final["SubState"] != "running" or
                              final.get("Result") != "success"):
        raise LifecycleError(f"service did not start cleanly: {unit}: {final}")


def service_stop_plan(initial: dict[str, dict[str, str]]) -> list[str]:
    return [unit for unit in SERVICES if initial[unit]["ActiveState"] == "active"]


def validate_initial_services(initial: dict[str, dict[str, str]]) -> None:
    for unit in SERVICES:
        value = initial[unit]
        if value["ActiveState"] == "active" and (
                value["SubState"] != "running" or value.get("Result") != "success"):
            raise LifecycleError(f"active service is not running/successful: {unit}: {value}")


def validate_local_sessions(sessions: list[dict[str, str]]) -> None:
    required = ("Id", "User", "Name", "Seat", "Class", "Type", "Service",
                "Active", "Remote", "State")
    for value in sessions:
        if any(name not in value for name in required):
            raise LifecycleError(f"login session record is incomplete: {value}")
        if value["Remote"] not in {"yes", "no"} or value["Active"] not in {"yes", "no"}:
            raise LifecycleError(f"login session locality/activity is ambiguous: {value}")
        if value["Remote"] == "no" and (
                not value["Seat"] or value["Class"] != "greeter" or
                value["Service"] != "gdm-launch-environment"):
            raise LifecycleError(f"local non-greeter session blocks GDM stop: {value}")


def local_sessions() -> list[dict[str, str]]:
    listing = checked_stdout(["loginctl", "list-sessions", "--no-legend", "--no-pager"])
    session_ids = [line.split()[0] for line in listing.splitlines() if line.split()]
    sessions = []
    properties = ("Id", "User", "Name", "Seat", "Class", "Type", "Service",
                  "Active", "Remote", "State")
    for session_id in session_ids:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", session_id):
            raise LifecycleError(f"unexpected login session identifier: {session_id!r}")
        raw = checked_stdout([
            "loginctl", "show-session", session_id, "--no-pager",
            *(part for name in properties for part in ("-p", name)),
        ])
        value = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
        if value.get("Id") != session_id:
            raise LifecycleError(f"login session identity changed: {session_id}: {value}")
        sessions.append(value)
    validate_local_sessions(sessions)
    return sessions


def service_restore_plan(initial: dict[str, dict[str, str]],
                         current: dict[str, dict[str, str]]) -> list[str]:
    # Persistenced precedes the display manager during restoration.
    return [unit for unit in reversed(SERVICES)
            if initial[unit]["ActiveState"] == "active" and
            current[unit]["ActiveState"] != "active"]


def service_stop_recheck(unit: str) -> list[dict[str, str]]:
    return local_sessions() if unit == "gdm.service" else []


def stop_service_after_recheck(unit: str,
                               record_gate: Callable[[list[dict[str, str]]], None]) -> None:
    cells.raise_if_interrupted()
    sessions = service_stop_recheck(unit)
    record_gate(sessions)
    cells.raise_if_interrupted()
    set_service(unit, "stop")


def validate_paths(candidate: Path, restore: Path, stage: Path, output: Path) -> None:
    for name, path in (("candidate", candidate), ("restore", restore),
                       ("stage", stage), ("output", output)):
        if not path.is_absolute():
            raise LifecycleError(f"{name} path must be absolute: {path}")
    if candidate.name != MODULE_FILENAME or restore.name != MODULE_FILENAME:
        raise LifecycleError("candidate and restore paths must name nvidia-uvm.ko")
    stage_parent = stage.parent.resolve(strict=True)
    if (stage_parent != STAGE_ROOT.resolve(strict=True) or
            not stage.name.startswith("prefetch-diagnostic-")):
        raise LifecycleError(f"stage must be a fresh direct child of {STAGE_ROOT}")
    output_parent = output.parent.resolve(strict=True)
    if (output_parent != RESULT_ROOT.resolve(strict=True) or
            output.name != "prefetch-invalid-575-02"):
        raise LifecycleError(f"output must be a fresh direct child of {RESULT_ROOT}")
    resolved_candidate = candidate.resolve(strict=True)
    resolved_restore = restore.resolve(strict=True)
    if resolved_restore != KNOWN_RESTORE.resolve(strict=True):
        raise LifecycleError(f"restore path is not the admitted old stage: {restore}")
    if resolved_candidate == resolved_restore:
        raise LifecycleError("candidate and restore module paths are identical")
    if stage.exists() or output.exists():
        raise LifecycleError("refusing to reuse the stage or output path")


def stage_candidate(candidate: Path, stage: Path) -> Path:
    run_command(["sudo", "-n", "mkdir", "--mode=0755", "--", str(stage)])
    destination = stage / MODULE_FILENAME
    run_command(["sudo", "-n", "install", "--mode=0644", "--",
                 str(candidate), str(destination)])
    run_command(["cmp", "--silent", "--", str(candidate), str(destination)])
    return destination


def lease_inventory() -> list[dict[str, Any]]:
    result = []
    for path in LEASE_PATHS:
        if path.is_symlink():
            raise LifecycleError(f"lease path is a symlink: {path}")
        info = path.stat()
        if (not stat.S_ISREG(info.st_mode) or info.st_size != 0 or
                (info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)) != (0, 0, 0o644)):
            raise LifecycleError(f"lease path identity/mode is unexpected: {path}")
        result.append({"path": str(path), "device": info.st_dev, "inode": info.st_ino,
                       "uid": info.st_uid, "gid": info.st_gid,
                       "mode": stat.S_IMODE(info.st_mode)})
    return result


def capture_core(output: Path | None = None) -> dict[str, Any]:
    if not CORE_MODULE.is_dir() or not CORE_BTF.is_file():
        raise LifecycleError("loaded NVIDIA core module/BTF is absent")
    version = (CORE_MODULE / "version").read_text().strip()
    if version != EXPECTED_DRIVER:
        raise LifecycleError(f"loaded core version differs: {version}")
    raw = btf_raw(CORE_BTF)
    required = ("nv_gpu_sched_ops", "bpf_nv_gpu_preempt_tsg")
    missing = [name for name in required if name not in raw]
    if missing:
        raise LifecycleError(f"loaded core BTF lacks expected interface names: {missing}")
    if output is not None:
        (output / "initial-core-btf.txt").write_text(raw + "\n")
    return {
        "version": version,
        "btf": file_inventory(CORE_BTF),
        "required_interface_names": list(required),
    }


def live_uvm_interface(*, diagnostic: bool) -> dict[str, Any]:
    if not LOADED_MODULE.is_dir() or not LOADED_UVM_BTF.is_file():
        raise LifecycleError("loaded UVM module/BTF is absent")
    version = (LOADED_MODULE / "version").read_text().strip()
    if version != EXPECTED_DRIVER:
        raise LifecycleError(f"loaded UVM version differs: {version}")
    raw = btf_raw(LOADED_UVM_BTF)
    interface = generic_uvm_interface(raw)
    present = "FUNC 'uvm_bpf_prefetch_diagnostic'" in raw
    if diagnostic:
        cells.validate_diagnostic_interface(raw)
    if present is not diagnostic:
        raise LifecycleError("loaded UVM diagnostic role differs")
    return {"version": version, "interface": interface,
            "diagnostic_present": present,
            "btf": file_inventory(LOADED_UVM_BTF)}


def require_boot(initial_boot: str) -> None:
    if BOOT_ID.read_text().strip() != initial_boot or os.uname().release != EXPECTED_KERNEL:
        raise LifecycleError("boot or kernel changed during the lifecycle")


def quiet_snapshot(initial_boot: str) -> dict[str, Any]:
    require_boot(initial_boot)
    snapshot = safety.safety_snapshot()
    gpu = snapshot["gpu"]
    if gpu["driver"] != EXPECTED_DRIVER:
        raise LifecycleError(f"GPU driver differs: {gpu['driver']}")
    if abs(float(snapshot["power_limit_w"]) - 400.0) > 0.01:
        raise LifecycleError(f"GPU power limit differs from 400 W: {snapshot['power_limit_w']}")
    if snapshot["uvm_refcount"] != 0:
        raise LifecycleError(f"UVM reference count is nonzero: {snapshot['uvm_refcount']}")
    if snapshot["struct_ops"]["maps"] or snapshot["struct_ops"]["links"]:
        raise LifecycleError(f"struct_ops state is nonempty: {snapshot['struct_ops']}")
    if gpu["compute_apps"] or gpu["memory_used_mib"] > 256 or gpu["utilization_gpu_percent"] != 0:
        raise LifecycleError(f"GPU is not idle: {gpu}")
    return snapshot


def validate_fuser_result(result: subprocess.CompletedProcess[str],
                          devices: list[Path]) -> dict[str, Any]:
    observed = {"devices": [str(path) for path in devices],
                "returncode": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    if result.returncode == 0:
        raise LifecycleError(f"UVM device holder exists: {observed}")
    if result.returncode != 1 or result.stdout or result.stderr:
        raise LifecycleError(f"fuser did not produce an unambiguous empty result: {observed}")
    return observed


def no_uvm_holders() -> dict[str, Any]:
    devices = [Path("/dev/nvidia-uvm"), Path("/dev/nvidia-uvm-tools")]
    missing = [str(path) for path in devices if not path.exists()]
    if missing:
        raise LifecycleError(f"required UVM device nodes are absent: {missing}")
    result = run_command(["sudo", "-n", "fuser", "-v", *map(str, devices)],
                         allowed=(0, 1))
    return validate_fuser_result(result, devices)


def remove_uvm(initial_boot: str,
               history_baseline: dict[str, Any] | None = None,
               *, honor_interrupt: bool = False) -> dict[str, Any]:
    snapshot = quiet_snapshot(initial_boot)
    if history_baseline is not None:
        safety.validate_post_server_safety(history_baseline, snapshot)
    holders = no_uvm_holders()
    pre_remove = quiet_snapshot(initial_boot)
    if history_baseline is not None:
        safety.validate_post_server_safety(history_baseline, pre_remove)
    if honor_interrupt:
        cells.raise_if_interrupted()
    run_command(["sudo", "-n", "rmmod", MODULE_NAME])
    wait_for("nvidia_uvm removal", lambda: not LOADED_MODULE.exists())
    if LOADED_UVM_BTF.exists():
        raise LifecycleError("UVM BTF survived module removal")
    return {"quiet": snapshot, "holders": holders, "pre_remove": pre_remove}


def insert_uvm(path: Path, parameters: dict[str, str], *, diagnostic: bool,
               expected_interface: dict[str, Any], initial_boot: str,
               honor_interrupt: bool = False) -> dict[str, Any]:
    require_boot(initial_boot)
    if LOADED_MODULE.exists():
        raise LifecycleError("refusing to insert UVM while a UVM module is loaded")
    argv = insmod_command(path, parameters)
    if honor_interrupt:
        cells.raise_if_interrupted()
    run_command(argv)
    wait_for("nvidia_uvm insertion", lambda: LOADED_MODULE.is_dir() and LOADED_UVM_BTF.is_file())
    loaded = live_uvm_interface(diagnostic=diagnostic)
    if loaded["interface"] != expected_interface:
        raise LifecycleError("loaded UVM ABI differs from the selected staged module")
    observed_parameters = read_parameters()
    if observed_parameters != parameters:
        raise LifecycleError("loaded UVM parameters differ from the captured initial values")
    return {"argv": argv, "loaded": loaded, "parameters": observed_parameters}


def module_recovery_action(state: RuntimeState, module_present: bool) -> str:
    if not state.destructive_started:
        return "validate_old"
    if not module_present:
        return "insert_old"
    if state.last_insert in {"initial", "restore"}:
        return "validate_old"
    if state.last_insert == "candidate":
        return "remove_candidate"
    raise LifecycleError(f"unclassified live UVM state: {asdict(state)}")


def campaign_complete(completed: list[str], restored: bool,
                      errors: list[dict[str, str]],
                      primary_error: BaseException | None = None) -> bool:
    return (primary_error is None and completed == list(MODES) and restored and
            not errors)


def promote_summary(candidate: Path, final: Path) -> None:
    os.replace(candidate, final)
    directory_fd = os.open(final.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def validate_services_restored(initial: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    current = {unit: service_state(unit) for unit in SERVICES}
    for unit in SERVICES:
        if current[unit]["ActiveState"] != initial[unit]["ActiveState"]:
            raise LifecycleError(f"service active state was not restored: {unit}")
        if current[unit]["SubState"] != initial[unit]["SubState"]:
            raise LifecycleError(f"service substate was not restored: {unit}")
        if current[unit]["UnitFileState"] != initial[unit]["UnitFileState"]:
            raise LifecycleError(f"service enablement changed: {unit}")
        if (initial[unit]["ActiveState"] == "active" and
                current[unit].get("Result") != "success"):
            raise LifecycleError(f"restarted service did not report success: {unit}")
    return current


def restore_runtime(recorder: Recorder, state: RuntimeState, initial: dict[str, Any],
                    restore_path: Path, restore_descriptor: dict[str, Any],
                    initial_parameters: dict[str, str], candidate_path: Path,
                    candidate_descriptor: dict[str, Any], staged_path: Path,
                    staged_descriptor: dict[str, Any] | None) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    blocking_errors: list[dict[str, str]] = []

    def record(name: str, status: str, **details: Any) -> None:
        try:
            recorder.transition(name, status, **details)
        except BaseException as error:
            item = {"stage": f"record_{name}",
                    "error": f"{type(error).__name__}: {error}"}
            errors.append(item)

    def attempt(name: str, operation: Callable[[], Any], *, blocking: bool = True) -> Any:
        try:
            result = operation()
        except BaseException as error:
            item = {"stage": name, "error": f"{type(error).__name__}: {error}"}
            errors.append(item)
            if blocking:
                blocking_errors.append(item)
            record(name, "failed", error=item["error"], runtime_state=asdict(state))
            return None
        record(name, "passed", runtime_state=asdict(state))
        return result

    record("recovery_started", "started", runtime_state=asdict(state))
    restore_artifact = attempt(
        "recovery_validate_restore_artifact",
        lambda: require_unchanged_module(
            restore_path, restore_descriptor, diagnostic=False),
    )
    attempt("recovery_validate_candidate_artifact",
            lambda: require_unchanged_module(
                candidate_path, candidate_descriptor, diagnostic=True),
            blocking=False)
    if staged_descriptor is not None:
        attempt("recovery_validate_staged_artifact",
                lambda: require_unchanged_module(
                    staged_path, staged_descriptor, diagnostic=True),
                blocking=False)
    action = attempt("recovery_classify_uvm", lambda: module_recovery_action(
        state, LOADED_MODULE.is_dir()))
    if action == "remove_candidate":
        removed = attempt("recovery_remove_candidate", lambda: remove_uvm(initial["boot_id"]))
        if removed is not None:
            state.last_insert = "none"
            state.candidate_loaded = False
            record("recovery_candidate_absent", "passed", runtime_state=asdict(state))
            action = "insert_old"
    if action == "insert_old" and not blocking_errors and restore_artifact is not None:
        state.last_insert = "restore"
        record("recovery_insert_old_started", "started",
               path=str(restore_path), runtime_state=asdict(state))
        inserted = attempt(
            "recovery_insert_old",
            lambda: insert_uvm(restore_path, initial_parameters, diagnostic=False,
                               expected_interface=restore_descriptor["interface"],
                               initial_boot=initial["boot_id"]),
        )
        if inserted is not None:
            state.old_restored = True
    elif action == "validate_old":
        def validate_old() -> dict[str, Any]:
            loaded = live_uvm_interface(diagnostic=False)
            if loaded["interface"] != restore_descriptor["interface"]:
                raise LifecycleError("live original/restored UVM ABI differs from old stage")
            if read_parameters() != initial_parameters:
                raise LifecycleError("live original/restored UVM parameters differ")
            return loaded
        if attempt("recovery_validate_old", validate_old) is not None:
            state.old_restored = True

    old_quiet = None
    if state.old_restored and not blocking_errors:
        def validate_old_pre_service() -> dict[str, Any]:
            core = capture_core()
            if core != initial["core"]:
                raise LifecycleError("NVIDIA core changed before service restoration")
            return {"core": core, "quiet": quiet_snapshot(initial["boot_id"]),
                    "holders": no_uvm_holders()}
        old_quiet = attempt(
            "recovery_old_quiet",
            validate_old_pre_service,
        )
    if state.old_restored and not blocking_errors and old_quiet is not None:
        current = attempt("recovery_read_services",
                          lambda: {unit: service_state(unit) for unit in SERVICES})
        if current is not None:
            for unit in service_restore_plan(initial["services"], current):
                attempt(f"recovery_start_{unit}", lambda unit=unit: set_service(unit, "start"))
            if attempt("recovery_validate_services",
                       lambda: validate_services_restored(initial["services"])) is not None:
                state.services_restored = True
    else:
        record("recovery_services", "withheld",
               reason="old UVM module was not restored and validated idle",
               runtime_state=asdict(state))

    def final_validation() -> dict[str, Any]:
        require_boot(initial["boot_id"])
        core = capture_core()
        if core != initial["core"]:
            raise LifecycleError("loaded NVIDIA core module/BTF identity changed")
        require_unchanged_module(candidate_path, candidate_descriptor, diagnostic=True)
        require_unchanged_module(restore_path, restore_descriptor, diagnostic=False)
        if staged_descriptor is None:
            raise LifecycleError("staged candidate never passed its admission checks")
        require_unchanged_module(staged_path, staged_descriptor, diagnostic=True)
        loaded = live_uvm_interface(diagnostic=False)
        if loaded["interface"] != restore_descriptor["interface"]:
            raise LifecycleError("final loaded UVM ABI differs from old stage")
        if read_parameters() != initial_parameters:
            raise LifecycleError("final UVM parameters differ from initial values")
        services = validate_services_restored(initial["services"])
        final_safety = quiet_snapshot(initial["boot_id"])
        safety.validate_post_server_safety(initial["safety"], final_safety)
        holders = no_uvm_holders()
        return {"core": core, "loaded_uvm": loaded, "services": services,
                "safety": final_safety, "holders": holders}

    if state.old_restored and state.services_restored:
        final = attempt("recovery_final_validation", final_validation)
        if final is not None:
            recorder.value["final"] = final
    return errors


def close_lease_and_publish(
        lease: Any, expected_leases: list[dict[str, Any]], recorder: Recorder | None,
        state: RuntimeState, restored: bool, recovery_errors: list[dict[str, str]],
        finalization_errors: list[dict[str, str]], primary_error: BaseException | None,
        output: Path, previous_handlers: dict[signal.Signals, Any]) -> BaseException | None:
    try:
        if lease_inventory() != expected_leases:
            raise LifecycleError("lease inode/identity changed during the campaign")
    except BaseException as error:
        finalization_errors.append({
            "stage": "lease_identity", "error": f"{type(error).__name__}: {error}",
        })
    try:
        if lease is not None:
            lease.close()
    except BaseException as error:
        finalization_errors.append({
            "stage": "lease_close", "error": f"{type(error).__name__}: {error}",
        })
    watched = set(previous_handlers)
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, watched)
    try:
        # The first snapshot rejects signals queued before publication.
        queued = set(cells.INTERRUPTED_SIGNALS)
        pending = {int(value) for value in signal.sigpending() if value in watched}
        prepublish_signals = sorted(queued | pending)
        if prepublish_signals:
            primary_error = primary_error or InterruptedError(
                f"signal {prepublish_signals[0]}")
        if recorder is None:
            return primary_error

        recorder.value["completion_linearization"] = {
            "signals_blocked_ns": time.time_ns(),
            "prepublish_signals": prepublish_signals,
            "original_handlers_restored_before_unblock": True,
        }
        if prepublish_signals:
            recorder.value["interrupt_signals"] = prepublish_signals
        recorder.value["runtime_state"] = asdict(state)
        recorder.value["recovery_errors"] = recovery_errors
        recorder.value["finalization_errors"] = finalization_errors
        recorder.value["restored"] = restored
        eligible = (
            campaign_complete(state.cells_completed, restored, recovery_errors,
                              primary_error) and not finalization_errors
        )
        # The durable pre-commit lifecycle record is always incomplete.
        recorder.value["complete"] = False
        recorder.value["finished_ns"] = time.time_ns()
        try:
            recorder.write()
        except BaseException as error:
            finalization_errors.append({
                "stage": "final_record", "error": f"{type(error).__name__}: {error}",
            })
            eligible = False

        summary = output / "summary.json"
        candidate_summary = output / ".summary.pending.json"
        if eligible:
            try:
                safety.atomic_write_json(candidate_summary, {
                    "complete": True, "modes": list(MODES), "restored": True,
                    "lifecycle": "lifecycle.json",
                })
            except BaseException as error:
                finalization_errors.append({
                    "stage": "summary_record",
                    "error": f"{type(error).__name__}: {error}",
                })
                eligible = False

        # This second snapshot is the completion commit point. Signals that
        # arrived while records were being published reject those records.
        # Later signals remain blocked until the original handlers are back.
        publish_window_pending = {
            int(value) for value in signal.sigpending() if value in watched
        }
        publish_window_signals = sorted(
            set(cells.INTERRUPTED_SIGNALS) | publish_window_pending
        )
        recorder.value["completion_linearization"].update({
            "commit_point_ns": time.time_ns(),
            "publish_window_signals": publish_window_signals,
        })
        if publish_window_signals:
            primary_error = primary_error or InterruptedError(
                f"signal {publish_window_signals[0]}")
            recorder.value["interrupt_signals"] = publish_window_signals
            eligible = False

        if eligible:
            recorder.value["complete"] = True
            recorder.value["finalization_errors"] = finalization_errors
            try:
                recorder.write()
                promote_summary(candidate_summary, summary)
            except BaseException as error:
                finalization_errors.append({
                    "stage": "completion_commit",
                    "error": f"{type(error).__name__}: {error}",
                })
                recorder.value["complete"] = False
                eligible = False

        if not eligible:
            recorder.value["complete"] = False
            try:
                summary.unlink(missing_ok=True)
                candidate_summary.unlink(missing_ok=True)
            except BaseException as error:
                finalization_errors.append({
                    "stage": "summary_reject",
                    "error": f"{type(error).__name__}: {error}",
                })
            recorder.value["finalization_errors"] = finalization_errors
            try:
                recorder.write()
            except BaseException as error:
                finalization_errors.append({
                    "stage": "rejected_record",
                    "error": f"{type(error).__name__}: {error}",
                })
        return primary_error
    finally:
        try:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)


def run_campaign(candidate: Path, restore: Path, stage: Path, output: Path) -> None:
    expected_leases = lease_inventory()
    recorder: Recorder | None = None
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    cells.INTERRUPTED_SIGNALS.clear()
    for sig in previous:
        signal.signal(sig, cells.note_interrupt)
    lease = None
    state = RuntimeState()
    initial: dict[str, Any] | None = None
    staged_descriptor: dict[str, Any] | None = None
    staged_path = stage / MODULE_FILENAME
    primary_error: BaseException | None = None
    recovery_errors: list[dict[str, str]] = []
    finalization_errors: list[dict[str, str]] = []
    restored = False
    try:
        lease = LifecycleLeases()
        cells.raise_if_interrupted()
        validate_paths(candidate, restore, stage, output)
        output.mkdir(parents=False, exist_ok=False)
        recorder = Recorder(output, {
            "candidate": str(candidate), "restore": str(restore),
            "stage": str(stage), "output": str(output),
        })
        acquired_leases = lease_inventory()
        if acquired_leases != expected_leases:
            raise LifecycleError("lease inode/identity changed while it was acquired")
        recorder.value["leases"] = acquired_leases
        recorder.transition("lease_acquired", "passed", runtime_state=asdict(state))
        if os.uname().release != EXPECTED_KERNEL:
            raise LifecycleError(f"kernel differs: {os.uname().release}")
        candidate_descriptor = module_descriptor(candidate, diagnostic=True)
        restore_descriptor = module_descriptor(restore, diagnostic=False)
        if candidate_descriptor["parameter_names"] != restore_descriptor["parameter_names"]:
            raise LifecycleError("candidate and restore module parameter inventories differ")
        services = {unit: service_state(unit) for unit in SERVICES}
        validate_initial_services(services)
        sessions = local_sessions()
        initial_parameters = read_parameters()
        if sorted(initial_parameters) != restore_descriptor["parameter_names"]:
            raise LifecycleError("live and old-stage module parameter inventories differ")
        live_initial = live_uvm_interface(diagnostic=False)
        if live_initial["interface"] != restore_descriptor["interface"]:
            raise LifecycleError("initial live UVM ABI differs from the old stage")
        boot_id = BOOT_ID.read_text().strip()
        core = capture_core(output)
        initial_safety = quiet_snapshot(boot_id)
        safety.validate_pre_server_safety(initial_safety)
        initial = {
            "boot_id": boot_id, "kernel": os.uname().release,
            "core": core, "services": services, "parameters": initial_parameters,
            "local_sessions": sessions,
            "live_uvm": live_initial, "safety": initial_safety,
            "candidate": candidate_descriptor, "restore": restore_descriptor,
        }
        recorder.value["initial"] = initial
        recorder.transition("admission", "passed", runtime_state=asdict(state))

        staged_path = stage_candidate(candidate, stage)
        candidate_after_stage = module_descriptor(candidate, diagnostic=True)
        if candidate_after_stage != candidate_descriptor:
            raise LifecycleError("candidate changed while it was staged")
        staged_descriptor = module_descriptor(staged_path, diagnostic=True)
        if comparable_descriptor(staged_descriptor) != comparable_descriptor(candidate_descriptor):
            raise LifecycleError("staged candidate differs in size, metadata, parameters, or BTF ABI")
        staged_inventory = staged_descriptor["inventory"]
        if (staged_inventory["uid"], staged_inventory["gid"], staged_inventory["mode"]) != (0, 0, 0o644):
            raise LifecycleError("staged candidate ownership or mode differs")
        recorder.value["staged_candidate"] = staged_descriptor
        recorder.transition("candidate_staged", "passed", path=str(staged_path),
                            runtime_state=asdict(state))

        for unit in service_stop_plan(services):
            stop_service_after_recheck(
                unit,
                lambda recheck, unit=unit: recorder.transition(
                    f"stop_{unit}", "started", local_sessions=recheck,
                    runtime_state=asdict(state)),
            )
            recorder.transition(f"stop_{unit}", "passed", runtime_state=asdict(state))
        quiet_snapshot(boot_id)
        recorder.transition("services_stopped_and_idle", "passed",
                            runtime_state=asdict(state))

        state.destructive_started = True
        recorder.transition("remove_old_started", "started", runtime_state=asdict(state))
        cells.raise_if_interrupted()
        removal = remove_uvm(boot_id, initial_safety, honor_interrupt=True)
        state.old_unloaded = True
        state.last_insert = "none"
        recorder.transition("old_removed", "passed", removal=removal,
                            runtime_state=asdict(state))
        cells.raise_if_interrupted()
        state.last_insert = "candidate"
        recorder.transition("insert_candidate_started", "started", path=str(staged_path),
                            runtime_state=asdict(state))
        candidate_load = insert_uvm(
            staged_path, initial_parameters, diagnostic=True,
            expected_interface=staged_descriptor["interface"], initial_boot=boot_id,
            honor_interrupt=True,
        )
        state.candidate_loaded = True
        recorder.value["candidate_load"] = candidate_load
        recorder.transition("candidate_loaded", "passed", runtime_state=asdict(state))

        for mode in MODES:
            cells.raise_if_interrupted()
            recorder.transition(f"cell_{mode}", "started", runtime_state=asdict(state))
            result = cells.run_cell(mode, output / mode)
            if result.get("complete") is not True:
                raise LifecycleError(f"cell did not complete: {mode}")
            state.cells_completed.append(mode)
            recorder.transition(f"cell_{mode}", "passed", runtime_state=asdict(state))
    except BaseException as error:
        primary_error = error
        if recorder is not None:
            recorder.value["primary_error"] = f"{type(error).__name__}: {error}"
            recorder.transition("campaign_body", "failed", error=recorder.value["primary_error"],
                                runtime_state=asdict(state))
    finally:
        try:
            if recorder is not None and initial is not None:
                recovery_errors = restore_runtime(
                    recorder, state, initial, restore, initial["restore"],
                    initial["parameters"], candidate, initial["candidate"],
                    staged_path, staged_descriptor,
                )
                restored = (not recovery_errors and state.old_restored and
                            state.services_restored and "final" in recorder.value)
        except BaseException as error:
            finalization_errors.append({
                "stage": "unhandled_recovery",
                "error": f"{type(error).__name__}: {error}",
            })
        finally:
            try:
                primary_error = close_lease_and_publish(
                    lease, expected_leases, recorder, state, restored,
                    recovery_errors, finalization_errors, primary_error, output,
                    previous,
                )
            except BaseException as error:
                finalization_errors.append({
                    "stage": "unhandled_publish",
                    "error": f"{type(error).__name__}: {error}",
                })
                # The publisher normally restores handlers while signals are
                # blocked. This fallback covers an exception before it entered
                # that protected region.
                for sig, handler in previous.items():
                    signal.signal(sig, handler)

    if primary_error is not None or recovery_errors or finalization_errors or not restored:
        details = []
        if primary_error is not None:
            details.append(f"campaign: {type(primary_error).__name__}: {primary_error}")
        details.extend(f"{item['stage']}: {item['error']}" for item in recovery_errors)
        details.extend(f"{item['stage']}: {item['error']}" for item in finalization_errors)
        if not restored and not recovery_errors and not finalization_errors:
            details.append("restoration was not proven")
        raise LifecycleError("; ".join(details)) from primary_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--restore", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_campaign(*(getattr(args, name).absolute()
                   for name in ("candidate", "restore", "stage", "output")))


if __name__ == "__main__":
    main()
