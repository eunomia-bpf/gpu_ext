#!/usr/bin/env python3
"""Run the scheduler-init matrix inside one reversible NVIDIA-core lifecycle.

This coordinator is root-only.  It stages explicit module files, swaps only the
NVIDIA modules that were present at admission, invokes the matrix in-process
under the already-held lease descriptors, and restores the exact admitted
module subset before publishing a successful result.
"""
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
import run_live as cells  # noqa: E402

safety = cells.safety

EXPECTED_KERNEL = "6.15.11-061511-generic"
EXPECTED_DRIVER = "575.57.08"
EXPECTED_VERMAGIC = (
    "6.15.11-061511-generic SMP preempt mod_unload modversions"
)
STAGE_ROOT = Path("/opt/gpubpf/modules/575.57.08")
KNOWN_RESTORE_DIR = STAGE_ROOT / "gpreempt-849ea75d-6.15.11"
RESULT_ROOT = ROOT / "docs/experiment/revision-safety"
BOOT_ID = Path("/proc/sys/kernel/random/boot_id")
CORE_PARAMETERS = Path("/proc/driver/nvidia/params")
SERVICES = ("gdm.service", "nvidia-persistenced.service")
LEASE_PATHS = cells.LEASE_PATHS
COMMAND_TIMEOUT = 45.0
STATE_TIMEOUT = 45.0
POWER_LIMIT_W = 400.0


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    filename: str
    dependencies: tuple[str, ...]
    required_btf_names: tuple[str, ...]

    @property
    def sysfs(self) -> Path:
        return Path("/sys/module") / self.name

    @property
    def loaded_btf(self) -> Path:
        return Path("/sys/kernel/btf") / self.name


MODULES = (
    ModuleSpec("nvidia", "nvidia.ko", (),
               ("nv_gpu_sched_ops", "bpf_nv_gpu_preempt_tsg")),
    ModuleSpec("nvidia_modeset", "nvidia-modeset.ko", ("nvidia", "video"),
               ("nvKmsKapiGetFunctionsTable", "nvkms_alloc")),
    ModuleSpec("nvidia_drm", "nvidia-drm.ko",
               ("drm_ttm_helper", "nvidia-modeset"),
               ("nv_drm_atomic_commit", "nv_drm_probe_devices")),
    ModuleSpec("nvidia_uvm", "nvidia-uvm.ko", ("nvidia",),
               ("gpu_mem_ops", "uvm_bpf_call_gpu_page_prefetch")),
)
MODULE_BY_NAME = {module.name: module for module in MODULES}
LOAD_ORDER = tuple(module.name for module in MODULES)
REMOVE_ORDER = tuple(reversed(LOAD_ORDER))
REQUIRED_MODULES = frozenset({"nvidia", "nvidia_uvm"})
CORE_STRING_PARAMETERS = frozenset({
    "RegistryDwords", "RegistryDwordsPerDevice", "RmMsg", "GpuBlacklist",
    "TemporaryFilePath", "ExcludedGpus",
})
CORE_PARAMETER_RENAMES = {
    "RmProfilingAdminOnly": "NVreg_RestrictProfilingToAdminUsers",
}


class LifecycleError(RuntimeError):
    pass


def demand(condition: bool, message: str) -> None:
    if not condition:
        raise LifecycleError(message)


class LifecycleLeases:
    """Acquire only the two pre-existing root-owned lease inodes, read-only."""

    def __init__(self, paths: tuple[Path, ...] = LEASE_PATHS, *,
                 expected_owner: tuple[int, int] = (0, 0),
                 expected_mode: int = 0o644):
        self.paths = paths
        self.expected_owner = expected_owner
        self.expected_mode = expected_mode
        self.descriptors: list[int] = []

    def acquire(self) -> None:
        demand(not self.descriptors, "lifecycle leases are already acquired")
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            for path in self.paths:
                before = path.lstat()
                demand(stat.S_ISREG(before.st_mode) and not path.is_symlink(),
                       f"lease is not a regular non-symlink: {path}")
                demand((before.st_uid, before.st_gid) == self.expected_owner and
                       stat.S_IMODE(before.st_mode) == self.expected_mode,
                       f"lease ownership/mode differs: {path}")
                descriptor = os.open(path, flags)
                try:
                    opened = os.fstat(descriptor)
                    demand((opened.st_dev, opened.st_ino) ==
                           (before.st_dev, before.st_ino),
                           f"lease inode changed while opening: {path}")
                    demand(stat.S_ISREG(opened.st_mode),
                           f"lease descriptor is not regular: {path}")
                    demand((fcntl.fcntl(descriptor, fcntl.F_GETFL) &
                            os.O_ACCMODE) == os.O_RDONLY,
                           f"lease descriptor is not read-only: {path}")
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BaseException:
                    os.close(descriptor)
                    raise
                self.descriptors.append(descriptor)
        except BaseException:
            self.close()
            raise

    def inventory(self) -> list[dict[str, int | str]]:
        return cells.validate_inherited_lease_fds(tuple(self.descriptors))

    def close(self) -> None:
        for descriptor in reversed(self.descriptors):
            os.close(descriptor)
        self.descriptors.clear()


@dataclass
class RuntimeState:
    destructive_started: bool = False
    old_removal_complete: bool = False
    candidate_insert_started: bool = False
    services_stopped: list[str] = field(default_factory=list)
    candidate_loaded: list[str] = field(default_factory=list)
    candidate_services_restored: bool = False
    native_preflight_complete: bool = False
    matrix_complete: bool = False
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
            "name": name, "status": status, "timestamp_ns": time.time_ns(),
            **details,
        })
        self.write()

    def write(self) -> None:
        safety.atomic_write_json(self.output / "lifecycle.json", self.value)


def validate_command(argv: list[str]) -> None:
    demand(argv and all(isinstance(value, str) and value for value in argv),
           "command contains an empty/non-string argument")
    lowered = [Path(value).name.lower() for value in argv]
    forbidden = {"modprobe", "depmod", "reboot", "shutdown", "pkill", "killall"}
    demand(not forbidden.intersection(lowered),
           f"forbidden lifecycle command: {argv!r}")
    demand("modules_install" not in argv,
           f"forbidden module-install target: {argv!r}")
    if "rmmod" in lowered:
        demand(not any(value in {"-f", "--force"} for value in argv),
               f"forced removal is forbidden: {argv!r}")


def run_command(argv: list[str], *, timeout: float = COMMAND_TIMEOUT,
                allowed: tuple[int, ...] = (0,),
                env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    validate_command(argv)
    process = subprocess.Popen(argv, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=True,
                               env=env)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.communicate(timeout=2)
        except (ProcessLookupError, subprocess.TimeoutExpired):
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


def wait_for(description: str, predicate: Callable[[], bool],
             timeout: float = STATE_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while True:
        if predicate():
            return
        if time.monotonic() >= deadline:
            raise LifecycleError(f"timed out waiting for {description}")
        time.sleep(0.2)


def file_inventory(path: Path) -> dict[str, Any]:
    demand(path.is_absolute(), f"module path is not absolute: {path}")
    demand(not path.is_symlink(), f"module path is a symlink: {path}")
    resolved = path.resolve(strict=True)
    info = resolved.stat()
    demand(stat.S_ISREG(info.st_mode) and info.st_size > 0,
           f"module path is not a nonempty regular file: {path}")
    return {
        "path": str(resolved), "size_bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns,
        "device": info.st_dev, "inode": info.st_ino,
        "uid": info.st_uid, "gid": info.st_gid,
        "mode": stat.S_IMODE(info.st_mode),
    }


def btf_raw(path: Path) -> str:
    raw = checked_stdout([
        "bpftool", "btf", "dump", "file", str(path), "format", "raw",
    ])
    demand(bool(raw), f"empty BTF dump: {path}")
    return raw


def btf_c(path: Path) -> str:
    value = checked_stdout([
        "bpftool", "btf", "dump", "file", str(path), "format", "c",
    ])
    demand(bool(value), f"empty BTF C dump: {path}")
    return value


def validate_uvm_interface(raw: str) -> dict[str, Any]:
    match = re.search(
        r"STRUCT 'gpu_mem_ops' size=48 vlen=6\n(?P<members>(?:\t[^\n]+\n){6})",
        raw,
    )
    expected = (
        "gpu_test_trigger", "gpu_page_prefetch", "gpu_page_prefetch_iter",
        "gpu_block_activate", "gpu_block_access", "gpu_evict_prepare",
    )
    members = tuple(re.findall(r"\t'([^']+)'", match["members"])) if match else ()
    demand(members == expected, f"gpu_mem_ops ABI differs: {members}")
    required = (
        "bpf_gpu_request_reorder", "bpf_gpu_set_prefetch_region",
        "uvm_bpf_call_gpu_page_prefetch",
    )
    missing = [name for name in required if f"FUNC '{name}'" not in raw]
    demand(not missing, f"UVM BTF lacks required functions: {missing}")
    return {"gpu_mem_ops_members": list(members),
            "required_functions": list(required)}


def validate_scheduler_base_interface(raw: str, c_declarations: str) -> dict[str, Any]:
    structure = re.search(
        r"^\[(\d+)\] STRUCT 'nv_gpu_sched_ops' size=32 vlen=4\n"
        r"((?:\t[^\n]+\n){4})", raw, re.MULTILINE,
    )
    members = (re.findall(r"\t'([^']+)' type_id=\d+ bits_offset=(\d+)",
                          structure.group(2)) if structure else [])
    expected_members = [
        ("on_task_init", "0"), ("on_bind", "64"),
        ("on_task_destroy", "128"), ("on_timeslice_control", "192"),
    ]
    demand(members == expected_members, f"nv_gpu_sched_ops ABI differs: {members}")
    function = re.search(
        r"^\[(\d+)\] FUNC 'bpf_nv_gpu_preempt_tsg' type_id=(\d+) linkage=\w+$",
        raw, re.MULTILINE,
    )
    demand(function is not None, "preemption kfunc is absent")
    function_id, prototype_id = function.groups()
    prototype = re.search(
        rf"^\[{prototype_id}\] FUNC_PROTO '\(anon\)' ret_type_id=(\d+) vlen=2\n"
        rf"\t'hClient' type_id=(\d+)\n\t'hTsg' type_id=(\d+)$",
        raw, re.MULTILINE,
    )
    demand(prototype is not None, "preemption kfunc prototype differs")
    _return_id, client_id, tsg_id = prototype.groups()
    demand(client_id == tsg_id, "preemption kfunc handle argument types differ")
    demand(re.search(
        r"^extern int bpf_nv_gpu_preempt_tsg\("
        r"(?P<handle>u32(?:___[0-9]+)?) hClient, (?P=handle) hTsg\) "
        r"__weak __ksym;$", c_declarations, re.MULTILINE,
    ) is not None, "preemption kfunc C signature differs")
    demand(re.search(
        rf"^\[\d+\] DECL_TAG 'bpf_kfunc' type_id={function_id} component_idx=-1$",
        raw, re.MULTILINE,
    ) is not None, "preemption function lacks the bpf_kfunc declaration tag")
    return {"struct_size": 32, "struct_members": expected_members,
            "preempt_signature": "s32(u32,u32)"}


def validate_module_btf(module: ModuleSpec, raw: str, c_declarations: str, *,
                        candidate: bool) -> dict[str, Any]:
    missing = [name for name in module.required_btf_names if name not in raw]
    demand(not missing, f"{module.name} BTF lacks required names: {missing}")
    if module.name == "nvidia":
        base = validate_scheduler_base_interface(raw, c_declarations)
        init_name = "nv_gpu_sched_init_diagnostic"
        gsp_name = "nv_gpu_sched_gsp_control_complete"
        present = [name for name in (init_name, gsp_name)
                   if f"FUNC '{name}'" in raw]
        if candidate:
            cells.validate_loaded_btf(raw)
            demand(present == [init_name, gsp_name],
                   "candidate core lacks scheduler diagnostics")
        else:
            # The known-good scheduling core already has the GSP completion
            # observation point.  The candidate role is distinguished by the
            # new constructor diagnostic and its exact ABI.
            gsp_id = cells.validate_btf_struct(
                raw, "nv_gpu_gsp_control_complete_ctx", 48,
                cells.BTF_GSP_FIELDS,
            )
            cells.validate_btf_hook(raw, gsp_id, gsp_name)
            demand(init_name not in present,
                   "restore core unexpectedly has scheduler-init diagnostic")
            demand(gsp_name in present,
                   "restore core lacks the admitted GSP completion hook")
        return {"required_names": list(module.required_btf_names),
                "scheduler_base": base, "scheduler_diagnostics": present}
    if module.name == "nvidia_uvm":
        return validate_uvm_interface(raw)
    return {"required_names": list(module.required_btf_names)}


def module_descriptor(path: Path, module: ModuleSpec, *, candidate: bool) -> dict[str, Any]:
    inventory = file_inventory(path)
    name = checked_stdout(["modinfo", "-F", "name", str(path)])
    version = checked_stdout(["modinfo", "-F", "version", str(path)])
    vermagic = checked_stdout(["modinfo", "-F", "vermagic", str(path)])
    depends = sorted(item for item in checked_stdout(
        ["modinfo", "-F", "depends", str(path)]).split(",") if item)
    parameter_names = sorted({
        line.split(":", 1)[0] for line in checked_stdout(
            ["modinfo", "-F", "parm", str(path)]).splitlines() if ":" in line
    })
    demand(name == module.name, f"module name differs for {path}: {name}")
    demand(version == EXPECTED_DRIVER,
           f"module version differs for {path}: {version}")
    demand(vermagic == EXPECTED_VERMAGIC,
           f"module vermagic differs for {path}: {vermagic}")
    demand(depends == sorted(module.dependencies),
           f"module dependencies differ for {path}: {depends}")
    interface = validate_module_btf(
        module, btf_raw(path), btf_c(path), candidate=candidate)
    return {
        "inventory": inventory, "name": name, "version": version,
        "vermagic": vermagic, "depends": depends,
        "parameter_names": parameter_names, "interface": interface,
        "candidate_role": candidate,
    }


def comparable_descriptor(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "size_bytes": value["inventory"]["size_bytes"],
        "name": value["name"], "version": value["version"],
        "vermagic": value["vermagic"], "depends": value["depends"],
        "parameter_names": value["parameter_names"],
        "interface": value["interface"],
        "candidate_role": value["candidate_role"],
    }


def require_unchanged_artifact(path: Path, module: ModuleSpec,
                               expected: dict[str, Any], *,
                               candidate: bool) -> dict[str, Any]:
    observed = module_descriptor(path, module, candidate=candidate)
    demand(observed == expected, f"module artifact changed during lifecycle: {path}")
    return observed


def loaded_module_names() -> tuple[str, ...]:
    rows = Path("/proc/modules").read_text().splitlines()
    loaded = {line.split()[0] for line in rows if line.split()}
    unknown = sorted(name for name in loaded
                     if name.startswith("nvidia") and name not in MODULE_BY_NAME)
    demand(not unknown, f"unmanaged NVIDIA modules are loaded: {unknown}")
    selected = tuple(name for name in LOAD_ORDER if name in loaded)
    demand(REQUIRED_MODULES.issubset(selected),
           f"required NVIDIA modules are absent: {sorted(REQUIRED_MODULES - set(selected))}")
    demand("nvidia_drm" not in selected or "nvidia_modeset" in selected,
           "loaded NVIDIA module subset is not dependency-closed")
    return selected


def loaded_subset_unchecked() -> tuple[str, ...]:
    rows = Path("/proc/modules").read_text().splitlines()
    loaded = {line.split()[0] for line in rows if line.split()}
    unknown = sorted(name for name in loaded
                     if name.startswith("nvidia") and name not in MODULE_BY_NAME)
    demand(not unknown, f"unmanaged NVIDIA modules are loaded: {unknown}")
    return tuple(name for name in LOAD_ORDER if name in loaded)


def module_holders(module: ModuleSpec) -> list[str]:
    directory = module.sysfs / "holders"
    demand(directory.is_dir(), f"module holders directory is absent: {module.name}")
    return sorted(path.name for path in directory.iterdir())


def parse_core_parameters(raw: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, line in enumerate(raw.splitlines(), 1):
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*): (.*)", line)
        demand(match is not None, f"invalid core parameter record at line {number}")
        name, value = match.groups()
        demand(name not in values, f"duplicate core parameter: {name}")
        if name in CORE_STRING_PARAMETERS:
            demand(len(value) >= 2 and value[0] == value[-1] == '"',
                   f"core string parameter is not quoted: {name}")
            value = value[1:-1]
        else:
            demand(re.fullmatch(r"[0-9]+", value) is not None,
                   f"core numeric parameter is not unsigned decimal: {name}")
            numeric = int(value)
            demand(0 <= numeric <= 0xFFFFFFFF,
                   f"core numeric parameter is out of range: {name}")
            if numeric > 0x7FFFFFFF:
                value = str(numeric - 0x100000000)
        values[name] = value
    demand(CORE_STRING_PARAMETERS.issubset(values),
           "core parameter report lacks a required string field")
    demand("RmProfilingAdminOnly" in values,
           "core parameter report lacks profiling state")
    return values


def read_sysfs_parameters(module: ModuleSpec) -> dict[str, str]:
    directory = module.sysfs / "parameters"
    demand(directory.is_dir(), f"parameter directory is absent: {module.name}")
    values: dict[str, str] = {}
    for path in sorted(directory.iterdir()):
        demand(path.is_file() and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", path.name),
               f"unexpected parameter entry: {path}")
        value = path.read_text().strip()
        demand("\x00" not in value and "\n" not in value,
               f"unsafe module parameter value: {module.name}.{path.name}")
        values[path.name] = value
    return values


def read_module_parameters(module: ModuleSpec) -> dict[str, str]:
    if module.name == "nvidia":
        demand(CORE_PARAMETERS.is_file(), "NVIDIA core parameter report is absent")
        return parse_core_parameters(CORE_PARAMETERS.read_text())
    return read_sysfs_parameters(module)


def core_parameter_argument(name: str) -> str:
    return CORE_PARAMETER_RENAMES.get(name, f"NVreg_{name}")


def parameter_arguments(module: ModuleSpec, parameters: dict[str, str]) -> list[str]:
    arguments = []
    for name, value in sorted(parameters.items()):
        demand(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is not None,
               f"invalid module parameter name: {name!r}")
        demand(isinstance(value, str) and "\x00" not in value and "\n" not in value,
               f"invalid module parameter value: {module.name}.{name}")
        if value == "(null)":
            continue
        if module.name == "nvidia" and name in CORE_STRING_PARAMETERS and value == "":
            # The core's proc report renders both its default null char pointer
            # and an empty string as "".  Omitting the empty value preserves the
            # known module default instead of manufacturing a non-null pointer.
            continue
        argument_name = core_parameter_argument(name) if module.name == "nvidia" else name
        arguments.append(f"{argument_name}={value}")
    return arguments


def module_parameter_inventory_matches(module: ModuleSpec, descriptor: dict[str, Any],
                                       values: dict[str, str]) -> None:
    names = set(descriptor["parameter_names"])
    if module.name == "nvidia":
        mapped = {core_parameter_argument(name) for name in values}
        demand(mapped.issubset(names),
               f"core runtime parameters are not accepted by artifact: {sorted(mapped - names)}")
    else:
        demand(set(values) == names,
               f"runtime/artifact parameter inventories differ for {module.name}")


def live_module_descriptor(module: ModuleSpec, *, candidate: bool,
                           parameters: dict[str, str]) -> dict[str, Any]:
    demand(module.sysfs.is_dir(), f"loaded module is absent: {module.name}")
    demand(module.loaded_btf.is_file(), f"loaded module BTF is absent: {module.name}")
    version = (module.sysfs / "version").read_text().strip()
    demand(version == EXPECTED_DRIVER,
           f"loaded module version differs for {module.name}: {version}")
    observed_parameters = read_module_parameters(module)
    demand(observed_parameters == parameters,
           f"loaded parameters differ for {module.name}")
    interface = validate_module_btf(
        module, btf_raw(module.loaded_btf), btf_c(module.loaded_btf),
        candidate=candidate)
    return {
        "name": module.name, "version": version,
        "parameters": observed_parameters, "holders": module_holders(module),
        "btf": file_inventory(module.loaded_btf), "interface": interface,
    }


def capture_runtime(subset: tuple[str, ...], *, candidate: bool,
                    parameters: dict[str, dict[str, str]]) -> dict[str, Any]:
    observed_subset = loaded_subset_unchecked()
    demand(observed_subset == subset,
           f"loaded NVIDIA subset differs: {observed_subset} != {subset}")
    return {
        name: live_module_descriptor(MODULE_BY_NAME[name], candidate=candidate,
                                     parameters=parameters[name])
        for name in subset
    }


def device_nodes() -> list[dict[str, Any]]:
    paths = set(Path("/dev").glob("nvidia*"))
    caps = Path("/dev/nvidia-caps")
    if caps.exists():
        demand(caps.is_dir() and not caps.is_symlink(),
               "NVIDIA capability-node directory is not a real directory")
        paths.update(caps.glob("nvidia*"))
    rows = []
    for path in sorted(paths, key=str):
        demand(not path.is_symlink(), f"NVIDIA device node is a symlink: {path}")
        if path.is_dir():
            continue
        info = path.stat()
        demand(stat.S_ISCHR(info.st_mode), f"NVIDIA path is not a character device: {path}")
        rows.append({
            "path": str(path), "major": os.major(info.st_rdev),
            "minor": os.minor(info.st_rdev), "uid": info.st_uid,
            "gid": info.st_gid, "mode": stat.S_IMODE(info.st_mode),
        })
    demand(rows, "no NVIDIA device nodes were found")
    return rows


def validate_fuser_result(result: subprocess.CompletedProcess[str],
                          paths: list[Path]) -> dict[str, Any]:
    observed = {"paths": [str(path) for path in paths],
                "returncode": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    demand(result.returncode == 1 and not result.stdout and not result.stderr,
           f"device holder check was not unambiguously empty: {observed}")
    return observed


def no_device_holders(expected_nodes: list[dict[str, Any]]) -> dict[str, Any]:
    paths = [Path(row["path"]) for row in expected_nodes if Path(row["path"]).exists()]
    demand(paths, "none of the admitted NVIDIA device nodes exists")
    result = run_command(["sudo", "-n", "fuser", "-v", *map(str, paths)],
                         allowed=(0, 1))
    return validate_fuser_result(result, paths)


def service_state(unit: str, *, allow_transitional: bool = False) -> dict[str, str]:
    raw = checked_stdout([
        "systemctl", "show", unit, "--no-pager", "-p", "LoadState",
        "-p", "ActiveState", "-p", "SubState", "-p", "Result",
        "-p", "UnitFileState",
    ])
    result = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
    demand(result.get("LoadState") == "loaded", f"service is not loaded: {unit}")
    active = result.get("ActiveState")
    if allow_transitional and active in {"activating", "deactivating", "reloading"}:
        return result
    demand(active in {"active", "inactive"},
           f"service state is unsafe/ambiguous: {unit}: {result}")
    return result


def validate_initial_services(values: dict[str, dict[str, str]]) -> None:
    for unit, value in values.items():
        if value["ActiveState"] == "active":
            demand(value["SubState"] == "running" and value.get("Result") == "success",
                   f"active service is not running/successful: {unit}: {value}")


def set_service(unit: str, action: str) -> None:
    demand(action in {"start", "stop"}, f"unsupported service action: {action}")
    target = "active" if action == "start" else "inactive"
    run_command(["sudo", "-n", "systemctl", "--no-block", action, unit])
    wait_for(f"{unit} to become {target}",
             lambda: service_state(unit, allow_transitional=True)["ActiveState"] == target)
    final = service_state(unit)
    if action == "start":
        demand(final["SubState"] == "running" and final.get("Result") == "success",
               f"service did not start cleanly: {unit}: {final}")


def local_sessions() -> list[dict[str, str]]:
    listing = checked_stdout(["loginctl", "list-sessions", "--no-legend", "--no-pager"])
    session_ids = [line.split()[0] for line in listing.splitlines() if line.split()]
    properties = ("Id", "User", "Name", "Seat", "Class", "Type", "Service",
                  "Active", "Remote", "State")
    sessions = []
    for session_id in session_ids:
        demand(re.fullmatch(r"[A-Za-z0-9_.-]+", session_id) is not None,
               f"invalid login session id: {session_id!r}")
        raw = checked_stdout([
            "loginctl", "show-session", session_id, "--no-pager",
            *(part for name in properties for part in ("-p", name)),
        ])
        value = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
        demand(value.get("Id") == session_id, f"login session identity changed: {session_id}")
        demand(all(name in value for name in properties),
               f"login session record is incomplete: {value}")
        demand(value["Remote"] in {"yes", "no"} and value["Active"] in {"yes", "no"},
               f"login session locality/activity is ambiguous: {value}")
        if value["Remote"] == "no":
            demand(bool(value["Seat"]) and value["Class"] == "greeter" and
                   value["Service"] == "gdm-launch-environment",
                   f"local non-greeter session blocks display stop: {value}")
        sessions.append(value)
    return sessions


def stop_service_after_session_gate(unit: str) -> list[dict[str, str]]:
    cells.raise_if_interrupted()
    sessions = local_sessions() if unit == "gdm.service" else []
    cells.raise_if_interrupted()
    set_service(unit, "stop")
    return sessions


def restore_services(initial: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    current = {unit: service_state(unit) for unit in SERVICES}
    for unit in SERVICES:
        if (initial[unit]["ActiveState"] == "inactive" and
                current[unit]["ActiveState"] == "active"):
            if unit == "gdm.service":
                local_sessions()
            set_service(unit, "stop")
    for unit in reversed(SERVICES):
        if (initial[unit]["ActiveState"] == "active" and
                current[unit]["ActiveState"] != "active"):
            set_service(unit, "start")
    final = {unit: service_state(unit) for unit in SERVICES}
    for unit in SERVICES:
        demand(final[unit]["ActiveState"] == initial[unit]["ActiveState"],
               f"service active state was not restored: {unit}")
        demand(final[unit]["SubState"] == initial[unit]["SubState"],
               f"service substate was not restored: {unit}")
        demand(final[unit]["UnitFileState"] == initial[unit]["UnitFileState"],
               f"service enablement changed: {unit}")
        if initial[unit]["ActiveState"] == "active":
            demand(final[unit].get("Result") == "success",
                   f"service restart failed: {unit}")
    return final


def stop_active_services() -> list[str]:
    stopped = []
    for unit in SERVICES:
        if service_state(unit)["ActiveState"] == "active":
            if unit == "gdm.service":
                local_sessions()
            set_service(unit, "stop")
            stopped.append(unit)
    return stopped


def require_boot(boot_id: str) -> None:
    demand(BOOT_ID.read_text().strip() == boot_id,
           "boot changed during NVIDIA module lifecycle")
    demand(os.uname().release == EXPECTED_KERNEL,
           "kernel changed during NVIDIA module lifecycle")


def quiet_snapshot(boot_id: str, baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    require_boot(boot_id)
    snapshot = safety.safety_snapshot()
    demand(snapshot["gpu"]["driver"] == EXPECTED_DRIVER,
           f"driver version differs: {snapshot['gpu']['driver']}")
    demand(abs(float(snapshot["power_limit_w"]) - POWER_LIMIT_W) <= 0.01,
           f"GPU power limit is not {POWER_LIMIT_W} W")
    safety.validate_pre_server_safety(snapshot)
    if baseline is not None:
        safety.validate_post_server_safety(baseline, snapshot)
    return snapshot


def ensure_power_limit() -> float:
    current = float(checked_stdout([
        "nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits",
    ]))
    if abs(current - POWER_LIMIT_W) > 0.01:
        run_command(["sudo", "-n", "nvidia-smi", "-i", "0", "-pl",
                     str(int(POWER_LIMIT_W))])
    observed = float(checked_stdout([
        "nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits",
    ]))
    demand(abs(observed - POWER_LIMIT_W) <= 0.01,
           f"unable to establish {POWER_LIMIT_W} W power limit: {observed}")
    return observed


def leaf_module_names(subset: tuple[str, ...]) -> tuple[str, ...]:
    loaded = set(subset)
    result = []
    for name in subset:
        has_loaded_dependent = any(
            name in {dependency.replace("-", "_")
                     for dependency in MODULE_BY_NAME[other].dependencies}
            for other in loaded if other != name
        )
        if not has_loaded_dependent:
            result.append(name)
    return tuple(result)


def removal_guard(boot_id: str, nodes: list[dict[str, Any]], *,
                  require_exact_nodes: bool) -> dict[str, Any]:
    require_boot(boot_id)
    inventory = safety.struct_ops_inventory()
    demand(not inventory["maps"] and not inventory["links"],
           f"struct_ops state blocks module removal: {inventory}")
    subset = loaded_subset_unchecked()
    if not subset:
        return {"struct_ops": inventory, "gpu": {"core_loaded": False},
                "holders": None, "loaded_subset": []}
    demand("nvidia" in subset, "dependent NVIDIA modules exist without the core")
    gpu = safety.gpu_state()
    demand(not gpu["compute_apps"] and gpu["memory_used_mib"] <= 256 and
           gpu["utilization_gpu_percent"] == 0,
           f"GPU is not idle for module removal: {gpu}")
    observed_nodes = device_nodes()
    if require_exact_nodes:
        demand(observed_nodes == nodes,
               "device-node set changed before destructive removal")
    holders = no_device_holders(
        nodes if require_exact_nodes else [
            row for row in nodes if Path(row["path"]).exists()
        ])
    leaves = []
    for name in leaf_module_names(subset):
        refcount = int((MODULE_BY_NAME[name].sysfs / "refcnt").read_text().strip())
        demand(refcount == 0,
               f"leaf module reference count blocks any removal: {name}={refcount}")
        leaves.append({"module": name, "refcount": refcount})
    return {"struct_ops": inventory, "gpu": gpu, "holders": holders,
            "loaded_subset": list(subset), "leaf_modules": leaves}


def remove_loaded_subset(boot_id: str, nodes: list[dict[str, Any]],
                         *, honor_interrupt: bool,
                         require_exact_nodes: bool = False) -> list[dict[str, Any]]:
    records = []
    guarded = removal_guard(boot_id, nodes,
                            require_exact_nodes=require_exact_nodes)
    if not guarded["loaded_subset"]:
        return records
    for name in REMOVE_ORDER:
        module = MODULE_BY_NAME[name]
        if not module.sysfs.is_dir():
            continue
        if honor_interrupt:
            cells.raise_if_interrupted()
        refcount = int((module.sysfs / "refcnt").read_text().strip())
        demand(refcount == 0, f"module reference count is nonzero: {name}={refcount}")
        run_command(["sudo", "-n", "rmmod", name])
        wait_for(f"{name} removal", lambda module=module: not module.sysfs.exists())
        demand(not module.loaded_btf.exists(), f"loaded BTF survived removal: {name}")
        records.append({"module": name, "refcount_before": refcount})
    demand(not loaded_subset_unchecked(), "NVIDIA managed modules survived removal")
    return records


def insmod_command(path: Path, module: ModuleSpec,
                   parameters: dict[str, str]) -> list[str]:
    return ["sudo", "-n", "insmod", str(path),
            *parameter_arguments(module, parameters)]


def insert_subset(directory: Path, subset: tuple[str, ...],
                  parameters: dict[str, dict[str, str]], *,
                  candidate: bool, honor_interrupt: bool) -> list[dict[str, Any]]:
    demand(not loaded_subset_unchecked(), "refusing to insert over NVIDIA modules")
    records = []
    for name in LOAD_ORDER:
        if name not in subset:
            continue
        module = MODULE_BY_NAME[name]
        path = directory / module.filename
        if honor_interrupt:
            cells.raise_if_interrupted()
        argv = insmod_command(path, module, parameters[name])
        run_command(argv)
        wait_for(f"{name} insertion",
                 lambda module=module: module.sysfs.is_dir() and module.loaded_btf.is_file())
        live = live_module_descriptor(module, candidate=(candidate and name == "nvidia"),
                                      parameters=parameters[name])
        records.append({"module": name, "argv": argv, "live": live})
    demand(loaded_subset_unchecked() == subset,
           "inserted NVIDIA subset differs from admitted subset")
    return records


def restore_partial_old_subset(directory: Path, subset: tuple[str, ...],
                               parameters: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    """Complete an interrupted old-stack removal without replacing survivors."""
    observed = loaded_subset_unchecked()
    demand(set(observed).issubset(subset),
           f"partial old subset has unexpected modules: {observed}")
    for name in observed:
        live_module_descriptor(MODULE_BY_NAME[name], candidate=False,
                               parameters=parameters[name])
    records = []
    for name in LOAD_ORDER:
        if name not in subset or name in observed:
            continue
        module = MODULE_BY_NAME[name]
        path = directory / module.filename
        argv = insmod_command(path, module, parameters[name])
        run_command(argv)
        wait_for(f"{name} restoration",
                 lambda module=module: module.sysfs.is_dir() and module.loaded_btf.is_file())
        live = live_module_descriptor(module, candidate=False,
                                      parameters=parameters[name])
        records.append({"module": name, "argv": argv, "live": live})
    demand(loaded_subset_unchecked() == subset,
           "completed old NVIDIA subset differs from admission")
    return records


def wait_device_nodes(expected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observed: list[dict[str, Any]] = []

    def exact() -> bool:
        nonlocal observed
        try:
            observed = device_nodes()
        except OSError:
            return False
        return observed == expected

    wait_for("exact NVIDIA device-node restoration", exact)
    return observed


def stage_modules(candidate_dir: Path, stage: Path,
                  subset: tuple[str, ...]) -> dict[str, Path]:
    run_command(["sudo", "-n", "mkdir", "--mode=0755", "--", str(stage)])
    result = {}
    for name in subset:
        module = MODULE_BY_NAME[name]
        source = candidate_dir / module.filename
        destination = stage / module.filename
        run_command(["sudo", "-n", "install", "--mode=0644", "--",
                     str(source), str(destination)])
        run_command(["cmp", "--silent", "--", str(source), str(destination)])
        result[name] = destination
    return result


def validate_paths(candidate_dir: Path, restore_dir: Path, stage: Path,
                   output: Path) -> None:
    for label, path in (("candidate", candidate_dir), ("restore", restore_dir),
                        ("stage", stage), ("output", output)):
        demand(path.is_absolute(), f"{label} path must be absolute: {path}")
    demand(candidate_dir.is_dir() and not candidate_dir.is_symlink(),
           "candidate directory is absent or a symlink")
    demand(restore_dir.resolve(strict=True) == KNOWN_RESTORE_DIR.resolve(strict=True),
           "restore directory is not the admitted known-good stage")
    demand(candidate_dir.resolve() != restore_dir.resolve(),
           "candidate and restore directories are identical")
    demand(stage.parent.resolve(strict=True) == STAGE_ROOT.resolve(strict=True) and
           stage.name.startswith("sched-init-candidate-"),
           f"stage is outside the scheduler-init namespace: {stage}")
    demand(output.parent.resolve(strict=True) == RESULT_ROOT.resolve(strict=True) and
           output.name.startswith("sched-init-live-575-"),
           f"output is outside the scheduler-init namespace: {output}")
    demand(not stage.exists() and not output.exists(),
           "stage and output must both be fresh")


def run_native_preflight(output: Path, boot_id: str,
                         baseline: dict[str, Any]) -> dict[str, Any]:
    output.mkdir(parents=False, exist_ok=False)
    before = quiet_snapshot(boot_id, baseline)
    command = ["taskset", "-c", "8-15", str(cells.TARGET)]
    completed = run_command(command, timeout=30, env=cells.command_env())
    (output / "stdout.jsonl").write_text(completed.stdout)
    (output / "stderr.log").write_text(completed.stderr)
    events = []
    for number, line in enumerate(completed.stdout.splitlines(), 1):
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise LifecycleError(f"native preflight JSON line {number} is invalid") from error
        demand(isinstance(value, dict), "native preflight JSON record is not an object")
        events.append(value)
    matches = [event for event in events if event.get("event") == "correctness"]
    demand(matches == [cells.EXPECTED_CORRECTNESS],
           f"native numerical preflight differs: {matches}")
    after = safety.wait_for_post_server_safety(before)
    result = {"complete": True, "command": command, "correctness": matches[0],
              "safety_before": before, "safety_after": after}
    safety.atomic_write_json(output / "result.json", result)
    return result


def describe_artifacts(directory: Path, subset: tuple[str, ...], *,
                       candidate: bool) -> dict[str, dict[str, Any]]:
    return {
        name: module_descriptor(directory / MODULE_BY_NAME[name].filename,
                                MODULE_BY_NAME[name],
                                candidate=(candidate and name == "nvidia"))
        for name in subset
    }


def require_artifacts_unchanged(directory: Path, subset: tuple[str, ...],
                                expected: dict[str, dict[str, Any]], *,
                                candidate: bool) -> None:
    for name in subset:
        require_unchanged_artifact(
            directory / MODULE_BY_NAME[name].filename, MODULE_BY_NAME[name],
            expected[name], candidate=(candidate and name == "nvidia"),
        )


def restoration_complete(state: RuntimeState, errors: list[dict[str, str]]) -> bool:
    untouched = not state.destructive_started and not state.services_stopped
    restored = state.old_restored and state.services_restored
    return (untouched or restored) and not errors


def recover(recorder: Recorder, state: RuntimeState, initial: dict[str, Any],
            candidate_dir: Path, restore_dir: Path, stage: Path,
            candidate_descriptors: dict[str, dict[str, Any]],
            restore_descriptors: dict[str, dict[str, Any]],
            staged_descriptors: dict[str, dict[str, Any]] | None) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    blocking: list[dict[str, str]] = []

    def record(name: str, status: str, **details: Any) -> None:
        try:
            recorder.transition(name, status, **details)
        except BaseException as error:
            errors.append({"stage": f"record_{name}",
                           "error": f"{type(error).__name__}: {error}"})

    def attempt(name: str, operation: Callable[[], Any], *,
                blocks: bool = True) -> Any:
        try:
            value = operation()
        except BaseException as error:
            item = {"stage": name, "error": f"{type(error).__name__}: {error}"}
            errors.append(item)
            if blocks:
                blocking.append(item)
            record(name, "failed", error=item["error"], runtime_state=asdict(state))
            return None
        record(name, "passed", runtime_state=asdict(state))
        return value

    subset = tuple(initial["module_subset"])
    parameters = initial["parameters"]
    boot_id = initial["boot_id"]
    nodes = initial["device_nodes"]
    record("recovery_started", "started", runtime_state=asdict(state))
    attempt("recovery_validate_restore_artifacts", lambda: require_artifacts_unchanged(
        restore_dir, subset, restore_descriptors, candidate=False))
    attempt("recovery_validate_candidate_artifacts", lambda: require_artifacts_unchanged(
        candidate_dir, subset, candidate_descriptors, candidate=True), blocks=False)
    if staged_descriptors is not None:
        attempt("recovery_validate_staged_artifacts", lambda: require_artifacts_unchanged(
            stage, subset, staged_descriptors, candidate=True), blocks=False)

    if state.destructive_started:
        attempt("recovery_stop_services", stop_active_services)
        replace_live = state.old_removal_complete or state.candidate_insert_started
        if not blocking and replace_live:
            attempt("recovery_remove_live_subset", lambda: remove_loaded_subset(
                boot_id, nodes, honor_interrupt=False,
                require_exact_nodes=False))
        if not blocking:
            operation = (
                (lambda: insert_subset(
                    restore_dir, subset, parameters, candidate=False,
                    honor_interrupt=False))
                if replace_live else
                (lambda: restore_partial_old_subset(
                    restore_dir, subset, parameters))
            )
            inserted = attempt("recovery_insert_old_subset", operation)
            if inserted is not None:
                state.old_restored = True
    else:
        validated = attempt("recovery_validate_initial_subset", lambda: capture_runtime(
            subset, candidate=False, parameters=parameters))
        if validated is not None:
            state.old_restored = True

    before_services = None
    if state.old_restored and not blocking:
        def validate_before_services() -> dict[str, Any]:
            ensure_power_limit()
            wait_device_nodes(nodes)
            runtime = capture_runtime(subset, candidate=False, parameters=parameters)
            snapshot = quiet_snapshot(boot_id, initial["safety"])
            no_device_holders(nodes)
            return {"runtime": runtime, "safety": snapshot,
                    "device_nodes": device_nodes()}
        before_services = attempt("recovery_validate_before_services",
                                  validate_before_services)
    if state.old_restored and before_services is not None and not blocking:
        services = attempt("recovery_restore_services",
                           lambda: restore_services(initial["services"]))
        if services is not None:
            state.services_restored = True
    else:
        record("recovery_services", "withheld",
               reason="known-good NVIDIA subset was not validated",
               runtime_state=asdict(state))

    if state.old_restored and state.services_restored and not blocking:
        def final_validation() -> dict[str, Any]:
            require_boot(boot_id)
            require_artifacts_unchanged(restore_dir, subset, restore_descriptors,
                                        candidate=False)
            runtime = capture_runtime(subset, candidate=False, parameters=parameters)
            demand(device_nodes() == nodes, "final device-node set differs")
            snapshot = quiet_snapshot(boot_id, initial["safety"])
            services = restore_services(initial["services"])
            return {"runtime": runtime, "device_nodes": nodes,
                    "safety": snapshot, "services": services}
        final = attempt("recovery_final_validation", final_validation)
        if final is not None:
            recorder.value["final"] = final
    return errors


def publish(recorder: Recorder | None, lease: LifecycleLeases | None,
            expected_leases: list[dict[str, Any]], state: RuntimeState,
            primary_error: BaseException | None,
            recovery_errors: list[dict[str, str]],
            finalization_errors: list[dict[str, str]], output: Path) -> BaseException | None:
    try:
        if lease is not None:
            demand(lease.inventory() == expected_leases,
                   "lease identity changed during lifecycle")
    except BaseException as error:
        finalization_errors.append({"stage": "lease_identity",
                                    "error": f"{type(error).__name__}: {error}"})
    try:
        if lease is not None:
            lease.close()
    except BaseException as error:
        finalization_errors.append({"stage": "lease_close",
                                    "error": f"{type(error).__name__}: {error}"})
    if recorder is None:
        return primary_error
    pending = sorted({int(value) for value in signal.sigpending()
                      if value in {signal.SIGINT, signal.SIGTERM}} |
                     set(cells.INTERRUPTED_SIGNALS))
    if pending and primary_error is None:
        primary_error = InterruptedError(f"signal {pending[0]}")
    recorder.value.update({
        "complete": False, "finished_ns": time.time_ns(),
        "runtime_state": asdict(state), "recovery_errors": recovery_errors,
        "finalization_errors": finalization_errors,
        "interrupt_signals": pending,
        "completion_linearization": {"leases_closed_ns": time.time_ns(),
                                     "prepublish_signals": pending},
    })
    try:
        recorder.write()
    except BaseException as error:
        finalization_errors.append({"stage": "final_record",
                                    "error": f"{type(error).__name__}: {error}"})
    eligible = (
        primary_error is None and state.native_preflight_complete and
        state.matrix_complete and restoration_complete(state, recovery_errors) and
        not finalization_errors and not pending and "final" in recorder.value
    )
    summary = output / "summary.json"
    pending_summary = output / ".summary.pending.json"
    if eligible:
        try:
            safety.atomic_write_json(pending_summary, {
                "complete": True, "passed_cells": len(cells.matrix_plan()),
                "restored": True, "lifecycle": "lifecycle.json",
            })
            commit_signals = sorted(
                {int(value) for value in signal.sigpending()
                 if value in {signal.SIGINT, signal.SIGTERM}} |
                set(cells.INTERRUPTED_SIGNALS)
            )
            recorder.value["completion_linearization"].update({
                "commit_point_ns": time.time_ns(),
                "publish_window_signals": commit_signals,
            })
            if commit_signals:
                primary_error = primary_error or InterruptedError(
                    f"signal {commit_signals[0]}")
                recorder.value["interrupt_signals"] = commit_signals
                eligible = False
        except BaseException as error:
            recorder.value["complete"] = False
            finalization_errors.append({"stage": "summary_record",
                                        "error": f"{type(error).__name__}: {error}"})
            eligible = False
    if eligible:
        try:
            recorder.value["complete"] = True
            recorder.write()
            os.replace(pending_summary, summary)
            directory_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException as error:
            recorder.value["complete"] = False
            finalization_errors.append({"stage": "completion_commit",
                                        "error": f"{type(error).__name__}: {error}"})
    if not recorder.value["complete"]:
        summary.unlink(missing_ok=True)
        pending_summary.unlink(missing_ok=True)
        recorder.value["finalization_errors"] = finalization_errors
        try:
            recorder.write()
        except BaseException:
            pass
    return primary_error


def run_campaign(candidate_dir: Path, restore_dir: Path, stage: Path,
                 output: Path) -> None:
    demand(os.geteuid() == 0, "full-core lifecycle is root-only")
    validate_paths(candidate_dir, restore_dir, stage, output)
    previous_handlers = {sig: signal.getsignal(sig)
                         for sig in (signal.SIGINT, signal.SIGTERM)}
    cells.INTERRUPTED_SIGNALS.clear()
    for sig in previous_handlers:
        signal.signal(sig, cells.note_interrupt)
    lease: LifecycleLeases | None = LifecycleLeases()
    recorder: Recorder | None = None
    state = RuntimeState()
    initial: dict[str, Any] | None = None
    candidate_descriptors: dict[str, dict[str, Any]] = {}
    restore_descriptors: dict[str, dict[str, Any]] = {}
    staged_descriptors: dict[str, dict[str, Any]] | None = None
    expected_leases: list[dict[str, Any]] = []
    primary_error: BaseException | None = None
    recovery_errors: list[dict[str, str]] = []
    finalization_errors: list[dict[str, str]] = []
    blocked_mask: set[signal.Signals] | None = None
    try:
        lease.acquire()
        expected_leases = lease.inventory()
        cells.raise_if_interrupted()
        output.mkdir(parents=False, exist_ok=False)
        recorder = Recorder(output, {
            "candidate_dir": str(candidate_dir), "restore_dir": str(restore_dir),
            "stage": str(stage), "output": str(output),
        })
        recorder.value["leases"] = expected_leases
        recorder.transition("lease_acquired", "passed", runtime_state=asdict(state))
        demand(os.uname().release == EXPECTED_KERNEL,
               f"running kernel differs: {os.uname().release}")
        subset = loaded_module_names()
        candidate_descriptors = describe_artifacts(candidate_dir, subset,
                                                    candidate=True)
        restore_descriptors = describe_artifacts(restore_dir, subset,
                                                  candidate=False)
        for name in subset:
            demand(candidate_descriptors[name]["parameter_names"] ==
                   restore_descriptors[name]["parameter_names"],
                   f"candidate/restore parameter inventories differ: {name}")
        services = {unit: service_state(unit) for unit in SERVICES}
        validate_initial_services(services)
        sessions = local_sessions()
        parameters = {name: read_module_parameters(MODULE_BY_NAME[name])
                      for name in subset}
        for name in subset:
            module_parameter_inventory_matches(
                MODULE_BY_NAME[name], restore_descriptors[name], parameters[name])
        boot_id = BOOT_ID.read_text().strip()
        nodes = device_nodes()
        runtime = capture_runtime(subset, candidate=False, parameters=parameters)
        safety_before = quiet_snapshot(boot_id)
        initial = {
            "boot_id": boot_id, "kernel": os.uname().release,
            "module_subset": list(subset), "parameters": parameters,
            "runtime": runtime, "services": services,
            "local_sessions": sessions, "device_nodes": nodes,
            "safety": safety_before, "candidate": candidate_descriptors,
            "restore": restore_descriptors,
        }
        recorder.value["initial"] = initial
        recorder.transition("admission", "passed", runtime_state=asdict(state))

        stage_modules(candidate_dir, stage, subset)
        require_artifacts_unchanged(candidate_dir, subset, candidate_descriptors,
                                    candidate=True)
        staged_descriptors = describe_artifacts(stage, subset, candidate=True)
        for name in subset:
            demand(comparable_descriptor(staged_descriptors[name]) ==
                   comparable_descriptor(candidate_descriptors[name]),
                   f"staged module differs from candidate: {name}")
            inventory = staged_descriptors[name]["inventory"]
            demand((inventory["uid"], inventory["gid"], inventory["mode"]) ==
                   (0, 0, 0o644), f"staged ownership/mode differs: {name}")
        recorder.value["staged_candidate"] = staged_descriptors
        recorder.transition("candidate_staged", "passed", runtime_state=asdict(state))

        for unit in SERVICES:
            if services[unit]["ActiveState"] == "active":
                gate_sessions = stop_service_after_session_gate(unit)
                state.services_stopped.append(unit)
                recorder.transition(f"stop_{unit}", "passed",
                                    local_sessions=gate_sessions,
                                    runtime_state=asdict(state))
        first_removal_gate = removal_guard(
            boot_id, nodes, require_exact_nodes=True)
        time.sleep(0.5)
        state.destructive_started = True
        recorder.transition("remove_old_started", "started",
                            first_removal_gate=first_removal_gate,
                            runtime_state=asdict(state))
        removed = remove_loaded_subset(
            boot_id, nodes, honor_interrupt=True, require_exact_nodes=True)
        state.old_removal_complete = True
        recorder.transition("old_subset_removed", "passed", modules=removed,
                            runtime_state=asdict(state))
        state.candidate_insert_started = True
        inserted = insert_subset(stage, subset, parameters, candidate=True,
                                 honor_interrupt=True)
        state.candidate_loaded = [row["module"] for row in inserted]
        ensure_power_limit()
        wait_device_nodes(nodes)
        candidate_runtime = capture_runtime(subset, candidate=True,
                                            parameters=parameters)
        candidate_safety = quiet_snapshot(boot_id, safety_before)
        no_device_holders(nodes)
        recorder.value["candidate"] = {
            "load": inserted, "runtime": candidate_runtime,
            "device_nodes": device_nodes(), "safety": candidate_safety,
        }
        recorder.transition("candidate_validated", "passed", runtime_state=asdict(state))
        restore_services(services)
        state.candidate_services_restored = True
        quiet_snapshot(boot_id, safety_before)
        recorder.transition("candidate_services_restored", "passed",
                            runtime_state=asdict(state))

        native = run_native_preflight(output / "native-preflight", boot_id,
                                      safety_before)
        state.native_preflight_complete = native["complete"] is True
        recorder.transition("native_preflight", "passed", runtime_state=asdict(state))
        matrix = cells.run_matrix(
            output / "matrix",
            inherited_lease_fds=tuple(lease.descriptors),
        )
        demand(matrix.get("complete") is True and
               matrix.get("passed_cells") == len(cells.matrix_plan()),
               "scheduler-init matrix did not complete every cell")
        demand(matrix.get("lease_mode") == "validated_inherited",
               "scheduler-init matrix did not use inherited lifecycle leases")
        state.matrix_complete = True
        recorder.transition("matrix", "passed", passed_cells=matrix["passed_cells"],
                            runtime_state=asdict(state))
    except BaseException as error:
        primary_error = error
        if recorder is not None:
            recorder.value["primary_error"] = f"{type(error).__name__}: {error}"
            try:
                recorder.transition("campaign_body", "failed",
                                    error=recorder.value["primary_error"],
                                    runtime_state=asdict(state))
            except BaseException as record_error:
                finalization_errors.append({
                    "stage": "record_campaign_failure",
                    "error": f"{type(record_error).__name__}: {record_error}",
                })
    finally:
        blocked_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        try:
            if recorder is not None and initial is not None:
                recovery_errors = recover(
                    recorder, state, initial, candidate_dir, restore_dir, stage,
                    candidate_descriptors, restore_descriptors, staged_descriptors,
                )
        except BaseException as error:
            finalization_errors.append({
                "stage": "unhandled_recovery",
                "error": f"{type(error).__name__}: {error}",
            })
        finally:
            try:
                primary_error = publish(
                    recorder, lease, expected_leases, state, primary_error,
                    recovery_errors, finalization_errors, output,
                )
            except BaseException as error:
                finalization_errors.append({
                    "stage": "unhandled_publish",
                    "error": f"{type(error).__name__}: {error}",
                })
                if lease is not None:
                    try:
                        lease.close()
                    except BaseException:
                        pass
            finally:
                for sig, handler in previous_handlers.items():
                    signal.signal(sig, handler)
                if blocked_mask is not None:
                    signal.pthread_sigmask(signal.SIG_SETMASK, blocked_mask)

    restored = restoration_complete(state, recovery_errors)
    if primary_error is not None or not restored or finalization_errors:
        details = []
        if primary_error is not None:
            details.append(f"campaign: {type(primary_error).__name__}: {primary_error}")
        details.extend(f"{item['stage']}: {item['error']}" for item in recovery_errors)
        details.extend(f"{item['stage']}: {item['error']}" for item in finalization_errors)
        if not restored and not recovery_errors:
            details.append("restoration was not proven")
        raise LifecycleError("; ".join(details)) from primary_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--restore-dir", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_campaign(*(getattr(args, name).absolute() for name in
                   ("candidate_dir", "restore_dir", "stage", "output")))


if __name__ == "__main__":
    main()
