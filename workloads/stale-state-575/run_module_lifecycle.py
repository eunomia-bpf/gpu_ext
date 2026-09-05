#!/usr/bin/env python3
"""Load the stale-state UVM candidate, run one preflight, and restore UVM."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any, Sequence

import live_runner
import protocol


HERE = Path(__file__).resolve().parent
GPU_EXT = HERE.parents[1]
BASE_PATH = GPU_EXT / "extension/revision-prefetch/run_lifecycle.py"
BASE_SPEC = importlib.util.spec_from_file_location("stale_state_lifecycle_base", BASE_PATH)
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise RuntimeError("cannot load reviewed UVM lifecycle base")
base = importlib.util.module_from_spec(BASE_SPEC)
sys.modules[BASE_SPEC.name] = base
BASE_SPEC.loader.exec_module(base)

EXPECTED_KERNEL = "6.15.11-061511-generic"
STAGE_ROOT = Path("/opt/gpubpf/modules/575.57.08")
RAW_ROOT = HERE / "raw"
MODULE_NAME = "nvidia_uvm"
MODULE_FILENAME = "nvidia-uvm.ko"


class LifecycleError(RuntimeError):
    pass


def demand(condition: bool, message: str) -> None:
    if not condition:
        raise LifecycleError(message)


@dataclass
class State:
    services_stopped: list[str] = field(default_factory=list)
    old_removed: bool = False
    candidate_loaded: bool = False
    old_restored: bool = False
    services_restored: bool = False
    child_complete: bool = False


def exact_stale_interface(raw: str) -> dict[str, Any]:
    match = re.search(
        r"STRUCT 'gpu_mem_ops' size=56 vlen=7\n(?P<members>(?:\t[^\n]+\n){7})",
        raw,
    )
    expected_members = (
        "gpu_test_trigger", "gpu_page_prefetch", "gpu_page_prefetch_iter",
        "gpu_block_activate", "gpu_block_access", "gpu_evict_prepare",
        "gpu_stale_state_prefetch_v1",
    )
    members = tuple(re.findall(r"\t'([^']+)'", match["members"])) if match else ()
    demand(members == expected_members, f"candidate gpu_mem_ops ABI differs: {members}")
    diagnostic_match = re.search(
        r"STRUCT 'uvm_stale_state_v1_diagnostic' size=176 vlen=17\n"
        r"(?P<members>(?:\t[^\n]+\n){17})",
        raw,
    )
    expected_diagnostic_members = (
        "input", "callback_return", "decision_age_ns", "requested_first",
        "requested_outer", "output_first", "output_outer", "diagnostic_phase",
        "mode", "status", "action", "action_attempted", "action_conflict",
        "action_request_calls", "region_result", "initial_effect", "owner_tgid",
    )
    diagnostic_members = tuple(re.findall(
        r"\t'([^']+)'", diagnostic_match["members"]
    )) if diagnostic_match else ()
    demand(diagnostic_members == expected_diagnostic_members,
           f"candidate diagnostic ABI differs: {diagnostic_members}")
    required = (
        "FUNC 'uvm_stale_state_v1_diagnostic'",
        "FUNC 'bpf_gpu_stale_state_v1_request'",
        "STRUCT 'uvm_stale_state_v1_input' size=88",
        "STRUCT 'uvm_stale_state_v1_diagnostic' size=176",
    )
    missing = [item for item in required if item not in raw]
    demand(not missing, f"candidate stale-state BTF lacks: {missing}")
    return {"gpu_mem_ops_members": list(members),
            "diagnostic_members": list(diagnostic_members),
            "required": list(required)}


def exact_restore_interface(raw: str) -> dict[str, Any]:
    """Describe either the admitted gpubpf base ABI or an unmodified stock ABI."""
    if "gpu_mem_ops" not in raw:
        return {"kind": "stock_no_gpu_mem_ops", "gpu_mem_ops_present": False}
    try:
        interface = base.generic_uvm_interface(raw)
    except (base.LifecycleError, RuntimeError) as exc:
        raise LifecycleError(f"restore gpu_mem_ops ABI differs: {exc}") from exc
    return {"kind": "gpubpf_base_v1", "gpu_mem_ops_present": True,
            **interface}


def restore_descriptor(path: Path) -> dict[str, Any]:
    inventory = base.file_inventory(path)
    name = base.checked_stdout(["modinfo", "-F", "name", str(path)])
    version = base.checked_stdout(["modinfo", "-F", "version", str(path)])
    vermagic = base.checked_stdout(["modinfo", "-F", "vermagic", str(path)])
    depends = sorted(item for item in base.checked_stdout(
        ["modinfo", "-F", "depends", str(path)]
    ).split(",") if item)
    parameters = sorted({line.split(":", 1)[0] for line in base.checked_stdout(
        ["modinfo", "-F", "parm", str(path)]
    ).splitlines() if ":" in line})
    raw = base.btf_raw(path)
    interface = exact_restore_interface(raw)
    demand(name == MODULE_NAME and version == protocol.EXPECTED_DRIVER,
           "restore module identity differs")
    demand(vermagic == base.EXPECTED_VERMAGIC, "restore vermagic differs")
    demand(depends == ["nvidia"], "restore dependency differs")
    demand(bool(parameters) and "uvm_perf_prefetch_enable" in parameters,
           "restore parameter inventory lacks prefetch control")
    forbidden = (
        "FUNC 'uvm_stale_state_v1_diagnostic'",
        "FUNC 'uvm_bpf_prefetch_diagnostic'",
    )
    demand(not any(item in raw for item in forbidden),
           "restore unexpectedly contains an experiment diagnostic")
    return {"inventory": inventory, "name": name, "version": version,
            "vermagic": vermagic, "depends": depends,
            "parameter_names": parameters, "interface": interface,
            "diagnostic_present": False}


def candidate_descriptor(path: Path) -> dict[str, Any]:
    inventory = base.file_inventory(path)
    name = base.checked_stdout(["modinfo", "-F", "name", str(path)])
    version = base.checked_stdout(["modinfo", "-F", "version", str(path)])
    vermagic = base.checked_stdout(["modinfo", "-F", "vermagic", str(path)])
    depends = sorted(item for item in base.checked_stdout(
        ["modinfo", "-F", "depends", str(path)]
    ).split(",") if item)
    parameters = sorted({line.split(":", 1)[0] for line in base.checked_stdout(
        ["modinfo", "-F", "parm", str(path)]
    ).splitlines() if ":" in line})
    interface = exact_stale_interface(base.btf_raw(path))
    demand(name == MODULE_NAME and version == protocol.EXPECTED_DRIVER,
           "candidate module identity differs")
    demand(vermagic == base.EXPECTED_VERMAGIC, "candidate vermagic differs")
    demand(depends == ["nvidia"], "candidate dependency differs")
    demand("uvm_perf_prefetch_enable" in parameters,
           "candidate parameter inventory lacks prefetch control")
    return {"inventory": inventory, "name": name, "version": version,
            "vermagic": vermagic, "depends": depends,
            "parameter_names": parameters, "interface": interface}


def comparable(value: dict[str, Any]) -> dict[str, Any]:
    return {"size_bytes": value["inventory"]["size_bytes"],
            "name": value["name"], "version": value["version"],
            "vermagic": value["vermagic"], "depends": value["depends"],
            "parameter_names": value["parameter_names"],
            "interface": value["interface"]}


def validate_paths(candidate: Path, restore: Path, stage: Path, output: Path,
                   record: Path) -> None:
    for label, path in (("candidate", candidate), ("restore", restore), ("stage", stage),
                        ("output", output), ("record", record)):
        demand(path.is_absolute(), f"{label} path must be absolute")
    demand(candidate.name == MODULE_FILENAME and candidate.is_file() and
           not candidate.is_symlink(), "candidate must be a regular nvidia-uvm.ko")
    demand(restore.name == MODULE_FILENAME and restore.is_file() and
           not restore.is_symlink(), "restore must be a regular nvidia-uvm.ko")
    demand(stage.parent.resolve(strict=True) == STAGE_ROOT.resolve(strict=True) and
           stage.name.startswith("stale-state-v1-stage-"), "stage namespace differs")
    demand(output.parent.resolve(strict=True) == RAW_ROOT.resolve(strict=True) and
           output.name.startswith("stale-state-575-preflight-"),
           "preflight output namespace differs")
    demand(record.parent.resolve(strict=True) == RAW_ROOT.resolve(strict=True) and
           record.name.startswith("stale-state-575-lifecycle-"),
           "lifecycle record namespace differs")
    demand(len({candidate.resolve(), restore.resolve(), stage.resolve(),
                output.resolve(), record.resolve()}) == 5, "lifecycle paths overlap")
    demand(not stage.exists() and not output.exists() and not record.exists(),
           "stage, preflight output, and lifecycle record must be fresh")


def loaded_candidate() -> dict[str, Any]:
    demand(base.LOADED_MODULE.is_dir() and base.LOADED_UVM_BTF.is_file(),
           "candidate UVM module/BTF is absent")
    demand((base.LOADED_MODULE / "version").read_text().strip() ==
           protocol.EXPECTED_DRIVER, "loaded candidate version differs")
    interface = exact_stale_interface(base.btf_raw(base.LOADED_UVM_BTF))
    demand(base.read_parameters(), "loaded candidate parameters are empty")
    demand(live_runner.coordinator.PROC_PATH.is_file(), "loaded proc bridge is absent")
    return {"version": protocol.EXPECTED_DRIVER, "interface": interface,
            "parameters": base.read_parameters()}


def loaded_restore() -> dict[str, Any]:
    demand(base.LOADED_MODULE.is_dir() and base.LOADED_UVM_BTF.is_file(),
           "restore UVM module/BTF is absent")
    version = (base.LOADED_MODULE / "version").read_text().strip()
    demand(version == protocol.EXPECTED_DRIVER, "loaded restore version differs")
    raw = base.btf_raw(base.LOADED_UVM_BTF)
    forbidden = (
        "FUNC 'uvm_stale_state_v1_diagnostic'",
        "FUNC 'uvm_bpf_prefetch_diagnostic'",
    )
    demand(not any(item in raw for item in forbidden),
           "loaded restore contains an experiment diagnostic")
    return {"version": version, "interface": exact_restore_interface(raw),
            "parameters": base.read_parameters()}


def demand_restore_matches_live(restore: dict[str, Any], live: dict[str, Any],
                                parameters: dict[str, str]) -> None:
    demand(live["version"] == restore["version"],
           "initial live UVM version differs from explicit restore")
    demand(live["interface"] == restore["interface"],
           "initial live UVM differs from explicit restore ABI")
    demand(sorted(parameters) == restore["parameter_names"],
           "live/restore parameter inventories differ")
    demand(live["parameters"] == parameters,
           "live UVM parameters changed during admission")


def require_restore_unchanged(path: Path,
                              expected: dict[str, Any]) -> dict[str, Any]:
    observed = restore_descriptor(path)
    demand(observed == expected,
           f"restore artifact changed during lifecycle: {path}")
    return observed


def insert_candidate(path: Path, parameters: dict[str, str],
                     expected_interface: dict[str, Any], boot_id: str) -> dict[str, Any]:
    base.require_boot(boot_id)
    demand(not base.LOADED_MODULE.exists(), "refusing candidate insertion over UVM")
    argv = base.insmod_command(path, parameters)
    base.run_command(argv)
    base.wait_for("stale-state UVM insertion",
                  lambda: base.LOADED_MODULE.is_dir() and base.LOADED_UVM_BTF.is_file())
    loaded = loaded_candidate()
    demand(loaded["interface"] == expected_interface,
           "loaded candidate interface differs from staged module")
    demand(loaded["parameters"] == parameters, "candidate parameters differ")
    return {"argv": argv, "loaded": loaded}


def insert_restore(path: Path, parameters: dict[str, str],
                   expected_interface: dict[str, Any], boot_id: str) -> dict[str, Any]:
    base.require_boot(boot_id)
    demand(not base.LOADED_MODULE.exists(), "refusing restore insertion over UVM")
    argv = base.insmod_command(path, parameters)
    base.run_command(argv)
    base.wait_for("restore UVM insertion",
                  lambda: base.LOADED_MODULE.is_dir() and base.LOADED_UVM_BTF.is_file())
    loaded = loaded_restore()
    demand(loaded["interface"] == expected_interface,
           "loaded restore interface differs from explicit module")
    demand(loaded["parameters"] == parameters, "restored UVM parameters differ")
    return {"argv": argv, "loaded": loaded}


def stage_candidate(candidate: Path, stage: Path) -> Path:
    base.run_command(["sudo", "-n", "mkdir", "--mode=0755", "--", str(stage)])
    destination = stage / MODULE_FILENAME
    base.run_command(["sudo", "-n", "install", "--mode=0644", "--",
                      str(candidate), str(destination)])
    base.run_command(["cmp", "--silent", "--", str(candidate), str(destination)])
    return destination


def child_command(output: Path, lease_fds: Sequence[int]) -> list[str]:
    demand(len(lease_fds) == 2, "child requires two inherited leases")
    return [sys.executable, "-B", str(HERE / "live_runner.py"),
            "execute-preflight", "--output", str(output),
            "--inherited-lease-fds", *(str(fd) for fd in lease_fds)]


def dry_run(candidate: Path, restore: Path, stage: Path, output: Path,
            record: Path) -> dict[str, Any]:
    validate_paths(candidate, restore, stage, output, record)
    candidate_value = candidate_descriptor(candidate)
    restore_value = restore_descriptor(restore)
    demand(candidate_value["parameter_names"] == restore_value["parameter_names"],
           "candidate/restore parameter inventories differ")
    return {"complete": True, "mode": "cpu-source-dry-run",
            "loads_modules": False, "executes_gpu": False,
            "candidate": candidate_value, "restore": restore_value,
            "stage": str(stage), "output": str(output), "record": str(record),
            "child": child_command(output, ("<lease-0>", "<lease-1>")),
            "recovery": "remove only the loaded candidate; restore admitted UVM and services"}


def execute(candidate: Path, restore: Path, stage: Path, output: Path,
            record: Path) -> None:
    demand(os.geteuid() == 0, "module lifecycle is root-only")
    validate_paths(candidate, restore, stage, output, record)
    state = State()
    record.mkdir(parents=False, exist_ok=False)
    lifecycle: dict[str, Any] = {"complete": False, "state": asdict(state),
                                 "events": [], "started_ns": time.time_ns()}

    def save() -> None:
        lifecycle["state"] = asdict(state)
        live_runner.atomic_json(record / "lifecycle.json", lifecycle)

    def event(name: str, status: str = "passed", **details: Any) -> None:
        lifecycle["events"].append({"name": name, "status": status,
                                    "timestamp_ns": time.time_ns(), **details})
        save()

    lease: Any | None = None
    initial: dict[str, Any] | None = None
    primary: BaseException | None = None
    recovery_errors: list[str] = []
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    base.cells.INTERRUPTED_SIGNALS.clear()
    for sig in previous:
        signal.signal(sig, base.cells.note_interrupt)
    try:
        lease = base.LifecycleLeases()
        event("leases_acquired", leases=base.lease_inventory())
        demand(os.uname().release == EXPECTED_KERNEL, "kernel release differs")
        candidate_value = candidate_descriptor(candidate)
        restore_value = restore_descriptor(restore)
        demand(candidate_value["parameter_names"] == restore_value["parameter_names"],
               "candidate/restore parameter inventories differ")
        services = {unit: base.service_state(unit) for unit in base.SERVICES}
        base.validate_initial_services(services)
        sessions = base.local_sessions()
        parameters = base.read_parameters()
        live_initial = loaded_restore()
        demand_restore_matches_live(restore_value, live_initial, parameters)
        boot_id = base.BOOT_ID.read_text().strip()
        initial_safety = base.quiet_snapshot(boot_id)
        initial = {"boot_id": boot_id, "services": services,
                   "sessions": sessions, "parameters": parameters,
                   "candidate": candidate_value, "restore": restore_value,
                   "live_uvm": live_initial, "safety": initial_safety}
        lifecycle["initial"] = initial
        event("admission")

        staged_path = stage_candidate(candidate, stage)
        staged = candidate_descriptor(staged_path)
        demand(comparable(staged) == comparable(candidate_value),
               "staged candidate differs")
        staged_inventory = staged["inventory"]
        demand((staged_inventory["uid"], staged_inventory["gid"],
                staged_inventory["mode"]) == (0, 0, 0o644),
               "staged candidate ownership/mode differs")
        event("candidate_staged", path=str(staged_path), descriptor=staged)
        require_restore_unchanged(restore, restore_value)
        event("restore_revalidated", path=str(restore))

        for unit in base.service_stop_plan(services):
            base.stop_service_after_recheck(
                unit, lambda recheck, unit=unit: event(
                    f"stop_{unit}_admission", local_sessions=recheck
                )
            )
            state.services_stopped.append(unit)
            event(f"stop_{unit}")
        base.quiet_snapshot(boot_id)
        base.remove_uvm(boot_id, initial_safety, honor_interrupt=False)
        state.old_removed = True
        event("old_uvm_removed")
        state.candidate_loaded = True
        loaded = insert_candidate(staged_path, parameters, staged["interface"], boot_id)
        event("candidate_loaded", loaded=loaded)

        child_stdout = (record / "child.stdout.log").open("x")
        child_stderr = (record / "child.stderr.log").open("x")
        try:
            command = child_command(output, lease.fds)
            child = subprocess.run(command, stdout=child_stdout, stderr=child_stderr,
                                   text=True, check=False, timeout=1800,
                                   pass_fds=tuple(lease.fds))
        finally:
            child_stdout.close()
            child_stderr.close()
        demand(child.returncode == 0, f"preflight child exited {child.returncode}")
        base.cells.raise_if_interrupted()
        validated = protocol.validate_preflight(output)
        state.child_complete = True
        event("preflight_complete", command=command, returncode=child.returncode,
              validation=validated)
    except BaseException as exc:
        primary = exc
        lifecycle["primary_error"] = f"{type(exc).__name__}: {exc}"
        event("lifecycle_body", "failed", error=lifecycle["primary_error"])
    finally:
        if initial is not None:
            try:
                if state.candidate_loaded and base.LOADED_MODULE.exists():
                    base.remove_uvm(initial["boot_id"], honor_interrupt=False)
                    state.candidate_loaded = False
                    event("candidate_removed")
                if not base.LOADED_MODULE.exists():
                    require_restore_unchanged(restore, initial["restore"])
                    insert_restore(restore, initial["parameters"],
                                   initial["restore"]["interface"],
                                   initial["boot_id"])
                    state.old_restored = True
                    event("old_uvm_restored")
                else:
                    live = loaded_restore()
                    demand_restore_matches_live(
                        initial["restore"], live, initial["parameters"])
                    state.old_restored = True
                require_restore_unchanged(restore, initial["restore"])
                current = {unit: base.service_state(unit) for unit in base.SERVICES}
                for unit in base.service_restore_plan(initial["services"], current):
                    base.set_service(unit, "start")
                base.validate_services_restored(initial["services"])
                state.services_restored = True
                final_safety = base.quiet_snapshot(initial["boot_id"])
                base.safety.validate_post_server_safety(initial["safety"], final_safety)
                lifecycle["final_safety"] = final_safety
                event("recovery_complete")
            except BaseException as exc:
                recovery_errors.append(f"{type(exc).__name__}: {exc}")
        if lease is not None:
            lease.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        lifecycle["recovery_errors"] = recovery_errors
        lifecycle["complete"] = (
            primary is None and not recovery_errors and state.child_complete and
            state.old_restored and state.services_restored
        )
        lifecycle["finished_ns"] = time.time_ns()
        save()
    if primary is not None or recovery_errors or not lifecycle["complete"]:
        raise LifecycleError(
            "; ".join(([f"body: {primary}"] if primary else []) + recovery_errors)
        ) from primary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("dry-run", "execute"))
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--restore", required=True, type=Path)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--record", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "dry-run":
            print(json.dumps(dry_run(args.candidate, args.restore, args.stage,
                                     args.output, args.record), indent=2,
                             sort_keys=True))
        else:
            execute(args.candidate, args.restore, args.stage, args.output,
                    args.record)
        return 0
    except (LifecycleError, base.LifecycleError, live_runner.LiveError,
            OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
