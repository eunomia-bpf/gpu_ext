#!/usr/bin/env python3
"""Run the endpoint-v1 probe inside one reversible four-module lifecycle.

Without --execute this command is CPU-only: it validates source, module
artifacts, the fixed probe, and path shape.  --execute is root-only, acquires
both shared experiment leases, temporarily removes the two node labels that can
schedule local GPU monitoring, swaps the exact four-module subset, runs the
fixed endpoint probe, and unconditionally attempts the admitted rollback.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
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
from typing import Any

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
GPU_EXT = HERE.parents[4]
WORKSPACE = GPU_EXT.parent
BASE_DIR = GPU_EXT / "extension/revision-init"
BASE_PATH = BASE_DIR / "run_lifecycle.py"

sys.path.insert(0, str(BASE_DIR))
_spec = importlib.util.spec_from_file_location("revision_init_lifecycle", BASE_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load lifecycle primitives: {BASE_PATH}")
base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)

EXPECTED_CANDIDATE_REVISION = "86e7e0dd2e7158d45382a0682682f2894a63c9a4"
EXPECTED_COMMAND = "0x20800408"
ENDPOINT_SYMBOL = "subdeviceCtrlCmdTimerGetGpuCpuTimeCorrelationEndpointsV1_IMPL"
RESTORE_DIR = Path(
    "/opt/gpubpf/modules/575.57.08/gpreempt-849ea75d-6.15.11"
)
STAGE_ROOT = Path("/opt/gpubpf/modules/575.57.08")
RAW_ROOT = HERE.parent / "raw"
PROBE = HERE / "rm_ptimer_correlation_sanity"
PROBE_ARGS = (
    "--samples", "200", "--control-transport", "direct",
    "--correlation-command", "endpoints-v1",
)
RUNNER = HERE.parent / "run_revision_rq4.py"
ANALYZER = HERE.parent / "analyze_revision_rq4.py"
BPFTIME_ROOT = WORKSPACE / "bpftime-table1-575"
BPFTIME_BUILD = BPFTIME_ROOT / "build-launchlate-575"
CHILD_MODES = ("none", "preflight", "preflight-full")
TARGET_LABELS = (
    "fleet.yunwei37.com/gpu",
    "monitoring.yunwei37.com/managed",
)
KUBECTL = (
    "sudo", "-n", "k3s", "kubectl", "--kubeconfig",
    "/var/lib/rancher/k3s/agent/kubelet.kubeconfig",
)
NODE = "lab"


class EndpointLifecycleError(RuntimeError):
    pass


def demand(condition: bool, message: str) -> None:
    if not condition:
        raise EndpointLifecycleError(message)


@dataclass
class State:
    labels_removed: bool = False
    services_stopped: list[str] = field(default_factory=list)
    destructive_started: bool = False
    candidate_loaded: bool = False
    probe_passed: bool = False
    child_passed: bool = False
    restore_loaded: bool = False
    services_restored: bool = False
    labels_restored: bool = False


def git_stdout(repo: Path, *args: str) -> str:
    return base.checked_stdout(["git", "-C", str(repo), *args])


def validate_candidate_source(candidate_dir: Path) -> dict[str, Any]:
    repo = candidate_dir.parent
    demand((repo / ".git").exists(), f"candidate repository is absent: {repo}")
    revision = git_stdout(repo, "rev-parse", "HEAD")
    demand(revision == EXPECTED_CANDIDATE_REVISION,
           f"candidate revision differs: {revision}")
    source_paths = (
        "src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080tmr.h",
        "src/nvidia/generated/g_subdevice_nvoc.c",
        "src/nvidia/generated/g_subdevice_nvoc.h",
        "src/nvidia/src/kernel/gpu/subdevice/subdevice_ctrl_timer_kernel.c",
    )
    for cached in (False, True):
        argv = ["git", "-C", str(repo), "diff", "--quiet"]
        if cached:
            argv.append("--cached")
        argv.extend(["--", *source_paths])
        base.run_command(argv)
    contents = {path: (repo / path).read_text() for path in source_paths}
    demand(EXPECTED_COMMAND in contents[source_paths[0]],
           "endpoint command is absent from the public control header")
    demand(EXPECTED_COMMAND in contents[source_paths[1]] and
           ENDPOINT_SYMBOL in contents[source_paths[1]],
           "endpoint command is absent from the generated export table")
    implementation = contents[source_paths[3]]
    for token in (ENDPOINT_SYMBOL, "osGetPerformanceCounter",
                  "tmrReadTimeLoReg_HAL", "cpuBeforeNs", "cpuAfterNs"):
        demand(token in implementation,
               f"endpoint implementation token is absent: {token}")
    core = candidate_dir / "nvidia.ko"
    symbols = base.checked_stdout(["nm", "-a", str(core)])
    demand(re.search(rf"^[0-9a-fA-F]+\s+[Tt]\s+{ENDPOINT_SYMBOL}$",
                     symbols, re.MULTILINE) is not None,
           "candidate core does not export the endpoint implementation symbol")
    return {"revision": revision, "command": EXPECTED_COMMAND,
            "symbol": ENDPOINT_SYMBOL, "source_paths": list(source_paths)}


def generic_interface(module: Any, raw: str, declarations: str,
                      *, endpoint_core: bool) -> dict[str, Any]:
    missing = [name for name in module.required_btf_names if name not in raw]
    demand(not missing, f"{module.name} BTF lacks required names: {missing}")
    if module.name == "nvidia":
        return {
            "required_names": list(module.required_btf_names),
            "scheduler_base": base.validate_scheduler_base_interface(
                raw, declarations),
            "endpoint_core": endpoint_core,
        }
    if module.name == "nvidia_uvm":
        return base.validate_uvm_interface(raw)
    return {"required_names": list(module.required_btf_names)}


def artifact_descriptor(path: Path, module: Any, *, endpoint_core: bool) -> dict[str, Any]:
    inventory = base.file_inventory(path)
    name = base.checked_stdout(["modinfo", "-F", "name", str(path)])
    version = base.checked_stdout(["modinfo", "-F", "version", str(path)])
    vermagic = base.checked_stdout(["modinfo", "-F", "vermagic", str(path)])
    depends = sorted(item for item in base.checked_stdout(
        ["modinfo", "-F", "depends", str(path)]).split(",") if item)
    parameter_names = sorted({
        line.split(":", 1)[0] for line in base.checked_stdout(
            ["modinfo", "-F", "parm", str(path)]).splitlines() if ":" in line
    })
    demand(name == module.name, f"module name differs for {path}: {name}")
    demand(version == base.EXPECTED_DRIVER,
           f"module version differs for {path}: {version}")
    demand(vermagic == base.EXPECTED_VERMAGIC,
           f"module vermagic differs for {path}: {vermagic}")
    demand(depends == sorted(module.dependencies),
           f"module dependencies differ for {path}: {depends}")
    interface = generic_interface(
        module, base.btf_raw(path), base.btf_c(path),
        endpoint_core=endpoint_core,
    )
    return {"inventory": inventory, "name": name, "version": version,
            "vermagic": vermagic, "depends": depends,
            "parameter_names": parameter_names, "interface": interface,
            "endpoint_core": endpoint_core}


def describe_directory(directory: Path, *, candidate: bool) -> dict[str, dict[str, Any]]:
    return {
        name: artifact_descriptor(
            directory / base.MODULE_BY_NAME[name].filename,
            base.MODULE_BY_NAME[name], endpoint_core=(candidate and name == "nvidia"),
        )
        for name in base.LOAD_ORDER
    }


def comparable(value: dict[str, Any]) -> dict[str, Any]:
    result = {key: value[key] for key in
            ("name", "version", "vermagic", "depends", "parameter_names",
             "interface", "endpoint_core")}
    result["size_bytes"] = value["inventory"]["size_bytes"]
    return result


def validate_probe_output(stdout: str) -> dict[str, Any]:
    """Require the fixed 200-sample endpoint-v1 correctness gate."""
    demand(bool(stdout.strip()), "endpoint probe emitted no JSON records")
    records: list[dict[str, Any]] = []
    for number, line in enumerate(stdout.splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise EndpointLifecycleError(
                f"endpoint probe line {number} is not JSON: {error}") from error
        demand(isinstance(value, dict),
               f"endpoint probe line {number} is not an object")
        records.append(value)
    samples = [value for value in records if value.get("record") == "sample"]
    summaries = [value for value in records if value.get("record") == "summary"]
    demand(len(samples) == 200 and len(summaries) == 1 and len(records) == 201,
           "endpoint probe did not emit exactly 200 samples and one summary")
    demand([value.get("index") for value in samples] == list(range(200)),
           "endpoint probe sample indices are incomplete or out of order")
    for value in samples:
        demand(value.get("control_transport") == "direct" and
               value.get("correlation_command") == "endpoints-v1" and
               value.get("valid") is True and value.get("rm_status") == 0 and
               value.get("cpu_midpoint_regression") is False and
               value.get("ptimer_regression") is False,
               f"endpoint probe sample failed its gate: {value.get('index')}")
    summary = summaries[0]
    expected = {
        "control_transport": "direct", "correlation_command": "endpoints-v1",
        "setup_error": 0, "cleanup_error": 0, "cleanup_rm_status": 0,
        "output_error": 0, "requested": 200, "attempted": 200,
        "accepted": 200, "rejected": 0, "cpu_midpoint_regressions": 0,
        "ptimer_regressions": 0, "gate_pass": True,
    }
    demand(all(summary.get(key) == value for key, value in expected.items()),
           "endpoint probe summary failed the fixed correctness/cleanup gate")
    return {"records": len(records), "samples": len(samples),
            "summary": summary}


def live_descriptor(module: Any, parameters: dict[str, str], *,
                    endpoint_core: bool) -> dict[str, Any]:
    demand(module.sysfs.is_dir(), f"loaded module is absent: {module.name}")
    demand(module.loaded_btf.is_file(), f"loaded BTF is absent: {module.name}")
    version = (module.sysfs / "version").read_text().strip()
    demand(version == base.EXPECTED_DRIVER,
           f"loaded version differs for {module.name}: {version}")
    observed_parameters = base.read_module_parameters(module)
    demand(observed_parameters == parameters,
           f"loaded parameters differ for {module.name}")
    interface = generic_interface(
        module, base.btf_raw(module.loaded_btf), base.btf_c(module.loaded_btf),
        endpoint_core=endpoint_core,
    )
    return {"name": module.name, "version": version,
            "parameters": observed_parameters,
            "holders": base.module_holders(module), "interface": interface}


def capture_runtime(subset: tuple[str, ...], parameters: dict[str, dict[str, str]],
                    *, candidate: bool) -> dict[str, Any]:
    demand(subset == base.LOAD_ORDER, f"four-module subset is required: {subset}")
    demand(base.loaded_subset_unchecked() == subset,
           "loaded module subset differs from the admitted four modules")
    return {
        name: live_descriptor(base.MODULE_BY_NAME[name], parameters[name],
                              endpoint_core=(candidate and name == "nvidia"))
        for name in subset
    }


def insert_directory(directory: Path, subset: tuple[str, ...],
                     parameters: dict[str, dict[str, str]], *,
                     candidate: bool) -> list[dict[str, Any]]:
    demand(not base.loaded_subset_unchecked(), "refusing to insert over modules")
    records = []
    for name in base.LOAD_ORDER:
        module = base.MODULE_BY_NAME[name]
        path = directory / module.filename
        argv = base.insmod_command(path, module, parameters[name])
        base.run_command(argv)
        base.wait_for(f"{name} insertion",
                      lambda module=module:
                      module.sysfs.is_dir() and module.loaded_btf.is_file())
        live = live_descriptor(module, parameters[name],
                               endpoint_core=(candidate and name == "nvidia"))
        records.append({"module": name, "argv": argv, "live": live})
    demand(base.loaded_subset_unchecked() == subset,
           "inserted module subset differs")
    return records


def kubectl_json(*args: str) -> dict[str, Any]:
    return json.loads(base.checked_stdout([*KUBECTL, *args]))


def cluster_snapshot() -> dict[str, Any]:
    node = kubectl_json("get", "node", NODE, "-o", "json")
    labels = node.get("metadata", {}).get("labels", {})
    demand(isinstance(labels, dict), "node labels are absent")
    ready = [item.get("status") for item in node.get("status", {}).get("conditions", [])
             if item.get("type") == "Ready"]
    demand(ready == ["True"], f"node is not uniquely Ready: {ready}")
    pods = kubectl_json("get", "pods", "--all-namespaces",
                        "--field-selector", f"spec.nodeName={NODE}", "-o", "json")
    running_gpu_pods = []
    for item in pods.get("items", []):
        name = item.get("metadata", {}).get("name", "")
        phase = item.get("status", {}).get("phase")
        if phase in {"Pending", "Running"} and re.search(
                r"nvidia.*device.*plugin|dcgm", name, re.IGNORECASE):
            running_gpu_pods.append(name)
    return {"target_labels": {key: labels.get(key) for key in TARGET_LABELS},
            "ready": ready[0], "gpu_pods": sorted(running_gpu_pods)}


def remove_target_labels(initial: dict[str, Any]) -> None:
    for key, value in initial["target_labels"].items():
        if value is not None:
            base.run_command([*KUBECTL, "label", "node", NODE, f"{key}-"])
    base.wait_for("target labels to be absent",
                  lambda: all(value is None for value in
                              cluster_snapshot()["target_labels"].values()))
    base.wait_for("GPU monitoring pods to stop",
                  lambda: not cluster_snapshot()["gpu_pods"])


def restore_target_labels(initial: dict[str, Any]) -> dict[str, Any]:
    for key, value in initial["target_labels"].items():
        argument = f"{key}-" if value is None else f"{key}={value}"
        base.run_command([*KUBECTL, "label", "node", NODE, argument,
                          "--overwrite"])
    final = cluster_snapshot()
    demand(final["target_labels"] == initial["target_labels"],
           "target node labels were not restored exactly")
    return final


def validate_paths(candidate_dir: Path, stage: Path, output: Path) -> None:
    for label, path in (("candidate", candidate_dir), ("stage", stage),
                        ("output", output)):
        demand(path.is_absolute(), f"{label} path must be absolute: {path}")
    demand(candidate_dir.resolve(strict=True) ==
           (WORKSPACE / "gpu_ext-kernel-575/kernel-open").resolve(strict=True),
           "candidate directory is not the admitted endpoint checkout")
    demand(RESTORE_DIR.is_dir() and not RESTORE_DIR.is_symlink(),
           "known-good restore directory is absent or a symlink")
    demand(stage.parent.resolve(strict=True) == STAGE_ROOT.resolve(strict=True) and
           stage.name.startswith("launchlate-endpoint-86e7e0dd-"),
           "stage path is outside the launchlate endpoint namespace")
    demand(output.parent.resolve(strict=True) == RAW_ROOT.resolve(strict=True) and
           output.name.startswith("rm-correlation-575-"),
           "output path is outside the RM-correlation namespace")
    demand(not stage.exists() and not output.exists(),
           "stage and output paths must be fresh")
    demand(PROBE.is_file() and not PROBE.is_symlink() and
           bool(PROBE.stat().st_mode & stat.S_IXUSR),
           f"fixed probe is absent or not executable: {PROBE}")
    for path in (RUNNER, ANALYZER):
        demand(path.is_file() and not path.is_symlink(),
               f"fixed campaign component is absent or a symlink: {path}")
    demand(BPFTIME_ROOT.is_dir() and BPFTIME_BUILD.is_dir(),
           "fixed launchlate bpftime source/build is absent")


def dry_run(candidate_dir: Path, stage: Path, output: Path,
            child_mode: str = "none") -> dict[str, Any]:
    demand(child_mode in CHILD_MODES, f"invalid child mode: {child_mode}")
    validate_paths(candidate_dir, stage, output)
    source = validate_candidate_source(candidate_dir)
    candidate = describe_directory(candidate_dir, candidate=True)
    restore = describe_directory(RESTORE_DIR, candidate=False)
    for name in base.LOAD_ORDER:
        demand(candidate[name]["parameter_names"] == restore[name]["parameter_names"],
               f"candidate/restore parameter inventories differ: {name}")
    return {"complete": True, "mode": "cpu-only-dry-run", "source": source,
            "candidate": candidate, "restore": restore,
            "load_order": list(base.LOAD_ORDER),
            "remove_order": list(base.REMOVE_ORDER),
            "probe": [str(PROBE), *PROBE_ARGS],
            "child_mode": child_mode,
            "stage": str(stage), "output": str(output)}


def child_environment() -> dict[str, str]:
    value = os.environ.copy()
    for key in list(value):
        if (key.startswith(("BPFTIME_", "OBS_", "NVBIT_", "GGML_")) or
                key in {"LD_PRELOAD", "LD_AUDIT", "CUDA_INJECTION64_PATH",
                        "CUDA_INJECTION32_PATH"}):
            value.pop(key)
    value["CUDA_VISIBLE_DEVICES"] = "0"
    value["PYTHONDONTWRITEBYTECODE"] = "1"
    return value


def campaign_argv(phase: str, directory: Path,
                  lease_fds: tuple[int, ...], preflight: Path | None = None
                  ) -> list[str]:
    demand(phase in {"preflight", "full"}, f"invalid campaign phase: {phase}")
    demand(len(lease_fds) == 2, "campaign requires both inherited leases")
    argv = [
        sys.executable, "-B", str(RUNNER), "--phase", phase,
        "--tools", "launchlate", "--output-dir", str(directory),
        "--bpftime-root", str(BPFTIME_ROOT),
        "--bpftime-build-dir", str(BPFTIME_BUILD),
        "--gpu-thread-count", "22528", "--inherited-lease-fds",
        *(str(value) for value in lease_fds),
    ]
    if phase == "full":
        demand(preflight is not None,
               "full campaign requires a passing preflight directory")
        argv.extend(["--preflight-dir", str(preflight)])
    else:
        demand(preflight is None, "preflight cannot name another preflight")
    return argv


def analyze_campaign(directory: Path, phase: str) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(
        "endpoint_lifecycle_revision_rq4_analyzer", ANALYZER)
    demand(spec is not None and spec.loader is not None,
           "cannot load fixed launchlate analyzer")
    analyzer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = analyzer
    spec.loader.exec_module(analyzer)
    result = analyzer.analyze(directory)
    demand(result.get("complete") is True and result.get("phase") == phase and
           list(result.get("tools", ())) == ["launchlate"] and
           list(result.get("configs", ())) == ["baseline", "gpubpf_launchlate",
                                            "nvbit_launchlate"],
           f"independent {phase} analyzer gate failed")
    return result


def run_campaign_child(phase: str, directory: Path,
                       lease_fds: tuple[int, ...],
                       preflight: Path | None = None) -> dict[str, Any]:
    argv = campaign_argv(phase, directory, lease_fds, preflight)
    process = subprocess.Popen(
        argv, cwd=HERE.parent, env=child_environment(), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,
        pass_fds=lease_fds, start_new_session=True,
    )
    interrupted = False
    timed_out = False
    deadline = time.monotonic() + 7200
    stdout = ""
    stderr = ""
    while True:
        try:
            stdout, stderr = process.communicate(timeout=0.2)
            break
        except subprocess.TimeoutExpired:
            pass
        if base.cells.INTERRUPTED_SIGNALS or time.monotonic() >= deadline:
            interrupted = bool(base.cells.INTERRUPTED_SIGNALS)
            timed_out = not interrupted
            os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate(timeout=10)
            break
    result = {"argv": argv, "returncode": process.returncode,
              "stdout": stdout, "stderr": stderr,
              "interrupted": interrupted, "timed_out": timed_out,
              "output": str(directory), "passed": False}
    if interrupted or timed_out or process.returncode != 0:
        result["failure"] = (
            f"fixed {phase} child failed with status {process.returncode}; "
            f"interrupted={interrupted}; timed_out={timed_out}")
        return result
    try:
        result["analysis"] = analyze_campaign(directory, phase)
        result["passed"] = True
    except BaseException as error:
        result["failure"] = f"{type(error).__name__}: {error}"
    return result


def execute(candidate_dir: Path, stage: Path, output: Path,
            child_mode: str = "none") -> None:
    demand(os.geteuid() == 0, "--execute is root-only")
    demand(child_mode in CHILD_MODES, f"invalid child mode: {child_mode}")
    validate_paths(candidate_dir, stage, output)
    state = State()
    leases = base.LifecycleLeases()
    initial: dict[str, Any] | None = None
    record: dict[str, Any] = {
        "complete": False, "state": asdict(state), "events": [],
        "started_ns": time.time_ns(),
        "arguments": {"candidate_dir": str(candidate_dir), "stage": str(stage),
                      "output": str(output), "child_mode": child_mode},
    }
    primary: BaseException | None = None
    recovery_errors: list[str] = []
    eligible = False

    def save() -> None:
        if output.exists():
            record["state"] = asdict(state)
            base.safety.atomic_write_json(output / "lifecycle.json", record)

    def event(name: str, **details: Any) -> None:
        record["events"].append({"name": name, "timestamp_ns": time.time_ns(),
                                 **details})
        save()

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    base.cells.INTERRUPTED_SIGNALS.clear()
    for sig in previous:
        signal.signal(sig, base.cells.note_interrupt)
    try:
        leases.acquire()
        output.mkdir(parents=False, exist_ok=False)
        event("leases_acquired", leases=leases.inventory())
        subset = base.loaded_module_names()
        demand(subset == base.LOAD_ORDER, "exact four-module subset is required")
        source = validate_candidate_source(candidate_dir)
        candidate = describe_directory(candidate_dir, candidate=True)
        restore = describe_directory(RESTORE_DIR, candidate=False)
        services = {unit: base.service_state(unit) for unit in base.SERVICES}
        base.validate_initial_services(services)
        k3s = base.service_state("k3s-agent.service")
        demand(k3s["ActiveState"] == "active" and k3s["SubState"] == "running",
               "k3s-agent is not active/running")
        cluster = cluster_snapshot()
        parameters = {name: base.read_module_parameters(base.MODULE_BY_NAME[name])
                      for name in subset}
        for name in subset:
            demand(candidate[name]["parameter_names"] == restore[name]["parameter_names"],
                   f"candidate/restore parameters differ: {name}")
            base.module_parameter_inventory_matches(
                base.MODULE_BY_NAME[name], restore[name], parameters[name])
        boot_id = base.BOOT_ID.read_text().strip()
        nodes = base.device_nodes()
        runtime = capture_runtime(subset, parameters, candidate=False)
        safety_before = base.quiet_snapshot(boot_id)
        initial = {"boot_id": boot_id, "subset": subset,
                   "parameters": parameters, "services": services,
                   "k3s": k3s, "cluster": cluster, "nodes": nodes,
                   "runtime": runtime, "safety": safety_before,
                   "candidate": candidate, "restore": restore,
                   "source": source}
        record["initial"] = initial
        event("admission_passed")

        base.stage_modules(candidate_dir, stage, subset)
        candidate_after_stage = describe_directory(candidate_dir, candidate=True)
        demand(candidate_after_stage == candidate,
               "candidate source artifacts changed while staging")
        staged = describe_directory(stage, candidate=True)
        for name in subset:
            demand(comparable(staged[name]) == comparable(candidate[name]),
                   f"staged candidate differs: {name}")
            info = staged[name]["inventory"]
            demand((info["uid"], info["gid"], info["mode"]) == (0, 0, 0o644),
                   f"staged ownership/mode differs: {name}")
        record["staged"] = staged
        event("candidate_staged")

        # Mark before the first mutation so partial label removal is recoverable.
        state.labels_removed = True
        remove_target_labels(cluster)
        event("labels_removed", cluster=cluster_snapshot())
        for unit in base.SERVICES:
            if services[unit]["ActiveState"] == "active":
                sessions = base.stop_service_after_session_gate(unit)
                state.services_stopped.append(unit)
                event(f"stopped_{unit}", sessions=sessions)
        holder_gate = base.removal_guard(boot_id, nodes, require_exact_nodes=True)
        time.sleep(0.5)
        state.destructive_started = True
        event("remove_initial_started", holder_gate=holder_gate)
        removed = base.remove_loaded_subset(
            boot_id, nodes, honor_interrupt=True, require_exact_nodes=True)
        event("initial_removed", modules=removed)
        loaded = insert_directory(stage, subset, parameters, candidate=True)
        state.candidate_loaded = True
        candidate_power_limit = base.ensure_power_limit()
        base.wait_device_nodes(nodes)
        candidate_runtime = capture_runtime(subset, parameters, candidate=True)
        candidate_safety = base.quiet_snapshot(boot_id, safety_before)
        candidate_holders = base.no_device_holders(nodes)
        event("candidate_validated", load=loaded, runtime=candidate_runtime,
              safety=candidate_safety, holders=candidate_holders,
              boot_id=boot_id, power_limit_w=candidate_power_limit)

        probe_dir = output / "probe"
        probe_dir.mkdir()
        completed = base.run_command(
            ["taskset", "-c", "8-15", str(PROBE), *PROBE_ARGS], timeout=60.0)
        (probe_dir / "stdout.jsonl").write_text(completed.stdout)
        (probe_dir / "stderr.log").write_text(completed.stderr)
        probe_gate = validate_probe_output(completed.stdout)
        state.probe_passed = True
        event("probe_passed", returncode=completed.returncode,
              command=[str(PROBE), *PROBE_ARGS], gate=probe_gate)

        if child_mode != "none":
            base.cells.raise_if_interrupted()
            lease_fds = tuple(leases.descriptors)
            preflight_dir = output / "launchlate-preflight"
            preflight_result = run_campaign_child(
                "preflight", preflight_dir, lease_fds)
            record["child_preflight"] = preflight_result
            event("child_preflight_finished",
                  returncode=preflight_result["returncode"],
                  passed=preflight_result["passed"], output=str(preflight_dir))
            demand(preflight_result["passed"],
                   preflight_result.get("failure", "preflight child failed"))
            base.cells.raise_if_interrupted()
            event("child_preflight_passed",
                  returncode=preflight_result["returncode"],
                  output=str(preflight_dir))
            if child_mode == "preflight-full":
                full_dir = output / "launchlate-full"
                full_result = run_campaign_child(
                    "full", full_dir, lease_fds, preflight=preflight_dir)
                record["child_full"] = full_result
                event("child_full_finished", returncode=full_result["returncode"],
                      passed=full_result["passed"], output=str(full_dir))
                demand(full_result["passed"],
                       full_result.get("failure", "full child failed"))
                base.cells.raise_if_interrupted()
                event("child_full_passed", returncode=full_result["returncode"],
                      output=str(full_dir))
            state.child_passed = True
        else:
            state.child_passed = True
    except BaseException as error:
        primary = error
        record["primary_error"] = f"{type(error).__name__}: {error}"
        save()
    finally:
        old_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        try:
            if initial is not None and state.destructive_started:
                try:
                    base.stop_active_services()
                    base.remove_loaded_subset(
                        initial["boot_id"], initial["nodes"], honor_interrupt=False,
                        require_exact_nodes=False)
                    insert_directory(RESTORE_DIR, initial["subset"],
                                     initial["parameters"], candidate=False)
                    state.restore_loaded = True
                    restored_power_limit = base.ensure_power_limit()
                    base.wait_device_nodes(initial["nodes"])
                    restored_runtime = capture_runtime(
                        initial["subset"], initial["parameters"], candidate=False)
                    restored_safety = base.quiet_snapshot(
                        initial["boot_id"], initial["safety"])
                    restored_holders = base.no_device_holders(initial["nodes"])
                    event("restore_validated_before_services",
                          runtime=restored_runtime, safety=restored_safety,
                          holders=restored_holders)
                    restored_services = base.restore_services(initial["services"])
                    state.services_restored = True
                    restored_cluster = restore_target_labels(initial["cluster"])
                    state.labels_restored = True
                    demand(base.service_state("k3s-agent.service") == initial["k3s"],
                           "k3s-agent state changed")
                    demand(base.BOOT_ID.read_text().strip() == initial["boot_id"],
                           "boot changed")
                    final = {"runtime": capture_runtime(
                                 initial["subset"], initial["parameters"],
                                 candidate=False),
                             "boot_id": base.BOOT_ID.read_text().strip(),
                             "power_limit_w": restored_power_limit,
                             "services": restored_services,
                             "cluster": restored_cluster,
                             "nodes": base.device_nodes(),
                             "safety": base.quiet_snapshot(
                                 initial["boot_id"], initial["safety"])}
                    demand(final["nodes"] == initial["nodes"],
                           "final device nodes differ")
                    record["final"] = final
                    event("rollback_complete")
                except BaseException as error:
                    recovery_errors.append(f"{type(error).__name__}: {error}")
                    # If exact restoration cannot be proven, contain the node:
                    # remove scheduling labels and stop admitted GPU services.
                    try:
                        remove_target_labels(initial["cluster"])
                    except BaseException as containment_error:
                        recovery_errors.append(
                            "label containment: "
                            f"{type(containment_error).__name__}: {containment_error}")
                    try:
                        base.stop_active_services()
                    except BaseException as containment_error:
                        recovery_errors.append(
                            "service containment: "
                            f"{type(containment_error).__name__}: {containment_error}")
            elif initial is not None:
                try:
                    if state.labels_removed:
                        restore_target_labels(initial["cluster"])
                        state.labels_restored = True
                    base.restore_services(initial["services"])
                    state.services_restored = True
                except BaseException as error:
                    recovery_errors.append(f"{type(error).__name__}: {error}")
            record["recovery_errors"] = recovery_errors
            record["finished_ns"] = time.time_ns()
            pending = sorted({int(value) for value in signal.sigpending()
                              if value in {signal.SIGINT, signal.SIGTERM}} |
                             set(base.cells.INTERRUPTED_SIGNALS))
            record["interrupt_signals"] = pending
            eligible = (primary is None and state.probe_passed and
                        state.child_passed and
                        state.restore_loaded and state.services_restored and
                        state.labels_restored and not recovery_errors and not pending and
                        "final" in record)
            # Never publish complete while either experiment lease is held.
            record["complete"] = False
            record["completion_eligible_before_lease_close"] = eligible
            save()
        finally:
            try:
                leases.close()
            except BaseException as error:
                recovery_errors.append(f"lease close: {type(error).__name__}: {error}")
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        record["recovery_errors"] = recovery_errors
        record["complete"] = bool(eligible and not recovery_errors)
        save()
    if primary is not None or recovery_errors or not record["complete"]:
        messages = [] if primary is None else [f"{type(primary).__name__}: {primary}"]
        messages.extend(recovery_errors)
        if not messages:
            messages.append("lifecycle did not reach complete rollback")
        raise EndpointLifecycleError("; ".join(messages)) from primary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child-mode", choices=CHILD_MODES, default="none")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    values = tuple(getattr(args, name).absolute()
                   for name in ("candidate_dir", "stage", "output"))
    if args.execute:
        execute(*values, child_mode=args.child_mode)
    else:
        print(json.dumps(dry_run(*values, child_mode=args.child_mode),
                         indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
