#!/usr/bin/env python3
"""Fail-closed smoke runner for the two safe-ABI example policies.

``admit`` is read-only. ``run`` is the only action that attaches a policy or
launches CUDA, and it always uses the repository's exclusive GPU/struct-ops
leases. Runtime cleanup is limited to process groups created by this runner.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
GPU_EXT = HERE.parents[1]
EXTENSION = GPU_EXT / "extension"
MOE_HARNESS_DIR = GPU_EXT / "workloads/moe-infinity"
if str(MOE_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(MOE_HARNESS_DIR))

import run_moe_head_to_head as shared  # noqa: E402


GateError = shared.GateError
WORKLOAD = HERE / "uvm_fault_stream"
WORKLOAD_GIB = 8
WORKLOAD_REGION_KIB = 64
WORKLOAD_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class PolicySpec:
    loader: Path
    bpf_object: Path
    loader_source: Path
    bpf_source: Path


POLICIES = {
    "prefetch_delta_markov": PolicySpec(
        loader=EXTENSION / "prefetch_delta_markov",
        bpf_object=EXTENSION / ".output/prefetch_delta_markov.bpf.o",
        loader_source=EXTENSION / "prefetch_delta_markov.c",
        bpf_source=EXTENSION / "prefetch_delta_markov.bpf.c",
    ),
    "eviction_2q_approx": PolicySpec(
        loader=EXTENSION / "eviction_2q_approx",
        bpf_object=EXTENSION / ".output/eviction_2q_approx.bpf.o",
        loader_source=EXTENSION / "eviction_2q_approx.c",
        bpf_source=EXTENSION / "eviction_2q_approx.bpf.c",
    ),
}


def policy_command(policy: str) -> list[str]:
    spec = POLICIES.get(policy)
    if spec is None:
        raise GateError(f"unsupported safe policy: {policy}")
    if policy == "prefetch_delta_markov":
        return [
            "sudo",
            "-n",
            str(spec.loader),
            "-c",
            "2",
            "-n",
            "2",
            "-m",
            "128",
            "-i",
            "1",
        ]
    return [
        "sudo",
        "-n",
        str(spec.loader),
        "-p",
        "2",
        "-g",
        "2",
        "-i",
        "1",
    ]


def workload_command(output: Path) -> list[str]:
    return [
        str(WORKLOAD),
        "--gib",
        str(WORKLOAD_GIB),
        "--region-kib",
        str(WORKLOAD_REGION_KIB),
        "--output",
        str(output),
    ]


def parse_json_events(text: str) -> list[dict[str, Any]]:
    events = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("event"), str):
            events.append(value)
    return events


def latest_event(text: str, event: str) -> dict[str, Any]:
    for value in reversed(parse_json_events(text)):
        if value["event"] == event:
            return value
    raise GateError(f"loader output has no {event} event")


def _integer(value: Any, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GateError(f"{field} must be an integer >= {minimum}")
    return value


def validate_ready(policy: str, ready: dict[str, Any]) -> dict[str, int]:
    if ready.get("event") != "ready":
        raise GateError("loader did not emit a ready event")
    result = {
        "pid": _integer(ready.get("pid"), "pid", 1),
        "struct_map_id": _integer(
            ready.get("struct_map_id"), "struct_map_id", 1
        ),
        "struct_link_id": _integer(
            ready.get("struct_link_id"), "struct_link_id", 1
        ),
    }
    if policy == "prefetch_delta_markov":
        result["kprobe_link_id"] = _integer(
            ready.get("kprobe_link_id"), "kprobe_link_id", 1
        )
        expected = {
            "confidence": 2,
            "prefetch_pages": 2,
            "maximum_delta": 128,
            "metrics_interval_seconds": 1,
        }
    elif policy == "eviction_2q_approx":
        expected = {
            "promote_after": 2,
            "maximum_generation_gap": 2,
            "metrics_interval_seconds": 1,
        }
    else:
        raise GateError(f"unsupported safe policy: {policy}")
    for field, expected_value in expected.items():
        if _integer(ready.get(field), field, 1) != expected_value:
            raise GateError(
                f"loader ready configuration differs for {field}: "
                f"expected {expected_value}, found {ready.get(field)!r}"
            )
    return result


def _counter_set(event: dict[str, Any], fields: tuple[str, ...]) -> dict[str, int]:
    return {field: _integer(event.get(field), field) for field in fields}


def validate_policy_metrics(policy: str, event: dict[str, Any]) -> dict[str, int]:
    if event.get("event") != "final_metrics":
        raise GateError("loader did not emit final_metrics")
    if policy == "eviction_2q_approx":
        metrics = _counter_set(
            event,
            (
                "activate_events",
                "access_events",
                "admissions",
                "identity_resets",
                "generation_resets",
                "same_episode_events",
                "probation_head_requests",
                "promotions",
                "protected_tail_requests",
                "reorder_errors",
                "eviction_prepares",
            ),
        )
        for field in (
            "activate_events",
            "admissions",
            "probation_head_requests",
        ):
            if metrics[field] == 0:
                raise GateError(f"2Q policy did not engage: {field} is zero")
        if metrics["reorder_errors"] != 0:
            raise GateError("2Q policy reported reorder request errors")
        return metrics
    if policy == "prefetch_delta_markov":
        metrics = _counter_set(
            event,
            (
                "context_captures",
                "callbacks",
                "blocks_initialized",
                "deltas_observed",
                "invalid_deltas",
                "transitions_created",
                "transition_matches",
                "transition_decays",
                "transition_replacements",
                "confident_predictions",
                "prefetch_requests",
                "empty_requests",
                "map_errors",
                "request_errors",
            ),
        )
        for field in (
            "context_captures",
            "callbacks",
            "blocks_initialized",
            "deltas_observed",
            "transitions_created",
            "confident_predictions",
            "prefetch_requests",
        ):
            if metrics[field] == 0:
                raise GateError(f"Markov policy did not engage: {field} is zero")
        if metrics["map_errors"] != 0 or metrics["request_errors"] != 0:
            raise GateError("Markov policy reported map or request errors")
        return metrics
    raise GateError(f"unsupported safe policy: {policy}")


def validate_workload_result(result: dict[str, Any]) -> dict[str, Any]:
    expected_bytes = WORKLOAD_GIB * 1024**3
    expected_region_bytes = WORKLOAD_REGION_KIB * 1024
    expected_regions = expected_bytes // expected_region_bytes
    for field, expected in (
        ("bytes", expected_bytes),
        ("region_bytes", expected_region_bytes),
        ("regions", expected_regions),
        ("mismatches", 0),
    ):
        if _integer(result.get(field), field) != expected:
            raise GateError(
                f"workload {field} differs: expected {expected}, "
                f"found {result.get(field)!r}"
            )
    if result.get("first_mismatch") is not None:
        raise GateError("zero-mismatch workload has a first_mismatch value")
    kernel_ms = result.get("kernel_ms")
    if (
        isinstance(kernel_ms, bool)
        or not isinstance(kernel_ms, (int, float))
        or not math.isfinite(float(kernel_ms))
        or float(kernel_ms) <= 0
    ):
        raise GateError("workload kernel_ms must be finite and positive")
    return {
        "bytes": expected_bytes,
        "region_bytes": expected_region_bytes,
        "regions": expected_regions,
        "kernel_ms": float(kernel_ms),
        "mismatches": 0,
        "first_mismatch": None,
    }


def validate_kprobe_link_ownership(
    ready: dict[str, int], links: list[dict[str, Any]]
) -> dict[str, Any]:
    link_id = ready.get("kprobe_link_id")
    if link_id is None:
        raise GateError("Markov ready event has no kprobe link ID")
    matches = [
        item
        for item in links
        if isinstance(item, dict) and item.get("id") == link_id
    ]
    if len(matches) != 1:
        raise GateError("owned Markov kprobe link is not uniquely enumerable")
    item = matches[0]
    owners = {
        int(owner["pid"])
        for owner in item.get("pids", ())
        if isinstance(owner, dict) and "pid" in owner
    }
    if owners and owners != {ready["pid"]}:
        raise GateError("Markov kprobe link PID ownership differs from ready PID")
    if link_id == ready["struct_link_id"]:
        raise GateError("Markov kprobe and struct_ops links share an ID")
    return {
        "kprobe_link_id": link_id,
        "owner_pid_enumerated": bool(owners),
    }


def _file_metric(path: Path, executable: bool = False) -> dict[str, Any]:
    if not path.is_file():
        raise GateError(f"required file is missing: {path}")
    if executable and not os.access(path, os.X_OK):
        raise GateError(f"required executable is not executable: {path}")
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size}


def runtime_files(policy: str) -> list[dict[str, Any]]:
    spec = POLICIES.get(policy)
    if spec is None:
        raise GateError(f"unsupported safe policy: {policy}")
    return [
        _file_metric(Path(__file__).resolve()),
        _file_metric(WORKLOAD, executable=True),
        _file_metric(HERE / "uvm_fault_stream.cu"),
        _file_metric(spec.loader, executable=True),
        _file_metric(spec.bpf_object),
        _file_metric(spec.loader_source),
        _file_metric(spec.bpf_source),
        _file_metric(EXTENSION / "safe_policy_models.h"),
    ]


def safety_metrics(snapshot: dict[str, Any]) -> dict[str, Any]:
    gpu = snapshot["gpu"]
    return {
        "power_limit_service_active": snapshot["power_limit_service"] == "active",
        "power_limit_w": float(snapshot["power_limit_w"]),
        "gpu_memory_used_mib": int(gpu["memory_used_mib"]),
        "gpu_utilization_percent": int(gpu["utilization_gpu_percent"]),
        "gpu_compute_processes": len(gpu["compute_apps"]),
        "uvm_refcount": int(snapshot["uvm_refcount"]),
        "struct_ops_maps": len(snapshot["struct_ops"]["maps"]),
        "struct_ops_links": len(snapshot["struct_ops"]["links"]),
        "dmesg_abnormal_records": len(snapshot["dmesg_abnormal"]),
        "journal_abnormal_records": len(snapshot["journal_abnormal"]),
        "xid_records": len(snapshot["xids"]),
    }


def validate_active_policy_safety(
    before: dict[str, Any], active: dict[str, Any], ready: dict[str, int]
) -> None:
    if active["power_limit_service"] != "active":
        raise GateError("power-limit service changed after policy attach")
    if abs(float(active["power_limit_w"]) - 400.0) > 0.01:
        raise GateError("power limit changed after policy attach")
    for field in ("dmesg_abnormal", "journal_abnormal", "xids"):
        if active[field] != before[field]:
            raise GateError(f"kernel safety history changed after attach: {field}")
    gpu = active["gpu"]
    if (
        gpu["compute_apps"]
        or gpu["memory_used_mib"] > 256
        or gpu["utilization_gpu_percent"] != 0
    ):
        raise GateError("GPU stopped being idle after policy attach")
    try:
        shared.validate_policy_ownership(ready, active["struct_ops"])
    except Exception as exc:
        raise GateError("active struct_ops ownership validation failed") from exc


def verify_policy_interface(
    policy: str, expected_driver: str | None = None
) -> dict[str, Any]:
    interface = shared.verify_loaded_uvm_interface(expected_driver)
    observation_hook_verified = False
    if policy == "prefetch_delta_markov":
        raw = shared.run_checked(
            [
                "sudo",
                "-n",
                "bpftool",
                "btf",
                "dump",
                "file",
                str(shared.LOADED_UVM_BTF),
                "format",
                "raw",
            ]
        )
        if "FUNC 'uvm_perf_prefetch_get_hint_va_block'" not in raw:
            raise GateError("loaded UVM BTF lacks the Markov observation hook")
        observation_hook_verified = True
    return {
        "btf_path": str(shared.LOADED_UVM_BTF.resolve()),
        "driver_revision": interface["version"],
        "gpu_mem_ops_members": len(interface["gpu_mem_ops_members"]),
        "required_policy_kfuncs": len(interface["required_kfuncs"]),
        "markov_observation_hook_verified": observation_hook_verified,
    }


def admission(
    policy: str | None, expected_driver: str | None = None
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    selected = tuple(POLICIES) if policy is None else (policy,)
    evidence: dict[str, Any] = {
        "policies": list(selected),
        "errors": [],
    }
    errors: list[str] = evidence["errors"]
    snapshot = None
    try:
        evidence["git_revision"] = shared.run_checked(
            ["git", "rev-parse", "HEAD"], cwd=GPU_EXT
        )
    except Exception as exc:
        errors.append(f"Git revision: {exc}")
    try:
        files: dict[str, dict[str, Any]] = {}
        for selected_policy in selected:
            for metric in runtime_files(selected_policy):
                files[metric["path"]] = metric
        evidence["files"] = [files[path] for path in sorted(files)]
    except Exception as exc:
        errors.append(f"runtime files: {exc}")
    try:
        interface_policy = (
            "prefetch_delta_markov"
            if "prefetch_delta_markov" in selected
            else selected[0]
        )
        evidence["interface"] = verify_policy_interface(interface_policy, expected_driver)
    except Exception as exc:
        errors.append(f"loaded safe ABI: {exc}")
    try:
        snapshot = shared.safety_snapshot()
        evidence["safety_metrics"] = safety_metrics(snapshot)
        shared.validate_pre_server_safety(snapshot)
    except Exception as exc:
        errors.append(f"pre-run safety gate failed: {type(exc).__name__}")
    evidence["admitted"] = not errors
    return evidence, snapshot


def _wait_loader_event(
    process: subprocess.Popen[Any], path: Path, event: str, timeout: float
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = path.read_text(errors="replace") if path.exists() else ""
        try:
            return latest_event(text, event)
        except GateError:
            pass
        if process.poll() is not None:
            raise GateError(
                f"policy loader exited {process.returncode} before {event}"
            )
        time.sleep(0.1)
    raise GateError(f"policy loader exceeded readiness timeout for {event}")


def _all_bpf_links() -> list[dict[str, Any]]:
    value = json.loads(
        shared.run_checked(["sudo", "-n", "bpftool", "link", "show", "-j"])
        or "[]"
    )
    if not isinstance(value, list):
        raise GateError("bpftool link inventory is not a list")
    return value


def _start_policy(
    policy: str, output: Path
) -> tuple[
    subprocess.Popen[Any], Any, Path, dict[str, int], dict[str, Any]
]:
    log_path = output / "policy.jsonl"
    log = log_path.open("x", buffering=1)
    try:
        process = subprocess.Popen(
            policy_command(policy),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except BaseException:
        log.close()
        raise
    try:
        ready_event = _wait_loader_event(process, log_path, "ready", 30)
        ready = validate_ready(policy, ready_event)
        owned_pids = set(shared.descendants(process.pid))
        if ready["pid"] not in owned_pids:
            raise GateError("ready PID is outside the owned loader process tree")
        if os.getpgid(ready["pid"]) != process.pid:
            raise GateError("ready PID is outside the owned loader process group")
        try:
            struct_ownership = shared.validate_policy_ownership(
                ready, shared.struct_ops_inventory()
            )
        except Exception as exc:
            raise GateError("struct_ops ownership validation failed") from exc
        ownership: dict[str, Any] = {
            "loader_pid": ready["pid"],
            "struct_map_id": int(struct_ownership["struct_map_id"]),
            "struct_link_id": ready["struct_link_id"],
            "struct_link_enumerated": bool(
                struct_ownership["link_enumerated"]
            ),
        }
        if policy == "prefetch_delta_markov":
            ownership.update(validate_kprobe_link_ownership(ready, _all_bpf_links()))
        return process, log, log_path, ready, ownership
    except BaseException:
        shared.stop_owned_process_group(process, timeout=60)
        log.close()
        raise


def _run_workload(output: Path) -> tuple[dict[str, Any], Path]:
    result_path = output / "workload-result.json"
    log_path = output / "workload.log"
    with log_path.open("x", buffering=1) as log:
        process = subprocess.Popen(
            workload_command(result_path),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            process.wait(timeout=WORKLOAD_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as exc:
            shared.stop_owned_process_group(process, timeout=60)
            raise GateError(
                f"owned workload exceeded the hard {WORKLOAD_TIMEOUT_SECONDS}-second timeout"
            ) from exc
        except BaseException:
            shared.stop_owned_process_group(process, timeout=60)
            raise
        if process.returncode != 0:
            raise GateError(f"owned workload exited {process.returncode}")
    if not result_path.is_file():
        raise GateError("workload did not create its result JSON")
    try:
        value = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError("workload result JSON is not parseable") from exc
    if not isinstance(value, dict):
        raise GateError("workload result JSON is not an object")
    return validate_workload_result(value), log_path


def run_smoke(
    policy: str, output: Path, expected_driver: str | None = None
) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise GateError(f"refusing to overwrite existing output path: {output}")
    lease = shared.LeaseSet.acquire()
    before = None
    policy_process = None
    policy_log = None
    policy_log_path = None
    ownership = None
    workload_metrics = None
    workload_log_path = None
    after = None
    errors: list[str] = []
    try:
        evidence, before = admission(policy, expected_driver)
        if not evidence["admitted"] or before is None:
            raise GateError("run admission failed: " + "; ".join(evidence["errors"]))
        output.mkdir(parents=True, exist_ok=False)
        try:
            (
                policy_process,
                policy_log,
                policy_log_path,
                ready,
                ownership,
            ) = _start_policy(policy, output)
            active = shared.safety_snapshot()
            validate_active_policy_safety(before, active, ready)
            if policy_process.poll() is not None:
                raise GateError("policy loader exited before workload launch")
            workload_metrics, workload_log_path = _run_workload(output)
            if policy_process.poll() is not None:
                raise GateError("policy loader exited while the workload ran")
        except BaseException as exc:
            errors.append(str(exc))
        finally:
            if policy_process is not None:
                try:
                    shared.stop_owned_process_group(policy_process, timeout=60)
                except Exception as exc:
                    errors.append(f"owned policy cleanup: {exc}")
            if policy_log is not None:
                policy_log.close()
        try:
            after = shared.wait_for_post_server_safety(before, timeout=60)
        except Exception as exc:
            errors.append(f"post-run safety gate failed: {type(exc).__name__}: {exc}")

        policy_metrics = None
        if policy_log_path is not None:
            try:
                final_event = latest_event(
                    policy_log_path.read_text(errors="replace"), "final_metrics"
                )
                policy_metrics = validate_policy_metrics(policy, final_event)
            except Exception as exc:
                errors.append(f"policy engagement: {exc}")
        if errors:
            try:
                failure_snapshot = shared.safety_snapshot()
            except Exception as exc:
                failure_snapshot = {"snapshot_error": str(exc)}
            shared.atomic_write_json(output / "failure.json", {
                "policy": policy,
                "errors": errors,
                "admission": evidence,
                "ownership": ownership,
                "workload_metrics": workload_metrics,
                "policy_metrics": policy_metrics,
                "post_failure_snapshot": failure_snapshot,
            })
            raise GateError("; ".join(errors))
        if (
            ownership is None
            or workload_metrics is None
            or workload_log_path is None
            or policy_metrics is None
            or after is None
            or policy_log_path is None
        ):
            raise GateError("smoke run ended without complete evidence")

        generated_files = [
            _file_metric(policy_log_path),
            _file_metric(workload_log_path),
            _file_metric(output / "workload-result.json"),
        ]
        result = {
            "schema": 1,
            "policy": policy,
            "git_revision": evidence["git_revision"],
            "files": evidence["files"] + generated_files,
            "interface": evidence["interface"],
            "ownership": ownership,
            "workload_metrics": workload_metrics,
            "policy_metrics": policy_metrics,
            "safety_metrics": {
                "before": safety_metrics(before),
                "after": safety_metrics(after),
            },
        }
        result_path = output / "run-result.json"
        shared.atomic_write_json(result_path, result)
        return result
    finally:
        lease.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed safe-ABI policy smoke; admit never launches work"
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    admit = subparsers.add_parser("admit", help="read-only safety admission")
    admit.add_argument(
        "--policy",
        choices=tuple(POLICIES),
        help="optionally admit only one policy; the default checks both",
    )
    run = subparsers.add_parser("run", help="attach one policy and run one smoke")
    run.add_argument("--policy", choices=tuple(POLICIES), required=True)
    run.add_argument("--output", type=Path, required=True)
    for command in (admit, run):
        command.add_argument(
            "--expected-driver",
            choices=("575.57.08", "610.43.02"),
            help="require this exact live driver version; default preserves the original experiment",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "admit":
            result, _ = admission(args.policy, args.expected_driver)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["admitted"] else 2
        result = run_smoke(args.policy, args.output, args.expected_driver)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except GateError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
