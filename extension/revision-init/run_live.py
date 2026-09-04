#!/usr/bin/env python3
"""Run the frozen scheduler-init functional matrix on an already loaded 575 candidate.

This cell runner never loads, unloads, or replaces a kernel module. A reviewed
full-core lifecycle coordinator must load the candidate and restore the known
driver around this runner.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import fcntl
import importlib.util
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
from typing import Any

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXTENSION = ROOT / "extension"
TARGET = ROOT / "workloads/bpftime-device-smoke/.output/vector"
OBSERVER = EXTENSION / "revision_init_trace"
LOADER = EXTENSION / "revision_init_loader"
OBJECT_DIR = EXTENSION / ".output"
COMPUTE_MONITOR = EXTENSION / "revision-prefetch/monitor_compute_apps.py"
COMPUTE_MAX_GAP_NS = 1_000_000_000
EXPECTED_KERNEL = "6.15.11-061511-generic"
EXPECTED_DRIVER = "575.57.08"
EXPECTED_CORRECTNESS = {
    "event": "correctness", "launches": 8,
    "checked_values": 32768, "mismatches": 0,
}
LEASE_PATHS = (
    Path("/tmp/gpubpf-revision-gpu0.lock"),
    Path("/tmp/gpubpf-revision-struct-ops.lock"),
)
STATUS_APPLY = 0
STATUS_DEFAULT = 1
STATUS_REPEAT = 2
STATUS_CONFLICT = 4
STATUS_RANGE = 6
STATUS_NOT_OBSERVED = 0xFFFFFFFF
PHASE_VALIDATED = 1
PHASE_NATIVE_RETURN = 2
PHASE_CONSTRUCTOR_RETURN = 3
FIELD_NONE = 0
FIELD_TIMESLICE = 1
FIELD_INTERLEAVE = 2
GSP_TIMESLICE = 0xA06C0103
GSP_INTERLEAVE = 0xA06C0107
GATE_CODE = (
    "import os,signal,sys;"
    "os.kill(os.getpid(),signal.SIGSTOP);"
    "os.execve(sys.argv[1],sys.argv[1:],os.environ)"
)
INTERRUPTED_SIGNALS: list[int] = []

BTF_DIAGNOSTIC_FIELDS = (
    ("abi_version", 0), ("abi_size", 32), ("phase", 64), ("field", 96),
    ("h_client", 128), ("h_resource", 160), ("gpu_instance", 192),
    ("subdevice_instance", 224), ("group_id", 256), ("runlist_id", 288),
    ("engine_type", 320), ("constructor_epoch", 352),
    ("default_timeslice", 384), ("minimum_timeslice", 448),
    ("default_interleave", 512), ("timeslice_attempted", 544),
    ("timeslice_conflict", 576), ("reserved0", 608),
    ("timeslice_request_value", 640), ("interleave_attempted", 704),
    ("interleave_conflict", 736), ("interleave_request_value", 768),
    ("timeslice_validation_result", 800),
    ("interleave_validation_result", 832), ("reserved1", 864),
    ("effective_timeslice", 896), ("effective_interleave", 960),
    ("timeslice_native_status", 992), ("timeslice_post_value", 1024),
    ("interleave_native_status", 1088), ("interleave_post_value", 1120),
    ("constructor_status", 1152), ("final_interleave", 1184),
    ("final_timeslice", 1216), ("final_snapshot_valid", 1280),
    ("reserved2", 1312),
)
BTF_GSP_FIELDS = (
    ("input_value", 0), ("hClient", 64), ("hObject", 96),
    ("command", 128), ("input_size", 160), ("wire_size", 192),
    ("input_valid", 224), ("transport_status", 256), ("gsp_status", 288),
    ("gsp_status_valid", 320), ("reserved", 352),
)

SAFETY_SPEC = importlib.util.spec_from_file_location(
    "revision_init_safety", ROOT / "workloads/moe-infinity/run_moe_head_to_head.py"
)
assert SAFETY_SPEC and SAFETY_SPEC.loader
safety = importlib.util.module_from_spec(SAFETY_SPEC)
sys.modules[SAFETY_SPEC.name] = safety
SAFETY_SPEC.loader.exec_module(safety)


class GateError(RuntimeError):
    pass


def note_interrupt(signum: int, _frame: Any) -> None:
    INTERRUPTED_SIGNALS.append(signum)


def raise_if_interrupted() -> None:
    if INTERRUPTED_SIGNALS:
        raise InterruptedError(f"signal {INTERRUPTED_SIGNALS[0]}; owned cleanup completed")


@dataclass(frozen=True)
class Row:
    name: str
    fixture: str | None
    fixture_id: int | None
    timeslice_returns: tuple[int, ...]
    interleave_returns: tuple[int, ...]
    timeslice_result: int
    interleave_result: int
    timeslice_conflict: int
    interleave_conflict: int
    interleave_request: int
    native_fields: tuple[int, ...]


ROWS = (
    Row("native_unattached", None, None, (), (), STATUS_DEFAULT, STATUS_DEFAULT,
        0, 0, 0, ()),
    Row("bpf_no_request", "no_request", 0, (), (), STATUS_DEFAULT, STATUS_DEFAULT,
        0, 0, 0, ()),
    Row("bpf_legal", "legal", 1, (STATUS_APPLY,), (STATUS_APPLY,),
        STATUS_APPLY, STATUS_APPLY, 0, 0, 0,
        (FIELD_TIMESLICE, FIELD_INTERLEAVE)),
    Row("bpf_duplicate", "duplicate", 3,
        (STATUS_APPLY, STATUS_REPEAT), (STATUS_APPLY, STATUS_REPEAT),
        STATUS_APPLY, STATUS_APPLY, 0, 0, 0,
        (FIELD_TIMESLICE, FIELD_INTERLEAVE)),
    Row("bpf_invalid_interleave", "invalid_interleave", 2, (), (STATUS_APPLY,),
        STATUS_DEFAULT, STATUS_RANGE, 0, 0, 3, ()),
    Row("bpf_conflict", "conflict", 4,
        (STATUS_APPLY, STATUS_CONFLICT, STATUS_CONFLICT),
        (STATUS_APPLY, STATUS_CONFLICT, STATUS_CONFLICT),
        STATUS_CONFLICT, STATUS_CONFLICT, 1, 1, 0, ()),
    Row("bpf_independent_interleave", "independent_interleave", 5,
        (STATUS_APPLY,), (STATUS_APPLY,), STATUS_APPLY, STATUS_RANGE,
        0, 0, 3, (FIELD_TIMESLICE,)),
    Row("bpf_independent_timeslice", "independent_timeslice", 6,
        (STATUS_APPLY, STATUS_CONFLICT), (STATUS_APPLY,),
        STATUS_CONFLICT, STATUS_APPLY, 1, 0, 0, (FIELD_INTERLEAVE,)),
)


def demand(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def matrix_plan() -> list[dict[str, Any]]:
    return [
        {"block": block, "cell": block * len(ROWS) + index,
         "row": row.name, "fixture": row.fixture}
        for block in range(2)
        for index, row in enumerate(ROWS)
    ]


def validate_btf_struct(raw: str, name: str, size: int,
                        fields: tuple[tuple[str, int], ...]) -> str:
    match = re.search(
        rf"^\[(\d+)\] STRUCT '{re.escape(name)}' size={size} vlen={len(fields)}\n"
        rf"((?:\t[^\n]+\n){{{len(fields)}}})", raw, re.MULTILINE,
    )
    observed = re.findall(r"\t'([^']+)' type_id=\d+ bits_offset=(\d+)",
                          match.group(2)) if match else []
    demand(observed == [(field, str(offset)) for field, offset in fields],
           f"loaded BTF layout mismatch for {name}")
    return match.group(1)


def validate_btf_hook(raw: str, structure_id: str, function: str) -> None:
    const_ids = re.findall(
        rf"^\[(\d+)\] CONST '\(anon\)' type_id={structure_id}$", raw, re.MULTILINE
    )
    pointer_ids = [pointer for const_id in const_ids for pointer in re.findall(
        rf"^\[(\d+)\] PTR '\(anon\)' type_id={const_id}$", raw, re.MULTILINE
    )]
    prototype_ids = [prototype for pointer in pointer_ids for prototype in re.findall(
        rf"^\[(\d+)\] FUNC_PROTO '\(anon\)' ret_type_id=0 vlen=1\n"
        rf"\t'ctx' type_id={pointer}$", raw, re.MULTILINE
    )]
    demand(any(re.search(
        rf"^\[\d+\] FUNC '{re.escape(function)}' type_id={prototype} linkage=",
        raw, re.MULTILINE,
    ) for prototype in prototype_ids), f"loaded BTF hook prototype mismatch for {function}")


def validate_loaded_btf(raw: str) -> None:
    diagnostic_id = validate_btf_struct(
        raw, "nv_gpu_sched_init_diagnostic_ctx", 168, BTF_DIAGNOSTIC_FIELDS
    )
    gsp_id = validate_btf_struct(
        raw, "nv_gpu_gsp_control_complete_ctx", 48, BTF_GSP_FIELDS
    )
    validate_btf_hook(raw, diagnostic_id, "nv_gpu_sched_init_diagnostic")
    validate_btf_hook(raw, gsp_id, "nv_gpu_sched_gsp_control_complete")


def json_events(path: Path, *, allow_partial: bool = False) -> list[dict[str, Any]]:
    records = []
    text = path.read_text(errors="strict")
    lines = text.splitlines(keepends=True)
    if allow_partial and lines and not lines[-1].endswith("\n"):
        lines.pop()
    for number, raw in enumerate(lines, 1):
        line = raw.rstrip("\r\n")
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise GateError(f"{path}:{number}: incomplete JSON record") from error
        demand(isinstance(value, dict), f"{path}:{number}: JSON record is not an object")
        records.append(value)
    return records


def one(records: list[dict[str, Any]], event: str) -> dict[str, Any]:
    selected = [record for record in records if record.get("event") == event]
    demand(len(selected) == 1, f"expected one {event}, found {len(selected)}")
    return selected[0]


def wait_compute_sample(path: Path, process: subprocess.Popen[Any], *,
                        after_ns: int = 0, empty: bool = False,
                        timeout: float = 10) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        raise_if_interrupted()
        demand(process.poll() is None,
               f"compute-process monitor exited early: {process.returncode}")
        for record in reversed(json_events(path, allow_partial=True)):
            started = record.get("query_started_mono_ns")
            finished = record.get("query_finished_mono_ns")
            pids = record.get("pids")
            if (record.get("event") == "sample" and type(started) is int and
                    type(finished) is int and 0 < started <= finished and
                    started > after_ns and "error" not in record and
                    isinstance(pids, list) and (not empty or pids == [])):
                return record
        time.sleep(0.05)
    raise GateError("compute-process monitor did not cover a required lifecycle point")


def wait_telemetry_sample(path: Path, process: subprocess.Popen[Any],
                          timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        raise_if_interrupted()
        demand(process.poll() is None,
               f"GPU telemetry exited early: {process.returncode}")
        if len([line for line in path.read_text(errors="replace").splitlines()
                if line.strip()]) >= 2:
            return
        time.sleep(0.05)
    raise GateError("GPU telemetry did not produce a pre-target sample")


def validate_compute_monitor(path: Path, target_pid: int | None,
                             window: dict[str, int] | None = None) -> dict[str, int]:
    records = json_events(path)
    samples = [record for record in records if record.get("event") == "sample"]
    finals = [record for record in records if record.get("event") == "final"]
    demand(samples and len(finals) == 1 and finals[0].get("errors") == 0 and
           records[-1] is finals[0],
           "continuous compute-process monitor was incomplete")
    starts = [record.get("query_started_mono_ns") for record in samples]
    finishes = [record.get("query_finished_mono_ns") for record in samples]
    demand(all(type(value) is int and value > 0 for value in starts + finishes) and
           all(start <= finish for start, finish in zip(starts, finishes)) and
           starts == sorted(starts) and len(set(starts)) == len(starts) and
           all(finish < next_start for finish, next_start in zip(finishes, starts[1:])),
           "invalid compute-process monitor timestamps")
    allowed = set() if target_pid is None else {target_pid}
    foreign: set[int] = set()
    for record in samples:
        pids = record.get("pids")
        demand("error" not in record and isinstance(pids, list) and
               all(type(pid) is int and pid > 0 for pid in pids) and
               pids == sorted(set(pids)), "invalid compute-process monitor sample")
        foreign.update(set(pids) - allowed)
    demand(not foreign, f"foreign compute clients appeared: {sorted(foreign)}")
    durations = [finish - start for start, finish in zip(starts, finishes)]
    gaps = [next_start - finish
            for finish, next_start in zip(finishes, starts[1:])]
    demand(all(value <= COMPUTE_MAX_GAP_NS for value in durations + gaps),
           "compute-process monitor has an uncovered sampling gap")
    if window is not None:
        required = (
            "pretarget_query_started_mono_ns", "pretarget_query_finished_mono_ns",
            "target_started_mono_ns", "target_exit_mono_ns", "owned_cleanup_mono_ns",
            "postcleanup_query_started_mono_ns", "postcleanup_query_finished_mono_ns",
        )
        demand(all(type(window.get(key)) is int and window[key] > 0
                   for key in required), "compute-process lifecycle markers are incomplete")
        values = [window[key] for key in required]
        demand(values == sorted(values),
               "compute-process lifecycle markers are out of order")
        by_start = {record["query_started_mono_ns"]: record for record in samples}
        pretarget = by_start.get(window["pretarget_query_started_mono_ns"])
        postcleanup = by_start.get(window["postcleanup_query_started_mono_ns"])
        demand(pretarget is not None and
               pretarget["query_finished_mono_ns"] ==
               window["pretarget_query_finished_mono_ns"] and pretarget["pids"] == [],
               "pre-target empty compute-process sample is absent")
        demand(postcleanup is not None and
               postcleanup["query_finished_mono_ns"] ==
               window["postcleanup_query_finished_mono_ns"] and postcleanup["pids"] == [],
               "post-cleanup empty compute-process sample is absent")
        demand(window["target_started_mono_ns"] -
               window["pretarget_query_finished_mono_ns"] <= COMPUTE_MAX_GAP_NS and
               window["postcleanup_query_started_mono_ns"] -
               window["owned_cleanup_mono_ns"] <= COMPUTE_MAX_GAP_NS,
               "compute-process lifecycle edge exceeds the sampling bound")
    return {
        "samples": len(samples),
        "max_query_duration_ns": max(durations),
        "max_idle_gap_ns": max(gaps) if gaps else 0,
        "foreign_pids": len(foreign),
    }


def validate_observer(records: list[dict[str, Any]], target_pid: int) -> tuple[
        list[dict[str, Any]], list[dict[str, Any]]]:
    ready = one(records, "scheduler_init_observer_ready")
    demand(ready.get("target_tgid") == target_pid, "observer ready TGID mismatch")
    summary = one(records, "scheduler_init_observer_summary")
    diagnostics = [r for r in records if r.get("event") == "scheduler_init_diagnostic"]
    gsp = [r for r in records if r.get("event") == "scheduler_init_gsp_completion"]
    demand(diagnostics, "no scheduler-init diagnostic events")
    for source, observed in (("diagnostic", diagnostics), ("gsp", gsp)):
        counters = summary.get(source)
        demand(isinstance(counters, dict), f"missing observer {source} counters")
        for key in ("observed", "emitted", "read_errors", "ring_drops", "received"):
            demand(type(counters.get(key)) is int and counters[key] >= 0,
                   f"invalid observer counter {source}.{key}")
        demand(counters["read_errors"] == counters["ring_drops"] == 0,
               f"observer {source} read/drop failure")
        demand(counters["observed"] == counters["emitted"] ==
               counters["received"] == len(observed),
               f"observer {source} accounting mismatch")
        demand(all(record.get("pid") == target_pid for record in observed),
               f"observer {source} retained a foreign TGID")
    return diagnostics, gsp


def validate_loader(row: Row, records: list[dict[str, Any]], target_pid: int) -> list[dict[str, Any]]:
    if row.fixture is None:
        demand(not records, "native row unexpectedly has loader records")
        return []
    ready = one(records, "scheduler_init_loader_ready")
    summary = one(records, "scheduler_init_loader_summary")
    requests = [r for r in records if r.get("event") == "scheduler_init_policy_request"]
    demand(ready.get("target_tgid") == summary.get("target_tgid") == target_pid,
           "loader target TGID mismatch")
    for key in ("struct_ops_map_id", "struct_ops_link_id"):
        demand(type(ready.get(key)) is int and ready[key] > 0 and
               ready[key] == summary.get(key), f"invalid loader {key}")
    demand(requests, "attached policy recorded no task-init callbacks")
    count = len(requests)
    demand(summary.get("init_seen") == summary.get("init_recorded") ==
           summary.get("request_records") == count,
           "policy callback/map accounting mismatch")
    demand(summary.get("init_record_error") == 0, "policy record error")
    demand(all(record.get("pid") == target_pid for record in requests),
           "policy map retained a foreign TGID")
    return requests


def expected_sequence(row: Row) -> tuple[list[int], list[int]]:
    return (
        [PHASE_VALIDATED] + [PHASE_NATIVE_RETURN] * len(row.native_fields) +
        [PHASE_CONSTRUCTOR_RETURN],
        [FIELD_NONE, *row.native_fields, FIELD_NONE],
    )


def validate_policy_record(row: Row, record: dict[str, Any], validated: dict[str, Any]) -> None:
    demand(record.get("fixture") == row.fixture_id and record.get("complete") == 1,
           "policy fixture/complete mismatch")
    demand(record.get("tsg_id") == validated["group_id"] and
           record.get("runlist_id") == validated["runlist_id"] and
           record.get("engine_type") == validated["engine_type"],
           "policy/diagnostic constructor identity mismatch")
    demand(record.get("default_timeslice") == validated["default_timeslice"] and
           record.get("default_interleave") == validated["default_interleave"],
           "policy/diagnostic defaults mismatch")
    demand(record.get("timeslice_count") == len(row.timeslice_returns) and
           record.get("interleave_count") == len(row.interleave_returns),
           "policy request count mismatch")
    demand(record.get("timeslice_returns") ==
           list(row.timeslice_returns) + [0] * (3 - len(row.timeslice_returns)),
           "policy timeslice recorder returns mismatch")
    demand(record.get("interleave_returns") ==
           list(row.interleave_returns) + [0] * (3 - len(row.interleave_returns)),
           "policy interleave recorder returns mismatch")
    demand(type(record.get("timestamp_ns")) is int and
           record["timestamp_ns"] <= validated["timestamp_ns"],
           "policy request did not precede validation")


def validate_epoch(row: Row, events: list[dict[str, Any]],
                   policy: dict[str, Any] | None,
                   gsp_events: list[dict[str, Any]]) -> dict[str, Any]:
    phases, fields = expected_sequence(row)
    demand([event.get("phase") for event in events] == phases,
           "diagnostic phase framing mismatch")
    demand([event.get("field") for event in events] == fields,
           "diagnostic field framing mismatch")
    diagnostic_timestamps = [event.get("timestamp_ns") for event in events]
    demand(all(type(timestamp) is int and timestamp > 0
               for timestamp in diagnostic_timestamps) and
           diagnostic_timestamps == sorted(diagnostic_timestamps),
           "diagnostic timestamps are out of order")
    validated = events[0]
    final = events[-1]
    immutable = ("pid", "tid", "h_client", "h_resource", "gpu_instance",
                 "subdevice_instance", "group_id", "runlist_id", "engine_type",
                 "constructor_epoch", "default_timeslice", "minimum_timeslice",
                 "default_interleave", "timeslice_attempted", "timeslice_conflict",
                 "timeslice_request_value", "interleave_attempted",
                 "interleave_conflict", "interleave_request_value",
                 "timeslice_validation_result", "interleave_validation_result",
                 "effective_timeslice", "effective_interleave")
    for event in events:
        demand(event.get("abi_version") == 1 and event.get("abi_size") == 168,
               "diagnostic ABI mismatch")
        demand(all(event.get(key) == validated.get(key) for key in immutable),
               "diagnostic constructor fields changed between phases")
    default_timeslice = validated["default_timeslice"]
    demand(validated.get("minimum_timeslice") == 0 and
           validated.get("default_interleave") == 1,
           "live 575 constructor defaults/minimum differ from frozen matrix")
    demand(validated.get("timeslice_attempted") == bool(row.timeslice_returns) and
           validated.get("interleave_attempted") == bool(row.interleave_returns),
           "diagnostic attempted flags mismatch")
    demand(validated.get("timeslice_conflict") == row.timeslice_conflict and
           validated.get("interleave_conflict") == row.interleave_conflict,
           "diagnostic conflict flags mismatch")
    demand(validated.get("timeslice_request_value") ==
           (default_timeslice if row.timeslice_returns else 0),
           "diagnostic timeslice request mismatch")
    demand(validated.get("interleave_request_value") ==
           (row.interleave_request if row.interleave_returns else 0),
           "diagnostic interleave request mismatch")
    demand(validated.get("timeslice_validation_result") == row.timeslice_result and
           validated.get("interleave_validation_result") == row.interleave_result,
           "diagnostic validation result mismatch")
    demand(validated.get("effective_timeslice") == default_timeslice and
           validated.get("effective_interleave") ==
           (0 if FIELD_INTERLEAVE in row.native_fields else 1),
           "diagnostic effective fields mismatch")
    demand(validated.get("timeslice_native_status") == STATUS_NOT_OBSERVED and
           validated.get("interleave_native_status") == STATUS_NOT_OBSERVED,
           "VALIDATED already contains a native return")

    for event, field in zip(events[1:-1], row.native_fields):
        if field == FIELD_TIMESLICE:
            demand(event.get("timeslice_native_status") == 0 and
                   event.get("timeslice_post_value") == default_timeslice,
                   "timeslice setter result/post-value mismatch")
        else:
            demand(event.get("interleave_native_status") == 0 and
                   event.get("interleave_post_value") == 0,
                   "interleave setter result/post-value mismatch")
    demand(final.get("constructor_status") == 0 and
           final.get("final_snapshot_valid") == 1 and
           final.get("final_timeslice") == default_timeslice and
           final.get("final_interleave") ==
           (0 if FIELD_INTERLEAVE in row.native_fields else 1),
           "constructor return/final fields mismatch")
    demand(final.get("timeslice_native_status") ==
           (0 if FIELD_TIMESLICE in row.native_fields else STATUS_NOT_OBSERVED) and
           final.get("interleave_native_status") ==
           (0 if FIELD_INTERLEAVE in row.native_fields else STATUS_NOT_OBSERVED),
           "constructor reports an unexpected or missing native setter")

    if row.fixture is None:
        demand(policy is None, "native constructor matched a policy record")
    else:
        demand(policy is not None, "BPF constructor has no policy record")
        validate_policy_record(row, policy, validated)

    expected_gsp = []
    for field in row.native_fields:
        expected_gsp.append((
            GSP_TIMESLICE if field == FIELD_TIMESLICE else GSP_INTERLEAVE,
            default_timeslice if field == FIELD_TIMESLICE else 0,
            8 if field == FIELD_TIMESLICE else 4,
        ))
    demand(len(gsp_events) == len(expected_gsp), "GSP completion count mismatch")
    gsp_timestamps = [event.get("timestamp_ns") for event in gsp_events]
    demand(all(type(timestamp) is int and timestamp > 0
               for timestamp in gsp_timestamps) and
           gsp_timestamps == sorted(gsp_timestamps),
           "GSP events are out of order")
    for event, (command, value, size) in zip(gsp_events, expected_gsp):
        demand(event.get("pid") == validated["pid"] and
               event.get("tid") == validated["tid"] and
               event.get("h_client") == validated["h_client"] and
               event.get("h_object") == validated["h_resource"],
               "GSP/constructor identity mismatch")
        demand(validated["timestamp_ns"] <= event.get("timestamp_ns", -1) <=
               final["timestamp_ns"], "GSP event lies outside constructor interval")
        demand(event.get("command") == command and event.get("input_value") == value and
               event.get("input_size") == size and event.get("wire_size") == size,
               "GSP command/value/size mismatch")
        demand(event.get("input_valid") == event.get("gsp_status_valid") == 1 and
               event.get("transport_status") == event.get("gsp_status") == 0,
               "GSP input/transport/status failure")
    return {
        "constructor_epoch": validated["constructor_epoch"],
        "pid": validated["pid"], "tid": validated["tid"],
        "group_id": validated["group_id"], "runlist_id": validated["runlist_id"],
        "native_fields": list(row.native_fields),
    }


def validate_cell(row: Row, target_records: list[dict[str, Any]],
                  observer_records: list[dict[str, Any]],
                  loader_records: list[dict[str, Any]], target_pid: int) -> dict[str, Any]:
    demand(target_records == [EXPECTED_CORRECTNESS], "target numerical oracle mismatch")
    diagnostics, gsp = validate_observer(observer_records, target_pid)
    policies = validate_loader(row, loader_records, target_pid)
    groups: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for event in diagnostics:
        key = (event["pid"], event["tid"], event["constructor_epoch"])
        groups.setdefault(key, []).append(event)
    demand(groups, "no complete constructor groups")

    policies_by_key: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for record in policies:
        key = (record["pid"], record["tid"], record["tsg_id"], record["runlist_id"])
        demand(key not in policies_by_key, "duplicate policy constructor identity")
        policies_by_key[key] = record

    used_policy: set[tuple[int, int, int, int]] = set()
    used_gsp: set[int] = set()
    constructor_intervals: list[tuple[int, int, int, int]] = []
    joined = []
    for events in groups.values():
        timestamps = [event.get("timestamp_ns") for event in events]
        demand(all(type(timestamp) is int and timestamp > 0
                   for timestamp in timestamps) and
               timestamps == sorted(timestamps),
               "diagnostic events are out of order")
        validated = events[0]
        constructor_intervals.append(
            (validated["pid"], validated["tid"],
             validated["timestamp_ns"], events[-1]["timestamp_ns"])
        )
        policy_key = (validated["pid"], validated["tid"],
                      validated["group_id"], validated["runlist_id"])
        policy = policies_by_key.get(policy_key)
        if policy is not None:
            used_policy.add(policy_key)
        interval_gsp = []
        for index, event in enumerate(gsp):
            if (event.get("pid") == validated["pid"] and
                    event.get("tid") == validated["tid"] and
                    event.get("h_client") == validated["h_client"] and
                    event.get("h_object") == validated["h_resource"] and
                    validated["timestamp_ns"] <= event.get("timestamp_ns", -1) <=
                    events[-1]["timestamp_ns"]):
                demand(index not in used_gsp, "GSP event matched multiple constructors")
                used_gsp.add(index)
                interval_gsp.append(event)
        joined.append(validate_epoch(row, events, policy, interval_gsp))
    demand(used_policy == set(policies_by_key), "unmatched policy request record")
    ignored_gsp = []
    for index, event in enumerate(gsp):
        if index in used_gsp:
            continue
        timestamp = event.get("timestamp_ns", -1)
        demand(not any(event.get("pid") == pid and event.get("tid") == tid and
                       start <= timestamp <= end
                       for pid, tid, start, end in constructor_intervals),
               "unmatched target GSP event inside a constructor interval")
        ignored_gsp.append(index)
    return {"row": row.name, "constructors": joined,
            "gsp_events_ignored_outside_constructor_intervals": len(ignored_gsp),
            "passed": True}


class ReadOnlyLeases:
    def __init__(self) -> None:
        self.descriptors: list[int] = []

    def acquire(self) -> None:
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            for path in LEASE_PATHS:
                descriptor = os.open(path, flags)
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    os.close(descriptor)
                    raise GateError(f"lease is not a regular file: {path}")
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BaseException:
                    os.close(descriptor)
                    raise
                self.descriptors.append(descriptor)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        for descriptor in reversed(self.descriptors):
            os.close(descriptor)
        self.descriptors.clear()


def atomic_json(path: Path, value: Any) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def group_members(pgid: int) -> list[int]:
    members = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if fields[0] != "Z" and int(fields[2]) == pgid and int(fields[3]) == pgid:
                members.append(int(path.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    return members


def stop_owned(process: subprocess.Popen[Any] | None) -> None:
    if process is None:
        return
    if group_members(process.pid):
        try:
            os.killpg(process.pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
    for sig, seconds in ((signal.SIGINT, 8), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        process.poll()
        if not group_members(process.pid):
            process.wait(timeout=1)
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            process.poll()
            if not group_members(process.pid):
                process.wait(timeout=1)
                return
            time.sleep(0.05)
    raise GateError(f"owned process group {process.pid} survived cleanup")


def wait_ready(path: Path, event: str, process: subprocess.Popen[Any], timeout: float = 15) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        raise_if_interrupted()
        matches = [record for record in json_events(path, allow_partial=True)
                   if record.get("event") == event]
        if len(matches) == 1:
            return matches[0]
        demand(len(matches) < 2, f"duplicate {event} records")
        if process.poll() is not None:
            raise GateError(f"process exited before {event}: rc={process.returncode}")
        time.sleep(0.05)
    raise GateError(f"timed out waiting for {event}")


def wait_stopped(process: subprocess.Popen[Any], timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        raise_if_interrupted()
        try:
            state = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError as error:
            raise GateError("exec-gated target exited before observation setup") from error
        if state in {"T", "t"}:
            return
        time.sleep(0.02)
    raise GateError("target did not enter the pre-CUDA exec gate")


def command_env() -> dict[str, str]:
    return {
        "PATH": "/usr/local/cuda-12.9/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "CUDA_VISIBLE_DEVICES": "0",
        "LD_LIBRARY_PATH": "/usr/local/cuda-12.9/lib64",
    }


def start(name: str, argv: list[str], directory: Path,
          streams: list[Any]) -> subprocess.Popen[Any]:
    stdout = (directory / f"{name}.jsonl").open("x")
    stderr = (directory / f"{name}.stderr.log").open("x")
    streams.extend((stdout, stderr))
    return subprocess.Popen(argv, stdout=stdout, stderr=stderr,
                            env=command_env(), start_new_session=True)


def struct_ops_ids() -> tuple[set[int], set[int]]:
    inventory = safety.struct_ops_inventory()
    return ({int(item["id"]) for item in inventory["maps"]},
            {int(item["id"]) for item in inventory["links"]})


def bpf_inventory() -> dict[str, list[int]]:
    inventory = {}
    for kind in ("prog", "map", "link"):
        output = safety.run_checked(["bpftool", kind, "show", "-j"])
        inventory[kind] = sorted(int(item["id"]) for item in json.loads(output or "[]"))
    return inventory


def run_cell(directory: Path, row: Row, block: int) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    streams: list[Any] = []
    target = observer = loader = telemetry = kernel_monitor = compute_monitor = None
    telemetry_path = None
    kernel_path = directory / "kernel-monitor.log"
    compute_path = directory / "compute-apps.jsonl"
    compute_window: dict[str, int] = {}
    bpf_before = None
    before = None
    pre_valid = False
    result: dict[str, Any] = {"status": "failed", "row": row.name, "block": block,
                              "safety_before": None,
                              "compute_window": compute_window}
    try:
        before = safety.safety_snapshot()
        result["safety_before"] = before
        safety.validate_pre_server_safety(before)
        bpf_before = bpf_inventory()
        result["bpf_before"] = bpf_before
        pre_valid = True
        telemetry, telemetry_stream, telemetry_path = safety.start_gpu_telemetry(directory)
        streams.append(telemetry_stream)
        kernel_stream = kernel_path.open("x")
        streams.append(kernel_stream)
        kernel_monitor = subprocess.Popen(
            ["journalctl", "-k", "-f", "-n", "0", "--no-pager",
             "-o", "short-monotonic"], stdout=kernel_stream,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        compute_monitor = start(
            "compute-apps",
            ["taskset", "-c", "16", sys.executable, "-B", str(COMPUTE_MONITOR)],
            directory, streams,
        )
        wait_telemetry_sample(telemetry_path, telemetry)
        pretarget = wait_compute_sample(compute_path, compute_monitor, empty=True)
        compute_window.update(
            pretarget_query_started_mono_ns=pretarget["query_started_mono_ns"],
            pretarget_query_finished_mono_ns=pretarget["query_finished_mono_ns"],
        )
        demand(kernel_monitor.poll() is None, "kernel monitor exited before target start")
        compute_window["target_started_mono_ns"] = time.monotonic_ns()
        target = start("target", ["taskset", "-c", "8-15", sys.executable,
                                  "-c", GATE_CODE, str(TARGET)], directory, streams)
        wait_stopped(target)
        observer = start("observer", ["taskset", "-c", "17", str(OBSERVER),
                                      str(target.pid), "120"], directory, streams)
        wait_ready(directory / "observer.jsonl", "scheduler_init_observer_ready", observer)
        if row.fixture is not None:
            object_path = OBJECT_DIR / f"revision_init_{row.fixture}.bpf.o"
            loader = start("loader", ["taskset", "-c", "17", str(LOADER),
                                      str(object_path), str(target.pid), "120"], directory, streams)
            ready = wait_ready(directory / "loader.jsonl", "scheduler_init_loader_ready", loader)
            maps, links = struct_ops_ids()
            demand(maps == {ready["struct_ops_map_id"]},
                   "loaded struct_ops map is not exactly the owned policy")
            demand(links == {ready["struct_ops_link_id"]},
                   "loaded struct_ops link is not exactly the owned policy")
        else:
            demand(struct_ops_ids() == (set(), set()), "native cell has struct_ops state")
        demand(all(process.poll() is None for process in
                   (telemetry, kernel_monitor, compute_monitor)),
               "continuous monitor exited before target release")
        raise_if_interrupted()
        os.kill(target.pid, signal.SIGCONT)
        demand(target.wait(timeout=30) == 0, "finite CUDA target failed")
        compute_window["target_exit_mono_ns"] = time.monotonic_ns()
        stop_owned(target)
        stop_owned(loader)
        stop_owned(observer)
        compute_window["owned_cleanup_mono_ns"] = time.monotonic_ns()
        postcleanup = wait_compute_sample(
            compute_path, compute_monitor,
            after_ns=compute_window["owned_cleanup_mono_ns"], empty=True,
        )
        compute_window.update(
            postcleanup_query_started_mono_ns=postcleanup["query_started_mono_ns"],
            postcleanup_query_finished_mono_ns=postcleanup["query_finished_mono_ns"],
        )
        result["monitors_alive_through_cleanup"] = {
            "telemetry": telemetry.poll() is None,
            "kernel": kernel_monitor.poll() is None,
            "compute_apps": compute_monitor.poll() is None,
        }
        demand(all(result["monitors_alive_through_cleanup"].values()),
               "continuous monitor exited before owned cleanup completed")
        stop_owned(compute_monitor)
        stop_owned(telemetry)
        stop_owned(kernel_monitor)
        demand(loader is None or loader.returncode == 0, "fixture loader failed")
        demand(observer.returncode == 0, "scheduler-init observer failed")
        joined = validate_cell(
            row, json_events(directory / "target.jsonl"),
            json_events(directory / "observer.jsonl"),
            [] if loader is None else json_events(directory / "loader.jsonl"), target.pid,
        )
        demand(struct_ops_ids() == (set(), set()), "owned policy survived loader exit")
        result.update(status="passed", target_pid=target.pid, joined=joined)
        return result
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        cleanup_errors = []
        for process in (target, loader, observer, compute_monitor, telemetry, kernel_monitor):
            try:
                stop_owned(process)
            except BaseException as error:
                cleanup_errors.append(str(error))
        for stream in streams:
            try:
                stream.close()
            except BaseException as error:
                cleanup_errors.append(str(error))
        if telemetry_path is not None:
            try:
                result["gpu_telemetry"] = safety.validate_gpu_telemetry(
                    telemetry_path, allow_fixed_power_cap=True
                )
            except BaseException as error:
                cleanup_errors.append(str(error))
        if compute_monitor is not None:
            try:
                result["compute_monitor"] = validate_compute_monitor(
                    compute_path, target.pid if target is not None else None,
                    compute_window if result["status"] == "passed" else None,
                )
            except BaseException as error:
                cleanup_errors.append(str(error))
        if kernel_path.exists():
            try:
                kernel_abnormal = safety.filtered_kernel_records(
                    kernel_path.read_text(errors="replace")
                )
                result["kernel_monitor_abnormal"] = kernel_abnormal
                if kernel_abnormal:
                    cleanup_errors.append(
                        f"kernel monitor observed: {kernel_abnormal}"
                    )
            except BaseException as error:
                cleanup_errors.append(str(error))
        if bpf_before is not None:
            try:
                bpf_after = bpf_inventory()
                result["bpf_after"] = bpf_after
                demand(bpf_after == bpf_before, "BPF object inventory changed after cleanup")
            except BaseException as error:
                cleanup_errors.append(str(error))
        if pre_valid:
            try:
                result["safety_after"] = safety.wait_for_post_server_safety(before)
            except BaseException as error:
                cleanup_errors.append(str(error))
        result["owned_group_survivors"] = {
            process.pid: group_members(process.pid)
            for process in (target, loader, observer, compute_monitor,
                            telemetry, kernel_monitor)
            if process is not None and group_members(process.pid)
        }
        if cleanup_errors or result["owned_group_survivors"]:
            result.update(status="failed", cleanup_errors=cleanup_errors)
        atomic_json(directory / "result.json", result)
        if cleanup_errors:
            raise GateError("; ".join(cleanup_errors))


def admission() -> dict[str, Any]:
    demand(os.geteuid() == 0, "live matrix is root-only")
    demand(os.uname().release == EXPECTED_KERNEL, "unexpected running kernel")
    missing = [str(path) for path in (TARGET, OBSERVER, LOADER, COMPUTE_MONITOR, *(
        OBJECT_DIR / f"revision_init_{row.fixture}.bpf.o"
        for row in ROWS if row.fixture is not None
    )) if not path.is_file()]
    demand(not missing, f"missing live artifact(s): {missing}")
    snapshot = safety.safety_snapshot()
    safety.validate_pre_server_safety(snapshot)
    demand(snapshot["gpu"]["driver"] == EXPECTED_DRIVER, "unexpected NVIDIA driver")
    raw_btf = subprocess.run(
        ["bpftool", "btf", "dump", "file", "/sys/kernel/btf/nvidia", "format", "raw"],
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    validate_loaded_btf(raw_btf)
    return {"kernel": os.uname().release, "driver": snapshot["gpu"]["driver"],
            "initial_safety": snapshot}


def run_matrix(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    leases = ReadOnlyLeases()
    result: dict[str, Any] = {
        "complete": False,
        "scope": "functional cells under an already loaded reviewed core candidate",
        "module_lifecycle_performed": False,
        "plan": matrix_plan(), "cells": [],
    }
    try:
        leases.acquire()
        raise_if_interrupted()
        result["admission"] = admission()
        raise_if_interrupted()
        for entry in result["plan"]:
            raise_if_interrupted()
            row = next(item for item in ROWS if item.name == entry["row"])
            cell_dir = output / f"block-{entry['block'] + 1:02d}-{row.name}"
            result["cells"].append(run_cell(cell_dir, row, entry["block"]))
            atomic_json(output / "result.json", result)
        raise_if_interrupted()
        result["complete"] = True
        result["passed_cells"] = len(result["cells"])
        return result
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        leases.close()
        atomic_json(output / "result.json", result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        demand(args.output is None, "--plan-only does not accept --output")
        print(json.dumps({"rows": [asdict(row) for row in ROWS],
                          "schedule": matrix_plan()}, indent=2))
        return 0
    demand(args.output is not None, "live execution requires --output")
    INTERRUPTED_SIGNALS.clear()
    signal.signal(signal.SIGINT, note_interrupt)
    signal.signal(signal.SIGTERM, note_interrupt)
    print(json.dumps(run_matrix(args.output.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GateError, InterruptedError) as error:
        print(f"scheduler-init gate failed: {error}", file=sys.stderr)
        raise SystemExit(1)
