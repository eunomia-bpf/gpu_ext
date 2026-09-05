#!/usr/bin/env python3
"""Fail-closed parsing and driver-counter reconciliation for the live observer."""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, Iterator

import coordinator
import protocol


READY_FIELDS = {
    "event",
    "pid",
    "target_pid",
    "implementation",
    "observer_link_id",
    "struct_link_id",
    "struct_map_id",
}
DECISION_FIELDS = {
    "event",
    "implementation",
    "decision_sequence",
    "snapshot_read_path",
    "decision_mono_ns",
    "snapshot_sequence",
    "snapshot_phase",
    "decision_age_ns",
    "action",
    "effect",
    "effect_source",
    "fault_page_index",
    "legal_max_first",
    "legal_max_outer",
    "output_first",
    "output_outer",
    "observer_mono_ns",
    "target_tgid",
}
FINAL_FIELDS = {
    "event",
    "implementation",
    "observer_link_id",
    "struct_link_id",
    "diagnostic_calls",
    "selected_seen",
    "finished_seen",
    "records_emitted",
    "foreign_tgid",
    "read_errors",
    "ringbuf_drops",
    "phase_errors",
    "valid",
}


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise protocol.ValidationError(f"duplicate observer JSON field: {key}")
        value[key] = item
    return value


def _load_object(line: str, line_number: int) -> dict[str, Any]:
    try:
        value = json.loads(line, object_pairs_hook=_strict_object)
    except (json.JSONDecodeError, protocol.ValidationError) as exc:
        raise protocol.ValidationError(
            f"invalid observer JSONL line {line_number}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise protocol.ValidationError(
            f"observer JSONL line {line_number} is not an object"
        )
    return value


def parse_jsonl(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        records.append(_load_object(line, line_number))
    if not records:
        raise protocol.ValidationError("observer produced no records")
    return records


def _position_event(record: dict[str, Any], index: int, *, last: bool) -> None:
    event = record.get("event")
    if index == 0 and event != "ready":
        raise protocol.ValidationError(
            "observer stream must begin with the ready record"
        )
    if last:
        if event != "observer_final":
            raise protocol.ValidationError(
                "observer stream must end with the final record"
            )
    elif index > 0 and event != "policy_decision":
        raise protocol.ValidationError(
            "observer decision records must occupy only the middle"
        )


def iter_jsonl(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    """Stream observer JSONL one record at a time with one-record lookahead."""
    pending: dict[str, Any] | None = None
    index = 0
    with open(path, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise protocol.ValidationError(
                    f"observer JSONL line {line_number} is blank"
                )
            record = _load_object(line, line_number)
            if pending is not None:
                _position_event(pending, index, last=False)
                yield pending
                index += 1
            pending = record
        if pending is not None:
            _position_event(pending, index, last=True)
            yield pending
        if index == 0:
            raise protocol.ValidationError("observer produced no records")


def _exact(record: dict[str, Any], fields: set[str], label: str) -> None:
    if set(record) != fields:
        raise protocol.ValidationError(
            f"{label} schema mismatch: missing={sorted(fields - set(record))}, "
            f"extra={sorted(set(record) - fields)}"
        )


def _integer(record: dict[str, Any], field: str, minimum: int = 0) -> int:
    value = record.get(field)
    if type(value) is not int or value < minimum:
        raise protocol.ValidationError(
            f"observer {field} must be an integer >= {minimum}"
        )
    return value


def validate_records(
    records: Iterable[dict[str, Any]], *, expected_pid: int, implementation: str
) -> dict[str, Any]:
    """Validate one complete loader stream after the owned workload is reaped."""

    values = list(records)
    if implementation not in protocol.IMPLEMENTATIONS:
        raise protocol.ValidationError("observer implementation is invalid")
    if type(expected_pid) is not int or expected_pid <= 0:
        raise protocol.ValidationError("observer target PID is invalid")
    if len(values) < 3:
        raise protocol.ValidationError("observer stream lacks ready/decision/final records")
    ready, final = values[0], values[-1]
    decisions = values[1:-1]
    _exact(ready, READY_FIELDS, "observer ready")
    _exact(final, FINAL_FIELDS, "observer final")
    if ready.get("event") != "ready" or final.get("event") != "observer_final":
        raise protocol.ValidationError("observer stream boundaries are invalid")
    if ready.get("implementation") != implementation or final.get(
        "implementation"
    ) != implementation:
        raise protocol.ValidationError("observer implementation identity differs")
    if _integer(ready, "pid", 1) == expected_pid:
        raise protocol.ValidationError("observer loader and workload PID are identical")
    if _integer(ready, "target_pid", 1) != expected_pid:
        raise protocol.ValidationError("observer ready target PID differs")
    observer_link = _integer(ready, "observer_link_id", 1)
    struct_link = _integer(ready, "struct_link_id")
    struct_map = _integer(ready, "struct_map_id")
    if (implementation == "native" and struct_link != 0) or (
        implementation == "bpf" and struct_link == 0
    ):
        raise protocol.ValidationError("observer struct_ops ownership differs")
    if (implementation == "native" and struct_map != 0) or (
        implementation == "bpf" and struct_map == 0
    ):
        raise protocol.ValidationError("observer struct_ops map ownership differs")
    if final.get("valid") is not True:
        raise protocol.ValidationError("observer final validity gate failed")
    if final.get("observer_link_id") != observer_link or final.get(
        "struct_link_id"
    ) != struct_link:
        raise protocol.ValidationError("observer link identity changed")

    previous_decision_ns = 0
    dense = discard = 0
    for sequence, decision in enumerate(decisions, 1):
        _exact(decision, DECISION_FIELDS, "observer decision")
        if decision.get("event") != "policy_decision" or decision.get(
            "implementation"
        ) != implementation:
            raise protocol.ValidationError("observer decision identity differs")
        if _integer(decision, "target_tgid", 1) != expected_pid:
            raise protocol.ValidationError("observer decision target PID differs")
        if _integer(decision, "decision_sequence", 1) != sequence:
            raise protocol.ValidationError("observer decision sequence is not contiguous")
        decision_ns = _integer(decision, "decision_mono_ns", 1)
        if decision_ns < previous_decision_ns:
            raise protocol.ValidationError("observer decision time regressed")
        previous_decision_ns = decision_ns
        observed_ns = _integer(decision, "observer_mono_ns", decision_ns)
        if observed_ns < decision_ns:
            raise protocol.ValidationError("observer timestamp predates driver decision")
        if decision.get("snapshot_read_path") != (
            f"driver_{implementation}_read_only_context"
        ) or decision.get("effect_source") != "driver_diagnostic":
            raise protocol.ValidationError("observer decision provenance differs")
        if decision.get("snapshot_phase") == "dense":
            dense += 1
        elif decision.get("snapshot_phase") == "sparse":
            discard += 1
        else:
            raise protocol.ValidationError("observer decision phase is invalid")

    count = len(decisions)
    expected_metrics = {
        "diagnostic_calls": 2 * count,
        "selected_seen": count,
        "finished_seen": count,
        "records_emitted": count,
        "foreign_tgid": 0,
        "read_errors": 0,
        "ringbuf_drops": 0,
        "phase_errors": 0,
    }
    for field, expected in expected_metrics.items():
        if _integer(final, field) != expected:
            raise protocol.ValidationError(f"observer final counter differs: {field}")
    if count == 0 or dense == 0 or discard == 0:
        raise protocol.ValidationError("observer did not retain both policy actions")
    return {
        "ready": ready,
        "decisions": decisions,
        "final": final,
        "dense": dense,
        "discard": discard,
    }


def reconcile_driver(
    observed: dict[str, Any], status: coordinator.BridgeStatus, *, implementation: str
) -> dict[str, Any]:
    """Create protocol policy-final only after observer and driver counters close."""

    decisions = observed.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise protocol.ValidationError("validated observer decisions are absent")
    count = len(decisions)
    if status.mode != implementation:
        raise protocol.ValidationError("driver mode differs from observer mode")
    expected = {
        "snapshot_updates": protocol.MEASURED_PHASES + 1,
        "callback_invocations": count,
        "snapshot_read_attempts": count,
        "snapshot_read_successes": count,
        "native_callback_invocations": count if implementation == "native" else 0,
        "bpf_callback_invocations": count if implementation == "bpf" else 0,
        "decision_requests": count,
        "decisions": count,
        "decision_records": count,
        "effect_requests": count,
        "effect_records": count,
        "selected_diagnostics": count,
        "finished_diagnostics": count,
        "dense_prefetch_decisions": observed["dense"],
        "discarded_prefetch_decisions": observed["discard"],
    }
    for field, value in expected.items():
        if getattr(status, field) != value:
            raise protocol.ValidationError(f"driver/observer counter differs: {field}")
    for field in (
        "snapshot_rejections",
        "missing_snapshot_decisions",
        "invalid_snapshot_decisions",
        "request_errors",
        "effect_errors",
    ):
        if getattr(status, field) != 0:
            raise protocol.ValidationError(f"driver reported an error: {field}")
    return {
        "event": "final_policy_stats",
        "implementation": implementation,
        **expected,
        "snapshot_rejections": 0,
        "missing_snapshot_decisions": 0,
        "invalid_snapshot_decisions": 0,
        "request_errors": 0,
        "effect_errors": 0,
        "decision_record_drops": 0,
        "effect_record_drops": 0,
    }
