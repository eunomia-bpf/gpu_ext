#!/usr/bin/env python3
"""Frozen matrix, CPU delay relay, and fail-closed raw-record validation."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


PROTOCOL = "stale-cross-layer-575-v1"
TIMELINE = "alternating-dense-sparse-40g-v1"
SEED = 20260903
DELAYS_MS = (0, 100, 1000)
IMPLEMENTATIONS = ("native", "bpf")
FORMAL_BLOCKS = 3
PREFLIGHT_BLOCKS = 1
ALLOCATION_BYTES = 40 * 1024**3
REGION_BYTES = 64 * 1024
REGIONS = ALLOCATION_BYTES // REGION_BYTES
SPARSE_STRIDE = 32
SPARSE_REGIONS = (REGIONS + SPARSE_STRIDE - 1) // SPARSE_STRIDE
DENSE_LAUNCH_REGIONS = 4096
BOOTSTRAP_NS = 1_200_000_000
PHASE_NS = 2_000_000_000
MEASURED_PHASES = 6
MAXIMUM_BOUNDARY_OVERRUN_NS = 500_000_000
LEASE_PATHS = (
    "/tmp/gpubpf-revision-gpu0.lock",
    "/tmp/gpubpf-revision-struct-ops.lock",
)
POLICY_ARTIFACT_NAMES = (
    "snapshot-publications.jsonl",
    "policy-decisions.jsonl",
    "policy-final.json",
    "policy-observer.jsonl",
    "policy-observer.stderr.log",
    "verifier.log",
)
EXPECTED_GPU = "NVIDIA GeForce RTX 5090"
EXPECTED_DRIVER = "575.57.08"
UVM_QUEUE_ENTRIES = 1 << 22
LIVE_BLOCKER = (
    "the 575 driver has no atomic timestamped snapshot readable by both a "
    "native same-algorithm consumer and the BPF prefetch callback, nor matched "
    "snapshot-aware decision diagnostics"
)


class ValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Condition:
    arm: str
    implementation: str | None
    delay_ms: int | None
    role: str


@dataclass(frozen=True)
class MatrixCell:
    ordinal: int
    block: int
    arm: str
    implementation: str | None
    delay_ms: int | None
    role: str


@dataclass(frozen=True)
class PendingSnapshot:
    sequence: int
    phase: str
    source_mono_ns: int
    scheduled_offset_ns: int
    eligible_mono_ns: int


class DelayedSnapshotRelay:
    """Pure fixed-delay relay used by CPU preflight and the future coordinator."""

    def __init__(self, delay_ms: int):
        if type(delay_ms) is not int or delay_ms not in DELAYS_MS:
            raise ValidationError(f"unsupported delay: {delay_ms!r}")
        self.delay_ns = delay_ms * 1_000_000
        self._last_sequence = 0
        self._last_source_ns = 0
        self._pending: list[PendingSnapshot] = []

    def observe(
        self,
        *,
        sequence: int,
        phase: str,
        source_mono_ns: int,
        scheduled_offset_ns: int,
    ) -> None:
        if sequence != self._last_sequence + 1:
            raise ValidationError("phase sequence is not contiguous")
        if phase not in {"dense", "sparse"}:
            raise ValidationError("phase is not dense or sparse")
        if type(source_mono_ns) is not int or source_mono_ns <= self._last_source_ns:
            raise ValidationError("source monotonic time did not advance")
        if type(scheduled_offset_ns) is not int or scheduled_offset_ns < 0:
            raise ValidationError("scheduled offset is invalid")
        self._pending.append(
            PendingSnapshot(
                sequence=sequence,
                phase=phase,
                source_mono_ns=source_mono_ns,
                scheduled_offset_ns=scheduled_offset_ns,
                eligible_mono_ns=source_mono_ns + self.delay_ns,
            )
        )
        self._last_sequence = sequence
        self._last_source_ns = source_mono_ns

    def drain(self, now_ns: int) -> list[dict[str, Any]]:
        if type(now_ns) is not int or now_ns <= 0:
            raise ValidationError("publication clock is invalid")
        publications = []
        while self._pending and self._pending[0].eligible_mono_ns <= now_ns:
            pending = self._pending.pop(0)
            publications.append(
                {
                    "event": "snapshot_published",
                    "sequence": pending.sequence,
                    "phase": pending.phase,
                    "source_mono_ns": pending.source_mono_ns,
                    "scheduled_offset_ns": pending.scheduled_offset_ns,
                    "eligible_mono_ns": pending.eligible_mono_ns,
                    "published_mono_ns": now_ns,
                    "delay_ns": self.delay_ns,
                }
            )
        return publications

    @property
    def pending(self) -> int:
        return len(self._pending)


def conditions() -> tuple[Condition, ...]:
    values = [Condition("uvm_default", None, None, "context_control")]
    for delay_ms in DELAYS_MS:
        label = "fresh" if delay_ms == 0 else f"delay_{delay_ms}ms"
        for implementation in IMPLEMENTATIONS:
            values.append(
                Condition(
                    arm=f"{implementation}_{label}",
                    implementation=implementation,
                    delay_ms=delay_ms,
                    role="paired_policy",
                )
            )
    return tuple(values)


def matrix(stage: str) -> list[MatrixCell]:
    if stage not in {"preflight", "full"}:
        raise ValidationError(f"unknown stage: {stage}")
    block_count = PREFLIGHT_BLOCKS if stage == "preflight" else FORMAL_BLOCKS
    rng = random.Random(SEED + (0 if stage == "full" else 1_000_000))
    result: list[MatrixCell] = []
    ordinal = 0
    expected_arms = {condition.arm for condition in conditions()}
    for block in range(1, block_count + 1):
        block_conditions = list(conditions())
        rng.shuffle(block_conditions)
        if {condition.arm for condition in block_conditions} != expected_arms:
            raise AssertionError("internal matrix arm loss")
        for condition in block_conditions:
            ordinal += 1
            result.append(MatrixCell(ordinal=ordinal, block=block, **asdict(condition)))
    expected = block_count * len(conditions())
    if len(result) != expected or len({(cell.block, cell.arm) for cell in result}) != expected:
        raise AssertionError("internal matrix duplication")
    return result


def lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def expected_phase(sequence: int) -> dict[str, Any]:
    if sequence == 1:
        return {
            "sequence": 1,
            "phase": "sparse",
            "measured": False,
            "scheduled_offset_ns": 0,
        }
    if sequence < 2 or sequence > MEASURED_PHASES + 1:
        raise ValidationError(f"unexpected phase sequence: {sequence}")
    measured_index = sequence - 1
    return {
        "sequence": sequence,
        "phase": "dense" if measured_index % 2 == 1 else "sparse",
        "measured": True,
        "measured_index": measured_index,
        "scheduled_offset_ns": BOOTSTRAP_NS + (measured_index - 1) * PHASE_NS,
    }


def paired_questions() -> dict[str, list[dict[str, Any]]]:
    mechanism = []
    information = []
    for block in range(1, FORMAL_BLOCKS + 1):
        for delay_ms in DELAYS_MS:
            label = "fresh" if delay_ms == 0 else f"delay_{delay_ms}ms"
            mechanism.append(
                {
                    "block": block,
                    "delay_ms": delay_ms,
                    "native": f"native_{label}",
                    "bpf": f"bpf_{label}",
                }
            )
        for implementation in IMPLEMENTATIONS:
            for delay_ms in DELAYS_MS[1:]:
                information.append(
                    {
                        "block": block,
                        "implementation": implementation,
                        "fresh": f"{implementation}_fresh",
                        "delayed": f"{implementation}_delay_{delay_ms}ms",
                        "delay_ms": delay_ms,
                    }
                )
    return {"mechanism_cost": mechanism, "information_cost": information}


def dry_run_plan(
    stage: str, output: Path, preflight: Path | None = None
) -> dict[str, Any]:
    output = lexical_absolute(output)
    preflight = lexical_absolute(preflight) if preflight is not None else None
    if stage == "full" and preflight is None:
        raise ValidationError("full dry-run requires an excluded preflight path")
    if stage == "preflight" and preflight is not None:
        raise ValidationError("preflight dry-run does not accept --preflight")

    cells = []
    for cell in matrix(stage):
        directory = output / f"block-{cell.block:02d}-{cell.arm}"
        workload = [
            "stale_state_workload",
            "--result",
            str(directory / "workload-result.json"),
            "--truth",
            str(directory / "phase-truth.jsonl"),
            "--release-fd",
            "<owned-release-fd>",
            "--truth-fd",
            "<owned-truth-fd>",
        ]
        cells.append(
            {
                **asdict(cell),
                "directory": str(directory),
                "fresh_process": True,
                "workload_command": workload,
                "snapshot_relay": (
                    None
                    if cell.role == "context_control"
                    else {
                        "delay_ms": cell.delay_ms,
                        "input": "workload-authored phase_start events",
                        "output": "driver-owned shared snapshot (missing interface)",
                    }
                ),
                "policy_consumer_command": None,
                "blocked_before_live": True,
                "uvm_monitor_command": [
                    "uvm_event_monitor",
                    "--uvm-fd",
                    "<coordinator-duplicated-inherited-uvm-fd>",
                    "--target-pid",
                    "<owned-workload-pid>",
                ],
            }
        )

    return {
        "dry_run": True,
        "writes_output": False,
        "inspects_runtime": False,
        "acquires_leases": False,
        "launches_processes": False,
        "executes_gpu_work": False,
        "experiment_evidence": False,
        "live_executable": False,
        "live_blocker": LIVE_BLOCKER,
        "protocol": PROTOCOL,
        "timeline": TIMELINE,
        "stage": stage,
        "seed": SEED,
        "blocks": PREFLIGHT_BLOCKS if stage == "preflight" else FORMAL_BLOCKS,
        "cell_count": len(cells),
        "output": str(output),
        "preflight": str(preflight) if preflight is not None else None,
        "conditions": [asdict(condition) for condition in conditions()],
        "frozen_workload": {
            "allocation_bytes": ALLOCATION_BYTES,
            "region_bytes": REGION_BYTES,
            "regions": REGIONS,
            "sparse_stride_regions": SPARSE_STRIDE,
            "sparse_regions": SPARSE_REGIONS,
            "dense_launch_regions": DENSE_LAUNCH_REGIONS,
            "bootstrap_ns": BOOTSTRAP_NS,
            "phase_ns": PHASE_NS,
            "measured_phases": MEASURED_PHASES,
            "phase_sequence": [expected_phase(i) for i in range(1, 8)],
        },
        "paired_questions": paired_questions() if stage == "full" else {},
        "leases": list(LEASE_PATHS),
        "required_records": [
            "execution.json",
            "phase-truth.jsonl",
            "workload-result.json",
            "uvm-events.jsonl",
            "gpu-telemetry.csv",
            "compute-apps.jsonl",
            "kernel-monitor.log",
            "safety-before.json",
            "safety-after.json",
            "policy rows: snapshot-publications.jsonl, policy-decisions.jsonl, policy-final.json",
        ],
        "cells": cells,
        "claim_boundary": (
            "The future result is limited to this alternating managed-memory "
            "workload and the pinned 575 stack. It does not establish universal "
            "stale-state robustness or generic BPF mechanism cost."
        ),
    }


def run_cpu_delay_preflight(
    *,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    sleep: Callable[[float], None] = time.sleep,
    samples_per_delay: int = 3,
) -> dict[str, Any]:
    if type(samples_per_delay) is not int or samples_per_delay < 1:
        raise ValidationError("samples_per_delay must be positive")
    rows = []
    for delay_ms in DELAYS_MS:
        relay = DelayedSnapshotRelay(delay_ms)
        for sample in range(1, samples_per_delay + 1):
            source_ns = clock_ns()
            relay.observe(
                sequence=sample,
                phase="dense" if sample % 2 else "sparse",
                source_mono_ns=source_ns,
                scheduled_offset_ns=(sample - 1) * PHASE_NS,
            )
            if delay_ms:
                sleep(delay_ms / 1000.0)
            now_ns = clock_ns()
            publications = relay.drain(now_ns)
            if len(publications) != 1:
                raise ValidationError(
                    f"delay {delay_ms} ms did not publish exactly one due snapshot"
                )
            publication = publications[0]
            rows.append(
                {
                    "delay_ms": delay_ms,
                    "sample": sample,
                    "source_mono_ns": source_ns,
                    "eligible_mono_ns": publication["eligible_mono_ns"],
                    "published_mono_ns": publication["published_mono_ns"],
                    "age_ns": publication["published_mono_ns"] - source_ns,
                }
            )
        if relay.pending:
            raise ValidationError("CPU relay retained a pending snapshot")
    grouped = {
        delay: [row["age_ns"] for row in rows if row["delay_ms"] == delay]
        for delay in DELAYS_MS
    }
    medians = {delay: int(statistics.median(values)) for delay, values in grouped.items()}
    for delay, ages in grouped.items():
        if any(age < delay * 1_000_000 for age in ages):
            raise ValidationError(f"delay {delay} ms published before eligibility")
    distinguishable = (
        medians[100] - medians[0] >= 75_000_000
        and medians[1000] - medians[100] >= 750_000_000
    )
    if not distinguishable:
        raise ValidationError(f"candidate decision ages are not distinguishable: {medians}")
    return {
        "protocol": PROTOCOL,
        "cpu_only": True,
        "experiment_evidence": False,
        "samples_per_delay": samples_per_delay,
        "rows": rows,
        "median_age_ns": {str(delay): value for delay, value in medians.items()},
        "distinguishable": True,
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"JSON value is not an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValidationError(f"cannot read JSONL {path}: {exc}") from exc
    values = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"invalid JSONL at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ValidationError(f"non-object JSONL record at {path}:{line_number}")
        values.append(value)
    if not values:
        raise ValidationError(f"JSONL file has no records: {path}")
    return values


def _integer(value: Any, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValidationError(f"{field} must be an integer >= {minimum}")
    return value


def _number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValidationError(f"{field} must be finite{' and positive' if positive else ''}")
    return result


def _require_true(value: Any, field: str) -> None:
    if value is not True:
        raise ValidationError(f"{field} must be true")


def _close(actual: float, expected: float, field: str) -> None:
    tolerance = max(1.0e-6, abs(expected) * 1.0e-6)
    if abs(actual - expected) > tolerance:
        raise ValidationError(f"{field} differs: expected {expected}, found {actual}")


def _condition_by_arm(arm: str) -> Condition:
    matches = [condition for condition in conditions() if condition.arm == arm]
    if len(matches) != 1:
        raise ValidationError(f"unknown arm: {arm}")
    return matches[0]


def _validate_execution(path: Path, cell: MatrixCell) -> dict[str, Any]:
    execution = _load_json(path / "execution.json")
    expected = {
        "protocol": PROTOCOL,
        "timeline": TIMELINE,
        "block": cell.block,
        "arm": cell.arm,
        "implementation": cell.implementation,
        "delay_ms": cell.delay_ms,
    }
    for field, value in expected.items():
        if execution.get(field) != value:
            raise ValidationError(
                f"execution {field} differs: expected {value!r}, found {execution.get(field)!r}"
            )
    if execution.get("status") != "passed" or execution.get("complete") is not True:
        raise ValidationError("execution did not finish with passed/complete")
    if execution.get("cleanup_errors") != []:
        raise ValidationError("execution has cleanup errors")
    if execution.get("lease_paths") != list(LEASE_PATHS) or execution.get(
        "lease_mode"
    ) != "read_only_exclusive":
        raise ValidationError("execution did not retain both read-only exclusive leases")
    target_pid = _integer(execution.get("target_pid"), "target_pid", 1)
    uvm_candidates = execution.get("uvm_fd_candidates")
    if not isinstance(uvm_candidates, list) or len(uvm_candidates) != 2:
        raise ValidationError("execution UVM candidate inventory is missing")
    candidate_fds = []
    for candidate in uvm_candidates:
        if not isinstance(candidate, dict) or set(candidate) != {"source_fd", "target"}:
            raise ValidationError("execution UVM candidate schema is invalid")
        candidate_fds.append(_integer(candidate.get("source_fd"), "candidate source_fd"))
        if candidate.get("target") != "/dev/nvidia-uvm":
            raise ValidationError("execution UVM candidate target differs")
    if candidate_fds != sorted(candidate_fds) or len(set(candidate_fds)) != 2:
        raise ValidationError("execution UVM candidate FDs are not ordered and unique")
    for group in ("monitor_coverage", "cleanup"):
        values = execution.get(group)
        if not isinstance(values, dict) or not values:
            raise ValidationError(f"execution {group} record is missing")
        for name, value in values.items():
            _require_true(value, f"{group}.{name}")
    safety = execution.get("safety")
    if not isinstance(safety, dict):
        raise ValidationError("execution safety record is missing")
    for field in ("pre_valid", "post_valid", "gpu_telemetry_valid"):
        _require_true(safety.get(field), f"safety.{field}")
    if safety.get("foreign_compute_pids") != [] or safety.get(
        "new_kernel_anomalies"
    ) != []:
        raise ValidationError("execution observed foreign compute or kernel anomalies")
    return {"value": execution, "target_pid": target_pid,
            "uvm_fd_candidates": uvm_candidates}


def _validate_safety(path: Path) -> None:
    before = _load_json(path / "safety-before.json")
    after = _load_json(path / "safety-after.json")
    for label, snapshot in (("before", before), ("after", after)):
        if snapshot.get("power_limit_service") != "active":
            raise ValidationError(f"{label} power-limit service is not active")
        if abs(_number(snapshot.get("power_limit_w"), f"{label}.power_limit_w") - 400.0) > 0.01:
            raise ValidationError(f"{label} power limit is not 400 W")
        gpu = snapshot.get("gpu")
        if (
            not isinstance(gpu, dict)
            or gpu.get("index") != 0
            or gpu.get("name") != EXPECTED_GPU
            or gpu.get("driver") != EXPECTED_DRIVER
        ):
            raise ValidationError(f"{label} GPU/driver snapshot is invalid")
        if gpu.get("compute_apps") != []:
            raise ValidationError(f"{label} snapshot has compute applications")
        if _integer(snapshot.get("uvm_refcount"), f"{label}.uvm_refcount") != 0:
            raise ValidationError(f"{label} UVM reference count is nonzero")
        struct_ops = snapshot.get("struct_ops")
        if not isinstance(struct_ops, dict) or struct_ops.get("maps") != [] or struct_ops.get("links") != []:
            raise ValidationError(f"{label} struct_ops state is not empty")
        for field in ("dmesg_abnormal", "journal_abnormal", "xids"):
            if snapshot.get(field) != []:
                raise ValidationError(f"{label} safety history is not clean: {field}")


def _validate_continuous_records(path: Path, target_pid: int) -> None:
    telemetry_path = path / "gpu-telemetry.csv"
    try:
        with telemetry_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
    except OSError as exc:
        raise ValidationError(f"cannot read GPU telemetry: {exc}") from exc
    if len(rows) < 3 or len(rows[0]) < 4:
        raise ValidationError("GPU telemetry lacks a header and two samples")
    if any(len(row) != len(rows[0]) for row in rows[1:]):
        raise ValidationError("GPU telemetry contains malformed rows")

    compute = _load_jsonl(path / "compute-apps.jsonl")
    if len(compute) < 3:
        raise ValidationError("compute-app monitor has fewer than three samples")
    observed_target = False
    previous_start = 0
    shutdown_interrupted_indices = []
    for index, row in enumerate(compute):
        start = _integer(row.get("query_started_mono_ns"), "query_started_mono_ns", 1)
        finish = _integer(row.get("query_finished_mono_ns"), "query_finished_mono_ns", start)
        if start <= previous_start:
            raise ValidationError("compute-app query starts are not strictly monotonic")
        previous_start = start
        if row.get("error") is not None:
            raise ValidationError("compute-app monitor recorded an error")
        shutdown_interrupted = row.get("shutdown_interrupted")
        shutdown_signal = row.get("shutdown_signal")
        if type(shutdown_interrupted) is not bool:
            raise ValidationError("compute-app shutdown marker is invalid")
        if shutdown_interrupted:
            if shutdown_signal not in (2, 15):
                raise ValidationError("compute-app shutdown signal is invalid")
            shutdown_interrupted_indices.append(index)
        elif shutdown_signal is not None:
            raise ValidationError("compute-app shutdown signal lacks an interruption")
        pids = row.get("pids")
        if not isinstance(pids, list) or any(type(pid) is not int for pid in pids):
            raise ValidationError("compute-app PID list is invalid")
        if any(pid != target_pid for pid in pids):
            raise ValidationError("compute-app monitor observed a foreign PID")
        if shutdown_interrupted and pids:
            raise ValidationError("interrupted shutdown query reported compute PIDs")
        observed_target = observed_target or pids == [target_pid]
        if finish < start:
            raise ValidationError("compute-app query finished before it started")
        if index in (0, len(compute) - 1) and pids:
            raise ValidationError("compute-app coverage is not empty at both boundaries")
    if not observed_target:
        raise ValidationError("compute-app monitor never observed the workload PID")
    if shutdown_interrupted_indices not in ([], [len(compute) - 2]):
        raise ValidationError("compute-app shutdown interruption is not penultimate")
    if not (path / "kernel-monitor.log").is_file():
        raise ValidationError("kernel-monitor.log is missing")


def _validate_truth(path: Path) -> list[dict[str, Any]]:
    records = _load_jsonl(path / "phase-truth.jsonl")
    if len(records) != 15 or records[0].get("event") != "workload_ready":
        raise ValidationError("phase truth lacks exactly one leading workload_ready record")
    for field, expected in (
        ("protocol", PROTOCOL),
        ("timeline", TIMELINE),
        ("allocation_bytes", ALLOCATION_BYTES),
        ("regions", REGIONS),
    ):
        if records[0].get(field) != expected:
            raise ValidationError(f"workload_ready {field} differs")
    _integer(records[0].get("pid"), "workload_ready pid", 1)
    expected_boundaries = [
        (event, sequence)
        for sequence in range(1, MEASURED_PHASES + 2)
        for event in ("phase_start", "phase_end")
    ]
    actual_boundaries = [
        (row.get("event"), row.get("sequence")) for row in records[1:]
    ]
    if actual_boundaries != expected_boundaries:
        raise ValidationError("phase truth is not the exact ordered start/end trace")
    starts = {row.get("sequence"): row for row in records if row.get("event") == "phase_start"}
    ends = {row.get("sequence"): row for row in records if row.get("event") == "phase_end"}
    expected_sequences = set(range(1, MEASURED_PHASES + 2))
    if set(starts) != expected_sequences or set(ends) != expected_sequences:
        raise ValidationError("phase truth does not contain every exact start/end pair")
    if sum(row.get("event") in {"phase_start", "phase_end"} for row in records) != 14:
        raise ValidationError("phase truth contains duplicate phase boundaries")
    intervals = []
    epoch = _integer(starts[1].get("mono_ns"), "phase 1 start", 1)
    previous_end = 0
    for sequence in range(1, MEASURED_PHASES + 2):
        expected = expected_phase(sequence)
        start = starts[sequence]
        end = ends[sequence]
        for row in (start, end):
            for field in ("sequence", "phase", "measured", "scheduled_offset_ns"):
                if row.get(field) != expected[field]:
                    raise ValidationError(f"phase {sequence} {field} differs")
        start_ns = _integer(start.get("mono_ns"), f"phase {sequence} start", 1)
        end_ns = _integer(end.get("mono_ns"), f"phase {sequence} end", start_ns + 1)
        scheduled_start = epoch + expected["scheduled_offset_ns"]
        duration = BOOTSTRAP_NS if sequence == 1 else PHASE_NS
        scheduled_end = scheduled_start + duration
        if start_ns < scheduled_start or start_ns > scheduled_start + MAXIMUM_BOUNDARY_OVERRUN_NS:
            raise ValidationError(f"phase {sequence} start is outside frozen schedule")
        if end_ns < scheduled_end or end_ns > scheduled_end + MAXIMUM_BOUNDARY_OVERRUN_NS:
            raise ValidationError(f"phase {sequence} end is outside frozen schedule")
        if previous_end and start_ns < previous_end:
            raise ValidationError("host phase intervals overlap")
        previous_end = end_ns
        intervals.append(
            {
                **expected,
                "start_mono_ns": start_ns,
                "end_mono_ns": end_ns,
            }
        )
    return intervals


def _validate_workload(path: Path, intervals: list[dict[str, Any]]) -> dict[str, Any]:
    result = _load_json(path / "workload-result.json")
    expected_scalars = {
        "protocol": PROTOCOL,
        "timeline": TIMELINE,
        "allocation_bytes": ALLOCATION_BYTES,
        "region_bytes": REGION_BYTES,
        "regions": REGIONS,
        "sparse_stride_regions": SPARSE_STRIDE,
        "sparse_regions": SPARSE_REGIONS,
        "dense_launch_regions": DENSE_LAUNCH_REGIONS,
        "bootstrap_ns": BOOTSTRAP_NS,
        "phase_ns": PHASE_NS,
        "measured_phases": MEASURED_PHASES,
        "epoch_mono_ns": intervals[0]["start_mono_ns"],
    }
    for field, expected in expected_scalars.items():
        if result.get(field) != expected:
            raise ValidationError(f"workload {field} differs")
    phases = result.get("phases")
    if not isinstance(phases, list) or len(phases) != MEASURED_PHASES:
        raise ValidationError("workload does not contain six measured phases")
    total_checked = 0
    total_kernel_ms = 0.0
    for measured_index, phase in enumerate(phases, 1):
        interval = intervals[measured_index]
        expected_count = DENSE_LAUNCH_REGIONS if interval["phase"] == "dense" else SPARSE_REGIONS
        for field, expected in (
            ("measured_index", measured_index),
            ("sequence", measured_index + 1),
            ("phase", interval["phase"]),
            ("scheduled_offset_ns", interval["scheduled_offset_ns"]),
            ("start_mono_ns", interval["start_mono_ns"]),
            ("end_mono_ns", interval["end_mono_ns"]),
        ):
            if phase.get(field) != expected:
                raise ValidationError(f"workload phase {measured_index} {field} differs")
        iterations = _integer(phase.get("iterations"), "phase iterations", 1)
        checked = _integer(phase.get("checked_values"), "phase checked_values", 1)
        if checked != expected_count * iterations:
            raise ValidationError("phase checked_values does not cover every returned word")
        if _integer(phase.get("mismatches"), "phase mismatches") != 0 or phase.get(
            "first_mismatch"
        ) is not None:
            raise ValidationError("workload phase has a numerical mismatch")
        kernel_ms = _number(phase.get("kernel_ms"), "phase kernel_ms", positive=True)
        wall_ms = (interval["end_mono_ns"] - interval["start_mono_ns"]) / 1.0e6
        _close(_number(phase.get("wall_ms"), "phase wall_ms", positive=True), wall_ms, "phase wall_ms")
        total_checked += checked
        total_kernel_ms += kernel_ms
    if _integer(result.get("checked_values"), "checked_values", 1) != total_checked:
        raise ValidationError("workload total checked_values differs from phases")
    if _integer(result.get("mismatches"), "mismatches") != 0 or result.get(
        "first_mismatch"
    ) is not None:
        raise ValidationError("workload has a numerical mismatch")
    _close(
        _number(result.get("total_kernel_ms"), "total_kernel_ms", positive=True),
        total_kernel_ms,
        "total_kernel_ms",
    )
    elapsed_ms = (intervals[-1]["end_mono_ns"] - intervals[1]["start_mono_ns"]) / 1.0e6
    _close(
        _number(result.get("end_to_end_ms"), "end_to_end_ms", positive=True),
        elapsed_ms,
        "end_to_end_ms",
    )
    expected_rate = total_checked * 1000.0 / elapsed_ms
    _close(
        _number(result.get("verified_words_per_second"), "verified_words_per_second", positive=True),
        expected_rate,
        "verified_words_per_second",
    )
    return {
        "end_to_end_ms": elapsed_ms,
        "total_kernel_ms": total_kernel_ms,
        "checked_values": total_checked,
        "verified_words_per_second": expected_rate,
    }


def _latest_final_uvm(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    finals = [row for row in records if row.get("event") == "final_uvm_stats"]
    if len(finals) != 1:
        raise ValidationError("UVM monitor lacks exactly one final_uvm_stats record")
    return finals[0]


def _validate_uvm(
    path: Path,
    elapsed_ms: float,
    target_pid: int,
    execution_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    records = _load_jsonl(path / "uvm-events.jsonl")
    ready = [row for row in records if row.get("event") == "ready"]
    if len(ready) != 1 or ready[0].get("target_pid") != target_pid:
        raise ValidationError("UVM monitor lacks one ready record for the owned target")
    if ready[0].get("queue_entries") != UVM_QUEUE_ENTRIES or ready[0].get(
        "entry_bytes"
    ) != 72:
        raise ValidationError("UVM monitor ready record has an unexpected queue ABI")
    _integer(ready[0].get("uvm_fd"), "UVM inherited fd")
    source_fds = ready[0].get("candidate_source_fds")
    if (
        not isinstance(source_fds, list)
        or len(source_fds) != 2
        or any(type(value) is not int or value < 0 for value in source_fds)
        or len(set(source_fds)) != 2
    ):
        raise ValidationError("UVM monitor candidate source FDs are invalid")
    if ready[0].get("candidate_targets") != [
        "/dev/nvidia-uvm", "/dev/nvidia-uvm"
    ]:
        raise ValidationError("UVM monitor candidate targets differ")
    if source_fds != [value["source_fd"] for value in execution_candidates] or ready[
        0
    ].get("candidate_targets") != [value["target"] for value in execution_candidates]:
        raise ValidationError("UVM monitor candidates differ from runner-owned inventory")
    selected_source = _integer(ready[0].get("selected_source_fd"), "selected_source_fd")
    rejected_source = _integer(ready[0].get("rejected_source_fd"), "rejected_source_fd")
    if (
        {selected_source, rejected_source} != set(source_fds)
        or ready[0].get("rejected_status") != 0x00000016
    ):
        raise ValidationError(
            "UVM monitor did not select exactly one driver-validated VA-space FD"
        )
    final = _latest_final_uvm(records)
    fields = (
        "gpu_faults",
        "migrations",
        "migrated_bytes",
        "prefetch_migrations",
        "prefetch_bytes",
        "thrashing_events",
        "eviction_events",
        "fault_buffer_overflows",
        "dropped_gpu_faults",
        "dropped_migrations",
        "dropped_thrashing",
        "dropped_evictions",
    )
    values = {field: _integer(final.get(field), field) for field in fields}
    for field in ("gpu_faults", "migrations", "migrated_bytes"):
        if values[field] == 0:
            raise ValidationError(f"UVM mechanism did not engage: {field} is zero")
    if values["prefetch_migrations"] > values["migrations"] or values[
        "prefetch_bytes"
    ] > values["migrated_bytes"]:
        raise ValidationError("prefetch migration totals exceed all migrations")
    for field in (
        "fault_buffer_overflows",
        "dropped_gpu_faults",
        "dropped_migrations",
        "dropped_thrashing",
        "dropped_evictions",
    ):
        if values[field] != 0:
            raise ValidationError(f"UVM monitor lost or overflowed events: {field}")
    seconds = elapsed_ms / 1000.0
    return {
        **values,
        "gpu_faults_per_second": values["gpu_faults"] / seconds,
        "migrated_bytes_per_second": values["migrated_bytes"] / seconds,
    }


def _truth_at(intervals: list[dict[str, Any]], timestamp_ns: int) -> str:
    matches = [
        interval["phase"]
        for interval in intervals
        if interval["start_mono_ns"] <= timestamp_ns < interval["end_mono_ns"]
    ]
    if len(matches) != 1:
        raise ValidationError("policy decision does not join to one host-truth interval")
    return matches[0]


def _validate_verifier_log(path: Path, implementation: str) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"cannot read verifier log: {exc}") from exc
    if lines[:2] != ["load_error=0", "program_count=2"]:
        raise ValidationError("verifier log load result/program count is invalid")
    headings = [index for index, line in enumerate(lines) if line.startswith("program=")]
    expected_names = {
        "stale_state_v1_diagnostic_observer",
        "stale_state_prefetch_v1",
    }
    if len(headings) != 2 or {lines[index][len("program="):] for index in headings} != expected_names:
        raise ValidationError("verifier log program inventory is invalid")
    bodies: dict[str, list[str]] = {}
    for position, start in enumerate(headings):
        end = headings[position + 1] if position + 1 < len(headings) else len(lines)
        bodies[lines[start][len("program="):]] = [
            line for line in lines[start + 1:end] if line.strip()
        ]
    if not bodies["stale_state_v1_diagnostic_observer"]:
        raise ValidationError("observer verifier transcript is empty")
    policy_body = bodies["stale_state_prefetch_v1"]
    if implementation == "bpf" and not policy_body:
        raise ValidationError("BPF policy verifier transcript is empty")
    if implementation == "native" and policy_body:
        raise ValidationError("native arm unexpectedly loaded the BPF policy")


def _validate_policy(
    path: Path, cell: MatrixCell, intervals: list[dict[str, Any]], target_pid: int
) -> dict[str, Any]:
    assert cell.implementation is not None and cell.delay_ms is not None
    import observer_protocol

    _validate_verifier_log(path / "verifier.log", cell.implementation)
    try:
        observer_stderr = (path / "policy-observer.stderr.log").read_text(
            encoding="utf-8"
        )
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"cannot read observer stderr: {exc}") from exc
    if any(word in observer_stderr.lower() for word in ("fatal", "segmentation fault")):
        raise ValidationError("observer stderr contains a fatal failure")
    observer_path = path / "policy-observer.jsonl"
    try:
        raw_observer = observer_protocol.parse_jsonl(
            observer_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"cannot read raw observer stream: {exc}") from exc
    observed = observer_protocol.validate_records(
        raw_observer,
        expected_pid=target_pid,
        implementation=cell.implementation,
    )
    publications = _load_jsonl(path / "snapshot-publications.jsonl")
    if any(row.get("event") != "snapshot_published" for row in publications):
        raise ValidationError("snapshot publication stream has an unknown record")
    if len(publications) != MEASURED_PHASES + 1:
        raise ValidationError("snapshot publication count differs from phase count")
    by_sequence: dict[int, dict[str, Any]] = {}
    for publication in publications:
        sequence = _integer(publication.get("sequence"), "snapshot sequence", 1)
        if sequence in by_sequence:
            raise ValidationError("duplicate snapshot publication sequence")
        if sequence > MEASURED_PHASES + 1:
            raise ValidationError("snapshot sequence is outside the phase trace")
        interval = intervals[sequence - 1]
        for field, expected in (
            ("publisher", "shared_driver_snapshot"),
            ("consumer_implementation", cell.implementation),
            ("phase", interval["phase"]),
            ("source_mono_ns", interval["start_mono_ns"]),
            ("scheduled_offset_ns", interval["scheduled_offset_ns"]),
            ("delay_ns", cell.delay_ms * 1_000_000),
        ):
            if publication.get(field) != expected:
                raise ValidationError(f"snapshot {sequence} {field} differs")
        eligible = interval["start_mono_ns"] + cell.delay_ms * 1_000_000
        if publication.get("eligible_mono_ns") != eligible:
            raise ValidationError("snapshot eligibility does not equal source plus delay")
        published = _integer(publication.get("published_mono_ns"), "published_mono_ns", eligible)
        status_observed = _integer(
            publication.get("status_observed_mono_ns"),
            "status_observed_mono_ns",
            published,
        )
        latest = eligible + MAXIMUM_BOUNDARY_OVERRUN_NS
        if published > latest or status_observed > latest:
            raise ValidationError("snapshot publication/status observation was too late")
        by_sequence[sequence] = publication
    if set(by_sequence) != set(range(1, MEASURED_PHASES + 2)):
        raise ValidationError("snapshot sequence is incomplete")

    decisions = _load_jsonl(path / "policy-decisions.jsonl")
    if any(row.get("event") != "policy_decision" for row in decisions):
        raise ValidationError("policy decision stream has an unknown record")
    if not decisions:
        raise ValidationError("policy has no decision records")
    if decisions != observed["decisions"]:
        raise ValidationError("normalized decisions differ from the raw observer stream")
    dense_actions = 0
    discard_actions = 0
    wrong = 0
    ages = []
    previous_decision_ns = 0
    expected_read_path = (
        "driver_native_read_only_context"
        if cell.implementation == "native"
        else "driver_bpf_read_only_context"
    )
    for decision_sequence, decision in enumerate(decisions, 1):
        if decision.get("implementation") != cell.implementation:
            raise ValidationError("decision implementation differs from cell")
        if decision.get("decision_sequence") != decision_sequence:
            raise ValidationError("policy decision sequence is not contiguous")
        if decision.get("snapshot_read_path") != expected_read_path:
            raise ValidationError("policy decision used the wrong snapshot read path")
        decision_ns = _integer(decision.get("decision_mono_ns"), "decision_mono_ns", 1)
        if decision_ns < previous_decision_ns:
            raise ValidationError("policy decisions are not monotonic")
        previous_decision_ns = decision_ns
        sequence = _integer(decision.get("snapshot_sequence"), "decision snapshot_sequence", 1)
        publication = by_sequence.get(sequence)
        if publication is None:
            raise ValidationError("policy decision names an unpublished snapshot")
        if decision_ns < publication["published_mono_ns"]:
            raise ValidationError("policy decision predates snapshot publication")
        observed_phase = decision.get("snapshot_phase")
        if observed_phase != publication["phase"]:
            raise ValidationError("policy decision snapshot phase differs from publication")
        action = decision.get("action")
        if decision.get("effect_source") != "driver_diagnostic":
            raise ValidationError("policy effect is not a matched driver diagnostic")
        fault_page = _integer(decision.get("fault_page_index"), "fault_page_index")
        maximum_first = _integer(decision.get("legal_max_first"), "legal_max_first")
        maximum_outer = _integer(decision.get("legal_max_outer"), "legal_max_outer", 1)
        if not (maximum_first <= fault_page < maximum_outer <= REGIONS):
            raise ValidationError("policy decision legal maximum does not contain the fault")
        if observed_phase == "dense":
            if action != "prefetch_max" or decision.get("effect") != "prefetch":
                raise ValidationError("dense snapshot did not produce real max-prefetch effect")
            first = _integer(decision.get("output_first"), "output_first")
            outer = _integer(decision.get("output_outer"), "output_outer")
            if first != maximum_first or outer != maximum_outer:
                raise ValidationError("dense prefetch output is not the full legal maximum")
            dense_actions += 1
        else:
            if action != "discard_prefetch" or decision.get("effect") != "discard":
                raise ValidationError("sparse snapshot did not produce real discard effect")
            if decision.get("output_first") != 0 or decision.get("output_outer") != 0:
                raise ValidationError("sparse discard output is not the empty region")
            discard_actions += 1
        age = decision_ns - publication["source_mono_ns"]
        if age < 0 or decision.get("decision_age_ns") != age:
            raise ValidationError("decision age was not recorded from source monotonic time")
        ages.append(age)
        truth = _truth_at(intervals, decision_ns)
        wrong += int(truth != observed_phase)

    final = _load_json(path / "policy-final.json")
    if final.get("event") != "final_policy_stats" or final.get(
        "implementation"
    ) != cell.implementation:
        raise ValidationError("policy-final identity is invalid")
    expected_counts = {
        "snapshot_updates": MEASURED_PHASES + 1,
        "callback_invocations": len(decisions),
        "snapshot_read_attempts": len(decisions),
        "snapshot_read_successes": len(decisions),
        "native_callback_invocations": (
            len(decisions) if cell.implementation == "native" else 0
        ),
        "bpf_callback_invocations": (
            len(decisions) if cell.implementation == "bpf" else 0
        ),
        "decision_requests": len(decisions),
        "decisions": len(decisions),
        "decision_records": len(decisions),
        "effect_requests": len(decisions),
        "effect_records": len(decisions),
        "selected_diagnostics": len(decisions),
        "finished_diagnostics": len(decisions),
        "dense_prefetch_decisions": dense_actions,
        "discarded_prefetch_decisions": discard_actions,
    }
    for field, expected in expected_counts.items():
        if final.get(field) != expected:
            raise ValidationError(f"policy final counter differs: {field}")
    for field in (
        "snapshot_rejections",
        "missing_snapshot_decisions",
        "invalid_snapshot_decisions",
        "request_errors",
        "effect_errors",
        "decision_record_drops",
        "effect_record_drops",
    ):
        if _integer(final.get(field), field) != 0:
            raise ValidationError(f"policy reported an invalid/lost action: {field}")
    if dense_actions == 0 or discard_actions == 0:
        raise ValidationError("both dense-prefetch and sparse-discard actions must engage")
    return {
        "decisions": len(decisions),
        "dense_prefetch_decisions": dense_actions,
        "discarded_prefetch_decisions": discard_actions,
        "decision_age_ns_mean": statistics.fmean(ages),
        "decision_age_ns_median": statistics.median(ages),
        "decision_age_ns_max": max(ages),
        "wrong_phase_decisions": wrong,
        "wrong_phase_fraction": wrong / len(decisions),
        "publication_lateness_ns": [
            publication["published_mono_ns"] - publication["eligible_mono_ns"]
            for publication in publications
        ],
    }


def validate_cell(path: Path, cell: MatrixCell) -> dict[str, Any]:
    if not path.is_dir():
        raise ValidationError(f"cell directory is missing: {path}")
    execution = _validate_execution(path, cell)
    _validate_safety(path)
    _validate_continuous_records(path, execution["target_pid"])
    intervals = _validate_truth(path)
    workload = _validate_workload(path, intervals)
    uvm = _validate_uvm(
        path,
        workload["end_to_end_ms"],
        execution["target_pid"],
        execution["uvm_fd_candidates"],
    )
    policy_paths = tuple(path / name for name in POLICY_ARTIFACT_NAMES)
    if cell.role == "context_control":
        if any(policy_path.exists() for policy_path in policy_paths):
            raise ValidationError("default-UVM control unexpectedly has policy records")
        policy = None
    else:
        policy = _validate_policy(path, cell, intervals, execution["target_pid"])
    return {
        "valid": True,
        "block": cell.block,
        "arm": cell.arm,
        "implementation": cell.implementation,
        "delay_ms": cell.delay_ms,
        "workload": workload,
        "uvm": uvm,
        "policy": policy,
    }


def _ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        raise ValidationError("comparison metric cannot form a finite positive ratio")
    return numerator / denominator


def _comparison_metrics(numerator: dict[str, Any], denominator: dict[str, Any]) -> dict[str, float]:
    return {
        "end_to_end_ratio": _ratio(
            numerator["workload"]["end_to_end_ms"],
            denominator["workload"]["end_to_end_ms"],
        ),
        "verified_throughput_ratio": _ratio(
            numerator["workload"]["verified_words_per_second"],
            denominator["workload"]["verified_words_per_second"],
        ),
        "gpu_fault_rate_ratio": _ratio(
            numerator["uvm"]["gpu_faults_per_second"],
            denominator["uvm"]["gpu_faults_per_second"],
        ),
        "migration_rate_ratio": _ratio(
            numerator["uvm"]["migrated_bytes_per_second"],
            denominator["uvm"]["migrated_bytes_per_second"],
        ),
    }


def _validate_manifest(path: Path, stage: str) -> tuple[list[MatrixCell], dict[str, Any]]:
    manifest = _load_json(path / "campaign.json")
    planned = matrix(stage)
    expected_order = [asdict(cell) for cell in planned]
    blocks = FORMAL_BLOCKS if stage == "full" else PREFLIGHT_BLOCKS
    for field, expected in (
        ("protocol", PROTOCOL),
        ("timeline", TIMELINE),
        ("stage", stage),
        ("seed", SEED),
        ("blocks", blocks),
        ("complete", True),
        ("order", expected_order),
        ("completed", expected_order),
    ):
        if manifest.get(field) != expected:
            raise ValidationError(f"campaign manifest {field} differs")
    return planned, manifest


def validate_preflight(path: Path) -> dict[str, Any]:
    planned, _ = _validate_manifest(path, "preflight")
    results = []
    for cell in planned:
        directory = path / f"block-{cell.block:02d}-{cell.arm}"
        results.append(validate_cell(directory, cell))
    return {
        "run_status": "valid",
        "protocol": PROTOCOL,
        "stage": "preflight",
        "cells": results,
    }


def validate_campaign(path: Path) -> dict[str, Any]:
    planned, manifest = _validate_manifest(path, "full")
    preflight_value = manifest.get("preflight")
    if not isinstance(preflight_value, str) or not preflight_value:
        raise ValidationError("formal campaign has no excluded preflight reference")
    preflight_path = Path(preflight_value)
    if not preflight_path.is_absolute():
        raise ValidationError("formal campaign preflight reference is not absolute")
    if lexical_absolute(preflight_path) == lexical_absolute(path):
        raise ValidationError("formal campaign cannot use itself as its preflight")
    preflight = validate_preflight(preflight_path)

    results = {}
    ordered_results = []
    for cell in planned:
        directory = path / f"block-{cell.block:02d}-{cell.arm}"
        result = validate_cell(directory, cell)
        results[(cell.block, cell.arm)] = result
        ordered_results.append(result)

    mechanism = []
    information = []
    for pair in paired_questions()["mechanism_cost"]:
        native = results[(pair["block"], pair["native"])]
        bpf = results[(pair["block"], pair["bpf"])]
        mechanism.append({**pair, **_comparison_metrics(bpf, native)})
    for pair in paired_questions()["information_cost"]:
        fresh = results[(pair["block"], pair["fresh"])]
        delayed = results[(pair["block"], pair["delayed"])]
        metrics = _comparison_metrics(delayed, fresh)
        wrong_delta = (
            delayed["policy"]["wrong_phase_fraction"]
            - fresh["policy"]["wrong_phase_fraction"]
        )
        degraded = (
            wrong_delta > 0
            and metrics["verified_throughput_ratio"] < 1
            and (
                metrics["gpu_fault_rate_ratio"] > 1
                or metrics["migration_rate_ratio"] > 1
            )
        )
        information.append(
            {
                **pair,
                **metrics,
                "wrong_phase_fraction_delta": wrong_delta,
                "observed_degradation": degraded,
                "negative_result": not degraded,
            }
        )
    return {
        "run_status": "valid",
        "protocol": PROTOCOL,
        "preflight": preflight,
        "cells": ordered_results,
        "mechanism_cost": mechanism,
        "information_cost": information,
        "negative_results_retained": sum(item["negative_result"] for item in information),
        "interpretation_boundary": (
            "Ratios are paired within three blocks and describe only the frozen "
            "workload. A negative row remains valid; no significance claim is "
            "inferred from three blocks."
        ),
    }
