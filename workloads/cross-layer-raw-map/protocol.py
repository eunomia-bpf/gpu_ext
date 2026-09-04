#!/usr/bin/env python3
"""Pure planning and validation for the cross-layer raw-record experiment."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable

PROTOCOL = "cross-layer-raw-record-v1"
SCHEMA = 1
SEED = 20260903
BLOCK_DIM = 128
RING_CAPACITY = 4
RECORD_U64_FIELDS = 7
RECORD_SIZE = RECORD_U64_FIELDS * 8


class EvidenceError(ValueError):
    """Raised when a record cannot support the declared evidence boundary."""


@dataclass(frozen=True)
class Arm:
    name: str
    threads: int
    launches: int
    expect_drop_rejection: bool

    @property
    def blocks(self) -> int:
        if self.threads <= 0 or self.threads % BLOCK_DIM:
            raise EvidenceError("thread count must be a positive multiple of BLOCK_DIM")
        return self.threads // BLOCK_DIM

    @property
    def callbacks(self) -> int:
        return self.threads * self.launches

    @property
    def retained_per_thread(self) -> int:
        return min(self.launches, RING_CAPACITY)

    @property
    def retained_records(self) -> int:
        return self.threads * self.retained_per_thread

    @property
    def full_drops(self) -> int:
        return self.callbacks - self.retained_records


ARMS = (
    Arm("small", 256, 3, False),
    Arm("large", 2048, 3, False),
    Arm("overflow_negative", 256, 6, True),
)
ARM_BY_NAME = {arm.name: arm for arm in ARMS}

RECORD_FIELDS = (
    "sequence",
    "block_x",
    "block_y",
    "block_z",
    "thread_x",
    "thread_y",
    "thread_z",
)


def blocks_for(mode: str) -> int:
    if mode == "preflight":
        return 1
    if mode == "full":
        return 5
    raise EvidenceError(f"unsupported mode: {mode}")


def campaign_order(mode: str) -> list[dict[str, Any]]:
    """Return seeded complete blocks; every cell is a fresh process pair."""
    rng = random.Random(SEED)
    result: list[dict[str, Any]] = []
    for block in range(1, blocks_for(mode) + 1):
        block_arms = list(ARMS)
        rng.shuffle(block_arms)
        for order, arm in enumerate(block_arms, 1):
            result.append({"block": block, "order": order, **asdict(arm)})
    return result


def dry_run_plan(mode: str, output: Path, runtime_build: Path,
                 preflight: Path | None = None) -> dict[str, Any]:
    """Describe the exact matrix without inspecting artifacts or touching output."""
    cells = []
    for item in campaign_order(mode):
        cell_name = (
            f"block-{item['block']:02d}-order-{item['order']:02d}-{item['name']}"
        )
        cells.append({
            **item,
            "directory": str(output.absolute() / cell_name),
            "fresh_native_process": True,
            "fresh_instrumented_process": True,
            "fresh_private_bpftime_segment": True,
        })
    expected = blocks_for(mode) * len(ARMS)
    if len(cells) != expected:
        raise EvidenceError("campaign matrix is incomplete")
    for block in range(1, blocks_for(mode) + 1):
        names = [cell["name"] for cell in cells if cell["block"] == block]
        if len(names) != len(ARMS) or set(names) != set(ARM_BY_NAME):
            raise EvidenceError("campaign block is not a complete arm permutation")
    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "dry_run": True,
        "executes_gpu_work": False,
        "writes_output": False,
        "inspects_runtime_artifacts": False,
        "mode": mode,
        "seed": SEED,
        "blocks": blocks_for(mode),
        "cell_count": len(cells),
        "block_dim": BLOCK_DIM,
        "ring_capacity_per_thread": RING_CAPACITY,
        "runtime_build": str(runtime_build.absolute()),
        "output": str(output.absolute()),
        "preflight": str(preflight.absolute()) if preflight else None,
        "cells": cells,
        "positive_gate": (
            "native CUDA truth equals instrumented CUDA truth; every device-BPF raw "
            "tuple and every aggregate shard matches that truth; all drop counters are zero"
        ),
        "negative_gate": (
            "the six-launch arm must retain exactly four tuples per thread, report every "
            "omitted tuple as a full drop, and be rejected as incomplete raw-stream evidence"
        ),
        "claim_boundary": (
            "This is a functional current-ABI test of raw GPU-to-host records plus an "
            "aggregated control. It is not a latency/bandwidth result, shared-memory-shard "
            "test, transparent-placement result, or proof for arbitrary data structures."
        ),
    }


def json_events(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("event"), str):
            result.append(value)
    return result


def one_event(events: Iterable[dict[str, Any]], name: str) -> dict[str, Any]:
    found = [event for event in events if event.get("event") == name]
    if len(found) != 1:
        raise EvidenceError(f"expected exactly one {name} event, found {len(found)}")
    return found[0]


def record_tuple(event: dict[str, Any], expected_event: str) -> tuple[int, ...]:
    if event.get("event") != expected_event:
        raise EvidenceError(f"unexpected record event: {event.get('event')}")
    values = tuple(event.get(field) for field in RECORD_FIELDS)
    if any(type(value) is not int or value < 0 for value in values):
        raise EvidenceError("record fields must be non-negative integers")
    return values  # type: ignore[return-value]


def expected_records(arm: Arm, retained_only: bool = False) -> list[tuple[int, ...]]:
    max_sequence = arm.retained_per_thread if retained_only else arm.launches
    return [
        (sequence, linear // BLOCK_DIM, 0, 0, linear % BLOCK_DIM, 0, 0)
        for sequence in range(1, max_sequence + 1)
        for linear in range(arm.threads)
    ]


def expected_aggregate(arm: Arm) -> dict[str, int]:
    sequence_sum_per_thread = arm.launches * (arm.launches + 1) // 2
    block_sum_per_launch = BLOCK_DIM * arm.blocks * (arm.blocks - 1) // 2
    thread_sum_per_block = BLOCK_DIM * (BLOCK_DIM - 1) // 2
    return {
        "thread_slots": arm.threads,
        "checked_slots": arm.threads,
        "callbacks": arm.callbacks,
        "sequence_sum": arm.threads * sequence_sum_per_thread,
        "block_x_sum": arm.launches * block_sum_per_launch,
        "thread_x_sum": arm.launches * arm.blocks * thread_sum_per_block,
        "slot_mismatches": 0,
    }


def expected_target_summary(arm: Arm) -> dict[str, int]:
    return {
        "threads": arm.threads,
        "blocks": arm.blocks,
        "threads_per_block": BLOCK_DIM,
        "launches": arm.launches,
        "truth_records": arm.callbacks,
        "checked_records": arm.callbacks,
        "mismatches": 0,
    }


def require_fields(event: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for key, value in expected.items():
        if event.get(key) != value:
            raise EvidenceError(
                f"{label}.{key}={event.get(key)!r}, expected {value!r}"
            )


def validate_target(events: list[dict[str, Any]], arm: Arm,
                    label: str) -> list[tuple[int, ...]]:
    summary = one_event(events, "cuda_summary")
    require_fields(summary, expected_target_summary(arm), f"{label}.cuda_summary")
    records = [record_tuple(event, "cuda_truth") for event in events
               if event.get("event") == "cuda_truth"]
    expected = expected_records(arm)
    if records != expected:
        raise EvidenceError(f"{label} CUDA truth does not exactly match launch geometry")
    if len(set(records)) != len(records):
        raise EvidenceError(f"{label} CUDA truth contains duplicate tuples")
    return records


def validate_probe(events: list[dict[str, Any]], arm: Arm) -> dict[str, Any]:
    ready = one_event(events, "ready")
    require_fields(ready, {
        "thread_slots": arm.threads,
        "threads_per_block": BLOCK_DIM,
        "launches": arm.launches,
        "ring_capacity_per_thread": RING_CAPACITY,
    }, "probe.ready")

    aggregate = one_event(events, "aggregate_summary")
    require_fields(aggregate, expected_aggregate(arm), "probe.aggregate_summary")

    ring = one_event(events, "ring_summary")
    require_fields(ring, {
        "value_size": RECORD_SIZE,
        "entries_per_thread": RING_CAPACITY,
        "allocated_thread_slots": arm.threads,
        "committed_records": arm.retained_records,
        "collected_records": arm.retained_records,
        "pending_records": 0,
        "oob_drops": 0,
        "full_drops": arm.full_drops,
        "bad_size_drops": 0,
        "other_drops": 0,
        "dirty_slots": 0,
        "callback_records": arm.retained_records,
        "malformed_records": 0,
    }, "probe.ring_summary")
    if ring["committed_records"] + ring["full_drops"] != arm.callbacks:
        raise EvidenceError("ring accounting does not cover every aggregate callback")

    raw = [record_tuple(event, "raw_record") for event in events
           if event.get("event") == "raw_record"]
    if len(raw) != arm.retained_records or len(set(raw)) != len(raw):
        raise EvidenceError("raw stream count is wrong or contains duplicates")
    expected_raw = expected_records(arm, retained_only=True)
    if sorted(raw) != sorted(expected_raw):
        raise EvidenceError("raw stream does not exactly match the retained CUDA tuples")

    if arm.expect_drop_rejection:
        if arm.full_drops <= 0 or raw == expected_records(arm):
            raise EvidenceError("overflow negative did not produce a provably incomplete stream")
        disposition = "rejected_incomplete_raw_stream"
    else:
        if arm.full_drops != 0 or sorted(raw) != sorted(expected_records(arm)):
            raise EvidenceError("positive raw stream is incomplete")
        disposition = "accepted_complete_raw_stream"
    return {
        "raw_records": len(raw),
        "aggregate_callbacks": aggregate["callbacks"],
        "full_drops": ring["full_drops"],
        "evidence_disposition": disposition,
        "negative_gate_passed": arm.expect_drop_rejection,
    }


def validate_cell_logs(native_log: Path, instrumented_log: Path,
                       probe_log: Path, arm: Arm) -> dict[str, Any]:
    native = validate_target(json_events(native_log), arm, "native")
    instrumented = validate_target(json_events(instrumented_log), arm, "instrumented")
    if native != instrumented:
        raise EvidenceError("instrumentation changed CUDA ground-truth tuples")
    probe = validate_probe(json_events(probe_log), arm)
    return {
        "arm": arm.name,
        "threads": arm.threads,
        "launches": arm.launches,
        "native_truth_records": len(native),
        "instrumented_truth_records": len(instrumented),
        "cuda_truth_exact": True,
        **probe,
    }


def validate_campaign_manifest(path: Path, mode: str) -> dict[str, Any]:
    blocks = blocks_for(mode)
    expected_cells = blocks * len(ARMS)
    manifest = json.loads((path / "manifest.json").read_text())
    if (manifest.get("schema") != SCHEMA or manifest.get("protocol") != PROTOCOL
            or manifest.get("mode") != mode or manifest.get("status") != "passed"
            or manifest.get("cell_count") != expected_cells
            or manifest.get("positive_cells") != blocks * 2
            or manifest.get("negative_drop_gates") != blocks):
        raise EvidenceError(f"expected one passed protocol-compatible {mode} campaign")
    cells = manifest.get("cells")
    if not isinstance(cells, list) or len(cells) != expected_cells:
        raise EvidenceError(f"{mode} campaign does not contain every planned cell")
    expected_order = campaign_order(mode)
    dispositions: dict[str, str] = {}
    for recorded, planned in zip(cells, expected_order, strict=True):
        arm = ARM_BY_NAME[planned["name"]]
        expected_directory = (
            f"block-{planned['block']:02d}-order-{planned['order']:02d}-{arm.name}"
        )
        if (recorded.get("block") != planned["block"]
                or recorded.get("order") != planned["order"]
                or recorded.get("arm") != arm.name
                or recorded.get("directory") != expected_directory):
            raise EvidenceError("campaign cell order/directory does not match the frozen plan")
        directory = path / expected_directory
        cell = json.loads((directory / "cell.json").read_text())
        if (cell.get("schema") != SCHEMA or cell.get("protocol") != PROTOCOL
                or cell.get("status") != "passed" or cell.get("arm") != arm.name
                or cell.get("cleanup_errors") != []
                or cell.get("owned_group_survivors") != {}
                or cell.get("private_segment_removed") is not True):
            raise EvidenceError("campaign cell lifecycle evidence is incomplete")
        revalidated = validate_cell_logs(
            directory / "native.log", directory / "instrumented.log",
            directory / "probe.log", arm,
        )
        if cell.get("validation") != revalidated:
            raise EvidenceError("campaign cell result does not match its raw logs")
        for key, value in revalidated.items():
            if recorded.get(key) != value:
                raise EvidenceError(f"campaign manifest disagrees with raw cell field {key}")
        dispositions[arm.name] = revalidated["evidence_disposition"]
    if dispositions != {
            "small": "accepted_complete_raw_stream",
            "large": "accepted_complete_raw_stream",
            "overflow_negative": "rejected_incomplete_raw_stream"}:
        raise EvidenceError("campaign evidence dispositions are incomplete")
    return manifest


def validate_preflight_manifest(path: Path) -> dict[str, Any]:
    return validate_campaign_manifest(path, "preflight")
