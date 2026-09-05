#!/usr/bin/env python3
"""CPU-only tests for the stale-state protocol and fail-closed boundary."""

from __future__ import annotations

import contextlib
import errno
import io
import json
import os
import subprocess
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

import coordinator
import live_runner
import observer_protocol
import protocol
import run_module_lifecycle
import run_study


EPOCH_NS = 10_000_000_000
TARGET_PID = 4242


class FastClock:
    def __init__(self, start_ns: int = EPOCH_NS):
        self.now_ns = start_ns

    def clock_ns(self) -> int:
        self.now_ns += 1_000
        return self.now_ns

    def sleep(self, seconds: float) -> None:
        self.now_ns += max(1, int(seconds * 1.0e9))


def status_text(**changes: int | str) -> str:
    values: dict[str, int | str] = {
        field: 0 for field in coordinator.STATUS_FIELDS
    }
    values.update(abi_version=1, mode="off")
    values.update(changes)
    return " ".join(f"{field}={values[field]}" for field in coordinator.STATUS_FIELDS)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, separators=(",", ":")) + "\n" for value in values),
        encoding="utf-8",
    )


def truth_intervals() -> list[dict]:
    result = []
    for sequence in range(1, protocol.MEASURED_PHASES + 2):
        expected = protocol.expected_phase(sequence)
        start = EPOCH_NS + expected["scheduled_offset_ns"]
        duration = protocol.BOOTSTRAP_NS if sequence == 1 else protocol.PHASE_NS
        result.append({**expected, "start_mono_ns": start, "end_mono_ns": start + duration})
    return result


def truth_fd_payload(*, pid: int = TARGET_PID) -> bytes:
    records = [
        {
            "event": "workload_ready",
            "pid": pid,
            "protocol": protocol.PROTOCOL,
            "timeline": protocol.TIMELINE,
            "allocation_bytes": protocol.ALLOCATION_BYTES,
            "regions": protocol.REGIONS,
        }
    ]
    for interval in truth_intervals():
        common = {
            "sequence": interval["sequence"],
            "phase": interval["phase"],
            "measured": interval["measured"],
            "scheduled_offset_ns": interval["scheduled_offset_ns"],
        }
        records.append(
            {"event": "phase_start", **common, "mono_ns": interval["start_mono_ns"]}
        )
        records.append(
            {"event": "phase_end", **common, "mono_ns": interval["end_mono_ns"]}
        )
    return b"".join(
        json.dumps(record, separators=(",", ":")).encode("utf-8") + b"\n"
        for record in records
    )


def pipe_with_payload(payload: bytes) -> int:
    read_fd, write_fd = os.pipe()
    try:
        view = memoryview(payload)
        while view:
            written = os.write(write_fd, view)
            view = view[written:]
    finally:
        os.close(write_fd)
    return read_fd


class ScriptedClock:
    def __init__(self, values: list[int]):
        self.values = iter(values)

    def clock_ns(self) -> int:
        try:
            return next(self.values)
        except StopIteration as exc:
            raise AssertionError("scripted clock was exhausted") from exc

    def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            raise AssertionError("scripted clock received a non-positive sleep")


def policy_clock_values(delay_ms: int) -> list[int]:
    delay_ns = delay_ms * 1_000_000
    values = []
    for interval in truth_intervals():
        source = interval["start_mono_ns"]
        eligible = source + delay_ns
        values.extend((source + 1_000, source + 2_000))
        if delay_ns:
            values.append(eligible)
            write_base = eligible
        else:
            write_base = source + 2_000
        values.extend(
            (
                write_base + 1_000,
                write_base + 2_000,
                write_base + 3_000,
                write_base + 4_000,
                interval["end_mono_ns"] + 1_000,
            )
        )
    return values


def safety_snapshot() -> dict:
    return {
        "power_limit_service": "active",
        "power_limit_w": 400.0,
        "gpu": {
            "index": 0,
            "name": protocol.EXPECTED_GPU,
            "driver": protocol.EXPECTED_DRIVER,
            "compute_apps": [],
        },
        "uvm_refcount": 0,
        "struct_ops": {"maps": [], "links": []},
        "dmesg_abnormal": [],
        "journal_abnormal": [],
        "xids": [],
    }


def policy_records(cell: protocol.MatrixCell, intervals: list[dict]) -> tuple[list[dict], list[dict], dict]:
    assert cell.delay_ms is not None and cell.implementation is not None
    delay_ns = cell.delay_ms * 1_000_000
    publications = []
    for interval in intervals:
        eligible = interval["start_mono_ns"] + delay_ns
        publications.append(
            {
                "event": "snapshot_published",
                "publisher": "shared_driver_snapshot",
                "consumer_implementation": cell.implementation,
                "sequence": interval["sequence"],
                "phase": interval["phase"],
                "source_mono_ns": interval["start_mono_ns"],
                "scheduled_offset_ns": interval["scheduled_offset_ns"],
                "eligible_mono_ns": eligible,
                "published_mono_ns": eligible,
                "status_observed_mono_ns": eligible,
                "delay_ns": delay_ns,
            }
        )

    decisions = []

    def add_decision(interval: dict, snapshot: dict, timestamp_ns: int) -> None:
        dense = snapshot["phase"] == "dense"
        decisions.append(
            {
                "event": "policy_decision",
                "implementation": cell.implementation,
                "decision_mono_ns": timestamp_ns,
                "snapshot_sequence": snapshot["sequence"],
                "snapshot_phase": snapshot["phase"],
                "decision_age_ns": timestamp_ns - snapshot["source_mono_ns"],
                "action": "prefetch_max" if dense else "discard_prefetch",
                "effect": "prefetch" if dense else "discard",
                "effect_source": "driver_diagnostic",
                "fault_page_index": 7,
                "legal_max_first": 0,
                "legal_max_outer": 16,
                "output_first": 0,
                "output_outer": 16 if dense else 0,
                "observer_mono_ns": timestamp_ns + 100,
                "target_tgid": TARGET_PID,
            }
        )

    for interval in intervals[1:]:
        sequence = interval["sequence"]
        if cell.delay_ms:
            # Before the new snapshot is eligible, the consumer really uses the
            # preceding published phase. The analyzer independently joins this
            # timestamp to host truth and counts the wrong-phase decision.
            add_decision(
                interval,
                publications[sequence - 2],
                interval["start_mono_ns"] + 1_000_000,
            )
        add_decision(
            interval,
            publications[sequence - 1],
            interval["start_mono_ns"] + delay_ns + 1_000_000,
        )

    for decision_sequence, decision in enumerate(decisions, 1):
        decision["decision_sequence"] = decision_sequence
        decision["snapshot_read_path"] = (
            "driver_native_read_only_context"
            if cell.implementation == "native"
            else "driver_bpf_read_only_context"
        )

    dense = sum(value["action"] == "prefetch_max" for value in decisions)
    discard = sum(value["action"] == "discard_prefetch" for value in decisions)
    final = {
        "event": "final_policy_stats",
        "implementation": cell.implementation,
        "snapshot_updates": len(publications),
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
        "dense_prefetch_decisions": dense,
        "discarded_prefetch_decisions": discard,
        "snapshot_rejections": 0,
        "missing_snapshot_decisions": 0,
        "invalid_snapshot_decisions": 0,
        "request_errors": 0,
        "effect_errors": 0,
        "decision_record_drops": 0,
        "effect_record_drops": 0,
    }
    return publications, decisions, final


def make_cell(root: Path, cell: protocol.MatrixCell) -> Path:
    path = root / f"block-{cell.block:02d}-{cell.arm}"
    path.mkdir(parents=True)
    intervals = truth_intervals()

    execution = {
        "protocol": protocol.PROTOCOL,
        "timeline": protocol.TIMELINE,
        "block": cell.block,
        "arm": cell.arm,
        "implementation": cell.implementation,
        "delay_ms": cell.delay_ms,
        "status": "passed",
        "complete": True,
        "cleanup_errors": [],
        "lease_paths": list(protocol.LEASE_PATHS),
        "lease_mode": "read_only_exclusive",
        "target_pid": TARGET_PID,
        "uvm_fd_candidates": [
            {"source_fd": 7, "target": "/dev/nvidia-uvm"},
            {"source_fd": 8, "target": "/dev/nvidia-uvm"},
        ],
        "monitor_coverage": {
            "uvm": True,
            "gpu_telemetry": True,
            "compute_apps": True,
            "kernel_log": True,
            "phase_truth": True,
            **(
                {"policy_artifact_absence": True}
                if cell.role == "context_control"
                else {"policy_diagnostics": True}
            ),
        },
        "cleanup": {
            "workload_reaped": True,
            "monitors_reaped": True,
            "policy_detached": True,
            "leases_released": True,
        },
        "safety": {
            "pre_valid": True,
            "post_valid": True,
            "gpu_telemetry_valid": True,
            "foreign_compute_pids": [],
            "new_kernel_anomalies": [],
        },
    }
    write_json(path / "execution.json", execution)
    write_json(path / "safety-before.json", safety_snapshot())
    write_json(path / "safety-after.json", safety_snapshot())

    truth = [
        {
            "event": "workload_ready",
            "pid": TARGET_PID,
            "protocol": protocol.PROTOCOL,
            "timeline": protocol.TIMELINE,
            "allocation_bytes": protocol.ALLOCATION_BYTES,
            "regions": protocol.REGIONS,
        }
    ]
    for interval in intervals:
        common = {
            "sequence": interval["sequence"],
            "phase": interval["phase"],
            "measured": interval["measured"],
            "scheduled_offset_ns": interval["scheduled_offset_ns"],
        }
        truth.append({"event": "phase_start", **common, "mono_ns": interval["start_mono_ns"]})
        truth.append({"event": "phase_end", **common, "mono_ns": interval["end_mono_ns"]})
    write_jsonl(path / "phase-truth.jsonl", truth)

    phases = []
    total_checked = 0
    total_kernel_ms = 0.0
    for measured_index, interval in enumerate(intervals[1:], 1):
        selected = (
            protocol.DENSE_LAUNCH_REGIONS
            if interval["phase"] == "dense"
            else protocol.SPARSE_REGIONS
        )
        iterations = 2
        checked = selected * iterations
        kernel_ms = 1.0 + measured_index / 10.0
        total_checked += checked
        total_kernel_ms += kernel_ms
        phases.append(
            {
                "measured_index": measured_index,
                "sequence": interval["sequence"],
                "phase": interval["phase"],
                "scheduled_offset_ns": interval["scheduled_offset_ns"],
                "start_mono_ns": interval["start_mono_ns"],
                "end_mono_ns": interval["end_mono_ns"],
                "wall_ms": protocol.PHASE_NS / 1.0e6,
                "kernel_ms": kernel_ms,
                "iterations": iterations,
                "checked_values": checked,
                "mismatches": 0,
                "first_mismatch": None,
            }
        )
    end_to_end_ms = protocol.MEASURED_PHASES * protocol.PHASE_NS / 1.0e6
    write_json(
        path / "workload-result.json",
        {
            "protocol": protocol.PROTOCOL,
            "timeline": protocol.TIMELINE,
            "allocation_bytes": protocol.ALLOCATION_BYTES,
            "region_bytes": protocol.REGION_BYTES,
            "regions": protocol.REGIONS,
            "sparse_stride_regions": protocol.SPARSE_STRIDE,
            "sparse_regions": protocol.SPARSE_REGIONS,
            "dense_launch_regions": protocol.DENSE_LAUNCH_REGIONS,
            "bootstrap_ns": protocol.BOOTSTRAP_NS,
            "phase_ns": protocol.PHASE_NS,
            "measured_phases": protocol.MEASURED_PHASES,
            "epoch_mono_ns": EPOCH_NS,
            "end_to_end_ms": end_to_end_ms,
            "total_kernel_ms": total_kernel_ms,
            "verified_words_per_second": total_checked * 1000.0 / end_to_end_ms,
            "checked_values": total_checked,
            "mismatches": 0,
            "first_mismatch": None,
            "phases": phases,
        },
    )

    write_jsonl(
        path / "uvm-events.jsonl",
        [
            {
                "event": "ready",
                "target_pid": TARGET_PID,
                "uvm_fd": 9,
                "candidate_source_fds": [7, 8],
                "candidate_targets": ["/dev/nvidia-uvm", "/dev/nvidia-uvm"],
                "selected_source_fd": 7,
                "rejected_source_fd": 8,
                "rejected_status": 0x00000016,
                "queue_entries": protocol.UVM_QUEUE_ENTRIES,
                "entry_bytes": 72,
            },
            {
                "event": "final_uvm_stats",
                "gpu_faults": 120,
                "migrations": 60,
                "migrated_bytes": 60 * protocol.REGION_BYTES,
                "prefetch_migrations": 20,
                "prefetch_bytes": 20 * protocol.REGION_BYTES,
                "thrashing_events": 4,
                "eviction_events": 2,
                "fault_buffer_overflows": 0,
                "dropped_gpu_faults": 0,
                "dropped_migrations": 0,
                "dropped_thrashing": 0,
                "dropped_evictions": 0,
            },
        ],
    )
    (path / "gpu-telemetry.csv").write_text(
        "timestamp,power.draw,memory.used,utilization.gpu\n"
        "2026-09-03T00:00:00,100,40960,90\n"
        "2026-09-03T00:00:01,110,40960,95\n",
        encoding="utf-8",
    )
    write_jsonl(
        path / "compute-apps.jsonl",
        [
            {
                "query_started_mono_ns": EPOCH_NS - 3,
                "query_finished_mono_ns": EPOCH_NS - 2,
                "pids": [],
                "error": None,
            },
            {
                "query_started_mono_ns": EPOCH_NS + 1,
                "query_finished_mono_ns": EPOCH_NS + 2,
                "pids": [TARGET_PID],
                "error": None,
            },
            {
                "query_started_mono_ns": intervals[-1]["end_mono_ns"] + 1,
                "query_finished_mono_ns": intervals[-1]["end_mono_ns"] + 2,
                "pids": [],
                "error": None,
            },
        ],
    )
    (path / "kernel-monitor.log").write_text("", encoding="utf-8")

    if cell.role == "paired_policy":
        publications, decisions, final = policy_records(cell, intervals)
        write_jsonl(path / "snapshot-publications.jsonl", publications)
        write_jsonl(path / "policy-decisions.jsonl", decisions)
        write_json(path / "policy-final.json", final)
        struct_link = 302 if cell.implementation == "bpf" else 0
        observer = [
            {
                "event": "ready",
                "pid": TARGET_PID + 1,
                "target_pid": TARGET_PID,
                "implementation": cell.implementation,
                "observer_link_id": 301,
                "struct_link_id": struct_link,
                "struct_map_id": 303 if cell.implementation == "bpf" else 0,
            },
            *decisions,
            {
                "event": "observer_final",
                "implementation": cell.implementation,
                "observer_link_id": 301,
                "struct_link_id": struct_link,
                "diagnostic_calls": 2 * len(decisions),
                "selected_seen": len(decisions),
                "finished_seen": len(decisions),
                "records_emitted": len(decisions),
                "foreign_tgid": 0,
                "read_errors": 0,
                "ringbuf_drops": 0,
                "phase_errors": 0,
                "valid": True,
            },
        ]
        write_jsonl(path / "policy-observer.jsonl", observer)
        (path / "policy-observer.stderr.log").write_text("", encoding="utf-8")
        policy_transcript = (
            "processed 20 insns\n" if cell.implementation == "bpf" else ""
        )
        (path / "verifier.log").write_text(
            "load_error=0\nprogram_count=2\n\n"
            "program=stale_state_v1_diagnostic_observer\nprocessed 12 insns\n\n"
            f"program=stale_state_prefetch_v1\n{policy_transcript}",
            encoding="utf-8",
        )
    return path


class MatrixTests(unittest.TestCase):
    def test_full_matrix_has_three_complete_randomized_blocks(self) -> None:
        cells = protocol.matrix("full")
        self.assertEqual(len(cells), 21)
        expected_arms = {condition.arm for condition in protocol.conditions()}
        for block in range(1, 4):
            block_cells = [cell for cell in cells if cell.block == block]
            self.assertEqual({cell.arm for cell in block_cells}, expected_arms)
            self.assertEqual(len(block_cells), 7)
        pairs = protocol.paired_questions()
        self.assertEqual(len(pairs["mechanism_cost"]), 9)
        self.assertEqual(len(pairs["information_cost"]), 12)

    def test_preflight_is_one_separate_block(self) -> None:
        cells = protocol.matrix("preflight")
        self.assertEqual(len(cells), 7)
        self.assertTrue(all(cell.block == 1 for cell in cells))
        self.assertNotEqual(
            [cell.arm for cell in cells],
            [cell.arm for cell in protocol.matrix("full")[:7]],
        )


class DelayRelayTests(unittest.TestCase):
    def test_publication_waits_for_exact_eligibility(self) -> None:
        relay = protocol.DelayedSnapshotRelay(100)
        relay.observe(sequence=1, phase="dense", source_mono_ns=1_000_000_000, scheduled_offset_ns=0)
        self.assertEqual(relay.drain(1_099_999_999), [])
        rows = relay.drain(1_100_000_000)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["eligible_mono_ns"], 1_100_000_000)
        self.assertEqual(relay.pending, 0)

    def test_relay_rejects_noncontiguous_or_invalid_truth(self) -> None:
        relay = protocol.DelayedSnapshotRelay(1000)
        with self.assertRaises(protocol.ValidationError):
            relay.observe(sequence=2, phase="dense", source_mono_ns=1, scheduled_offset_ns=0)
        relay.observe(sequence=1, phase="sparse", source_mono_ns=2, scheduled_offset_ns=0)
        with self.assertRaises(protocol.ValidationError):
            relay.observe(sequence=2, phase="unknown", source_mono_ns=3, scheduled_offset_ns=1)

    def test_cpu_preflight_with_deterministic_clock(self) -> None:
        class FakeTime:
            now_ns = 1_000_000_000

            def clock(self) -> int:
                self.now_ns += 1_000_000
                return self.now_ns

            def sleep(self, seconds: float) -> None:
                self.now_ns += int(seconds * 1.0e9)

        fake = FakeTime()
        result = protocol.run_cpu_delay_preflight(
            clock_ns=fake.clock, sleep=fake.sleep, samples_per_delay=2
        )
        self.assertTrue(result["distinguishable"])
        self.assertTrue(result["cpu_only"])
        self.assertFalse(result["experiment_evidence"])
        self.assertEqual(len(result["rows"]), 6)


class BridgeStatusParserTests(unittest.TestCase):
    def test_exact_29_field_status_parses(self) -> None:
        text = status_text()
        self.assertEqual(len(text.split()), 29)
        parsed = coordinator.parse_status(text)
        self.assertEqual(parsed.abi_version, 1)
        self.assertEqual(parsed.mode, "off")

    def test_status_schema_rejects_malformed_missing_extra_and_duplicate(self) -> None:
        valid = status_text()
        cases = (
            valid + " malformed",
            " ".join(valid.split()[1:]),
            valid + " extra=0",
            valid + " mode=off",
        )
        for value in cases:
            with self.subTest(value=value), self.assertRaises(protocol.ValidationError):
                coordinator.parse_status(value)

    def test_status_rejects_non_integer_negative_and_overflow(self) -> None:
        cases = (
            status_text(snapshot_updates="not-a-number"),
            status_text(snapshot_updates=-1),
            status_text(snapshot_updates=coordinator.UINT64_MAX + 1),
            status_text(active_callbacks=coordinator.INT32_MAX + 1),
        )
        for value in cases:
            with self.subTest(value=value), self.assertRaises(protocol.ValidationError):
                coordinator.parse_status(value)

    def test_status_rejects_incoherent_snapshot(self) -> None:
        cases = (
            status_text(snapshot_present=2),
            status_text(snapshot_present=0, snapshot_sequence=1),
            status_text(
                snapshot_present=1,
                snapshot_sequence=1,
                snapshot_phase=1,
                source_mono_ns=100,
                published_mono_ns=99,
            ),
        )
        for value in cases:
            with self.subTest(value=value), self.assertRaises(protocol.ValidationError):
                coordinator.parse_status(value)


class BridgeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FastClock()
        self.bridge = coordinator.InMemoryContractBridge(clock_ns=self.clock.clock_ns)

    def test_contract_errno_classes(self) -> None:
        invalid_calls = (
            (errno.EINVAL, lambda: self.bridge.configure("invalid", 1)),
            (errno.ESTALE, lambda: self.bridge.publish(1, 1, 1, 1)),
        )
        for expected_errno, operation in invalid_calls:
            with self.subTest(expected_errno=expected_errno), self.assertRaises(OSError) as caught:
                operation()
            self.assertEqual(caught.exception.errno, expected_errno)

        self.bridge.configure("native", 7)
        for expected_errno, operation in (
            (errno.EINVAL, lambda: self.bridge.publish(7, 0, 1, 1)),
            (errno.EINVAL, lambda: self.bridge.publish(7, 1, True, 1)),
            (errno.EINVAL, lambda: self.bridge.publish(7, 1, 3, 1)),
            (errno.ESTALE, lambda: self.bridge.publish(8, 1, 1, 1)),
            (errno.ERANGE, lambda: self.bridge.publish(7, 2, 1, 1)),
            (
                errno.ERANGE,
                lambda: self.bridge.publish(
                    7, 1, 1, self.clock.now_ns + 1_000_000_000
                ),
            ),
        ):
            with self.subTest(expected_errno=expected_errno), self.assertRaises(OSError) as caught:
                operation()
            self.assertEqual(caught.exception.errno, expected_errno)
        with self.assertRaises(OSError) as caught:
            self.bridge.disable(8)
        self.assertEqual(caught.exception.errno, errno.ESTALE)

    def test_all_six_conditions_close_sequence_ack_counters_and_cleanup(self) -> None:
        result = run_study.run_bridge_preflight(clock_factory=FastClock)
        self.assertEqual(result["schema"], run_study.BRIDGE_PREFLIGHT_SCHEMA)
        self.assertFalse(result["live_bridge"])
        self.assertFalse(result["experiment_evidence"])
        self.assertTrue(result["synthetic_source"])
        self.assertEqual(result["condition_count"], 6)
        self.assertEqual(
            {(row["implementation"], row["delay_ms"]) for row in result["conditions"]},
            {(mode, delay) for mode in protocol.IMPLEMENTATIONS for delay in protocol.DELAYS_MS},
        )
        for row in result["conditions"]:
            self.assertFalse(row["live_bridge"])
            self.assertFalse(row["experiment_evidence"])
            self.assertTrue(row["synthetic_source"])
            self.assertEqual(len(row["publications"]), 7)
            for sequence, publication in enumerate(row["publications"], 1):
                expected = protocol.expected_phase(sequence)
                self.assertEqual(publication["sequence"], sequence)
                self.assertEqual(publication["phase"], expected["phase"])
                self.assertEqual(
                    publication["eligible_mono_ns"]
                    - publication["source_mono_ns"],
                    row["delay_ms"] * 1_000_000,
                )
                self.assertLessEqual(
                    publication["write_started_mono_ns"],
                    publication["published_mono_ns"],
                )
                self.assertLessEqual(
                    publication["published_mono_ns"],
                    publication["write_finished_mono_ns"],
                )
            self.assertEqual(row["final_enabled_status"]["snapshot_updates"], 7)
            self.assertEqual(row["final_enabled_status"]["snapshot_rejections"], 0)
            self.assertEqual(row["disabled_status"]["mode"], "off")
            self.assertEqual(row["disabled_status"]["snapshot_present"], 0)
            self.assertEqual(row["disabled_status"]["snapshot_updates"], 7)

    def test_selects_one_condition(self) -> None:
        result = run_study.run_bridge_preflight(
            implementation="bpf", delay_ms=100, clock_factory=FastClock
        )
        self.assertEqual(result["condition_count"], 1)
        self.assertEqual(result["conditions"][0]["implementation"], "bpf")
        self.assertEqual(result["conditions"][0]["delay_ms"], 100)
        with self.assertRaises(protocol.ValidationError):
            run_study.run_bridge_preflight(implementation="native")

    def test_busy_endpoint_is_rejected_without_overwriting_it(self) -> None:
        self.bridge.configure("native", 55)
        runner = coordinator.Coordinator(
            self.bridge, clock_ns=self.clock.clock_ns, sleep=self.clock.sleep
        )
        with self.assertRaisesRegex(protocol.ValidationError, "not idle"):
            runner.replay(implementation="bpf", generation=56, delay_ms=0)
        self.assertEqual(self.bridge.status().mode, "native")
        self.assertEqual(self.bridge.status().generation, 55)

    def test_counter_contamination_fails_closed_and_cleans_up(self) -> None:
        class ContaminatedBridge(coordinator.InMemoryContractBridge):
            def configure(self, mode: str, generation: int) -> None:
                super().configure(mode, generation)
                self._counters["callback_invocations"] = 1

        bridge = ContaminatedBridge(clock_ns=self.clock.clock_ns)
        runner = coordinator.Coordinator(
            bridge, clock_ns=self.clock.clock_ns, sleep=self.clock.sleep
        )
        with self.assertRaises(ExceptionGroup) as caught:
            runner.replay(implementation="native", generation=1, delay_ms=0)
        self.assertIn("callback or decision", str(caught.exception.exceptions[0]))
        self.assertEqual(bridge.status().mode, "off")

    def test_publication_outside_write_ack_window_fails_closed(self) -> None:
        class OutsideAckBridge(coordinator.InMemoryContractBridge):
            def publish(self, generation: int, sequence: int, phase: int, source_mono_ns: int) -> None:
                super().publish(generation, sequence, phase, source_mono_ns)
                current_sequence, current_phase, current_source, published = self._snapshot
                self._snapshot = (
                    current_sequence,
                    current_phase,
                    current_source,
                    published + 1_000_000_000,
                )

        bridge = OutsideAckBridge(clock_ns=self.clock.clock_ns)
        runner = coordinator.Coordinator(
            bridge, clock_ns=self.clock.clock_ns, sleep=self.clock.sleep
        )
        with self.assertRaisesRegex(protocol.ValidationError, "acknowledgement"):
            runner.replay(implementation="native", generation=1, delay_ms=0)
        self.assertEqual(bridge.status().mode, "off")

    def test_late_boundary_fails_closed(self) -> None:
        class LateClock(FastClock):
            def clock_ns(self) -> int:
                self.now_ns += protocol.MAXIMUM_BOUNDARY_OVERRUN_NS + 1
                return self.now_ns

        clock = LateClock()
        bridge = coordinator.InMemoryContractBridge(clock_ns=clock.clock_ns)
        with self.assertRaisesRegex(protocol.ValidationError, "boundary"):
            coordinator.Coordinator(
                bridge, clock_ns=clock.clock_ns, sleep=clock.sleep
            ).replay(implementation="native", generation=1, delay_ms=0)
        self.assertEqual(bridge.status().mode, "off")

    def test_generation_mismatch_fails_closed(self) -> None:
        class WrongGenerationBridge(coordinator.InMemoryContractBridge):
            def configure(self, mode: str, generation: int) -> None:
                super().configure(mode, generation + 1)

        bridge = WrongGenerationBridge(clock_ns=self.clock.clock_ns)
        with self.assertRaises(ExceptionGroup) as caught:
            coordinator.Coordinator(
                bridge, clock_ns=self.clock.clock_ns, sleep=self.clock.sleep
            ).replay(implementation="bpf", generation=7, delay_ms=0)
        self.assertIn("configure acknowledgement", str(caught.exception.exceptions[0]))
        self.assertEqual(caught.exception.exceptions[1].errno, errno.ESTALE)
        self.assertEqual(bridge.status().mode, "bpf")
        self.assertEqual(bridge.status().generation, 8)

    def test_missing_proc_endpoint_fails_without_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "missing-proc-endpoint"
            bridge = coordinator.ProcBridge(path)
            with self.assertRaises(FileNotFoundError):
                bridge.status()
            self.assertFalse(path.exists())

    def test_publish_failure_still_disables_bridge(self) -> None:
        class PublishFailureBridge(coordinator.InMemoryContractBridge):
            def publish(self, generation: int, sequence: int, phase: int, source_mono_ns: int) -> None:
                if sequence == 3:
                    raise OSError(errno.EIO, "injected publish failure")
                super().publish(generation, sequence, phase, source_mono_ns)

        bridge = PublishFailureBridge(clock_ns=self.clock.clock_ns)
        with self.assertRaises(OSError) as caught:
            coordinator.Coordinator(
                bridge, clock_ns=self.clock.clock_ns, sleep=self.clock.sleep
            ).replay(implementation="native", generation=3, delay_ms=0)
        self.assertEqual(caught.exception.errno, errno.EIO)
        self.assertEqual(bridge.status().mode, "off")
        self.assertEqual(bridge.status().snapshot_present, 0)

    def test_dirty_cleanup_is_rejected(self) -> None:
        class DirtyDisableBridge(coordinator.InMemoryContractBridge):
            def disable(self, generation: int) -> None:
                if generation != self._generation:
                    raise OSError(errno.ESTALE, "stale bridge generation")

        bridge = DirtyDisableBridge(clock_ns=self.clock.clock_ns)
        with self.assertRaisesRegex(protocol.ValidationError, "clean disabled state"):
            coordinator.Coordinator(
                bridge, clock_ns=self.clock.clock_ns, sleep=self.clock.sleep
            ).replay(implementation="native", generation=4, delay_ms=0)
        self.assertEqual(bridge.status().mode, "native")
        self.assertEqual(bridge.status().snapshot_present, 1)

    def test_primary_and_cleanup_failures_form_exception_group(self) -> None:
        class DualFailureBridge(coordinator.InMemoryContractBridge):
            def publish(self, generation: int, sequence: int, phase: int, source_mono_ns: int) -> None:
                raise OSError(errno.EIO, "injected publish failure")

            def disable(self, generation: int) -> None:
                raise OSError(errno.EBUSY, "injected cleanup failure")

        bridge = DualFailureBridge(clock_ns=self.clock.clock_ns)
        with self.assertRaises(ExceptionGroup) as caught:
            coordinator.Coordinator(
                bridge, clock_ns=self.clock.clock_ns, sleep=self.clock.sleep
            ).replay(implementation="bpf", generation=5, delay_ms=0)
        self.assertEqual(len(caught.exception.exceptions), 2)
        self.assertEqual(caught.exception.exceptions[0].errno, errno.EIO)
        self.assertEqual(caught.exception.exceptions[1].errno, errno.EBUSY)

    def test_cli_emits_separate_non_live_schema(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = run_study.main(
                ["bridge-preflight", "--implementation", "bpf", "--delay-ms", "1000"]
            )
        self.assertEqual(status, 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["schema"], run_study.BRIDGE_PREFLIGHT_SCHEMA)
        self.assertEqual(result["condition_count"], 1)
        self.assertFalse(result["live_bridge"])
        self.assertFalse(result["experiment_evidence"])
        self.assertTrue(result["synthetic_source"])

    def test_bridge_preflight_rejects_backend_that_relabels_replay(self) -> None:
        class RelabeledBridge(coordinator.InMemoryContractBridge):
            def configure(self, mode: str, generation: int) -> None:
                super().configure(mode, generation)
                self.live = True
                self.experiment_evidence = True

        with self.assertRaisesRegex(protocol.ValidationError, "evidence boundary"):
            run_study.run_bridge_preflight(
                implementation="native",
                delay_ms=0,
                clock_factory=FastClock,
                bridge_factory=lambda clock_ns: RelabeledBridge(clock_ns=clock_ns),
            )


class LiveTruthFDCoordinatorTests(unittest.TestCase):
    class EngagedBridge(coordinator.InMemoryContractBridge):
        def publish(
            self,
            generation: int,
            sequence: int,
            phase: int,
            source_mono_ns: int,
        ) -> None:
            super().publish(generation, sequence, phase, source_mono_ns)
            if sequence == 1:
                return
            for field in (
                "callback_invocations",
                "snapshot_read_attempts",
                "snapshot_read_successes",
                "decision_requests",
                "decisions",
                "decision_records",
                "effect_requests",
                "effect_records",
                "selected_diagnostics",
                "finished_diagnostics",
            ):
                self._counters[field] += 1
            self._counters[f"{self._mode}_callback_invocations"] += 1
            action = (
                "dense_prefetch_decisions"
                if phase == coordinator.PHASE_IDS["dense"]
                else "discarded_prefetch_decisions"
            )
            self._counters[action] += 1

    def test_native_and_bpf_consume_real_fd_truth_and_close_live_contract(self) -> None:
        for implementation in protocol.IMPLEMENTATIONS:
            for delay_ms in protocol.DELAYS_MS:
                with self.subTest(implementation=implementation, delay_ms=delay_ms):
                    clock = ScriptedClock(policy_clock_values(delay_ms))
                    bridge = self.EngagedBridge(clock_ns=clock.clock_ns)
                    releases = []
                    read_fd = pipe_with_payload(truth_fd_payload())
                    try:
                        result = coordinator.TruthFDCoordinator(
                            bridge, clock_ns=clock.clock_ns, sleep=clock.sleep
                        ).run(
                            truth_fd=read_fd,
                            expected_pid=TARGET_PID,
                            release=lambda: releases.append("R"),
                            implementation=implementation,
                            generation=2026090401,
                            delay_ms=delay_ms,
                        )
                    finally:
                        os.close(read_fd)

                    self.assertEqual(
                        result["schema"], coordinator.LIVE_COORDINATOR_SCHEMA
                    )
                    self.assertEqual(result["truth_source"], "workload_phase_fd")
                    self.assertFalse(result["synthetic_source"])
                    self.assertFalse(result["experiment_evidence"])
                    self.assertEqual(
                        result["evidence_scope"],
                        "coordinator_only_not_complete_cell",
                    )
                    self.assertEqual(result["truth_record_count"], 15)
                    self.assertEqual(releases, ["R"])
                    self.assertEqual(len(result["publications"]), 7)
                    self.assertEqual(
                        [row["source_mono_ns"] for row in result["publications"]],
                        [row["start_mono_ns"] for row in truth_intervals()],
                    )
                    for row in result["publications"]:
                        self.assertEqual(
                            row["eligible_mono_ns"] - row["source_mono_ns"],
                            delay_ms * 1_000_000,
                        )
                        self.assertLessEqual(
                            row["write_started_mono_ns"],
                            row["published_mono_ns"],
                        )
                        self.assertLessEqual(
                            row["published_mono_ns"],
                            row["write_finished_mono_ns"],
                        )
                        self.assertLessEqual(
                            row["write_finished_mono_ns"],
                            row["status_observed_mono_ns"],
                        )
                    self.assertEqual(
                        result["final_enabled_status"]["decisions"], 6
                    )
                    self.assertEqual(result["disabled_status"]["mode"], "off")
                    self.assertEqual(
                        result["disabled_status"]["snapshot_updates"], 7
                    )

    def test_bootstrap_release_precedes_configuration_to_avoid_missing_snapshot(self) -> None:
        clock = ScriptedClock(policy_clock_values(1000))
        bridge = self.EngagedBridge(clock_ns=clock.clock_ns)
        modes_at_release: list[str] = []
        read_fd = pipe_with_payload(truth_fd_payload())
        try:
            result = coordinator.TruthFDCoordinator(
                bridge, clock_ns=clock.clock_ns, sleep=clock.sleep
            ).run(
                truth_fd=read_fd,
                expected_pid=TARGET_PID,
                release=lambda: modes_at_release.append(bridge.status().mode),
                implementation="bpf",
                generation=2026090402,
                delay_ms=1000,
            )
        finally:
            os.close(read_fd)
        self.assertEqual(modes_at_release, ["off"])
        self.assertEqual(result["configured_status"]["snapshot_present"], 0)
        self.assertEqual(result["publications"][0]["sequence"], 1)

    def test_baseline_never_configures_or_publishes_policy_state(self) -> None:
        class UntouchedBridge(coordinator.InMemoryContractBridge):
            def configure(self, mode: str, generation: int) -> None:
                raise AssertionError("baseline configured a policy")

            def publish(
                self,
                generation: int,
                sequence: int,
                phase: int,
                source_mono_ns: int,
            ) -> None:
                raise AssertionError("baseline published a snapshot")

            def disable(self, generation: int) -> None:
                raise AssertionError("baseline disabled policy state it did not own")

        values = []
        for interval in truth_intervals():
            values.extend(
                (interval["start_mono_ns"] + 1_000, interval["end_mono_ns"] + 1_000)
            )
        clock = ScriptedClock(values)
        bridge = UntouchedBridge(clock_ns=clock.clock_ns)
        read_fd = pipe_with_payload(truth_fd_payload())
        try:
            result = coordinator.TruthFDCoordinator(
                bridge, clock_ns=clock.clock_ns, sleep=clock.sleep
            ).run(
                truth_fd=read_fd,
                expected_pid=TARGET_PID,
                release=lambda: None,
                implementation=None,
                generation=None,
                delay_ms=None,
            )
        finally:
            os.close(read_fd)
        self.assertFalse(result["policy_configured"])
        self.assertFalse(result["baseline_policy_artifacts"])
        self.assertEqual(result["publications"], [])
        self.assertIsNone(result["configured_status"])
        self.assertIsNone(result["final_enabled_status"])
        self.assertIsNone(result["disabled_status"])
        self.assertEqual(bridge.status().mode, "off")

    def test_malformed_phase_after_ready_fails_and_disables_owned_generation(self) -> None:
        records = truth_fd_payload().splitlines(keepends=True)
        malformed = json.loads(records[1])
        malformed["unexpected"] = 1
        records[1] = json.dumps(malformed).encode("utf-8") + b"\n"
        clock = FastClock()
        bridge = coordinator.InMemoryContractBridge(clock_ns=clock.clock_ns)
        read_fd = pipe_with_payload(b"".join(records))
        try:
            with self.assertRaisesRegex(protocol.ValidationError, "schema mismatch"):
                coordinator.TruthFDCoordinator(
                    bridge, clock_ns=clock.clock_ns, sleep=clock.sleep
                ).run(
                    truth_fd=read_fd,
                    expected_pid=TARGET_PID,
                    release=lambda: None,
                    implementation="native",
                    generation=99,
                    delay_ms=0,
                )
        finally:
            os.close(read_fd)
        status = bridge.status()
        self.assertEqual(status.mode, "off")
        self.assertEqual(status.generation, 0)
        self.assertEqual(status.snapshot_present, 0)

    def test_partial_truth_record_times_out_without_touching_baseline_state(self) -> None:
        read_fd, write_fd = os.pipe()
        bridge = coordinator.InMemoryContractBridge()
        try:
            os.write(write_fd, b'{"event":')
            with self.assertRaisesRegex(protocol.ValidationError, "timed out"):
                coordinator.TruthFDCoordinator(
                    bridge, truth_timeout_seconds=0.001
                ).run(
                    truth_fd=read_fd,
                    expected_pid=TARGET_PID,
                    release=lambda: None,
                    implementation=None,
                    generation=None,
                    delay_ms=None,
                )
        finally:
            os.close(write_fd)
            os.close(read_fd)
        self.assertEqual(bridge.status().mode, "off")
        self.assertEqual(bridge.status().snapshot_updates, 0)

    def test_before_release_hook_runs_after_ready_and_fails_before_policy_state(self) -> None:
        bridge = coordinator.InMemoryContractBridge()
        read_fd = pipe_with_payload(truth_fd_payload())
        released = False
        observed: list[dict] = []

        def before_release(ready: dict) -> None:
            observed.append(ready)
            raise RuntimeError("observer attach failed")

        def release() -> None:
            nonlocal released
            released = True

        try:
            with self.assertRaisesRegex(RuntimeError, "observer attach failed"):
                coordinator.TruthFDCoordinator(bridge).run(
                    truth_fd=read_fd,
                    expected_pid=TARGET_PID,
                    release=release,
                    implementation="bpf",
                    generation=103,
                    delay_ms=0,
                    before_release=before_release,
                )
        finally:
            os.close(read_fd)
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0]["event"], "workload_ready")
        self.assertFalse(released)
        self.assertEqual(bridge.status().mode, "off")
        self.assertEqual(bridge.status().generation, 0)

    def test_late_status_observation_fails_and_disables_generation(self) -> None:
        values = policy_clock_values(100)
        values[6] = (
            EPOCH_NS
            + 100_000_000
            + protocol.MAXIMUM_BOUNDARY_OVERRUN_NS
            + 1
        )
        clock = ScriptedClock(values)
        bridge = self.EngagedBridge(clock_ns=clock.clock_ns)
        read_fd = pipe_with_payload(truth_fd_payload())
        try:
            with self.assertRaisesRegex(protocol.ValidationError, "acknowledgement"):
                coordinator.TruthFDCoordinator(
                    bridge, clock_ns=clock.clock_ns, sleep=clock.sleep
                ).run(
                    truth_fd=read_fd,
                    expected_pid=TARGET_PID,
                    release=lambda: None,
                    implementation="native",
                    generation=102,
                    delay_ms=100,
                )
        finally:
            os.close(read_fd)
        self.assertEqual(bridge.status().mode, "off")

    def test_late_truth_event_and_dirty_live_counter_fail_and_clean_up(self) -> None:
        late_values = [
            EPOCH_NS + protocol.MAXIMUM_BOUNDARY_OVERRUN_NS + 1
        ]
        read_fd = pipe_with_payload(truth_fd_payload())
        bridge = coordinator.InMemoryContractBridge()
        try:
            with self.assertRaisesRegex(protocol.ValidationError, "delivered"):
                coordinator.TruthFDCoordinator(
                    bridge, clock_ns=ScriptedClock(late_values).clock_ns
                ).run(
                    truth_fd=read_fd,
                    expected_pid=TARGET_PID,
                    release=lambda: None,
                    implementation="bpf",
                    generation=100,
                    delay_ms=0,
                )
        finally:
            os.close(read_fd)
        self.assertEqual(bridge.status().mode, "off")

        class DirtyBridge(self.EngagedBridge):
            def publish(
                self,
                generation: int,
                sequence: int,
                phase: int,
                source_mono_ns: int,
            ) -> None:
                super().publish(generation, sequence, phase, source_mono_ns)
                if sequence == 3:
                    self._counters["effect_errors"] += 1

        clock = ScriptedClock(policy_clock_values(100))
        bridge = DirtyBridge(clock_ns=clock.clock_ns)
        read_fd = pipe_with_payload(truth_fd_payload())
        try:
            with self.assertRaises(ExceptionGroup) as caught:
                coordinator.TruthFDCoordinator(
                    bridge, clock_ns=clock.clock_ns, sleep=clock.sleep
                ).run(
                    truth_fd=read_fd,
                    expected_pid=TARGET_PID,
                    release=lambda: None,
                    implementation="native",
                    generation=101,
                    delay_ms=100,
                )
        finally:
            os.close(read_fd)
        self.assertIn("invalid action", str(caught.exception.exceptions[0]))
        self.assertIn("invalid action", str(caught.exception.exceptions[1]))
        self.assertEqual(bridge.status().mode, "off")

    def test_ready_identity_duplicate_json_and_trailing_records_fail_closed(self) -> None:
        duplicate = (
            b'{"event":"workload_ready","event":"workload_ready","pid":4242,'
            b'"protocol":"stale-cross-layer-575-v1",'
            b'"timeline":"alternating-dense-sparse-40g-v1",'
            b'"allocation_bytes":42949672960,"regions":655360}\n'
        )
        with self.assertRaisesRegex(protocol.ValidationError, "duplicate"):
            coordinator._decode_truth_record(duplicate)

        float_ready = json.loads(truth_fd_payload().splitlines()[0])
        float_ready["pid"] = float(TARGET_PID)
        with self.assertRaisesRegex(protocol.ValidationError, "identity"):
            coordinator._require_workload_ready(float_ready, TARGET_PID)

        wrong_ready = truth_fd_payload(pid=TARGET_PID + 1)
        read_fd = pipe_with_payload(wrong_ready)
        try:
            with self.assertRaisesRegex(protocol.ValidationError, "identity"):
                coordinator.TruthFDCoordinator(
                    coordinator.InMemoryContractBridge()
                ).run(
                    truth_fd=read_fd,
                    expected_pid=TARGET_PID,
                    release=lambda: None,
                    implementation=None,
                    generation=None,
                    delay_ms=None,
                )
        finally:
            os.close(read_fd)

        values = []
        for interval in truth_intervals():
            values.extend(
                (interval["start_mono_ns"] + 1_000, interval["end_mono_ns"] + 1_000)
            )
        read_fd = pipe_with_payload(truth_fd_payload() + b'{"event":"extra"}\n')
        try:
            with self.assertRaisesRegex(protocol.ValidationError, "trailing"):
                coordinator.TruthFDCoordinator(
                    coordinator.InMemoryContractBridge(),
                    clock_ns=ScriptedClock(values).clock_ns,
                ).run(
                    truth_fd=read_fd,
                    expected_pid=TARGET_PID,
                    release=lambda: None,
                    implementation=None,
                    generation=None,
                    delay_ms=None,
                )
        finally:
            os.close(read_fd)


class ObserverProtocolTests(unittest.TestCase):
    @staticmethod
    def records(implementation: str = "bpf") -> list[dict]:
        struct_link = 302 if implementation == "bpf" else 0
        values = [
            {
                "event": "ready",
                "pid": 5000,
                "target_pid": TARGET_PID,
                "implementation": implementation,
                "observer_link_id": 301,
                "struct_link_id": struct_link,
                "struct_map_id": 303 if implementation == "bpf" else 0,
            }
        ]
        for sequence, phase in ((1, "dense"), (2, "sparse")):
            decision_ns = EPOCH_NS + sequence * 1000
            values.append(
                {
                    "event": "policy_decision",
                    "implementation": implementation,
                    "decision_sequence": sequence,
                    "snapshot_read_path": f"driver_{implementation}_read_only_context",
                    "decision_mono_ns": decision_ns,
                    "snapshot_sequence": sequence,
                    "snapshot_phase": phase,
                    "decision_age_ns": 1000,
                    "action": "prefetch_max" if phase == "dense" else "discard_prefetch",
                    "effect": "prefetch" if phase == "dense" else "discard",
                    "effect_source": "driver_diagnostic",
                    "fault_page_index": 7,
                    "legal_max_first": 0,
                    "legal_max_outer": 16,
                    "output_first": 0,
                    "output_outer": 16 if phase == "dense" else 0,
                    "observer_mono_ns": decision_ns + 100,
                    "target_tgid": TARGET_PID,
                }
            )
        values.append(
            {
                "event": "observer_final",
                "implementation": implementation,
                "observer_link_id": 301,
                "struct_link_id": struct_link,
                "diagnostic_calls": 4,
                "selected_seen": 2,
                "finished_seen": 2,
                "records_emitted": 2,
                "foreign_tgid": 0,
                "read_errors": 0,
                "ringbuf_drops": 0,
                "phase_errors": 0,
                "valid": True,
            }
        )
        return values

    @staticmethod
    def driver_status(implementation: str = "bpf") -> coordinator.BridgeStatus:
        count = 2
        return coordinator.parse_status(
            status_text(
                mode=implementation,
                generation=99,
                snapshot_updates=7,
                callback_invocations=count,
                snapshot_read_attempts=count,
                snapshot_read_successes=count,
                native_callback_invocations=count if implementation == "native" else 0,
                bpf_callback_invocations=count if implementation == "bpf" else 0,
                decision_requests=count,
                decisions=count,
                decision_records=count,
                effect_requests=count,
                effect_records=count,
                dense_prefetch_decisions=1,
                discarded_prefetch_decisions=1,
                selected_diagnostics=count,
                finished_diagnostics=count,
            )
        )

    def test_complete_native_and_bpf_streams_reconcile_exactly(self) -> None:
        for implementation in protocol.IMPLEMENTATIONS:
            with self.subTest(implementation=implementation):
                encoded = "".join(json.dumps(row) + "\n" for row in self.records(implementation))
                parsed = observer_protocol.parse_jsonl(encoded)
                observed = observer_protocol.validate_records(
                    parsed, expected_pid=TARGET_PID, implementation=implementation
                )
                final = observer_protocol.reconcile_driver(
                    observed,
                    self.driver_status(implementation),
                    implementation=implementation,
                )
                self.assertEqual(final["effect_records"], 2)
                self.assertEqual(final["decision_record_drops"], 0)
                self.assertEqual(final["effect_record_drops"], 0)

    def test_duplicate_json_loss_wrong_owner_and_counter_drift_fail(self) -> None:
        with self.assertRaisesRegex(protocol.ValidationError, "duplicate"):
            observer_protocol.parse_jsonl('{"event":"ready","event":"ready"}\n')

        for mutation in ("loss", "wrong_owner", "driver_drift"):
            with self.subTest(mutation=mutation):
                records = self.records()
                status = self.driver_status()
                if mutation == "loss":
                    records[-1]["ringbuf_drops"] = 1
                    records[-1]["valid"] = False
                elif mutation == "wrong_owner":
                    records[0]["struct_link_id"] = 0
                else:
                    status = replace(status, effect_records=1)
                if mutation == "driver_drift":
                    observed = observer_protocol.validate_records(
                        records, expected_pid=TARGET_PID, implementation="bpf"
                    )
                    with self.assertRaisesRegex(protocol.ValidationError, "effect_records"):
                        observer_protocol.reconcile_driver(
                            observed, status, implementation="bpf"
                        )
                else:
                    with self.assertRaises(protocol.ValidationError):
                        observer_protocol.validate_records(
                            records, expected_pid=TARGET_PID, implementation="bpf"
                        )


class BoundaryTests(unittest.TestCase):
    def test_uvm_queue_can_buffer_the_complete_owner06_event_volume(self) -> None:
        retained = 607_035 + 946_061
        dropped = 229_387 + 343_462
        complete_owner06_events = retained + dropped
        self.assertEqual(protocol.UVM_QUEUE_ENTRIES, 1 << 22)
        self.assertEqual(
            protocol.UVM_QUEUE_ENTRIES & (protocol.UVM_QUEUE_ENTRIES - 1), 0
        )
        self.assertLess(complete_owner06_events, protocol.UVM_QUEUE_ENTRIES)
        self.assertGreater(protocol.UVM_QUEUE_ENTRIES - 1, complete_owner06_events)
        source = (Path(__file__).parent / "uvm_event_monitor.c").read_text(
            encoding="utf-8"
        )
        self.assertIn("#define PROBE_QUEUE_ENTRIES 2U", source)
        self.assertIn(".queue_buffer_size = PROBE_QUEUE_ENTRIES", source)

    def test_uvm_monitor_keeps_json_off_the_event_drain_path(self) -> None:
        source = (Path(__file__).parent / "uvm_event_monitor.c").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('emit("uvm_stats"', source)
        self.assertEqual(source.count('emit("final_uvm_stats"'), 1)

    def test_struct_ops_ownership_preserves_separate_id_namespaces(self) -> None:
        ready = {
            "pid": 5000,
            "struct_map_id": 16905,
            "struct_link_id": 16905,
        }
        inventory = {
            "maps": [{"id": 16905, "type": "struct_ops",
                      "pids": [{"pid": 5000, "comm": "live_loader"}]}],
            "links": [{"id": 16905, "type": "struct_ops", "map_id": 16905}],
        }
        self.assertEqual(
            live_runner.validate_struct_ops_ownership(ready, inventory, "bpf"),
            {"map_id": 16905, "link_id": 16905, "owner_pid": 5000},
        )
        for mutation in ("missing_link", "wrong_link_map", "wrong_owner"):
            with self.subTest(mutation=mutation):
                changed = json.loads(json.dumps(inventory))
                if mutation == "missing_link":
                    changed["links"] = []
                elif mutation == "wrong_link_map":
                    changed["links"][0]["map_id"] = 16906
                else:
                    changed["maps"][0]["pids"][0]["pid"] = 5001
                with self.assertRaises(live_runner.LiveError):
                    live_runner.validate_struct_ops_ownership(ready, changed, "bpf")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            execution = {"status": "running"}
            missing_link = {"maps": inventory["maps"], "links": []}
            with mock.patch.object(
                live_runner, "struct_ops_inventory", return_value=missing_link
            ), self.assertRaisesRegex(live_runner.LiveError, "link ownership"):
                live_runner.preserve_and_validate_struct_ops(
                    path, execution, ready, "bpf"
                )
            retained = json.loads((path / "execution.json").read_text())
            self.assertEqual(
                retained["struct_ops_at_ready"],
                {"observer_ready": ready, "inventory": missing_link},
            )

    def test_cuda_12_9_uvm_candidates_are_exact_and_ordered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fd_root = root / str(TARGET_PID) / "fd"
            fd_root.mkdir(parents=True)
            os.symlink("/dev/nvidia-uvm", fd_root / "8")
            os.symlink("/dev/null", fd_root / "6")
            os.symlink("/dev/nvidia-uvm", fd_root / "7")
            candidates = live_runner.discover_workload_uvm_fds(
                TARGET_PID, proc_root=root
            )
        self.assertEqual(candidates, [
            {"source_fd": 7, "target": "/dev/nvidia-uvm"},
            {"source_fd": 8, "target": "/dev/nvidia-uvm"},
        ])
        inherited = [
            {**candidates[0], "inherited_fd": 17},
            {**candidates[1], "inherited_fd": 18},
        ]
        self.assertEqual(
            live_runner.uvm_monitor_command(inherited, TARGET_PID)[1:],
            ["--uvm-candidate", "7:17", "--uvm-candidate", "8:18",
             "--target-pid", str(TARGET_PID)],
        )

    def test_cuda_uvm_candidate_discovery_rejects_wrong_cardinality(self) -> None:
        for count in (1, 3):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fd_root = root / str(TARGET_PID) / "fd"
                fd_root.mkdir(parents=True)
                for index in range(count):
                    os.symlink("/dev/nvidia-uvm", fd_root / str(index + 7))
                with self.assertRaises(live_runner.LiveError):
                    live_runner.discover_workload_uvm_fds(TARGET_PID, proc_root=root)

    def test_live_preflight_dry_run_is_side_effect_free_and_keeps_baseline_clean(self) -> None:
        with (
            mock.patch.object(Path, "exists", side_effect=AssertionError("exists")),
            mock.patch.object(Path, "mkdir", side_effect=AssertionError("mkdir")),
            mock.patch.object(subprocess, "run", side_effect=AssertionError("run")),
            mock.patch.object(subprocess, "Popen", side_effect=AssertionError("Popen")),
            mock.patch.object(os, "open", side_effect=AssertionError("open")),
        ):
            result = live_runner.dry_run(Path("relative/preflight"), (11, 12))
        self.assertFalse(result["experiment_evidence"])
        self.assertFalse(result["executes_gpu"])
        self.assertFalse(result["loads_modules"])
        self.assertFalse(result["baseline_policy_artifacts"])
        self.assertEqual(len(result["order"]), 7)
        self.assertIsNone(result["policy_loader"]["baseline"])
        self.assertEqual(result["policy_loader"]["native"], "fentry observer only")

    def test_module_lifecycle_interface_and_child_lease_contract(self) -> None:
        members = (
            "gpu_test_trigger", "gpu_page_prefetch", "gpu_page_prefetch_iter",
            "gpu_block_activate", "gpu_block_access", "gpu_evict_prepare",
            "gpu_stale_state_prefetch_v1",
        )
        raw = "STRUCT 'gpu_mem_ops' size=56 vlen=7\n" + "".join(
            f"\t'{name}' type_id=1 bits_offset=0\n" for name in members
        )
        raw += (
            "FUNC 'uvm_stale_state_v1_diagnostic' type_id=1 linkage=global\n"
            "FUNC 'bpf_gpu_stale_state_v1_request' type_id=1 linkage=global\n"
            "STRUCT 'uvm_stale_state_v1_input' size=88 vlen=1\n"
            "STRUCT 'uvm_stale_state_v1_diagnostic' size=176 vlen=17\n"
        )
        diagnostic_members = (
            "input", "callback_return", "decision_age_ns", "requested_first",
            "requested_outer", "output_first", "output_outer", "diagnostic_phase",
            "mode", "status", "action", "action_attempted", "action_conflict",
            "action_request_calls", "region_result", "initial_effect", "owner_tgid",
        )
        raw += "".join(
            f"\t'{name}' type_id=1 bits_offset=0\n"
            for name in diagnostic_members
        )
        interface = run_module_lifecycle.exact_stale_interface(raw)
        self.assertEqual(interface["gpu_mem_ops_members"], list(members))
        self.assertEqual(interface["diagnostic_members"], list(diagnostic_members))
        with self.assertRaises(run_module_lifecycle.LifecycleError):
            run_module_lifecycle.exact_stale_interface(
                raw.replace("gpu_stale_state_prefetch_v1", "wrong_callback")
            )
        with self.assertRaises(run_module_lifecycle.LifecycleError):
            run_module_lifecycle.exact_stale_interface(
                raw.replace("owner_tgid", "reserved")
            )
        command = run_module_lifecycle.child_command(Path("/tmp/output"), (31, 32))
        self.assertEqual(command[-2:], ["31", "32"])
        with self.assertRaises(run_module_lifecycle.LifecycleError):
            run_module_lifecycle.child_command(Path("/tmp/output"), (31,))

    def test_dry_run_has_no_filesystem_or_process_side_effects(self) -> None:
        output = io.StringIO()
        errors = io.StringIO()
        with (
            mock.patch.object(Path, "exists", side_effect=AssertionError("exists")),
            mock.patch.object(Path, "mkdir", side_effect=AssertionError("mkdir")),
            mock.patch.object(subprocess, "run", side_effect=AssertionError("run")),
            mock.patch.object(subprocess, "Popen", side_effect=AssertionError("Popen")),
            mock.patch.object(os, "open", side_effect=AssertionError("open")),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(errors),
        ):
            status = run_study.main(
                [
                    "dry-run",
                    "full",
                    "--output",
                    "relative/raw",
                    "--preflight",
                    "relative/preflight",
                ]
            )
        self.assertEqual(status, 0)
        self.assertEqual(errors.getvalue(), "")
        plan = json.loads(output.getvalue())
        self.assertFalse(plan["live_executable"])
        self.assertEqual(plan["cell_count"], 21)
        self.assertTrue(plan["dry_run"])
        self.assertFalse(plan["writes_output"])

    def test_live_refuses_before_path_or_process_access(self) -> None:
        errors = io.StringIO()
        with (
            mock.patch.object(Path, "exists", side_effect=AssertionError("exists")),
            mock.patch.object(Path, "mkdir", side_effect=AssertionError("mkdir")),
            mock.patch.object(subprocess, "run", side_effect=AssertionError("run")),
            mock.patch.object(subprocess, "Popen", side_effect=AssertionError("Popen")),
            mock.patch.object(os, "open", side_effect=AssertionError("open")),
            contextlib.redirect_stderr(errors),
        ):
            status = run_study.main(["live", "--output", "/must/not/be/touched"])
        self.assertEqual(status, 2)
        self.assertIn(protocol.LIVE_BLOCKER, errors.getvalue())

    def test_full_dry_run_requires_excluded_preflight(self) -> None:
        with self.assertRaises(protocol.ValidationError):
            protocol.dry_run_plan("full", Path("raw"))


class RawValidationTests(unittest.TestCase):
    def test_valid_policy_cell_records_truth_age_effects_and_real_uvm_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cell = next(
                value
                for value in protocol.matrix("full")
                if value.block == 1 and value.arm == "bpf_delay_1000ms"
            )
            path = make_cell(root, cell)
            result = protocol.validate_cell(path, cell)
            self.assertEqual(result["policy"]["wrong_phase_decisions"], 6)
            self.assertEqual(result["policy"]["decisions"], 12)
            self.assertEqual(result["uvm"]["thrashing_events"], 4)
            self.assertEqual(result["uvm"]["eviction_events"], 2)
            self.assertGreater(result["workload"]["checked_values"], 0)

    def test_decision_after_publication_before_status_observation_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cell = next(
                value
                for value in protocol.matrix("full")
                if value.block == 1 and value.arm == "bpf_delay_100ms"
            )
            path = make_cell(root, cell)
            publications = [
                json.loads(line)
                for line in (path / "snapshot-publications.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            for publication in publications:
                publication["status_observed_mono_ns"] = (
                    publication["published_mono_ns"] + 10_000_000
                )
            write_jsonl(path / "snapshot-publications.jsonl", publications)
            result = protocol.validate_cell(path, cell)
            self.assertTrue(result["valid"])

    def test_default_control_rejects_policy_artifacts(self) -> None:
        for artifact in protocol.POLICY_ARTIFACT_NAMES:
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                cell = next(
                    value
                    for value in protocol.matrix("full")
                    if value.block == 1 and value.arm == "uvm_default"
                )
                path = make_cell(root, cell)
                (path / artifact).write_text("unexpected\n", encoding="utf-8")
                with self.assertRaises(protocol.ValidationError):
                    protocol.validate_cell(path, cell)

    def test_policy_requires_raw_observer_and_verifier_evidence(self) -> None:
        for mutation in (
            "missing_observer", "duplicate_observer_key", "decision_mismatch",
            "missing_verifier",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                cell = next(
                    value
                    for value in protocol.matrix("full")
                    if value.block == 1 and value.arm == "bpf_fresh"
                )
                path = make_cell(root, cell)
                if mutation == "missing_observer":
                    (path / "policy-observer.jsonl").unlink()
                elif mutation == "duplicate_observer_key":
                    observer_path = path / "policy-observer.jsonl"
                    lines = observer_path.read_text(encoding="utf-8").splitlines()
                    lines[0] = lines[0].replace(
                        f'"target_pid":{TARGET_PID}',
                        f'"target_pid":{TARGET_PID},"target_pid":{TARGET_PID}',
                    )
                    observer_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                elif mutation == "decision_mismatch":
                    decisions = [
                        json.loads(line)
                        for line in (path / "policy-decisions.jsonl")
                        .read_text(encoding="utf-8")
                        .splitlines()
                    ]
                    decisions[0]["output_outer"] -= 1
                    write_jsonl(path / "policy-decisions.jsonl", decisions)
                else:
                    (path / "verifier.log").unlink()
                with self.assertRaises(protocol.ValidationError):
                    protocol.validate_cell(path, cell)

    def test_validator_rejects_numerical_error_event_loss_and_cleanup_failure(self) -> None:
        mutations = (
            "numerical", "event_loss", "uvm_fd_inventory_missing",
            "uvm_fd_inventory_mismatch", "uvm_fd_selection", "cleanup",
            "effect_error",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                cell = next(
                    value
                    for value in protocol.matrix("full")
                    if value.block == 1 and value.arm == "native_fresh"
                )
                path = make_cell(root, cell)
                if mutation == "numerical":
                    value = json.loads((path / "workload-result.json").read_text())
                    value["mismatches"] = 1
                    value["first_mismatch"] = 7
                    write_json(path / "workload-result.json", value)
                elif mutation == "event_loss":
                    values = [json.loads(line) for line in (path / "uvm-events.jsonl").read_text().splitlines()]
                    values[-1]["dropped_gpu_faults"] = 1
                    write_jsonl(path / "uvm-events.jsonl", values)
                elif mutation == "uvm_fd_inventory_missing":
                    value = json.loads((path / "execution.json").read_text())
                    del value["uvm_fd_candidates"]
                    write_json(path / "execution.json", value)
                elif mutation == "uvm_fd_inventory_mismatch":
                    value = json.loads((path / "execution.json").read_text())
                    value["uvm_fd_candidates"][1]["source_fd"] = 9
                    write_json(path / "execution.json", value)
                elif mutation == "uvm_fd_selection":
                    values = [json.loads(line) for line in (path / "uvm-events.jsonl").read_text().splitlines()]
                    values[0]["selected_source_fd"] = values[0]["rejected_source_fd"]
                    write_jsonl(path / "uvm-events.jsonl", values)
                elif mutation == "cleanup":
                    value = json.loads((path / "execution.json").read_text())
                    value["cleanup"]["policy_detached"] = False
                    write_json(path / "execution.json", value)
                else:
                    value = json.loads((path / "policy-final.json").read_text())
                    value["effect_errors"] = 1
                    write_json(path / "policy-final.json", value)
                with self.assertRaises(protocol.ValidationError):
                    protocol.validate_cell(path, cell)

    def test_validator_rejects_early_snapshot_and_unengaged_policy(self) -> None:
        for mutation in ("early", "late_status", "unengaged"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                cell = next(
                    value
                    for value in protocol.matrix("full")
                    if value.block == 1 and value.arm == "native_delay_100ms"
                )
                path = make_cell(root, cell)
                if mutation == "early":
                    values = [json.loads(line) for line in (path / "snapshot-publications.jsonl").read_text().splitlines()]
                    values[0]["published_mono_ns"] -= 1
                    values[0]["status_observed_mono_ns"] -= 1
                    write_jsonl(path / "snapshot-publications.jsonl", values)
                elif mutation == "late_status":
                    values = [json.loads(line) for line in (path / "snapshot-publications.jsonl").read_text().splitlines()]
                    values[0]["status_observed_mono_ns"] = (
                        values[0]["eligible_mono_ns"]
                        + protocol.MAXIMUM_BOUNDARY_OVERRUN_NS
                        + 1
                    )
                    write_jsonl(path / "snapshot-publications.jsonl", values)
                else:
                    decisions = [json.loads(line) for line in (path / "policy-decisions.jsonl").read_text().splitlines()]
                    decisions = [row for row in decisions if row["snapshot_phase"] == "dense"]
                    write_jsonl(path / "policy-decisions.jsonl", decisions)
                with self.assertRaises(protocol.ValidationError):
                    protocol.validate_cell(path, cell)

    def test_three_block_campaign_retains_negative_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            preflight_root = temporary_root / "preflight"
            preflight_root.mkdir()
            preflight_cells = protocol.matrix("preflight")
            for cell in preflight_cells:
                make_cell(preflight_root, cell)
            preflight_order = [asdict(cell) for cell in preflight_cells]
            write_json(
                preflight_root / "campaign.json",
                {
                    "protocol": protocol.PROTOCOL,
                    "timeline": protocol.TIMELINE,
                    "stage": "preflight",
                    "seed": protocol.SEED,
                    "blocks": protocol.PREFLIGHT_BLOCKS,
                    "complete": True,
                    "order": preflight_order,
                    "completed": preflight_order,
                },
            )

            root = temporary_root / "full"
            root.mkdir()
            cells = protocol.matrix("full")
            for cell in cells:
                make_cell(root, cell)
            order = [asdict(cell) for cell in cells]
            write_json(
                root / "campaign.json",
                {
                    "protocol": protocol.PROTOCOL,
                    "timeline": protocol.TIMELINE,
                    "stage": "full",
                    "seed": protocol.SEED,
                    "blocks": protocol.FORMAL_BLOCKS,
                    "complete": True,
                    "order": order,
                    "completed": order,
                    "preflight": str(preflight_root),
                },
            )
            result = protocol.validate_campaign(root)
            self.assertEqual(result["run_status"], "valid")
            self.assertEqual(result["preflight"]["stage"], "preflight")
            self.assertEqual(len(result["cells"]), 21)
            self.assertEqual(len(result["mechanism_cost"]), 9)
            self.assertEqual(len(result["information_cost"]), 12)
            self.assertEqual(result["negative_results_retained"], 12)
            self.assertTrue(all(row["negative_result"] for row in result["information_cost"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
