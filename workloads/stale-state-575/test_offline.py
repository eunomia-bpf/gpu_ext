#!/usr/bin/env python3
"""CPU-only tests for the stale-state protocol and fail-closed boundary."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import protocol
import run_study


EPOCH_NS = 10_000_000_000
TARGET_PID = 4242


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
                "consumer_ack_mono_ns": eligible,
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
                "host_phase_fixture": interval["phase"],
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
            "native_atomic_snapshot_read"
            if cell.implementation == "native"
            else "bpf_snapshot_helper"
        )

    dense = sum(value["action"] == "prefetch_max" for value in decisions)
    discard = sum(value["action"] == "discard_prefetch" for value in decisions)
    final = {
        "event": "final_policy_stats",
        "implementation": cell.implementation,
        "snapshot_updates": len(publications),
        "callback_invocations": len(decisions),
        "decisions": len(decisions),
        "decision_records": len(decisions),
        "effect_requests": len(decisions),
        "effect_records": len(decisions),
        "snapshot_reads": len(decisions),
        "native_snapshot_reads": len(decisions) if cell.implementation == "native" else 0,
        "snapshot_helper_calls": len(decisions) if cell.implementation == "bpf" else 0,
        "snapshot_helper_successes": len(decisions) if cell.implementation == "bpf" else 0,
        "dense_prefetch_decisions": dense,
        "discarded_prefetch_decisions": discard,
        "snapshot_rejections": 0,
        "missing_snapshot_decisions": 0,
        "invalid_snapshot_decisions": 0,
        "request_errors": 0,
        "decision_record_drops": 0,
        "effect_record_drops": 0,
        "snapshot_helper_errors": 0,
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
        "monitor_coverage": {
            "uvm": True,
            "gpu_telemetry": True,
            "compute_apps": True,
            "kernel_log": True,
            "phase_truth": True,
            "policy_diagnostics": True,
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
                "queue_entries": 65536,
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


class BoundaryTests(unittest.TestCase):
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

    def test_default_control_rejects_policy_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cell = next(
                value
                for value in protocol.matrix("full")
                if value.block == 1 and value.arm == "uvm_default"
            )
            path = make_cell(root, cell)
            write_json(path / "policy-final.json", {"event": "unexpected"})
            with self.assertRaises(protocol.ValidationError):
                protocol.validate_cell(path, cell)

    def test_validator_rejects_numerical_error_event_loss_and_cleanup_failure(self) -> None:
        mutations = ("numerical", "event_loss", "cleanup", "helper_error")
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
                elif mutation == "cleanup":
                    value = json.loads((path / "execution.json").read_text())
                    value["cleanup"]["policy_detached"] = False
                    write_json(path / "execution.json", value)
                else:
                    value = json.loads((path / "policy-final.json").read_text())
                    value["snapshot_helper_errors"] = 1
                    write_json(path / "policy-final.json", value)
                with self.assertRaises(protocol.ValidationError):
                    protocol.validate_cell(path, cell)

    def test_validator_rejects_early_snapshot_and_unengaged_policy(self) -> None:
        for mutation in ("early", "unengaged"):
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
                    values[0]["consumer_ack_mono_ns"] -= 1
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
