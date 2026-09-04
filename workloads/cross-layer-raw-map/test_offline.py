#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import analyze_raw_map
import protocol
import run_raw_map as runner


def target_events(arm: protocol.Arm) -> list[dict]:
    events = [
        {"event": "cuda_truth", **dict(zip(protocol.RECORD_FIELDS, record))}
        for record in protocol.expected_records(arm)
    ]
    events.append({"event": "cuda_summary", **protocol.expected_target_summary(arm)})
    return events


def probe_events(arm: protocol.Arm) -> list[dict]:
    events = [{
        "event": "ready",
        "thread_slots": arm.threads,
        "threads_per_block": protocol.BLOCK_DIM,
        "launches": arm.launches,
        "ring_capacity_per_thread": protocol.RING_CAPACITY,
    }]
    # The real drain order is per-thread; intentionally differ from CUDA's
    # launch-major order so validation cannot rely on line ordering.
    retained = protocol.expected_records(arm, retained_only=True)
    retained.sort(key=lambda item: (item[1] * protocol.BLOCK_DIM + item[4], item[0]))
    events.extend(
        {"event": "raw_record", **dict(zip(protocol.RECORD_FIELDS, record))}
        for record in retained
    )
    events.append({"event": "aggregate_summary", **protocol.expected_aggregate(arm)})
    events.append({
        "event": "ring_summary",
        "value_size": protocol.RECORD_SIZE,
        "entries_per_thread": protocol.RING_CAPACITY,
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
    })
    return events


def write_events(path: Path, events: list[dict]) -> None:
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


class ProtocolTests(unittest.TestCase):
    def test_retained_full_campaign_revalidates(self):
        path = Path(__file__).resolve().parent / "raw/full-575-02"
        if not path.exists():
            self.skipTest("retained formal evidence is not present")
        summary = analyze_raw_map.summarize(path)
        self.assertEqual(summary["cells"], 15)
        self.assertEqual(summary["positive_cells"], 10)
        self.assertEqual(summary["negative_cells"], 5)
        self.assertEqual(summary["positive_exact_raw_records"], 34560)
        self.assertEqual(summary["negative_accounted_drops"], 2560)

    def test_segment_identity_requires_live_child_mapping(self):
        path = Path("/dev/shm") / f"raw_map_identity_test_{os.getpid()}"
        self.assertFalse(path.exists())
        identity = None
        process = subprocess.Popen([
            sys.executable, "-c",
            ("import mmap, os, sys, time; "
             "fd=os.open(sys.argv[1], os.O_CREAT|os.O_EXCL|os.O_RDWR, 0o600); "
             "os.ftruncate(fd, 4096); mapping=mmap.mmap(fd, 4096); "
             "print('ready', flush=True); time.sleep(30)"),
            str(path),
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
           start_new_session=True)
        try:
            identity = runner.wait_owned_segment(process, path, timeout=5)
            self.assertTrue(runner.process_holds_segment(process, identity))
            replacement = (identity[0], identity[1] + 1, identity[2])
            self.assertFalse(runner.process_holds_segment(process, replacement))
        finally:
            runner.stop_owned(process)
            assert process.stdout is not None and process.stderr is not None
            process.stdout.close()
            process.stderr.close()
            if identity is not None:
                runner.unlink_owned_segment(path, identity)
        self.assertFalse(path.exists())

    def test_machine_stdout_is_separate_from_diagnostic_stderr(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processes, streams = [], []
            process = runner.run_process(
                root / "machine.log",
                [sys.executable, "-c",
                 "import sys; print('machine'); print('diagnostic', file=sys.stderr)"],
                dict(os.environ), processes, streams,
                stderr_log=root / "diagnostic.log",
            )
            self.assertEqual(process.wait(timeout=5), 0)
            for stream in streams:
                stream.close()
            self.assertEqual((root / "machine.log").read_text(), "machine\n")
            self.assertEqual((root / "diagnostic.log").read_text(), "diagnostic\n")

    def test_matrix_is_complete_and_seeded(self):
        self.assertEqual(protocol.campaign_order("preflight"),
                         protocol.campaign_order("preflight"))
        full = protocol.campaign_order("full")
        self.assertEqual(len(full), 15)
        for block in range(1, 6):
            names = [item["name"] for item in full if item["block"] == block]
            self.assertEqual(set(names), set(protocol.ARM_BY_NAME))

    def test_dry_run_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must-not-exist"
            plan = protocol.dry_run_plan(
                "full", output, Path("/missing/runtime"), Path("/missing/preflight")
            )
            self.assertFalse(output.exists())
            self.assertFalse(plan["writes_output"])
            self.assertFalse(plan["inspects_runtime_artifacts"])
            self.assertEqual(plan["cell_count"], 15)

    def test_positive_and_overflow_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for arm in protocol.ARMS:
                native = root / f"{arm.name}-native.log"
                instrumented = root / f"{arm.name}-instrumented.log"
                probe = root / f"{arm.name}-probe.log"
                write_events(native, target_events(arm))
                write_events(instrumented, target_events(arm))
                write_events(probe, probe_events(arm))
                result = protocol.validate_cell_logs(native, instrumented, probe, arm)
                expected = ("rejected_incomplete_raw_stream" if arm.expect_drop_rejection
                            else "accepted_complete_raw_stream")
                self.assertEqual(result["evidence_disposition"], expected)

    def mutation_rejected(self, arm: protocol.Arm, mutate):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native_events = target_events(arm)
            instrumented_events = copy.deepcopy(native_events)
            observed_events = probe_events(arm)
            mutate(native_events, instrumented_events, observed_events)
            native, instrumented, probe = root / "n", root / "i", root / "p"
            write_events(native, native_events)
            write_events(instrumented, instrumented_events)
            write_events(probe, observed_events)
            with self.assertRaises(protocol.EvidenceError):
                protocol.validate_cell_logs(native, instrumented, probe, arm)

    def test_missing_raw_tuple_rejected(self):
        arm = protocol.ARM_BY_NAME["small"]
        self.mutation_rejected(arm, lambda _n, _i, p: p.pop(1))

    def test_duplicate_raw_tuple_rejected(self):
        arm = protocol.ARM_BY_NAME["small"]
        self.mutation_rejected(arm, lambda _n, _i, p: p.insert(2, copy.deepcopy(p[1])))

    def test_cuda_truth_mutation_rejected(self):
        arm = protocol.ARM_BY_NAME["small"]
        self.mutation_rejected(
            arm, lambda _n, i, _p: i[0].__setitem__("thread_x", 99)
        )

    def test_silent_drop_rejected(self):
        arm = protocol.ARM_BY_NAME["small"]
        def mutate(_n, _i, p):
            p.pop(1)
            ring = next(event for event in p if event["event"] == "ring_summary")
            ring["committed_records"] -= 1
            ring["collected_records"] -= 1
            ring["callback_records"] -= 1
        self.mutation_rejected(arm, mutate)

    def test_unexpected_overflow_rejected(self):
        arm = protocol.ARM_BY_NAME["small"]
        def mutate(_n, _i, p):
            ring = next(event for event in p if event["event"] == "ring_summary")
            ring["full_drops"] = 1
        self.mutation_rejected(arm, mutate)

    def test_overflow_without_exact_drop_accounting_rejected(self):
        arm = protocol.ARM_BY_NAME["overflow_negative"]
        def mutate(_n, _i, p):
            ring = next(event for event in p if event["event"] == "ring_summary")
            ring["full_drops"] -= 1
        self.mutation_rejected(arm, mutate)

    def test_oob_and_bad_size_rejected(self):
        arm = protocol.ARM_BY_NAME["small"]
        for field in ("oob_drops", "bad_size_drops", "other_drops", "dirty_slots"):
            def mutate(_n, _i, p, field=field):
                ring = next(event for event in p if event["event"] == "ring_summary")
                ring[field] = 1
            self.mutation_rejected(arm, mutate)

    def test_aggregate_mutation_rejected(self):
        arm = protocol.ARM_BY_NAME["large"]
        def mutate(_n, _i, p):
            aggregate = next(event for event in p if event["event"] == "aggregate_summary")
            aggregate["sequence_sum"] += 1
        self.mutation_rejected(arm, mutate)

    def test_preflight_manifest_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cells = []
            for item in protocol.campaign_order("preflight"):
                arm = protocol.ARM_BY_NAME[item["name"]]
                name = f"block-{item['block']:02d}-order-{item['order']:02d}-{arm.name}"
                directory = root / name
                directory.mkdir()
                write_events(directory / "native.log", target_events(arm))
                write_events(directory / "instrumented.log", target_events(arm))
                write_events(directory / "probe.log", probe_events(arm))
                validation = protocol.validate_cell_logs(
                    directory / "native.log", directory / "instrumented.log",
                    directory / "probe.log", arm,
                )
                cell = {
                    "schema": protocol.SCHEMA,
                    "protocol": protocol.PROTOCOL,
                    "status": "passed",
                    "arm": arm.name,
                    "cleanup_errors": [],
                    "owned_group_survivors": {},
                    "private_segment_removed": True,
                    "validation": validation,
                }
                (directory / "cell.json").write_text(json.dumps(cell))
                cells.append({
                    "block": item["block"], "order": item["order"],
                    "directory": name, **validation,
                })
            manifest = {
                "schema": protocol.SCHEMA,
                "protocol": protocol.PROTOCOL,
                "mode": "preflight",
                "status": "passed",
                "cell_count": 3,
                "positive_cells": 2,
                "negative_drop_gates": 1,
                "cells": cells,
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            protocol.validate_preflight_manifest(root)
            manifest["cells"][0]["evidence_disposition"] = "wrong"
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(protocol.EvidenceError):
                protocol.validate_preflight_manifest(root)

    def test_preflight_raw_mutation_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cells = []
            for item in protocol.campaign_order("preflight"):
                arm = protocol.ARM_BY_NAME[item["name"]]
                name = f"block-{item['block']:02d}-order-{item['order']:02d}-{arm.name}"
                directory = root / name
                directory.mkdir()
                write_events(directory / "native.log", target_events(arm))
                write_events(directory / "instrumented.log", target_events(arm))
                write_events(directory / "probe.log", probe_events(arm))
                validation = protocol.validate_cell_logs(
                    directory / "native.log", directory / "instrumented.log",
                    directory / "probe.log", arm,
                )
                (directory / "cell.json").write_text(json.dumps({
                    "schema": protocol.SCHEMA, "protocol": protocol.PROTOCOL,
                    "status": "passed", "arm": arm.name, "cleanup_errors": [],
                    "owned_group_survivors": {}, "private_segment_removed": True,
                    "validation": validation,
                }))
                cells.append({"block": item["block"], "order": item["order"],
                              "directory": name, **validation})
            (root / "manifest.json").write_text(json.dumps({
                "schema": protocol.SCHEMA, "protocol": protocol.PROTOCOL,
                "mode": "preflight", "status": "passed", "cell_count": 3,
                "positive_cells": 2, "negative_drop_gates": 1, "cells": cells,
            }))
            first = root / cells[0]["directory"] / "probe.log"
            mutated = protocol.json_events(first)
            mutated.pop(1)
            write_events(first, mutated)
            with self.assertRaises(protocol.EvidenceError):
                protocol.validate_preflight_manifest(root)


if __name__ == "__main__":
    unittest.main()
