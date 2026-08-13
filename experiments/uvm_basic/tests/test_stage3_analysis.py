#!/usr/bin/env python3

from __future__ import annotations

import csv
import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def module(name: str):
    path = ROOT / "analysis" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(loaded)
    return loaded


DECISIONS = module("analyze_prefetch_decisions")
REFAULT = module("analyze_eviction_refault")
ARRAYS = module("analyze_array_migrations")


class Stage3AnalysisTests(unittest.TestCase):
    def test_prefetch_callback_and_decision_are_correlated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "results/stage3/trace_semantics/prefetch_always_max/na/run1"
            run.mkdir(parents=True)
            (run / "manifest.json").write_text(json.dumps({
                "experiment": "trace_semantics", "policy": "prefetch_always_max", "ratio": "na"}))
            fields = ["event_type", "call_id", "action_name", "max_candidate_first",
                      "max_candidate_outer", "policy_result_first", "policy_result_outer",
                      "final_effective_first", "final_effective_outer", "final_pages"]
            with (run / "prefetch_decision_trace.csv").open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                writer.writerow({"event_type": "CALLBACK", "call_id": "7"})
                writer.writerow({"event_type": "DECISION", "call_id": "7", "action_name": "BYPASS",
                                 "max_candidate_first": "0", "max_candidate_outer": "512",
                                 "policy_result_first": "0", "policy_result_outer": "512",
                                 "final_effective_first": "0", "final_effective_outer": "512",
                                 "final_pages": "512"})
            row = DECISIONS.analyze(root)[0]
            self.assertEqual(row["matched_call_ids"], 1)
            self.assertEqual(row["bypass_count"], 1)
            self.assertEqual(row["final_pages_mean"], 512)
            self.assertEqual(row["policy_final_equal_count"], 1)

    def test_eviction_refault_requires_same_va_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "manifest.json").write_text(json.dumps({
                "experiment": "oversub", "policy": "custom_no_policy", "ratio": "1.10"}))
            phases = [
                {"phase": "phase_A_first", "monotonic_start_ns": 100, "monotonic_end_ns": 200},
                {"phase": "phase_B_first", "monotonic_start_ns": 201, "monotonic_end_ns": 300},
                {"phase": "phase_A_reuse", "monotonic_start_ns": 301, "monotonic_end_ns": 400},
            ]
            (run / "program.jsonl").write_text("".join(json.dumps(row) + "\n" for row in phases))
            with (run / "prefetch_decision_trace.csv").open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=["event_type", "timestamp_ns", "va_start", "va_end"])
                writer.writeheader()
                writer.writerow({"event_type": "DECISION", "timestamp_ns": 150, "va_start": "0x1000", "va_end": "0x1fff"})
                writer.writerow({"event_type": "DECISION", "timestamp_ns": 350, "va_start": "0x1000", "va_end": "0x1fff"})
            with (run / "chunk_trace.csv").open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=["hook_type", "timestamp_ns", "va_start", "va_end"])
                writer.writeheader()
                writer.writerow({"hook_type": "EVICTION_SELECTED", "timestamp_ns": 250,
                                 "va_start": "0x1000", "va_end": "0x1fff"})
            row = REFAULT.analyze_run(run)
            self.assertEqual(row["refaulted_block_count"], 1)
            self.assertEqual(row["refaulted_bytes"], 4096)
            self.assertEqual(row["eviction_to_refault_us_mean"], 0.1)

    def test_nsys_migrations_are_classified_by_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "manifest.json").write_text(json.dumps({"policy": "prefetch_always_max"}))
            allocation = {"phase": "allocation_addresses", "kernel_mode": "read-a",
                          "a_base_u64": 0x1000, "a_end_u64": 0x2000,
                          "b_base_u64": 0x3000, "b_end_u64": 0x4000,
                          "c_base_u64": 0x5000, "c_end_u64": 0x6000}
            kernel = {"phase": "kernel_1_demand", "kernel_mode": "read-a", "elapsed_ms": 2.0}
            (run / "program.jsonl").write_text(json.dumps(allocation) + "\n" + json.dumps(kernel) + "\n")
            connection = sqlite3.connect(run / "representative.sqlite")
            connection.execute("create table CUPTI_ACTIVITY_KIND_MEMCPY(copyKind int, bytes int, virtualAddress int)")
            connection.execute("insert into CUPTI_ACTIVITY_KIND_MEMCPY values(11, 4096, 4352)")
            connection.execute("create table CUDA_UM_GPU_PAGE_FAULT_EVENTS(address int, numberOfPageFaults int)")
            connection.execute("insert into CUDA_UM_GPU_PAGE_FAULT_EVENTS values(4352, 3)")
            connection.commit(); connection.close()
            rows = ARRAYS.analyze(run)
            a = next(row for row in rows if row["allocation"] == "A")
            self.assertEqual(a["h2d_bytes"], 4096)
            self.assertEqual(a["gpu_faults"], 3)

    def test_timeout_manifest_is_not_complete_or_correct(self) -> None:
        rows = [{"correct": True}, {"correct": True}]
        exit_code = 124
        completed = exit_code == 0
        correct = exit_code == 0 and bool(rows) and all(row["correct"] for row in rows)
        self.assertFalse(completed)
        self.assertFalse(correct)


if __name__ == "__main__":
    unittest.main()
