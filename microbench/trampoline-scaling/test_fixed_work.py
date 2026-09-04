#!/usr/bin/env python3
"""CPU-only tests for the fixed-work profile and analyzer."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import analyze_fixed_work as analyzer
import run_fixed_work as fixed


runner = fixed.runner


def measurement(cell: dict, elapsed_ms: float) -> dict:
    phase = runner.phase_parameters("full")
    return {
        "event": "measurement",
        **analyzer.expected_measurement(cell, phase),
        "elapsed_ms": elapsed_ms,
    }


def result_fixture(endpoint_effect_pct: float = 0.0) -> dict:
    phase = runner.phase_parameters("full")
    schedule = runner.frozen_schedule("full")
    by_id = {cell["id"]: cell for cell in runner.CELLS}
    records = []
    for item in schedule:
        arm = item["arm"]
        values = []
        for cell_id in item["cell_ids"]:
            cell = by_id[cell_id]
            native = 10.0 + cell_id / 1000.0 + item["block"] / 10_000.0
            delta = 0.0
            if arm == "noop":
                delta = 0.1
                if cell_id == analyzer.HIGH_BLOCK_CELL:
                    low_native = 10.0 + analyzer.LOW_BLOCK_CELL / 1000.0
                    high_native = 10.0 + analyzer.HIGH_BLOCK_CELL / 1000.0
                    delta += endpoint_effect_pct * (low_native + high_native) / 200.0
            elif arm == "counter":
                delta = 0.2
            values.append(measurement(cell, native + delta))
        record = {
            "valid": True, "block": item["block"], "order": item["order"],
            "arm": arm, "measurements": values,
            "telemetry": {"summary": {"samples": 1}},
            "safety_after": {"clean": True},
        }
        if arm != "baseline":
            record.update(
                private_segment_removed=True,
                owned_group_survivors={},
                engagement={
                    "marker_callbacks": 32,
                    "target_counter_exact": arm == "counter",
                },
                agent_gate={"routing_order_valid": True},
            )
        records.append(record)
    return {
        "kind": runner.EXPERIMENT_KIND,
        "status": "complete",
        "failures": [],
        "params": {
            "kind": runner.EXPERIMENT_KIND,
            "phase": "full", "blocks": phase["blocks"],
            "cell_ids": list(phase["cell_ids"]),
            "warmup": phase["warmup"], "launches": phase["launches"],
            "hook_repeats": phase["hook_repeats"],
            "schedule_seed": runner.SEED,
            "expected_driver": runner.EXPECTED_DRIVER,
            "expected_gpu": runner.EXPECTED_GPU,
            "matrix": [dict(cell) for cell in runner.CELLS],
            "randomize_cell_order": True,
        },
        "schedule": schedule,
        "records": records,
        "safety_after": {"clean": True},
    }


class FixedWorkProfileTests(unittest.TestCase):
    def test_matrix_holds_total_work_and_dynamic_warps_fixed(self) -> None:
        self.assertEqual([cell["blocks"] for cell in runner.CELLS],
                         [128, 256, 1024, 2048, 4096])
        self.assertEqual(
            {cell["blocks"] * cell["threads_per_block"] for cell in runner.CELLS},
            {131_072},
        )
        self.assertEqual({cell["active_threads"] for cell in runner.CELLS}, {131_072})
        self.assertEqual({cell["active_threads"] // 32 for cell in runner.CELLS}, {4096})
        self.assertTrue(all(cell["threads_per_block"] % 32 == 0 for cell in runner.CELLS))

    def test_compiled_header_contains_exact_profile(self) -> None:
        text = (Path(__file__).resolve().parent / "fixed_work_matrix.h").read_text()
        for cell in runner.CELLS:
            marker = (
                f"X({cell['id']}, {cell['blocks']}, {cell['threads_per_block']}, "
                f"{cell['active_threads']}, {cell['counter_key']})"
            )
            self.assertIn(marker, text)
            geometry = (
                f"X({cell['blocks']}, {cell['threads_per_block']}, "
                f"{cell['counter_key']})"
            )
            self.assertIn(geometry, text)
        self.assertIn("#define SCALE_CELL_COUNT 5", text)
        self.assertEqual(len({(cell["blocks"], cell["threads_per_block"])
                              for cell in runner.CELLS}), 5)
        self.assertEqual({cell["counter_key"] for cell in runner.CELLS}, set(range(5)))

    def test_profile_uses_separate_compiled_artifacts(self) -> None:
        self.assertEqual(runner.APPLICATION_BINARY.name, "fixed-work-scaling")
        self.assertEqual(runner.COMPILED_PTX.name, "fixed-work-scaling.ptx")
        self.assertEqual(runner.LOADER_BINARY.name, "fixed-work-probe")
        self.assertEqual(runner.BPF_OBJECT_PREFIX, "fixed-work-probe")
        self.assertEqual(runner.MATRIX_HEADER.name, "fixed_work_matrix.h")

    def test_cell_order_is_randomized_per_block_and_shared_across_arms(self) -> None:
        schedule = runner.frozen_schedule("full")
        orders = []
        for block in range(10):
            cells = [tuple(item["cell_ids"]) for item in schedule if item["block"] == block]
            self.assertEqual(len(cells), 3)
            self.assertEqual(len(set(cells)), 1)
            orders.append(cells[0])
        self.assertGreater(len(set(orders)), 1)
        self.assertEqual(schedule, runner.frozen_schedule("full"))

    def test_core_application_gate_accepts_variable_block_dimensions(self) -> None:
        phase = runner.phase_parameters("full")
        records = [{
            "event": "device", "name": runner.EXPECTED_GPU,
            "major": 12, "minor": 0, "warp_size": 32,
            "max_threads_per_block": 1024, "max_grid_x": 2_147_483_647,
        }, {"event": "marker", "threads": 32, "mismatches": 0}]
        records.extend(
            measurement(cell, 1.0 + cell["id"] / 100.0)
            for cell in runner.CELLS
        )
        records.append({"event": "complete", "cells": 5, "run_id": 0})
        actual = runner.validate_application_events(
            records, phase["cell_ids"], phase["warmup"], phase["launches"],
            phase["hook_repeats"], 0,
        )
        self.assertEqual([item["threads_per_block"] for item in actual],
                         [1024, 512, 128, 64, 32])

    def test_full_counter_oracle_is_exact_for_every_fixed_work_cell(self) -> None:
        phase = runner.phase_parameters("full")
        oracle = runner.expected_counter_segments(
            phase["cell_ids"], phase["warmup"], phase["launches"],
            phase["hook_repeats"],
        )
        for key in range(5):
            self.assertEqual(oracle[("target_count", key)], [
                {"begin": 0, "end": 131_072, "value": 160},
                {"begin": 131_072, "end": runner.MAX_THREADS, "value": 0},
            ])


class FixedWorkAnalysisTests(unittest.TestCase):
    def test_constant_increment_supports_bounded_hypothesis(self) -> None:
        result = analyzer.analyze(result_fixture())
        self.assertEqual(result["run_status"], "valid")
        self.assertEqual(result["tested_hypothesis"],
                         "supported_within_predeclared_bound")
        self.assertAlmostEqual(result["primary_metric"]["median"], 0.0, places=9)
        self.assertEqual(len(result["cells"]), 5)

    def test_material_endpoint_effect_contradicts_hypothesis(self) -> None:
        result = analyzer.analyze(result_fixture(endpoint_effect_pct=3.0))
        self.assertEqual(result["tested_hypothesis"], "contradicted")
        self.assertGreater(result["primary_metric"]["ci95_low"], 1.0)

    def test_incomplete_or_mutated_evidence_fails_closed(self) -> None:
        missing = result_fixture()
        missing["records"].pop()
        with self.assertRaisesRegex(analyzer.AnalysisError, "30 arm"):
            analyzer.analyze(missing)

        mutated = result_fixture()
        mutated["records"][0]["measurements"][0]["active_warps"] = 2048
        with self.assertRaisesRegex(analyzer.AnalysisError, "measurement invariant"):
            analyzer.analyze(mutated)

    def test_render_and_json_round_trip_are_cpu_only(self) -> None:
        result = analyzer.analyze(result_fixture())
        rendered = analyzer.render_markdown(result)
        self.assertIn("Endpoint effect", rendered)
        self.assertEqual(json.loads(json.dumps(result)), result)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "analysis.md"
            path.write_text(rendered)
            self.assertTrue(path.read_text().endswith("\n"))


if __name__ == "__main__":
    unittest.main()
