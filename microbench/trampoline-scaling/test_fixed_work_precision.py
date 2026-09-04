#!/usr/bin/env python3
"""CPU-only tests for the frozen fixed-work precision follow-up."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import analyze_fixed_work as base_analyzer
import analyze_fixed_work_precision as precision_analyzer
import run_fixed_work as fixed
import run_fixed_work_precision as precision
import test_fixed_work as fixtures


runner = precision.runner


class PrecisionProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        precision.configure()

    def tearDown(self) -> None:
        fixed.configure()

    def test_profile_preserves_per_kernel_work_and_freezes_budget(self) -> None:
        phase = runner.phase_parameters("full")
        self.assertEqual(phase, {
            "blocks": 48,
            "cell_ids": (0, 1, 2, 3, 4),
            "warmup": 16,
            "launches": 512,
            "hook_repeats": 16,
        })
        self.assertEqual(
            {cell["blocks"] * cell["threads_per_block"] for cell in runner.CELLS},
            {131_072},
        )
        self.assertEqual({cell["active_threads"] for cell in runner.CELLS}, {131_072})
        self.assertEqual({cell["active_threads"] // 32 for cell in runner.CELLS}, {4096})
        self.assertEqual(runner.APPLICATION_BINARY.name, "fixed-work-scaling")
        self.assertEqual(runner.BPF_OBJECT_PREFIX, "fixed-work-probe")
        self.assertEqual(
            {path.name for path in runner.EXTRA_SOURCE_PATHS},
            {
                "run_fixed_work.py", "analyze_fixed_work.py",
                "run_fixed_work_precision.py", "analyze_fixed_work_precision.py",
                "fixed-work-precision-plan.md",
            },
        )

    def test_schedule_has_all_permutations_and_exact_position_balance(self) -> None:
        schedule = runner.frozen_schedule("full")
        self.assertEqual(schedule, runner.frozen_schedule("full"))
        self.assertEqual(len(schedule), 144)
        permutations: Counter[tuple[str, ...]] = Counter()
        positions = {arm: [0, 0, 0] for arm in runner.ARMS}
        cell_orders = []
        for block in range(48):
            items = [item for item in schedule if item["block"] == block]
            self.assertEqual([item["order"] for item in items], [0, 1, 2])
            permutation = tuple(item["arm"] for item in items)
            permutations[permutation] += 1
            for item in items:
                positions[item["arm"]][item["order"]] += 1
            orders = {tuple(item["cell_ids"]) for item in items}
            self.assertEqual(len(orders), 1)
            cell_orders.append(next(iter(orders)))
        self.assertEqual(len(permutations), 6)
        self.assertEqual(set(permutations.values()), {8})
        self.assertTrue(all(counts == [16, 16, 16] for counts in positions.values()))
        self.assertGreater(len(set(cell_orders)), 1)
        for first in range(0, 48, 6):
            group = {
                tuple(item["arm"] for item in schedule if item["block"] == block)
                for block in range(first, first + 6)
            }
            self.assertEqual(len(group), 6)

    def test_counter_oracle_covers_every_launch_exactly(self) -> None:
        phase = runner.phase_parameters("full")
        oracle = runner.expected_counter_segments(
            phase["cell_ids"], phase["warmup"], phase["launches"],
            phase["hook_repeats"],
        )
        for key in range(5):
            self.assertEqual(oracle[("target_count", key)], [
                {"begin": 0, "end": 131_072, "value": 8_448},
                {"begin": 131_072, "end": runner.MAX_THREADS, "value": 0},
            ])

    def test_ten_block_schedule_remains_replay_compatible(self) -> None:
        fixed.configure()
        old_result = Path(__file__).resolve().parent / (
            "raw/fixed-work-full-575-01/result.json"
        )
        result = json.loads(old_result.read_text())
        self.assertEqual(result["schedule"], runner.frozen_schedule("full"))


class PrecisionRawReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        precision.configure()

    def tearDown(self) -> None:
        fixed.configure()

    def test_complete_raw_fixture_replays_all_720_cells(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result, path = fixtures.result_fixture(Path(temporary) / "raw")
            result["records"][0].update(
                valid=False,
                measurements=[{"elapsed_ms": 999_999.0}],
                telemetry={"summary": None},
                engagement={"untrusted": "ignored"},
            )
            analysis = precision_analyzer.analyze(result, path)
            application = (
                Path(result["records"][0]["directory"]) / "application.log"
            )
            fixtures.mutate_json_event(
                application, "measurement", "active_warps", 2_048,
            )
            with self.assertRaisesRegex(
                base_analyzer.AnalysisError, "application.*gate failed",
            ):
                precision_analyzer.analyze(result, path)
        self.assertEqual(analysis["run_status"], "valid")
        self.assertEqual(
            analysis["tested_hypothesis"], "supported_within_predeclared_bound",
        )
        self.assertEqual(analysis["primary_metric"]["pairs"], 48)
        self.assertEqual(analysis["raw_evidence_audit"]["arm_directories"], 144)
        self.assertEqual(analysis["raw_evidence_audit"]["timed_cells"], 720)
        self.assertFalse(analysis["precision_design"]["prior_campaign_pooled"])
        self.assertTrue(
            analysis["precision_design"]["fixed_sample_no_optional_stopping"]
        )

    def test_endpoint_effect_still_contradicts_instead_of_forcing_equivalence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result, path = fixtures.result_fixture(
                Path(temporary) / "raw", endpoint_effect_pct=3.0,
            )
            analysis = precision_analyzer.analyze(result, path)
        self.assertEqual(analysis["tested_hypothesis"], "contradicted")
        self.assertGreater(analysis["primary_metric"]["ci95_low"], 1.0)

    def test_incomplete_record_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result, path = fixtures.result_fixture(Path(temporary) / "raw")
            removed = result["records"].pop()
            with self.assertRaisesRegex(
                base_analyzer.AnalysisError, "144 arm records",
            ):
                precision_analyzer.analyze(result, path)
            self.assertIsNotNone(removed)

    def test_old_ten_block_result_is_rejected_not_pooled(self) -> None:
        old_path = Path(__file__).resolve().parent / (
            "raw/fixed-work-full-575-01/result.json"
        )
        old_result = json.loads(old_path.read_text())
        with self.assertRaisesRegex(
            base_analyzer.AnalysisError, "wrong experiment kind|parameter",
        ):
            precision_analyzer.analyze(copy.deepcopy(old_result), old_path)


if __name__ == "__main__":
    unittest.main()
