#!/usr/bin/env python3
"""CPU-only regression tests: do not attach BPF or launch CUDA."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import run_xsched_rq3 as runner


class RunnerTests(unittest.TestCase):
    def test_clock_offset_is_bounded(self):
        before = {"offset_ns": -300000000, "uncertainty_ns": 3000}
        after = {"offset_ns": -300010000, "uncertainty_ns": 4000}
        self.assertEqual(runner.validate_clock_pair(before, after), {
            "offset_drift_ns": 10000, "conservative_error_bound_ns": 17000,
        })
        with self.assertRaises(RuntimeError):
            runner.validate_clock_pair(before, {"offset_ns": 0, "uncertainty_ns": 1})

    def test_decision_categories_are_disjoint(self):
        cases = [
            ([-20, -10], [0, .02], "positive"),
            ([-20, -10], [-.20, -.10], "mixed"),
            ([10, 20], [0, .02], "negative"),
            ([-20, 10], [-.20, -.10], "negative"),
            ([-20, -10], [-.06, -.04], "inconclusive"),
            ([-20, 10], [-.01, .02], "inconclusive"),
        ]
        for latency, throughput, expected in cases:
            with self.subTest(latency=latency, throughput=throughput):
                self.assertEqual(runner.classify_comparison(latency, throughput), expected)

    def test_short_budget_p99_is_maximum(self):
        self.assertEqual(runner.percentile(list(range(40)), .99), 39)
        self.assertEqual(runner.percentile(list(range(400)), .99), 395)

    def test_workload_environment_uses_upstream_role_settings(self):
        with mock.patch.dict(runner.os.environ, {"XSCHED_CUDA_LV3_IMPL": "trap", "LD_PRELOAD": "foreign"}):
            for role, threshold, batch in (("lc", "16", "8"), ("be", "4", "2")):
                env = runner.workload_env("xsched", role)
                self.assertNotIn("XSCHED_CUDA_LV3_IMPL", env)
                self.assertNotIn("LD_PRELOAD", env)
                self.assertEqual(env["XSCHED_AUTO_XQUEUE_LEVEL"], "1")
                self.assertEqual(env["XSCHED_AUTO_XQUEUE_THRESHOLD"], threshold)
                self.assertEqual(env["XSCHED_AUTO_XQUEUE_BATCH_SIZE"], batch)

    def test_partial_campaign_is_not_a_completed_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "protocol.json").write_text(json.dumps({
                "phase": "pilot", "repetitions": 5, "tasks_per_stream": 5,
            }))
            with self.assertRaisesRegex(RuntimeError, "5 complete"):
                runner.analyze(root)

    def test_complete_pilot_preserves_scope_and_all_three_configs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "protocol.json").write_text(json.dumps({
                "phase": "pilot", "repetitions": 5, "tasks_per_stream": 5,
                "full_50_kernel_10_block_protocol": False,
            }))
            for block in range(5):
                for config, latency in (("native", 300), ("xsched", 200), ("gpubpf", 100)):
                    directory = root / f"{block}-{config}"
                    directory.mkdir()
                    (directory / "result.json").write_text(json.dumps({
                        "block": block, "config": config, "tasks_per_stream": 5,
                        "lc_p99_us": latency, "be_throughput_kernels_s": 10,
                        "lc_samples": 40, "be_completed": 80,
                        "lc_p99_is_sample_maximum": True,
                    }))
            with mock.patch.object(runner, "bootstrap_ci", side_effect=lambda values, seed: [min(values), max(values)]):
                summary = runner.analyze(root)
            self.assertEqual(summary["complete_blocks"], 5)
            self.assertEqual(set(summary["configs"]), set(runner.CONFIGS))
            self.assertFalse(summary["protocol"]["full_50_kernel_10_block_protocol"])
            self.assertEqual(summary["predeclared_decision"]["classification"], "positive")

    def test_cell_records_failure_and_checks_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(runner.shared, "safety_snapshot", return_value={"state": "before"}), \
                 mock.patch.object(runner.shared, "validate_pre_server_safety"), \
                 mock.patch.object(runner.shared, "wait_for_post_server_safety", return_value={"state": "after"}) as after:
                with self.assertRaisesRegex(ValueError, "test failure"):
                    with runner.safe_cell(root):
                        raise ValueError("test failure")
            after.assert_called_once()
            self.assertEqual(json.loads((root / "failure.json").read_text())["error_type"], "ValueError")
            self.assertTrue((root / "safety-after.json").exists())


if __name__ == "__main__":
    unittest.main()
