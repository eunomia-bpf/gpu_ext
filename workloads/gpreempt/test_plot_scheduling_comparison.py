"""Synthetic contract tests only; never render figures during GPU timing.

Temporary fixtures contain invented values and are not experimental evidence.
The rendering routine is mocked: these tests do not import matplotlib.
"""
import copy
import json
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch

import plot_scheduling_comparison as plot


def gp_fixture():
    audit = {"complete": True, "formal_eligible": True, "rejected_cells": [],
             "incomplete_cells": [], "scenarios": {}}
    for scenario in plot.gp_plot.SCENARIOS:
        audit["scenarios"][scenario] = {"complete": True, "valid_paired_blocks": 5,
            "per_cell_points": [{"block": block, "arm": arm, "metrics": {
                "vgg_rt": {"p99_response_us": 2000 + block, "completion_coverage": 1.,
                           "p99_is_conditional": False},
                "resnet152_be": {"goodput_rps": 50 + block}}}
                for block in range(5) for arm in plot.gp_plot.ARMS]}
    return audit


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def xs_fixture(directory):
    protocol = {"phase": "full", "repetitions": 10, "tasks_per_stream": 50,
                "streams_per_process": 4, "lc_processes": 2, "be_processes": 4,
                "xsched_level": 1, "full_50_kernel_10_block_protocol": True,
                "short_budget_difference": None, "configs": list(plot.XS_ALL_ARMS),
                "schedule": [list(plot.XS_ALL_ARMS) for _ in range(10)],
                "reps": 123, "blocks": 2, "threads": 8}
    audit = {"status": "passed", "audited_cells": 46, "complete_blocks": 10,
             "mixed_cells": 40, "isolated_controls": 6, "actual_order_verified": True,
             "aggregate_recomputation_equal": True,
             "per_cell_policy_engagement_and_safety_verified": True,
             "per_worker_argv_environment_clock_numerics_verified": True,
             "verified_directories": []}
    for role in ("lc", "be"):
        for index in range(3):
            name = f"control-{role}-{index}"
            audit["verified_directories"].append(name)
            write_json(directory / name / "result.json", {"synthetic": True})
    summary = {"protocol": protocol, "complete_blocks": 10, "configs": {}}
    for arm_index, arm in enumerate(plot.XS_ALL_ARMS):
        latencies, rates = [], []
        for block in range(10):
            latency, rate = (arm_index + 1) * 1_000_000 + block * 1000, 10 + arm_index + block / 10
            latencies.append(latency)
            rates.append(rate)
            name = f"block-{block:02d}-{arm_index}-{arm}"
            audit["verified_directories"].append(name)
            write_json(directory / name / "result.json", {
                "config": arm, "block": block, "order_index": arm_index,
                "reps": 123, "tasks_per_stream": 50, "blocks": 2, "threads": 8,
                "lc_samples": 400, "be_completed": 800, "outputs_validated_per_process": 3200,
                "lc_p99_us": latency, "lc_completion_p99_us": 999_000_000,
                "be_throughput_kernels_s": rate})
        summary["configs"][arm] = {"lc_p99_median_us": statistics.median(latencies),
                                   "be_throughput_median": statistics.median(rates)}
    write_json(directory / "protocol.json", protocol)
    write_json(directory / "summary.json", summary)
    path = directory / "independent-raw-audit.json"
    write_json(path, audit)
    return path


class ComparisonFigureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-scheduling-contract-")
        self.root = Path(self.temporary.name)
        self.xs = xs_fixture(self.root / "xs")

    def tearDown(self):
        self.temporary.cleanup()

    def test_full_inventory_units_and_excluded_different_driver_policy(self):
        data = plot.comparison_data(self.xs, gp_fixture())
        self.assertEqual(len(data["xsched"]), 30)
        self.assertEqual(len(data["gpreempt"]), 45)
        self.assertEqual({p["arm"] for p in data["xsched"]}, set(plot.XS_ARMS))
        self.assertEqual(data["xsched"][0]["queue_p99_s"], 1.)
        self.assertEqual(data["gpreempt"][0]["response_p99_ms"], 2.)
        self.assertFalse(data["scope"]["cross_workload_latency_comparison"])

    def test_missing_or_partial_xs_raw_audit_rejected(self):
        audit = json.loads(self.xs.read_text())
        for key, value in (("complete_blocks", 5), ("status", "failed"),
                           ("aggregate_recomputation_equal", False), ("audited_cells", 45)):
            changed = {**audit, key: value}
            write_json(self.xs, changed)
            with self.assertRaises(ValueError):
                plot.xsched_points(self.xs)
        write_json(self.xs, audit)
        (self.xs.parent / "block-00-0-native/result.json").unlink()
        with self.assertRaises(ValueError):
            plot.xsched_points(self.xs)

    def test_rejects_unverified_duplicate_and_unlisted_results(self):
        audit = json.loads(self.xs.read_text())
        audit["verified_directories"][-1] = audit["verified_directories"][0]
        write_json(self.xs, audit)
        with self.assertRaises(ValueError):
            plot.xsched_points(self.xs)
        self.xs = xs_fixture(self.root / "xs")
        write_json(self.xs.parent / "block-10-0-native/result.json", {})
        with self.assertRaises(ValueError):
            plot.xsched_points(self.xs)

    def test_xs_wrong_arm_counter_nan_and_aggregate_drift_rejected(self):
        path = self.xs.parent / "block-00-0-native/result.json"
        row = json.loads(path.read_text())
        for key, value in (("config", "gpubpf"), ("lc_samples", 20),
                           ("be_completed", 799), ("lc_p99_us", float("nan")),
                           ("be_throughput_kernels_s", 0)):
            write_json(path, {**row, key: value})
            with self.assertRaises(ValueError):
                plot.xsched_points(self.xs)
        write_json(path, row)
        summary_path = self.xs.parent / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["configs"]["native"]["lc_p99_median_us"] *= 2
        write_json(summary_path, summary)
        with self.assertRaises(ValueError):
            plot.xsched_points(self.xs)

    def test_reuses_gp_final_45_cell_contract(self):
        for mutation in (lambda a: a.update(complete=False),
                         lambda a: a.update(formal_eligible=False),
                         lambda a: a["scenarios"]["be100"]["per_cell_points"].pop(),
                         lambda a: a.update(rejected_cells=[{"cell": 1}])):
            audit = gp_fixture()
            mutation(audit)
            with self.assertRaises(ValueError):
                plot.comparison_data(self.xs, audit)

    def test_caption_distinguishes_metrics_scope_variation_and_censoring(self):
        audit = gp_fixture()
        foreground = audit["scenarios"]["be100"]["per_cell_points"][0]["metrics"]["vgg_rt"]
        foreground.update(completion_coverage=.5, p99_is_conditional=True)
        text = plot.caption(plot.comparison_data(self.xs, audit))
        for phrase in ("Level-1", "50 kernels per stream", "first CTA entry", "in seconds",
                       "in milliseconds", "FIFO waiting", "host-mapped", "not original GDRCopy",
                       "conditional", "50.0%", "different workloads", "No confidence intervals"):
            self.assertIn(phrase, text)
        self.assertIn("driver-only gpubpf", text)

    def test_prepare_outputs_with_mocked_drawing_and_no_overwrite(self):
        audit_path = self.root / "synthetic-gp-audit.json"
        write_json(audit_path, gp_fixture())
        prefix = self.root / "not-results/comparison"
        with patch.object(plot, "_draw") as draw:
            outputs = plot.render(self.xs, audit_path, prefix)
        draw.assert_called_once()
        self.assertEqual(len(outputs), 4)
        # No real PDF/PNG is generated by CPU contract tests.
        self.assertFalse(outputs[0].exists())
        self.assertFalse(outputs[1].exists())
        self.assertTrue(outputs[2].exists())
        points = json.loads(outputs[3].read_text())
        self.assertEqual(len(points["xsched"]), 30)
        self.assertEqual(len(points["gpreempt"]), 45)
        with self.assertRaises(FileExistsError), patch.object(plot, "_draw"):
            plot.render(self.xs, audit_path, prefix)

    def test_reject_small_canvas_instead_of_unreadable_downscale(self):
        with self.assertRaises(ValueError):
            plot.render(self.xs, self.root / "unused.json", self.root / "unused", 3.3)


if __name__ == "__main__":
    unittest.main()
