#!/usr/bin/env python3
"""CPU-only tests for the five-arm LMCache GDS performance runner."""

import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("run_gds_five_arm", HERE / "run_gds_five_arm.py")
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class FiveArmTests(unittest.TestCase):
    def test_five_rotations_are_complete_and_position_balanced(self):
        orders = runner.rotation_orders(5)
        self.assertEqual(len(orders), 5)
        for order in orders:
            self.assertEqual(set(order), set(runner.CONFIGS))
        for position in range(5):
            self.assertEqual({order[position] for order in orders}, set(runner.CONFIGS))

    def test_gds_environment_replaces_disk_backend_and_activates_adapter(self):
        base = {
            "LMCACHE_LOCAL_CPU": "False",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": "2.0",
            "LMCACHE_LOCAL_DISK": "file:///old",
            "LMCACHE_MAX_LOCAL_DISK_SIZE": "16.0",
            "PYTHONPATH": "/existing",
        }
        with mock.patch.object(runner, "_BASE_SERVER_ENVIRONMENT", return_value=base.copy()):
            env = runner.gds_server_environment(
                "gds_bpf", Path("/cache"), "610.43.02", 512
            )
        self.assertNotIn("LMCACHE_LOCAL_DISK", env)
        self.assertNotIn("LMCACHE_MAX_LOCAL_DISK_SIZE", env)
        self.assertEqual(env["LMCACHE_GDS_PATH"], "/cache")
        self.assertEqual(env["LMCACHE_GDS_BUFFER_SIZE"], "512")
        self.assertEqual(env["LMCACHE_USE_GDS"], "True")
        self.assertEqual(env["LMCACHE_GDS_BACKEND"], "cufile")
        self.assertEqual(env["LMCACHE_GDS_POLICY_MODE"], "bpf")
        self.assertEqual(env["LMCACHE_LOCAL_CPU"], "False")
        self.assertEqual(env["PYTHONPATH"].split(":"), [
            str(runner.BOOTSTRAP), str(runner.GDS_CONTROL), "/existing"
        ])

    def test_non_gds_environment_is_unchanged(self):
        expected = {"ordinary": "environment"}
        with mock.patch.object(runner, "_BASE_SERVER_ENVIRONMENT", return_value=expected.copy()):
            actual = runner.gds_server_environment(
                "lmcache_cpu", Path("/cache"), "610.43.02", 512
            )
        self.assertEqual(actual, expected)

    def test_summary_uses_numeric_cell_medians(self):
        cells = []
        for index, value in enumerate((3.0, 1.0, 2.0)):
            cells.append({
                "config": "gds_native",
                "metrics": {
                    "warm_ttft_median_ms": value,
                    "warm_requests_per_s": 10.0 + value,
                    "warm_output_tokens_per_s": 100.0 + value,
                },
            })
        summary = runner.median_summary(cells)
        native = summary["per_arm"]["gds_native"]
        self.assertEqual(native["cells_attempted"], 3)
        self.assertEqual(native["cells_measured"], 3)
        self.assertEqual(native["medians"]["warm_ttft_median_ms"], 2.0)
        self.assertIsNone(
            summary["per_arm"]["gds_fifo"]["medians"]["warm_requests_per_s"]
        )

    def test_jsonl_is_one_raw_object_per_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.jsonl"
            with path.open("x", encoding="utf-8") as stream:
                runner.write_jsonl_record(stream, {"config": "gds_fifo", "value": 1})
                runner.write_jsonl_record(stream, {"config": "gds_native", "value": 2})
            records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([record["config"] for record in records],
                         ["gds_fifo", "gds_native"])

    def test_dry_run_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "unused"
            stdout = io.StringIO()
            with mock.patch("sys.stdout", stdout):
                status = runner.main(["--dry-run", "--output", str(output)])
            self.assertEqual(status, 0)
            self.assertFalse(output.exists())
            plan = json.loads(stdout.getvalue())
            self.assertEqual(plan["configs"], list(runner.CONFIGS))
            self.assertEqual(plan["gds"]["buffer_size_mib"], 512)

    def test_default_driver_is_the_live_575_campaign_driver(self):
        args = runner.parse_args([])
        self.assertEqual(args.expected_driver, "575.57.08")


if __name__ == "__main__":
    unittest.main()
