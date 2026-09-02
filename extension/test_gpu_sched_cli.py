#!/usr/bin/env python3
"""Exercise policy parsing without loading BPF or opening a GPU device."""
from pathlib import Path
import subprocess
import unittest


LOADER = Path(__file__).resolve().parent / "gpu_sched_set_timeslices"


class SchedulerPolicyCli(unittest.TestCase):
    def test_valid_priorities_and_timeslices(self):
        # -h exits after parsing all preceding policies, before skeleton load.
        for policy in ("bench_be:0", "bench_lc:1", "bench_lc:2"):
            with self.subTest(policy=policy):
                result = subprocess.run(
                    [str(LOADER), "-i", policy, "-p", "bench_lc:1000000", "-h"],
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_policies_fail_before_load(self):
        cases = {
            "-i": ("bench:3", "bench:-1", "bench:", "bench:1x", "bench:+1",
                   ":0", "abcdefghijklmnop:1", "bench:18446744073709551616"),
            "-p": ("bench:0", "bench:-1", "bench:", "bench:20x", ":200"),
        }
        for option, policies in cases.items():
            for policy in policies:
                with self.subTest(option=option, policy=policy):
                    result = subprocess.run(
                        [str(LOADER), option, policy, "-h"],
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertNotIn("libbpf:", result.stderr)


if __name__ == "__main__":
    unittest.main()
