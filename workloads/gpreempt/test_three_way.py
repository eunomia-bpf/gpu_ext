#!/usr/bin/env python3
"""CPU-only parser, protocol, pairing, environment and owned-process tests."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import run_three_way as runner


def report_fixture(count=150):
    results, checks = [], []
    for name in runner.TASKS:
        analyzer = {"type": "basic", "completedRequests": count,
                    "avgThroughput(req/s)": count / 60,
                    "requestLatencyNs": [(index + 1) * 1000 for index in range(count)],
                    "latencyDefinition": "sum_of_original_six_recorded_stages"}
        results.append({"clientName": name, "analyzers": [analyzer]})
        checks.append({"task": name, "checked": count + 110, "timed_checked": count,
                       "atol": 1e-6, "rtol": 1e-4, "max_absolute_error": 0.0})
    return {"benchmarkTime(s)": 60, "results": results}, checks


def report_log(report, checks):
    return json.dumps(report, indent=2) + "\n" + "\n".join(
        "GPREEMPT_VALIDATION " + json.dumps(check) for check in checks)


def engagement_fixture():
    bridge = ("gpreempt_hint_ready: backend=ubpf-jit\n"
              "gpreempt_context_registered: role=0 hclient=1 htsg=2 tsg_id=10 engine=1 timeslice_us=1000000 cuda_context=100\n"
              "gpreempt_context_registered: role=1 hclient=1 htsg=3 tsg_id=11 engine=1 timeslice_us=1 cuda_context=101\n"
              "gpreempt_bridge_stats: backend=ubpf-jit preprocess=500 due=1000 infer=500 reset=150 hint=150 "
              "block=150 release=150 scopes=2 registered=2 ended=2 errors=0\n")
    kernel = ("gpreempt_policy_stats: scope_enter=2 scope_leave=2 gr_init=2 timeslice_ok=2 alloc_captured=2 "
              "registered=2 destroy=2 unknown_engine=0 setter_error=0 alloc_error=0 register_error=0 "
              "bind_shadow_mismatch=0 map_error=0 scope_error=0\n")
    return bridge, kernel


class ComparisonTests(unittest.TestCase):
    def test_balanced_five_blocks_are_deterministic_and_complete(self):
        selected = runner.orders(5, 20260902)
        self.assertEqual(selected, runner.orders(5, 20260902))
        self.assertEqual(len({tuple(order) for order in selected}), 5)
        for order in selected:
            self.assertEqual(set(order), set(runner.ARMS))
        for arm in runner.ARMS:
            positions = [sum(order[index] == arm for order in selected) for index in range(3)]
            self.assertLessEqual(max(positions) - min(positions), 1)

    def test_actual_counts_and_nearest_rank_p99_not_fixed_6000(self):
        report, checks = report_fixture()
        metrics = runner.parse_report(report_log(report, checks))["metrics"]
        self.assertEqual(metrics[runner.TASKS[0]]["p99_latency_us"], 149)
        self.assertEqual(metrics[runner.TASKS[1]]["completed_requests"], 150)
        self.assertIsNone(metrics[runner.TASKS[0]]["observed_arrivals"])

    def test_incorrect_numerics_count_tolerance_or_nan_rejected(self):
        for key, value in (("checked", 150), ("timed_checked", 149), ("atol", 1e-4),
                           ("max_absolute_error", float("nan"))):
            report, checks = report_fixture()
            checks[0][key] = value
            with self.assertRaises(ValueError):
                runner.parse_report(report_log(report, checks))

    def test_wrong_duration_missing_samples_or_wrong_throughput_rejected(self):
        report, checks = report_fixture()
        for mutate in (lambda r: r.update({"benchmarkTime(s)": 5}),
                       lambda r: r["results"][0]["analyzers"][0].update({"requestLatencyNs": []}),
                       lambda r: r["results"][0]["analyzers"][0].update({"avgThroughput(req/s)": 100})):
            altered = copy.deepcopy(report)
            mutate(altered)
            with self.assertRaises(ValueError):
                runner.parse_report(report_log(altered, checks))
        short_report, short_checks = report_fixture(99)
        with self.assertRaises(ValueError):
            runner.parse_report(report_log(short_report, short_checks))

    def test_bpf_full_engagement_and_zero_events_are_distinct(self):
        client, loader = engagement_fixture()
        self.assertEqual(runner.check_engagement("bpf_gpreempt", client, loader)["backend"], "ubpf-jit")
        for old, new in (("gr_init=2", "gr_init=0"), ("destroy=2", "destroy=0"),
                         ("setter_error=0", "setter_error=1")):
            with self.assertRaises(ValueError):
                runner.check_engagement("bpf_gpreempt", client, loader.replace(old, new))
        for old, new in (("role=1", "role=0"), ("cuda_context=101", "cuda_context=100"),
                         ("backend=ubpf-jit", "backend=original-c"), ("errors=0", "errors=1")):
            with self.assertRaises(ValueError):
                runner.check_engagement("bpf_gpreempt", client.replace(old, new), loader)

    def test_original_backend_does_not_silently_use_bpf(self):
        original = ("gpreempt_bridge_stats: backend=original-c preprocess=500 due=100 infer=500 "
                    "reset=150 hint=150 block=150 release=150 scopes=0 registered=0 ended=0 errors=0\n")
        self.assertEqual(runner.check_engagement("original_gpreempt", original, "")["backend"], "original-c")
        with self.assertRaises(ValueError):
            runner.check_engagement("native", original, "")

    def test_environment_never_inherits_preload_and_selects_private_gdr(self):
        env = runner.environment("bpf_gpreempt", Path("/sys/fs/bpf/fixture"), Path("/private/gdr"))
        self.assertNotIn("LD_PRELOAD", env)
        self.assertIn("/private/gdr/src", env["LD_LIBRARY_PATH"])
        self.assertEqual(env["GPREEMPT_POLICY"], "bpf")

    def test_partial_blocks_are_not_completion(self):
        report, checks = report_fixture()
        metrics = runner.parse_report(report_log(report, checks))["metrics"]
        rows = [{"arm": arm, "block": block, "status": "passed", "metrics": metrics}
                for block in range(5) for arm in runner.ARMS]
        self.assertTrue(runner.summarize(rows, 5)["formal_5_block_complete"])
        self.assertFalse(runner.summarize(rows[:-1], 5)["formal_5_block_complete"])

    def test_cleanup_finds_orphan_after_owned_leader_exits(self):
        # Only a finite, owned CPU sleeping child; no CUDA imports or device access.
        code = "import os,time\npid=os.fork()\nif pid: os._exit(0)\ntime.sleep(15)\n"
        child = subprocess.Popen([sys.executable, "-c", code], start_new_session=True,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            child.wait(timeout=2)
            self.assertTrue(runner.group_members(child.pid))
            runner.stop_owned(child)
            self.assertEqual(runner.group_members(child.pid), [])
        finally:
            runner.stop_owned(child)


if __name__ == "__main__":
    unittest.main()
