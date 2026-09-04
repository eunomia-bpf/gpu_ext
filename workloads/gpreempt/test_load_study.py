"""CPU-only parser/plan/lifecycle checks. No CUDA, device or real child execution."""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import run_load_study as run


def fixture(scenario="be100", seconds=10, count=150):
    config = run.make_config(scenario, seconds)
    begin, end = 2_000_000_000, (seconds + 2) * 1_000_000_000
    report = {"benchmarkTime(s)": seconds, "loadStudyBeginNs": begin,
              "loadStudyEndNs": end, "loadStudyClock": "steady_clock", "results": []}
    checks, loads = [], []
    for task in config["tasks"]:
        name, periodic = task["id"], task["load"]["type"] == "periodic_fifo"
        interval = 1_000_000_000 // task["load"]["frequency"] if periodic else 0
        requests = []
        for index in range(count):
            start = begin + index * (interval or 2_000_000)
            requests.append([index, start, start, start + 1_000_000])
        loads.append({"task": name, "clock": "steady_clock", "mode": "periodic_fifo" if periodic else "continuous_closed_loop",
                      "phase_ns": 0, "begin_ns": begin, "end_ns": end, "interval_ns": interval,
                      "offered": seconds * task["load"]["frequency"] if periodic else None,
                      "started": count, "request_fields": "id,scheduled_ns,started_ns,verified_ready_ns", "requests": requests})
        checks.append({"task": name, "checked": count + 110, "timed_checked": count,
                       "max_absolute_error": 0.0, "atol": 1e-6, "rtol": 1e-4})
        report["results"].append({"clientName": name, "analyzers": [{"type": "basic", "completedRequests": count,
            "avgThroughput(req/s)": count / seconds, "requestLatencyNs": [900_000] * count,
            "latencyDefinition": "sum_of_original_six_recorded_stages"}]})
    return config, report, checks, loads


def log_fixture(report, checks, loads, native=True):
    pieces = [json.dumps(report, indent=2)]
    pieces += ["GPREEMPT_VALIDATION " + json.dumps(c) for c in checks]
    pieces += ["GPREEMPT_LOAD_STUDY " + json.dumps(row) for row in loads]
    if native:
        pieces += [f"GPREEMPT_LOAD_PRIORITY task={name} role={role} actual={0 if role else -2} least=0 greatest=-2"
                   for role, name in enumerate(run.TASKS)]
    return "\n".join(pieces) + "\n"


class LoadStudyTests(unittest.TestCase):
    def parse(self, values, native=True):
        config, report, checks, loads = values
        return run.parse_report(log_fixture(report, checks, loads, native), config,
                                "native" if native else "original_gpreempt")

    def test_fixed_plan_has_45_unique_cells_balanced_scenarios_and_arms(self):
        plan = run.make_plan("full")
        self.assertEqual(plan, run.make_plan("full"))
        self.assertEqual(plan["required_cells"], 45)
        keys = [(r["block"], r["scenario"], r["arm"]) for r in plan["orders"]]
        self.assertEqual(len(set(keys)), 45)
        self.assertEqual([r["cell"] for r in plan["orders"]], list(range(45)))
        for scenario in run.SCENARIOS:
            positions = [sum(order[index] == scenario for order in plan["scenario_orders"]) for index in range(3)]
            self.assertLessEqual(max(positions) - min(positions), 1)
            for arm in run.ARMS:
                counts = [sum([r["arm"] for r in plan["orders"] if r["block"] == block and r["scenario"] == scenario][i] == arm
                              for block in range(5)) for i in range(3)]
                self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertEqual(plan["statistics"]["draws"], 10000)

    def test_preflight_is_only_continuous_10s_and_never_formal(self):
        plan = run.make_plan("preflight")
        self.assertEqual(plan["required_cells"], 3)
        self.assertEqual(plan["scenarios"], ["be_continuous"])
        self.assertEqual(plan["timed_seconds_per_cell"], 10)
        rows = [{**spec, "status": "passed", "metrics": {}} for spec in plan["orders"]]
        self.assertTrue(run.summarize(rows, plan)["complete"])
        self.assertFalse(run.summarize(rows, plan)["formal_complete"])

    def test_lc_knee_plan_is_prespecified_supporting_evidence(self):
        plan = run.make_plan("full", study="lc-knee")
        self.assertEqual(plan["required_cells"], 27)
        self.assertEqual(plan["blocks"], 3)
        self.assertEqual(plan["scenarios"], ["lc500", "lc625", "lc800"])
        self.assertEqual(plan["prespecified_lc_rates_rps"], [500, 625, 800])
        self.assertEqual(plan["evidence_role"], "supporting")
        self.assertFalse(plan["post_hoc_rate_additions_allowed"])
        keys = [(row["block"], row["scenario"], row["arm"]) for row in plan["orders"]]
        self.assertEqual(len(keys), len(set(keys)))
        for scenario in run.LC_KNEE_SCENARIOS:
            config = plan["configs"][scenario]
            self.assertEqual(config["tasks"][0]["load"]["frequency"], int(scenario[2:]))
            self.assertEqual(config["tasks"][1]["load"], {"type": "continuous"})
            for arm in run.ARMS:
                positions = [[row["arm"] for row in plan["orders"]
                              if row["block"] == block and row["scenario"] == scenario].index(arm)
                             for block in range(3)]
                self.assertEqual(sorted(positions), [0, 1, 2])
        for scenario in run.LC_KNEE_SCENARIOS:
            positions = [order.index(scenario) for order in plan["scenario_orders"]]
            self.assertEqual(sorted(positions), [0, 1, 2])
        rows = [{**spec, "status": "passed", "metrics": {}} for spec in plan["orders"]]
        self.assertTrue(run.summarize(rows, plan)["formal_complete"])

    def test_lc_knee_preflight_is_one_lc800_three_arm_block(self):
        plan = run.make_plan("preflight", study="lc-knee")
        self.assertEqual(plan["required_cells"], 3)
        self.assertEqual(plan["blocks"], 1)
        self.assertEqual(plan["scenarios"], ["lc800"])
        self.assertEqual(plan["timed_seconds_per_cell"], 10)
        self.assertEqual({row["arm"] for row in plan["orders"]}, set(run.ARMS))
        rows = [{**spec, "status": "passed", "metrics": {}} for spec in plan["orders"]]
        self.assertTrue(run.summarize(rows, plan)["complete"])
        self.assertFalse(run.summarize(rows, plan)["formal_complete"])

    def test_lc_knee_full_requires_explicit_preflight_but_plan_only_does_not_read_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "full"
            with patch("sys.argv", ["run_load_study.py", "full", "--study", "lc-knee",
                                    "--output", str(output)]), \
                 patch.object(run.os, "geteuid", return_value=0), \
                 contextlib.redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit):
                run.main()
            self.assertFalse(output.exists())
            missing = root / "not-present"
            with patch("sys.argv", ["run_load_study.py", "full", "--study", "lc-knee",
                                    "--preflight", str(missing), "--plan"]), \
                 patch.object(run, "validate_completed_preflight",
                              side_effect=AssertionError("plan-only read preflight")) as gate, \
                 contextlib.redirect_stdout(io.StringIO()):
                run.main()
            gate.assert_not_called()

    def test_completed_preflight_gate_delegates_to_independent_analyzer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = {"campaign": str((root / "preflight").resolve()), "complete": True}
            with patch("analyze_load_study.validate_completed_preflight", return_value=expected) as audit:
                self.assertEqual(run.validate_completed_preflight(root / "preflight", root / "full"), expected)
            audit.assert_called_once_with((root / "preflight").resolve())
            with self.assertRaises(ValueError):
                run.validate_completed_preflight(root / "preflight", root / "preflight/child")

    def test_configs_reject_policy_model_repeat_type_or_rate_drift(self):
        for scenario in run.SCENARIOS:
            config = run.make_config(scenario, 60)
            self.assertEqual(run.validate_config(config), scenario)
            for field, value in (("preprocess_time", 201), ("batch_size", 2), ("use_cuda_graph", 1), ("priority", False)):
                changed = copy.deepcopy(config)
                changed["tasks"][0]["client"][field] = value
                with self.assertRaises(ValueError):
                    run.validate_config(changed)
        with self.assertRaises(ValueError):
            run.make_config("be100", True)
        for scenario in run.LC_KNEE_SCENARIOS:
            config = run.make_config(scenario, 60, "lc-knee")
            self.assertEqual(run.validate_config(config, study="lc-knee"), scenario)
            changed = copy.deepcopy(config)
            changed["tasks"][1]["load"] = {"type": "periodic_fifo", "frequency": 200, "priority": 0}
            with self.assertRaises(ValueError):
                run.validate_config(changed, scenario, "lc-knee")

    def test_fifo_response_includes_wait_and_backlog_without_discarding(self):
        values = fixture()
        values[3][0]["requests"][-1][2] += 2_000_000
        values[3][0]["requests"][-1][3] += 2_000_000
        result = self.parse(values)["metrics"][run.TASKS[0]]
        self.assertEqual(result["started_requests"], 150)
        self.assertEqual(result["offered_requests"], 1000)
        self.assertEqual(result["never_started_backlog"], 850)
        self.assertEqual(result["goodput_rps"], 15)
        self.assertEqual(result["completion_coverage"], .15)
        self.assertTrue(result["response_p99_conditional"])
        self.assertGreater(result["mean_response_us"], 1000)

    def test_exact_cutoff_completion_excluded_from_goodput_not_numerics(self):
        values = fixture(count=1000)
        values[3][0]["requests"][-1][3] = values[1]["loadStudyEndNs"]
        metric = self.parse(values)["metrics"][run.TASKS[0]]
        self.assertEqual(metric["completed_in_window"], 999)
        self.assertEqual(metric["completed_after_window"], 1)
        self.assertEqual(metric["goodput_rps"], 99.9)
        self.assertEqual(metric["unfinished_offered_at_deadline"], 1)
        self.assertEqual(metric["completion_coverage"], 1)
        self.assertEqual(metric["window_completion_fraction"], .999)
        self.assertFalse(metric["response_p99_conditional"])

    def test_continuous_has_no_fictitious_offered_denominator(self):
        metric = self.parse(fixture("be_continuous"))["metrics"][run.TASKS[1]]
        for field in ("offered_requests", "never_started_backlog", "unfinished_offered_at_deadline",
                      "completion_coverage", "window_completion_fraction"):
            self.assertIsNone(metric[field])
        self.assertEqual(metric["goodput_rps"], 15)
        self.assertFalse(metric["response_p99_conditional"])

    def test_zero_or_small_sample_count_is_retained_and_explicit(self):
        for count in (0, 1, 99):
            metric = self.parse(fixture(count=count))["metrics"][run.TASKS[0]]
            self.assertEqual(metric["started_requests"], count)
            self.assertTrue(metric["p99_less_than_100_samples"])
            self.assertTrue(metric["response_p99_conditional"])
            self.assertEqual(metric["response_p99_us"], 1000 if count else None)

    def test_request_gap_wrong_epoch_late_start_overlap_and_unfinished_rejected(self):
        mutations = [lambda row: row["requests"][3].__setitem__(0, 4),
                     lambda row: row["requests"][3].__setitem__(1, row["requests"][3][1] + 1),
                     lambda row: row["requests"][3].__setitem__(2, row["end_ns"]),
                     lambda row: row["requests"][3].__setitem__(2, row["requests"][2][2]),
                     lambda row: row["requests"][3].__setitem__(3, 0),
                     lambda row: row.update(begin_ns=row["begin_ns"] + 1),
                     lambda row: row.update(interval_ns=1),
                     lambda row: row.update(phase_ns=True),
                     lambda row: row.update(offered=1000.0)]
        for mutate in mutations:
            values = fixture()
            mutate(values[3][0])
            with self.assertRaises(ValueError):
                self.parse(values)

    def test_counts_and_invalid_numeric_checks_or_service_fail(self):
        for key, value in (("checked", 150), ("timed_checked", 149), ("atol", 1e-4),
                           ("max_absolute_error", float("nan")), ("max_absolute_error", -1)):
            values = fixture()
            values[2][0][key] = value
            with self.assertRaises(ValueError):
                self.parse(values)
        values = fixture()
        values[1]["results"][0]["analyzers"][0]["requestLatencyNs"] = []
        with self.assertRaises(ValueError):
            self.parse(values)

    def test_duplicate_rows_wrong_duration_or_old_binary_rejected(self):
        values = fixture()
        config, report, checks, loads = values
        for log in (log_fixture(report, checks, loads + [loads[0]]),
                    log_fixture(report, checks, loads).replace('"loadStudyClock": "steady_clock"', '"loadStudyClock": "system_clock"'),
                    log_fixture(report, checks, loads).replace("GPREEMPT_LOAD_STUDY ", "OLD ")):
            with self.assertRaises(ValueError):
                run.parse_report(log, config, "native")

    def test_native_priorities_actual_values_required(self):
        values = fixture()
        config, report, checks, loads = values
        log = log_fixture(report, checks, loads)
        with self.assertRaises(ValueError):
            run.parse_report(log.replace("actual=-2", "actual=0"), config, "native")
        with self.assertRaises(ValueError):
            run.parse_report(log, config, "original_gpreempt")
        self.assertEqual(self.parse(values, native=False)["native_priorities"], {})

    def test_environment_and_commands_only_use_new_private_build(self):
        for arm in run.ARMS:
            env = run.environment(arm, Path("/sys/fs/bpf/fixture"))
            self.assertNotIn("build/ninja", env["LD_LIBRARY_PATH"])
            self.assertIn("build/load-study", env["LD_LIBRARY_PATH"])
            self.assertNotIn("LD_PRELOAD", env)
            command = run.client_command(arm, Path("/tmp/config.json"))
            self.assertEqual(Path(command[0]).parent, run.BUILD)
            if arm != "native":
                self.assertEqual(command[-2:], ["--flag-transport", "host_mapped"])

    def test_partial_duplicate_or_foreign_cells_do_not_complete(self):
        plan = run.make_plan("full")
        rows = [{**spec, "status": "passed", "metrics": {}} for spec in plan["orders"]]
        self.assertTrue(run.summarize(rows, plan)["formal_complete"])
        self.assertFalse(run.summarize(rows[:-1], plan)["formal_complete"])
        with self.assertRaises(ValueError):
            run.summarize(rows + [rows[0]], plan)
        rows[0]["block"] = 9
        with self.assertRaises(ValueError):
            run.summarize(rows, plan)

    def test_mock_native_cell_saves_raw_and_cleans_up_without_gpu(self):
        values = fixture("be_continuous")
        config, report, checks, loads = values
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            config_path = parent / "config.json"
            # Test fixture write is JSON serialization, not editing project files.
            config_path.write_text(json.dumps(config))
            client, telemetry = Mock(), Mock()
            client.returncode = 0
            client.poll.return_value = 0
            def popen(command, **kwargs):
                self.assertEqual(Path(command[0]).parent, run.BUILD)
                kwargs["stdout"].write(log_fixture(report, checks, loads))
                kwargs["stdout"].flush()
                return client
            def start_telemetry(directory):
                path = directory / "gpu-telemetry.csv"
                return telemetry, path.open("x"), path
            with patch.object(run, "runtime_inventory", return_value={"fixture": {"bytes": 1}}), \
                 patch.object(run.safety, "safety_snapshot", return_value={"gpu": {"driver": "575.57.08"}}), \
                 patch.object(run.safety, "validate_pre_server_safety"), \
                 patch.object(run.safety, "wait_for_post_server_safety", return_value={"mock_idle": True}), \
                 patch.object(run.safety, "start_gpu_telemetry", side_effect=start_telemetry), \
                 patch.object(run.safety, "validate_gpu_telemetry", return_value={"samples": 1}), \
                 patch.object(run.subprocess, "Popen", side_effect=popen), \
                 patch.object(run, "stop_owned") as stop:
                result = run.run_cell(parent / "cell", {"cell": 0, "block": 0, "scenario": "be_continuous", "arm": "native"}, config_path, 240)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["runtime_before"], result["runtime_after"])
            self.assertIn(client, [call.args[0] for call in stop.call_args_list])
            self.assertIn(telemetry, [call.args[0] for call in stop.call_args_list])
            self.assertTrue((parent / "cell/request-report.json").exists())
            self.assertTrue((parent / "cell/load-study-report.json").exists())


if __name__ == "__main__":
    unittest.main()
