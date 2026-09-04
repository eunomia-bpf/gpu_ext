"""CPU-only adversarial tests of raw FIFO/continuous measurement auditing."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analyze_load_study as audit
import run_load_study as runner


def fixture(scenario="be100", count=3, seconds=60, arm="native", study="load"):
    begin, end = 1_000_000_000, (seconds + 1) * 1_000_000_000
    rows, checks, results = [], [], []
    expected_loads = audit.expected_loads(study, scenario)
    for role, task in enumerate(audit.base.TASKS):
        expected_load = expected_loads[role]
        continuous = expected_load["type"] == "continuous"
        interval = 0 if continuous else audit.NS // expected_load["frequency"]
        requests = []
        for identifier in range(count):
            scheduled = begin + identifier * (interval or 1_000_000)
            requests.append([identifier, scheduled, scheduled, scheduled + 100_000])
        rows.append({"task": task, "clock": "steady_clock",
                     "mode": "continuous_closed_loop" if continuous else "periodic_fifo",
                     "phase_ns": 0, "begin_ns": begin, "end_ns": end, "interval_ns": interval,
                     "offered": None if continuous else seconds * audit.NS // interval,
                     "started": count, "request_fields": "id,scheduled_ns,started_ns,verified_ready_ns",
                     "requests": requests})
        checks.append({"task": task, "checked": count + 110, "timed_checked": count,
                       "atol": 1e-6, "rtol": 1e-4, "max_absolute_error": 0.0})
        results.append({"clientName": task, "analyzers": [{"type": "basic",
                        "completedRequests": count, "requestLatencyNs": [50_000] * count,
                        "avgThroughput(req/s)": count / seconds,
                        "latencyDefinition": "sum_of_original_six_recorded_stages"}]})
    report = {"benchmarkTime(s)": seconds, "loadStudyBeginNs": begin, "loadStudyEndNs": end,
              "loadStudyClock": "steady_clock", "results": results}
    priorities = [f"GPREEMPT_LOAD_PRIORITY task={task} role={role} actual={-5 if role == 0 else 0} least=0 greatest=-5"
                  for role, task in enumerate(audit.base.TASKS)] if arm == "native" else []
    return {"report": report, "rows": rows, "checks": checks, "priorities": priorities}


def log(data):
    return "\n".join([json.dumps(data["report"]),
                      *("GPREEMPT_LOAD_STUDY " + json.dumps(row) for row in data["rows"]),
                      *("GPREEMPT_VALIDATION " + json.dumps(row) for row in data["checks"]),
                      *data["priorities"]])


class RawAuditTests(unittest.TestCase):
    def parse(self, data, scenario="be100", seconds=60, arm="native", study="load"):
        return audit.parse_client(log(data), scenario, seconds, arm, study)

    def test_fifo_backlog_is_not_a_dropped_or_hidden_request(self):
        parsed = self.parse(fixture())
        for values in parsed["metrics"].values():
            self.assertEqual(values["offered_requests"], 6000)
            self.assertEqual(values["never_started_backlog"], 5997)
            self.assertEqual(values["goodput_rps"], .05)
            self.assertEqual(values["response_p99_us"], 100)
            self.assertTrue(values["response_p99_conditional"])
            self.assertTrue(values["p99_less_than_100_samples"])
            self.assertEqual(values["offered_requests"], sum(values[key] for key in
                             ("completed_in_window", "completed_after_window", "started_unfinished", "never_started_backlog")))
        plotted = audit.plotting_metrics(parsed["metrics"])["vgg_rt"]
        self.assertEqual(plotted["completion_coverage"], 3 / 6000)
        self.assertIsNone(plotted["all_offered_p99_response_us"])

    def test_be200_and_closed_loop_have_distinct_denominators(self):
        be = self.parse(fixture("be200"), "be200")["metrics"]["resnet152_be"]
        self.assertEqual(be["offered_requests"], 12000)
        be = self.parse(fixture("be_continuous"), "be_continuous")["metrics"]["resnet152_be"]
        for field in ("offered_requests", "never_started_backlog", "unfinished_offered_at_deadline", "window_completion_fraction"):
            self.assertIsNone(be[field])
        self.assertFalse(be["response_p99_conditional"])

    def test_lc_knee_uses_requested_foreground_rate_and_continuous_background(self):
        for scenario, rate in zip(audit.LC_KNEE_SCENARIOS, (500, 625, 800)):
            data = fixture(scenario, seconds=60, study="lc-knee")
            parsed = self.parse(data, scenario, study="lc-knee")["metrics"]
            with self.subTest(scenario=scenario):
                self.assertEqual(parsed["vgg_rt"]["offered_requests"], rate * 60)
                self.assertIsNone(parsed["resnet152_be"]["offered_requests"])
                producer = runner.parse_report(log(data), runner.make_config(scenario, 60, "lc-knee"),
                                               "native", "lc-knee")
                audit.require_equal(producer["metrics"], parsed, "knee producer/independent metrics")

    def test_completion_at_cutoff_is_not_window_goodput(self):
        data = fixture()
        last = data["rows"][0]["requests"][-1]
        last[2:] = [data["report"]["loadStudyEndNs"] - 1, data["report"]["loadStudyEndNs"]]
        values = self.parse(data)["metrics"]["vgg_rt"]
        self.assertEqual(values["completed_in_window"], 2)
        self.assertEqual(values["completed_after_window"], 1)
        self.assertEqual(values["goodput_rps"], 2 / 60)
        self.assertEqual(values["numerics"]["timed_checked"], 3)
        self.assertGreater(values["response_p99_us"], values["in_window_response_p99_us"])

    def test_full_coverage_including_tail_is_not_survivor_p99(self):
        data = fixture(count=6000)
        last = data["rows"][0]["requests"][-1]
        last[3] = data["report"]["loadStudyEndNs"]
        values = audit.plotting_metrics(self.parse(data)["metrics"])["vgg_rt"]
        self.assertEqual(values["completion_coverage"], 1)
        self.assertLess(values["window_completion_fraction"], 1)
        self.assertFalse(values["p99_is_conditional"])
        self.assertEqual(values["all_offered_p99_response_us"], values["p99_response_us"])

    def test_zero_completed_requests_remain_an_adverse_observation(self):
        values = self.parse(fixture(count=0))["metrics"]["vgg_rt"]
        self.assertEqual(values["goodput_rps"], 0)
        self.assertEqual(values["never_started_backlog"], 6000)
        self.assertIsNone(values["response_p99_us"])

    def test_raw_timing_and_conservation_tampering_is_rejected(self):
        mutations = [
            lambda d: d["rows"][0].update(started=4),
            lambda d: d["rows"][0].update(offered=5999),
            lambda d: d["rows"][0].update(begin_ns=1),
            lambda d: d["rows"][0].update(phase_ns=1000),
            lambda d: d["rows"][0].update(mode="periodic"),
            lambda d: d["rows"][0]["requests"][1].__setitem__(0, 2),
            lambda d: d["rows"][0]["requests"][1].__setitem__(1, 1000000000),
            lambda d: d["rows"][0]["requests"][1].__setitem__(2, d["report"]["loadStudyEndNs"]),
            lambda d: d["rows"][0]["requests"][1].__setitem__(3, 1),
            lambda d: d["rows"][0]["requests"][0].__setitem__(3, d["rows"][0]["requests"][1][2] + 1),
            lambda d: d["rows"][0]["requests"][0].__setitem__(0, False),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                data = fixture()
                mutate(data)
                with self.assertRaises(ValueError):
                    self.parse(data)

    def test_numerical_stage_counts_and_native_priority_are_checked(self):
        mutations = [
            lambda d: d["checks"][0].update(timed_checked=2),
            lambda d: d["checks"][0].update(checked=3),
            lambda d: d["checks"][0].update(max_absolute_error=float("nan")),
            lambda d: d["report"]["results"][0]["analyzers"][0].update(requestLatencyNs=[]),
            lambda d: d["report"]["results"][0]["analyzers"][0].update(**{"avgThroughput(req/s)": 100}),
            lambda d: d["priorities"].__setitem__(0, d["priorities"][0].replace("actual=-5", "actual=0")),
            lambda d: d["priorities"].pop(),
            lambda d: d["rows"].append(d["rows"][0]),
            lambda d: d["report"].update(loadStudyClock="system_clock"),
        ]
        for mutate in mutations:
            data = fixture()
            mutate(data)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.parse(data)

    def test_preflight_is_not_accepted_as_a_sixty_second_cell(self):
        data = fixture(seconds=10)
        self.assertEqual(self.parse(data, seconds=10)["metrics"]["vgg_rt"]["offered_requests"], 1000)
        with self.assertRaises(ValueError):
            self.parse(data)

    def test_independent_and_producer_parsers_agree_at_the_whole_schema(self):
        for scenario in audit.SCENARIOS:
            for seconds in (10, 60):
                for arm in audit.base.ARMS:
                    data = fixture(scenario, seconds=seconds, arm=arm)
                    independent = self.parse(data, scenario, seconds, arm)
                    producer = runner.parse_report(log(data), runner.make_config(scenario, seconds), arm)
                    with self.subTest(scenario=scenario, seconds=seconds, arm=arm):
                        audit.require_equal(producer, independent, "whole parser result")

    def test_zero_response_and_fabricated_continuous_offered_are_rejected(self):
        data = fixture()
        data["rows"][0]["requests"][0][3] = data["rows"][0]["requests"][0][1]
        with self.assertRaises(ValueError):
            self.parse(data)
        data = fixture("be_continuous")
        data["rows"][1]["offered"] = 6000
        with self.assertRaises(ValueError):
            self.parse(data, "be_continuous")


def blocks(count=5):
    original = audit.plotting_metrics(audit.parse_client(log(fixture()), "be100", 60, "native")["metrics"])
    result = []
    for block in range(count):
        cells = {}
        for arm, factor in zip(audit.base.ARMS, (1., .5, .25)):
            cells[arm] = copy.deepcopy(original)
            cells[arm]["vgg_rt"]["p99_response_us"] = factor * (1000 + block)
        result.append({"block": block, "cells": cells})
    return result


class EstimateTests(unittest.TestCase):
    def test_paired_gm_ci_and_conditional_population_are_explicit(self):
        result = audit.summarize_scenario(blocks(), [])
        self.assertTrue(result["complete"])
        pair = result["paired"]["bpf_gpreempt/original_gpreempt:vgg_rt:p99_response_us"]
        self.assertEqual(pair["geometric_ratio"], .5)
        self.assertEqual(pair["paired_block_bootstrap_ci95"], [.5, .5])
        self.assertTrue(pair["conditional_response_population"])
        self.assertFalse(pair["all_offered_latency_comparison"])

    def test_partial_or_duplicate_blocks_are_not_final(self):
        self.assertFalse(audit.summarize_scenario(blocks(1), [])["complete"])
        with self.assertRaises(ValueError):
            audit.summarize_scenario(blocks() + blocks(1), [])
        rows = blocks()
        rows[0]["cells"].pop("native")
        with self.assertRaises(ValueError):
            audit.summarize_scenario(rows, [])

    def test_zero_goodput_is_not_dropped_to_form_a_favorable_ratio(self):
        rows = blocks()
        rows[0]["cells"]["bpf_gpreempt"]["resnet152_be"]["goodput_rps"] = 0
        result = audit.summarize_scenario(rows, [])
        self.assertEqual(result["valid_paired_blocks"], 5)
        pair = result["paired"]["bpf_gpreempt/original_gpreempt:resnet152_be:goodput_rps"]
        self.assertEqual(len(pair["block_ratios"]), 5)
        self.assertEqual(pair["block_ratios"][0], 0)
        self.assertIsNone(pair["geometric_ratio"])


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def campaign_fixture(path, mode="preflight", study="load"):
    preflight = path / "separate-preflight" if mode == "full" and study == "lc-knee" else None
    plan = runner.make_plan(mode, study=study, preflight=preflight)
    write_json(path / "plan.json", plan)
    for scenario, config in plan["configs"].items():
        write_json(path / "configs" / f"{scenario}.json", config)
    paths = [audit.base.HERE / "build/load-study" / name for name in
             ("baseclient", "gpreemptclient", "libgpreempt.so", "block.cubin")]
    paths += [audit.base.EXTENSION / name for name in
              ("gpreempt_policy", "libgpreempt_bridge.so", "gpreempt_hint.bin")]
    files = {str(item): {"path": str(item), "bytes": 128, "mtime_ns": 100} for item in paths}
    write_json(path / "build-inventory.json", {"files": files,
               "cmake": {"GPREEMPT_LOAD_STUDY": "ON", "GPREEMPT_CUDA_ARCH": "120",
                         "CMAKE_CUDA_ARCHITECTURES": "120", "CMAKE_CXX_COMPILER": "/usr/bin/g++-13",
                         "CMAKE_C_COMPILER": "/usr/bin/gcc-13", "CMAKE_CUDA_COMPILER": "/usr/local/cuda-12.9/bin/nvcc"},
               "source_revisions": {"gpu_ext_head": "fixture_revision", "upstream": "249ee3e",
                                    "expected_driver_port": "849ea75d"}})
    assets = {}
    for name, layers in (("vgg", 19), ("resnet152", 152)):
        assets[name] = {"specification": {"model": name, "layers": layers, "architecture": "sm_120",
                         "dtype": "float32", "input_shape": [1, 3, 224, 224], "output_shape": [1, 1000],
                         "parameter_seed": 0, "input_formula": "((element_index % 257) - 128) / 128.0"},
                        "inventory": {filename: {"path": str(Path("/fixture") / name / filename),
                                      "bytes": 4000 if filename == "reference.f32" else 128}
                                      for filename in ("mod.cu", "mod.cubin", "mod.json", "host.json", "mod.params", "reference.f32")}}
    write_json(path / "model-assets.json", assets)
    return plan, files


def native_cell_fixture(campaign, plan, inventory):
    specification = plan["orders"][0]
    assert specification["arm"] == "native"
    scenario = specification["scenario"]
    config = campaign / "configs" / f"{scenario}.json"
    directory = campaign / "block-00" / scenario / "native"
    directory.mkdir(parents=True)
    study = plan.get("study", "load")
    client_log = log(fixture(scenario, seconds=plan["timed_seconds_per_cell"], study=study))
    parsed = audit.parse_client(client_log, scenario, plan["timed_seconds_per_cell"], "native", study)
    (directory / "client.log").write_text(client_log)
    write_json(directory / "request-report.json", parsed["report"])
    write_json(directory / "load-study-report.json", parsed["load_reports"])
    result = {**specification, "status": "passed", "returncode": 0, "config_path": str(config),
              "config": plan["configs"][scenario], "timed_seconds": plan["timed_seconds_per_cell"],
              "timeout_seconds": plan["cell_timeout_seconds"], "flag_transport": "not_used",
              "comparison_variant": "host_mapped_compatibility", "command": runner.client_command("native", config),
              "environment": runner.environment("native", Path("/unused"), Path(plan["gdrcopy_directory"])),
              "runtime_before": inventory, "runtime_after": inventory, "metrics": parsed["metrics"],
              "native_priorities": parsed["native_priorities"], "engagement": audit.base.check_engagement("native", client_log, ""),
              "safety_before": {"gpu": {"driver": "575.57.08"}}, "safety_after": {}, "telemetry": {"samples": 1}}
    write_json(directory / "result.json", result)
    return directory, result


class CampaignAuditTests(unittest.TestCase):
    def test_every_policy_action_is_bound_to_real_request_counts(self):
        metrics = audit.parse_client(log(fixture()), "be100", 60, "native")["metrics"]
        fields = {"preprocess": "226", "infer": "226", "reset": "3", "hint": "3",
                  "block": "3", "release": "3", "due": "1200", "scopes": "0",
                  "registered": "0", "ended": "0"}
        for arm in ("original_gpreempt", "bpf_gpreempt"):
            result = audit.policy_action_coverage(arm, {"bridge": fields}, metrics)
            self.assertEqual(result["foreground_started"], 3)
            for key in ("preprocess", "infer", "reset", "hint", "block", "release", "due"):
                altered = {**fields, key: "1"}
                with self.subTest(arm=arm, key=key), self.assertRaises(ValueError):
                    audit.policy_action_coverage(arm, {"bridge": altered}, metrics)
        with self.assertRaises(ValueError):
            audit.policy_action_coverage("original_gpreempt", {"bridge": {**fields, "scopes": "2"}}, metrics)
        self.assertEqual(audit.policy_action_coverage("native", {}, metrics), {"applicable": False})

    def test_frozen_manifest_and_preflight_cannot_become_formal(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            plan, _ = campaign_fixture(path)
            self.assertFalse(audit.validate_plan(plan, path))
            result = audit.analyze(path)
            self.assertFalse(result["formal_eligible"])
            self.assertFalse(result["complete"])
            self.assertEqual(len(result["incomplete_cells"]), 3)
            for key, replacement in (("seed", 7), ("timed_seconds_per_cell", 60), ("mode", "full"),
                                     ("kernel_repetition", 2), ("required_cells", 45)):
                altered = copy.deepcopy(plan)
                altered[key] = replacement
                with self.subTest(key=key), self.assertRaises(ValueError):
                    audit.validate_plan(altered, path)

    def test_full_order_rates_and_bootstrap_settings_are_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            plan, _ = campaign_fixture(path, "full")
            self.assertTrue(audit.validate_plan(plan, path))
            mutations = [lambda p: p["orders"].reverse(), lambda p: p["orders"][0].update(cell=1),
                         lambda p: p["statistics"].update(draws=100),
                         lambda p: p["configs"]["be200"]["tasks"][1]["load"].update(frequency=100)]
            for mutate in mutations:
                altered = copy.deepcopy(plan)
                mutate(altered)
                with self.assertRaises(ValueError):
                    audit.validate_plan(altered, path)

    def test_lc_knee_manifest_rejects_rate_scope_or_evidence_role_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            plan, _ = campaign_fixture(path, "full", "lc-knee")
            self.assertTrue(audit.validate_plan(plan, path))
            self.assertEqual(plan["required_cells"], 27)
            mutations = [lambda p: p["prespecified_lc_rates_rps"].append(900),
                         lambda p: p.update(evidence_role="decisive"),
                         lambda p: p.update(post_hoc_rate_additions_allowed=True),
                         lambda p: p["configs"]["lc625"]["tasks"][0]["load"].update(frequency=626),
                         lambda p: p["configs"]["lc625"]["tasks"][1].update(
                             load={"type": "periodic_fifo", "frequency": 200, "priority": 0}),
                         lambda p: p.update(preflight_campaign=None),
                         lambda p: p["orders"].pop()]
            for mutate in mutations:
                altered = copy.deepcopy(plan)
                mutate(altered)
                with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                    audit.validate_plan(altered, path)

    def test_lc_knee_preflight_is_auditable_but_never_formal(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            plan, _ = campaign_fixture(path, "preflight", "lc-knee")
            self.assertFalse(audit.validate_plan(plan, path))
            result = audit.analyze(path)
            self.assertEqual(result["study"], "lc-knee")
            self.assertEqual(result["evidence_role"], "supporting")
            self.assertFalse(result["formal_eligible"])
            self.assertEqual(len(result["incomplete_cells"]), 3)

    def test_completed_preflight_gate_checks_summary_and_independent_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            campaign_fixture(path, "preflight", "lc-knee")
            summary = {"status": "completed", "error": None, "mode": "preflight",
                       "completed_cells": 3, "required_cells": 3, "valid_paired_groups": 1,
                       "complete": True, "formal_complete": False}
            write_json(path / "summary.json", summary)
            accepted = {"schema": "gpreempt_lc_knee_audit_v1", "study": "lc-knee",
                        "evidence_role": "supporting", "mode": "preflight", "complete": True,
                        "formal_eligible": False, "formal_complete": False, "valid_cells": 3,
                        "required_cells": 3, "rejected_cells": [], "incomplete_cells": [],
                        "unexpected_cells": []}
            with patch.object(audit, "analyze", return_value=accepted) as independent:
                gate = audit.validate_completed_preflight(path)
            independent.assert_called_once_with(path.resolve())
            self.assertEqual(gate["scenario"], "lc800")
            self.assertFalse(gate["formal_complete"])
            for key, value in (("status", "failed"), ("completed_cells", 2),
                               ("complete", False), ("formal_complete", True)):
                altered = {**summary, key: value}
                write_json(path / "summary.json", altered)
                with self.subTest(key=key), patch.object(audit, "analyze", return_value=accepted), \
                     self.assertRaises(ValueError):
                    audit.validate_completed_preflight(path)

    def test_raw_native_cell_metadata_is_checked_and_missing_arms_stay_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            plan, inventory = campaign_fixture(path)
            directory, original = native_cell_fixture(path, plan, inventory)
            with patch.object(audit.base.safety, "validate_pre_server_safety") as pre, \
                 patch.object(audit.base.safety, "validate_post_server_safety") as post, \
                 patch.object(audit.base.safety, "validate_gpu_telemetry", return_value={"samples": 1}) as telemetry:
                result = audit.analyze(path)
                self.assertEqual(result["valid_cells"], 1)
                self.assertEqual(result["rejected_cells"], [])
                self.assertEqual(len(result["incomplete_cells"]), 2)
                self.assertFalse(result["scenarios"]["be_continuous"]["complete"])
                self.assertTrue(pre.called and post.called and telemetry.called)
                mutations = [lambda r: r["environment"].update(LD_PRELOAD="/wrong.so"),
                             lambda r: r["command"].__setitem__(0, "/wrong/baseclient"),
                             lambda r: next(iter(r["runtime_after"].values())).update(bytes=129),
                             lambda r: r["metrics"]["vgg_rt"].update(goodput_rps=100),
                             lambda r: r["native_priorities"]["vgg_rt"].update(actual="0"),
                             lambda r: r.update(status="failed", error="retained failure"),
                             lambda r: r.update(returncode=1), lambda r: r.update(cell=9)]
                for mutate in mutations:
                    altered = copy.deepcopy(original)
                    mutate(altered)
                    write_json(directory / "result.json", altered)
                    with self.subTest(mutate=mutate):
                        result = audit.analyze(path)
                        self.assertEqual(result["valid_cells"], 0)
                        self.assertEqual(len(result["rejected_cells"]), 1)
                        self.assertFalse(result["complete"])

    def test_unplanned_attempt_is_retained_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            campaign_fixture(path)
            write_json(path / "failed-retry" / "result.json", {"status": "failed"})
            result = audit.analyze(path)
            self.assertEqual(result["unexpected_cells"], [str(path / "failed-retry")])
            self.assertFalse(result["complete"])

    def test_integer_tampering_is_not_hidden_by_float_tolerance(self):
        audit.require_equal({"metric": 1.00000000000001}, {"metric": 1.0}, "rounding")
        with self.assertRaises(ValueError):
            audit.require_equal({"id": 1.0}, {"id": 1}, "integer")
        with self.assertRaises(ValueError):
            audit.require_equal({"metric": 1.001}, {"metric": 1.0}, "wrong metric")


if __name__ == "__main__":
    unittest.main()
