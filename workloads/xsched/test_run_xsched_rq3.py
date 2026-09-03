#!/usr/bin/env python3
"""CPU-only regression tests: do not attach BPF or launch CUDA."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import run_xsched_rq3 as runner


class RunnerTests(unittest.TestCase):
    def resume_fixture(self, root):
        protocol = {"phase": "full", "repetitions": 10,
                    "schedule": runner.fixed_schedule(("native", "xsched", "bpftime_hpf", "gpubpf"), 10)}
        checks = {key: {"test_value": 1} for key in runner.FROZEN_ADMISSION_FIELDS}
        (root / "protocol.json").write_text(json.dumps(protocol))
        (root / "admission.json").write_text(json.dumps(checks))
        (root / "pause-01.json").write_text(json.dumps({"state": "paused_after_complete_block", "completed_blocks": 1}))
        for role in ("lc", "be"):
            for i in range(3):
                (root / f"control-{role}-{i}").mkdir()
        for index, config in enumerate(protocol["schedule"][0]):
            directory = root / f"block-00-{index}-{config}"
            directory.mkdir()
            (directory / "result.json").write_text(json.dumps({"block": 0, "order_index": index, "config": config}))
        return protocol, checks

    def test_full_resume_audits_all_controls_and_completed_block(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, checks = self.resume_fixture(root)
            with mock.patch.object(runner, "audit_saved_cell") as audit:
                self.assertEqual(runner.validate_resume(root, protocol, checks), 1)
                self.assertEqual(audit.call_count, 10)
            with mock.patch.object(runner, "audit_saved_cell", side_effect=RuntimeError("raw sample invalid")):
                with self.assertRaisesRegex(RuntimeError, "raw sample invalid"):
                    runner.validate_resume(root, protocol, checks)

    def test_resume_rejects_metadata_protocol_partial_and_failed_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, checks = self.resume_fixture(root)
            changed = dict(checks, runner_file={"test_value": 2})
            with self.assertRaisesRegex(RuntimeError, "metadata changed"):
                runner.validate_resume(root, protocol, changed)
            with self.assertRaisesRegex(RuntimeError, "protocol/schedule"):
                runner.validate_resume(root, dict(protocol, repetitions=5), checks)
            with mock.patch.object(runner, "audit_saved_cell"):
                (root / "block-01-0-native").mkdir()
                with self.assertRaisesRegex(RuntimeError, "partial"):
                    runner.validate_resume(root, protocol, checks)
                (root / "failure.json").write_text('{"error":"synthetic failure"}')
                with self.assertRaisesRegex(RuntimeError, "failed/interrupted"):
                    runner.validate_resume(root, protocol, checks)

    def test_resume_rejects_reordered_completed_cells(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, checks = self.resume_fixture(root)
            name = protocol["schedule"][0][0]
            (root / f"block-00-0-{name}" / "result.json").write_text(json.dumps({"block": 0, "order_index": 1, "config": name}))
            with mock.patch.object(runner, "audit_saved_cell"):
                with self.assertRaisesRegex(RuntimeError, "randomized order"):
                    runner.validate_resume(root, protocol, checks)

    def test_fixed_schedule_is_full_and_deterministic(self):
        configs = ("native", "xsched", "bpftime_hpf", "gpubpf")
        schedule = runner.fixed_schedule(configs, 10)
        self.assertEqual(len(schedule), 10)
        self.assertTrue(all(set(order) == set(configs) and len(order) == 4 for order in schedule))
        self.assertEqual(schedule, runner.fixed_schedule(configs, 10))

    def test_saved_raw_worker_failure_not_hidden_by_cell_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for phase in ("before", "after"):
                (root / f"safety-{phase}.json").write_text('{}')
            clock = {"offset_ns": 0, "uncertainty_ns": 1}
            (root / "result.json").write_text(json.dumps({"clock_calibration": clock,
                "clock_calibration_after": clock, "clock_error": runner.validate_clock_pair(clock, clock)}))
            (root / "be1.json").write_text(json.dumps({"returncode": 2}))
            with mock.patch.object(runner.shared, "validate_pre_server_safety"), \
                 mock.patch.object(runner.shared, "validate_post_server_safety"):
                with self.assertRaisesRegex(RuntimeError, "saved worker failed"):
                    runner.audit_saved_cell(root, {"tasks_per_stream": 50, "blocks": 340, "threads": 256}, "native")

    def test_saved_raw_timestamp_argv_and_numerics_are_checked(self):
        for damage, expected_error in (("clock", "sample clock/order"), ("argv", "command differs"),
                                       ("values", "dimensions/identity")):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                for phase in ("before", "after"):
                    (root / f"safety-{phase}.json").write_text('{}')
                clock = {"offset_ns": 2, "uncertainty_ns": 1}
                (root / "result.json").write_text(json.dumps({"clock_calibration": clock,
                    "clock_calibration_after": clock, "clock_error": runner.validate_clock_pair(clock, clock)}))
                sample = {"submit_ns": 12, "submit_cupti_ns": 10, "entry_ns": 13, "exit_ns": 14}
                if damage == "clock": sample["submit_ns"] = 11
                record = {"event": "result", "role": "be", "process_id": 1, "clock_offset_ns": 2,
                          "outputs_validated": 3 if damage == "values" else 4,
                          "samples": [sample] * 4}
                argv = ["taskset", "-c", ",".join(map(str, runner.allowed_cpus(10)[2:])),
                        str(runner.HERE / "build" / "bench_be"), "be", "1", "4", "1", "100", "1", "1", "1", "2"]
                if damage == "argv": argv[2] = "999"
                (root / "be1.json").write_text(json.dumps({"returncode": 0, "command": argv,
                    "stdout": [json.dumps({"event": "ready", "role": "be", "process_id": 1, "streams": 4, "tasks": 1}), json.dumps(record)]}))
                with mock.patch.object(runner.shared, "validate_pre_server_safety"), \
                     mock.patch.object(runner.shared, "validate_post_server_safety"):
                    with self.assertRaisesRegex(RuntimeError, expected_error):
                        runner.audit_saved_cell(root, {"tasks_per_stream": 1, "blocks": 1, "threads": 1, "reps": 100}, "native")

    def test_isolated_role_cannot_be_changed_under_another_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "control-lc-0"
            root.mkdir()
            for phase in ("before", "after"):
                (root / f"safety-{phase}.json").write_text('{}')
            clock = {"offset_ns": 0, "uncertainty_ns": 1}
            (root / "result.json").write_text(json.dumps({"control": "isolated-be",
                "clock_calibration": clock, "clock_calibration_after": clock,
                "clock_error": runner.validate_clock_pair(clock, clock)}))
            with mock.patch.object(runner.shared, "validate_pre_server_safety"), \
                 mock.patch.object(runner.shared, "validate_post_server_safety"):
                with self.assertRaisesRegex(RuntimeError, "role differs from directory"):
                    runner.audit_saved_cell(root, {"tasks_per_stream": 50, "blocks": 340, "threads": 256})

    def test_runtime_environment_filters_credentials_and_keeps_execution_settings(self):
        self.assertEqual(runner.runtime_environment({"API_KEY": "must-not-record", "PATH": "/bin",
            "LD_LIBRARY_PATH": "/xsched", "XSCHED_AUTO_XQUEUE": "ON", "CUDA_MODULE_LOADING": "LAZY"}),
            {"LD_LIBRARY_PATH": "/xsched", "XSCHED_AUTO_XQUEUE": "ON", "CUDA_MODULE_LOADING": "LAZY"})

    def test_bpftime_uses_identical_worker_environment(self):
        for role in ("lc", "be"):
            self.assertEqual(runner.workload_env("xsched", role), runner.workload_env("bpftime_hpf", role))
        original, original_env = runner.xsched_server("xsched")
        candidate, candidate_env = runner.xsched_server("bpftime_hpf")
        self.assertEqual(original[1:], candidate[1:])
        self.assertNotEqual(original[0], candidate[0])
        self.assertNotIn("GPUBPF_HPF_CODE", original_env)
        self.assertEqual(candidate_env.pop("GPUBPF_HPF_CODE"), str(runner.HERE / "build/bpftime_hpf.bin"))
        self.assertEqual(original_env, candidate_env)

    def test_bpftime_requires_jit_and_both_decision_types(self):
        ready = "bpftime_hpf_ready: backend=ubpf-jit max_queues=64\n"
        result = runner.validate_bpftime_hpf_engagement(ready + "bpftime_hpf_stats: calls=7 queues=100 suspend=40 resume=60")
        self.assertEqual(result["queues"], 100)
        for invalid in (
            "bpftime_hpf_stats: calls=7 queues=100 suspend=40 resume=60",
            ready + "bpftime_hpf_stats: calls=7 queues=100 suspend=0 resume=100",
            ready + "bpftime_hpf_stats: calls=7 queues=100 suspend=40 resume=50",
        ):
            with self.assertRaises(RuntimeError):
                runner.validate_bpftime_hpf_engagement(invalid)

    def test_candidate_commands_do_not_change_the_old_policy(self):
        old_timeslice, old_preempt = runner.gpubpf_policy_commands("gpubpf")
        no_timeslice, no_preempt = runner.gpubpf_policy_commands("gpubpf_nocooldown")
        interleave_timeslice, interleave_preempt = runner.gpubpf_policy_commands("gpubpf_interleave")
        self.assertEqual(no_timeslice, old_timeslice)
        self.assertEqual(no_preempt[:-1], old_preempt[:-1])
        self.assertEqual((old_preempt[-1], no_preempt[-1]), ("100", "0"))
        self.assertEqual(interleave_timeslice, old_timeslice + ["-i", "bench_lc:2", "-i", "bench_be:0"])
        self.assertEqual(interleave_preempt, old_preempt)
        self.assertEqual(runner.CONFIGS, ("native", "xsched", "gpubpf"))

    def test_configuration_lists_are_explicit_and_unique(self):
        self.assertEqual(runner.parse_configs("native,gpubpf_nocooldown"), ("native", "gpubpf_nocooldown"))
        for invalid in ("", "native,native", "unknown"):
            with self.assertRaises(Exception):
                runner.parse_configs(invalid)

    def test_no_cooldown_requires_every_lc_launch_and_four_targets(self):
        counts = {
            "uprobe_hit": 120, "preempt_ok": 160, "preempt_err": 0,
            "skipped": 80, "cooldown_skip": 0, "targets_hit": 160,
            "tsg_captured": 4, "active_targets": 4,
        }
        text = lambda values: "\n".join(f"  {key}: {value}" for key, value in values.items())
        result = runner.validate_gpubpf_engagement(
            "gpubpf_nocooldown", "timeslice_mod: 20", text(counts), 5)
        self.assertEqual(result["expected_preemptions"], 160)
        counts.update(preempt_ok=8, targets_hit=8, cooldown_skip=38)
        with self.assertRaisesRegex(RuntimeError, "exactly 160"):
            runner.validate_gpubpf_engagement(
                "gpubpf_nocooldown", "timeslice_mod: 20", text(counts), 5)
        self.assertEqual(runner.validate_gpubpf_engagement(
            "gpubpf", "timeslice_mod: 20", text(counts), 5)["preempt_ok"], 8)

    def test_interleave_must_be_observed_without_setter_errors(self):
        preempt = "\n".join(f"{key}: {value}" for key, value in {
            "uprobe_hit": 120, "preempt_ok": 8, "preempt_err": 0, "skipped": 80,
            "cooldown_skip": 38, "targets_hit": 8, "tsg_captured": 4, "active_targets": 4,
        }.items())
        timeslice = "timeslice_mod: 20\ninterleave_mod: 20\ninterleave_observed: 20\ninterleave_mismatch: 0\nsetter_error: 0"
        self.assertEqual(runner.validate_gpubpf_engagement(
            "gpubpf_interleave", timeslice, preempt, 5)["interleave_observed"], 20)
        with self.assertRaisesRegex(RuntimeError, "not confirmed"):
            runner.validate_gpubpf_engagement(
                "gpubpf_interleave", timeslice.replace("interleave_mismatch: 0", "interleave_mismatch: 1"), preempt, 5)

    def test_new_runtime_requires_persistent_control_engagement(self):
        preempt = "\n".join(f"{key}: {value}" for key, value in {
            "uprobe_hit": 120, "preempt_ok": 8, "preempt_err": 0, "skipped": 80,
            "cooldown_skip": 38, "targets_hit": 8, "tsg_captured": 4, "active_targets": 4,
        }.items())
        timeslice = "timeslice_mod: 20\ncontrol_override: 6\nsetter_error: 0"
        self.assertEqual(runner.validate_gpubpf_engagement(
            "gpubpf", timeslice, preempt, 5, require_persistent=True)["control_override"], 6)
        for bad in (timeslice.replace("control_override: 6", "control_override: 0"),
                    timeslice.replace("setter_error: 0", "setter_error: 1")):
            with self.assertRaisesRegex(RuntimeError, "persistent"):
                runner.validate_gpubpf_engagement("gpubpf", bad, preempt, 5, require_persistent=True)

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

    def test_full_analysis_requires_all_six_correct_isolated_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "protocol.json").write_text(json.dumps({
                "phase": "full", "repetitions": 10, "tasks_per_stream": 50,
                "blocks": 340, "threads": 256,
            }))
            with self.assertRaisesRegex(RuntimeError, "requires isolated control"):
                runner.analyze(root)
            for role in ("lc", "be"):
                for repetition in range(3):
                    directory = root / f"control-{role}-{repetition}"
                    directory.mkdir()
                    (directory / "result.json").write_text(json.dumps({
                        "control": f"isolated-{role}", "samples": 200,
                        "outputs_validated": 200 * 340 * 256,
                        "p99_us": 10, "throughput_kernels_s": 10,
                    }))
            with self.assertRaisesRegex(RuntimeError, "10 complete"):
                runner.analyze(root)
            (directory / "result.json").write_text(json.dumps({
                "control": "isolated-be", "samples": 20, "outputs_validated": 20 * 340 * 256,
            }))
            with self.assertRaisesRegex(RuntimeError, "isolated control workload mismatch"):
                runner.analyze(root)

    def test_analysis_does_not_accept_a_result_from_a_failed_cell(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "failure.json").write_text(json.dumps({"error": "cleanup failed"}))
            with self.assertRaisesRegex(RuntimeError, "failed cell"):
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
