"""CPU-only CLI orchestration tests; raw-artifact validation has its own suite."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import analyze_paper_comparison as offline


class OfflineAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="moe-offline-analysis-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = {
            "protocol": offline.runner.PROTOCOL, "required_blocks": 5,
            "seed": offline.runner.SEED, "schedule": offline.runner.schedule(),
            "warmup_prompt": 0, "measured_input_output_tokens": [512, 64],
            "memory_budget": 0.75, "kv_blocks": 128,
            "timing_shadow_verification": False, "driver_stage": "/fixture-driver",
            "runtime_inventory": {"driver_stage": "/fixture-driver"}}
        self.write_manifest()
        self.auditor = mock.patch.object(offline.audit, "audit_block",
            side_effect=lambda path, *_: json.loads((path / "result.json").read_text())).start()
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(offline.runner.paper, "admit", side_effect=AssertionError("GPU admission")).start()
        mock.patch.object(offline.runner.base, "run_checked", side_effect=AssertionError("subprocess")).start()

    def write_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def attempt(self, number=1, attempt=1, passed=True):
        item = self.manifest["schedule"][number - 1]
        path = self.root / f"block-{number:02d}-attempt-{attempt:02d}"
        path.mkdir()
        factors = {"native-off": 1., "paper-native": 2., "paper-bpf": 4.}
        block = {**item, "passed": passed, "cells": [{
            "mode": mode, "passed": True, "shadow_verification": False,
            "verified_output_tokens": 512, "requests": [{}] * 8,
            "prompt_order": item["prompts"],
            "output_throughput_tokens_per_s": factors[mode] * number,
            "first_text_ttft_median_ms": 20., "e2e_median_ms": 100.,
            "final_drain_s": .01} for mode in item["modes"]]}
        (path / "result.json").write_text(json.dumps(block), encoding="utf-8")
        return path

    def test_empty_and_partial_campaign_stay_incomplete(self):
        attempt = self.root / "block-01-attempt-01"
        attempt.mkdir()
        (attempt / "paper-bpf").mkdir()
        (attempt / "paper-bpf/result.json").write_text('{"passed": true}')
        report = offline.analyze_campaign(self.root)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["unverified_blocks"], [1, 2, 3, 4, 5])
        self.assertEqual(report["analysis"]["valid_blocks"], 0)
        self.assertEqual(len(report["incomplete_attempts"]), 1)
        self.auditor.assert_not_called()

    def test_all_five_are_audited_and_statistics_recomputed(self):
        paths = [self.attempt(number) for number in range(1, 6)]
        (self.root / "analysis.json").write_text('{"complete": false, "valid_blocks": 0}')
        report = offline.analyze_campaign(self.root)
        self.assertTrue(report["complete"])
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["analysis"]["valid_blocks"], 5)
        self.assertEqual(len(report["accepted_attempts"]), 5)
        self.assertEqual(report["unverified_blocks"], [])
        self.assertEqual(self.auditor.call_args_list, [
            mock.call(path, item, self.manifest["runtime_inventory"])
            for path, item in zip(paths, self.manifest["schedule"])])
        pair = report["analysis"]["paired"]["paper-bpf/paper-native"]
        self.assertAlmostEqual(pair["geometric_throughput_ratio"], 2.)
        for bound in pair["paired_block_bootstrap_ci95"]:
            self.assertAlmostEqual(bound, 2.)

    def test_duplicate_success_claims_are_both_audited_and_excluded(self):
        self.attempt(attempt=1)
        self.attempt(attempt=2)
        report = offline.analyze_campaign(self.root)
        self.assertEqual(self.auditor.call_count, 2)
        self.assertEqual(report["status"], "rejected")
        self.assertEqual(len(report["rejected_attempts"]), 2)
        self.assertEqual(report["accepted_attempts"], [])

    def test_duplicate_with_corrupt_success_cannot_select_the_survivor(self):
        first = self.attempt(attempt=1)
        self.attempt(attempt=2)

        def audit_one_corrupt(path, *_):
            if path == first:
                raise offline.audit.AuditError("corrupt SSE")
            return json.loads((path / "result.json").read_text())

        self.auditor.side_effect = audit_one_corrupt
        report = offline.analyze_campaign(self.root)
        self.assertEqual(self.auditor.call_count, 2)
        self.assertEqual(report["analysis"]["valid_blocks"], 0)
        self.assertEqual(len(report["rejected_attempts"]), 2)
        self.assertIn("corrupt SSE", report["rejected_attempts"][0]["audit_error"])

    def test_success_flag_does_not_override_raw_audit_failure(self):
        self.attempt()
        self.auditor.side_effect = offline.audit.AuditError("missing raw request")
        report = offline.analyze_campaign(self.root)
        self.assertFalse(report["complete"])
        self.assertEqual(report["status"], "rejected")
        self.assertIn("missing raw request", report["rejected_attempts"][0]["reason"])

    def test_secondary_ttft_constant_ratios_and_single_block_has_no_ci(self):
        blocks = []
        for number in range(1, 6):
            path = self.attempt(number)
            block = json.loads((path / "result.json").read_text())
            for cell in block["cells"]:
                cell["first_text_ttft_median_ms"] = {
                    "native-off": 400., "paper-native": 200., "paper-bpf": 100.
                }[cell["mode"]] * number
            (path / "result.json").write_text(json.dumps(block))
            blocks.append(block)
        report = offline.analyze_campaign(self.root)
        metric = report["secondary"]["first_visible_text_ttft"]
        self.assertEqual(metric["priority"], "secondary")
        self.assertEqual(metric["direction"], "lower_is_better")
        self.assertEqual(metric["blocks"], 5)
        for pair, expected in (("paper-bpf/paper-native", .5),
                               ("paper-native/native-off", .5), ("paper-bpf/native-off", .25)):
            self.assertAlmostEqual(metric["paired"][pair]["geometric_ttft_ratio"], expected)
            for bound in metric["paired"][pair]["paired_block_bootstrap_ci95"]:
                self.assertAlmostEqual(bound, expected)
        primary = report["analysis"]["paired"]["paper-bpf/paper-native"]
        self.assertAlmostEqual(primary["geometric_throughput_ratio"], 2.)
        single = offline.secondary_ttft(blocks[:1])["paired"]["paper-bpf/paper-native"]
        self.assertAlmostEqual(single["geometric_ttft_ratio"], .5)
        self.assertIsNone(single["paired_block_bootstrap_ci95"])

    def test_failed_attempt_is_retained_beside_a_successful_retry(self):
        failed = self.attempt(attempt=1, passed=False)
        (failed / "result.json").write_text('{"passed": false, "error": "retained timeout"}')
        self.attempt(attempt=2)
        report = offline.analyze_campaign(self.root)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["analysis"]["valid_blocks"], 1)
        self.assertEqual(report["failed_attempts"][0]["failure"]["error"], "retained timeout")
        self.auditor.assert_called_once()

    def test_malformed_result_is_rejected_not_treated_as_a_retry_failure(self):
        path = self.attempt()
        for text in ('{', '{"passed": true, "passed": false}', '{"passed": 1}'):
            with self.subTest(text=text):
                (path / "result.json").write_text(text)
                report = offline.analyze_campaign(self.root)
                self.assertEqual(report["status"], "rejected")
                self.assertEqual(report["failed_attempts"], [])
        self.auditor.assert_not_called()

    def test_unexpected_and_aliased_attempts_are_rejected(self):
        (self.root / "block-06-attempt-01").mkdir()
        (self.root / "block-01-attempt-001").mkdir()
        (self.root / "block-02-attempt-01").symlink_to(self.root, target_is_directory=True)
        report = offline.analyze_campaign(self.root)
        self.assertEqual(len(report["rejected_attempts"]), 3)
        self.auditor.assert_not_called()

    def test_manifest_cannot_shrink_or_change_the_frozen_schedule(self):
        original = copy.deepcopy(self.manifest)
        for key, value in (("required_blocks", 1), ("seed", True),
                           ("schedule", original["schedule"][:4]),
                           ("timing_shadow_verification", True)):
            with self.subTest(key=key):
                self.manifest = {**original, key: value}
                self.write_manifest()
                with self.assertRaises(offline.audit.AuditError):
                    offline.analyze_campaign(self.root)

    def test_cli_output_is_exclusive_and_incomplete_exit_is_one(self):
        output = self.root / "offline-report.json"
        with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
            self.assertEqual(offline.main([str(self.root), "--output", str(output)]), 1)
        self.assertEqual(json.loads(output.read_text()), json.loads(stdout.getvalue()))
        original = output.read_text()
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(offline.main([str(self.root), "--output", str(output)]), 2)
        self.assertEqual(output.read_text(), original)
        self.auditor.assert_not_called()

    def test_cli_rejected_exit_and_no_heavy_imports(self):
        self.attempt()
        self.auditor.side_effect = offline.audit.AuditError("bad output")
        with mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(offline.main([str(self.root)]), 2)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("numpy", sys.modules)


if __name__ == "__main__":
    unittest.main()
