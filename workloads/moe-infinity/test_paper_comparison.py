"""Independent CPU-only protocol, accounting and SSE regression checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import run_paper_comparison as comparison


def valid_blocks(count=5):
    factors = {"native-off": 1., "paper-native": 2., "paper-bpf": 4.}
    blocks = []
    for item in comparison.schedule()[:count]:
        block = {**item, "passed": True, "cells": []}
        for mode in item["modes"]:
            block["cells"].append({
                "mode": mode, "passed": True, "shadow_verification": False,
                "verified_output_tokens": 512, "requests": [{}] * 8,
                "prompt_order": item["prompts"],
                "output_throughput_tokens_per_s": factors[mode] * item["block"],
                "first_text_ttft_median_ms": 20., "e2e_median_ms": 100.,
                "final_drain_s": .01})
        blocks.append(block)
    return blocks


def activation_pair(mode):
    bpf = mode == "paper-bpf"
    controller = {"completed_requests": 1, "matched_predictions": 0,
        "prefetch_candidates_selected": 0, "rank_calls": 10 if bpf else 0,
        "bpf_match_calls": 0, "aborted_requests": 0,
        "rank_mismatches": 0, "match_mismatches": 0}
    dispatcher = {"mode": 2 if bpf else 1, "eviction_selections": 1,
        "bpf_eviction_calls": 1 if bpf else 0, "eviction_mismatches": 0,
        "prefetch_submitted": 0, "prefetch_completed": 0, "prefetch_bytes": 0,
        "prefetch_hits": 0, "prefetch_wasted": 0, "prefetch_wasted_bytes": 0,
        "prefetch_unused_resident": 0}
    before = {"mode": mode, "controller": controller, "dispatcher": dispatcher}
    after = copy.deepcopy(before)
    after["controller"].update(completed_requests=9, matched_predictions=100,
        prefetch_candidates_selected=200, rank_calls=30 if bpf else 0,
        bpf_match_calls=20 if bpf else 0)
    after["dispatcher"].update(eviction_selections=21, bpf_eviction_calls=21 if bpf else 0,
        prefetch_submitted=200, prefetch_completed=12, prefetch_bytes=1200,
        prefetch_hits=9, prefetch_wasted=2, prefetch_wasted_bytes=200,
        prefetch_unused_resident=1)
    return before, after


def sse_lines(*, tokens=64, done=True, text="x", early_finish=False):
    lines = []
    for i in range(tokens):
        finished = (i == 0) if early_finish else (i == tokens - 1)
        payload = {"choices": [{"index": 0, "text": text,
                                "finish_reason": "length" if finished else None}]}
        lines.extend([b"data: " + json.dumps(payload).encode() + b"\n", b"\n"])
    if done:
        lines.extend([b"data: [DONE]\n", b"\n"])
    return lines


class FakeResponse:
    def __init__(self, lines, status=200):
        self.lines = iter(lines)
        self.status = status

    def readline(self):
        return next(self.lines, b"")

    def read(self):
        return b"HTTP failure diagnostic"


class FakeConnection:
    def __init__(self, lines, status=200):
        self.response = FakeResponse(lines, status)
        self.closed = False

    def request(self, *args, **kwargs):
        pass

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


class ComparisonTests(unittest.TestCase):
    def test_schedule_is_fixed_five_unique_mode_permutations_with_paired_prompts(self):
        schedule = comparison.schedule()
        self.assertEqual(schedule, comparison.schedule())
        self.assertEqual(len(schedule), 5)
        self.assertEqual(len({tuple(item["modes"]) for item in schedule}), 5)
        for item in schedule:
            self.assertEqual(set(item["modes"]), set(comparison.paper.MODES))
            self.assertEqual(sorted(item["prompts"]), list(range(1, 9)))

    def test_full_analysis_pairs_blocks_and_gets_known_geometric_ratios(self):
        result = comparison.analyze(valid_blocks())
        self.assertTrue(result["complete"])
        self.assertEqual(result["valid_blocks"], 5)
        pair = result["paired"]["paper-bpf/paper-native"]
        self.assertAlmostEqual(pair["geometric_throughput_ratio"], 2.)
        for bound in pair["paired_block_bootstrap_ci95"]:
            self.assertAlmostEqual(bound, 2.)

    def test_one_block_is_not_complete_and_does_not_claim_ci(self):
        result = comparison.analyze(valid_blocks(1))
        self.assertFalse(result["complete"])
        self.assertIsNone(result["paired"]["paper-bpf/paper-native"]["paired_block_bootstrap_ci95"])

    def test_duplicate_block_is_not_silently_counted_twice(self):
        blocks = valid_blocks()
        blocks.append(copy.deepcopy(blocks[0]))
        result = comparison.analyze(blocks)
        self.assertFalse(result["complete"])
        self.assertEqual(result["valid_blocks"], 4)

    def test_shadow_and_missing_cells_invalidate_block(self):
        blocks = valid_blocks(1)
        blocks[0]["cells"][0]["shadow_verification"] = True
        self.assertEqual(comparison.analyze(blocks)["valid_blocks"], 0)
        blocks = valid_blocks(1)
        blocks[0]["cells"].pop()
        self.assertEqual(comparison.analyze(blocks)["valid_blocks"], 0)

    def test_bpf_requires_all_three_measured_window_programs(self):
        before, after = activation_pair("paper-bpf")
        self.assertEqual(comparison.activation_delta("paper-bpf", before, after)["controller"]["completed_requests"], 8)
        after["controller"]["bpf_match_calls"] = 0
        with self.assertRaises(comparison.base.GateError):
            comparison.activation_delta("paper-bpf", before, after)

    def test_native_policy_rejects_bpf_calls(self):
        before, after = activation_pair("paper-native")
        comparison.activation_delta("paper-native", before, after)
        after["dispatcher"]["bpf_eviction_calls"] = 1
        with self.assertRaises(comparison.base.GateError):
            comparison.activation_delta("paper-native", before, after)

    def test_warmup_engagement_cannot_substitute_for_measured_engagement(self):
        _, before = activation_pair("paper-native")
        after = copy.deepcopy(before)
        after["controller"]["completed_requests"] += 8
        with self.assertRaises(comparison.base.GateError):
            comparison.activation_delta("paper-native", before, after)

    def run_stream(self, directory, lines, golden="x" * 64, status=200):
        connection = FakeConnection(lines, status)
        path = Path(directory) / "request.sse"
        with mock.patch.object(comparison.http.client, "HTTPConnection", return_value=connection):
            result = comparison.stream_request(18230, [17] * 512, golden, path)
        self.assertTrue(connection.closed)
        return result, path

    def test_full_stream_retains_raw_and_exact_text(self):
        with tempfile.TemporaryDirectory() as directory:
            result, path = self.run_stream(directory, sse_lines())
            self.assertTrue(result["passed"])
            self.assertEqual(len(result["frames"]), 65)
            self.assertEqual(result["text"], "x" * 64)
            self.assertEqual(result["request_payload"],
                             comparison.base.completion_payload(comparison.CONFIG, [17] * 512, True))
            self.assertTrue(path.read_bytes().endswith(b"data: [DONE]\n\n"))
            self.assertTrue(json.loads(path.with_suffix(".json").read_text())["passed"])

    def test_missing_done_wrong_text_or_bad_count_keeps_failed_evidence(self):
        variants = [sse_lines(done=False), sse_lines(text="y"), sse_lines(tokens=63)]
        for lines in variants:
            with self.subTest(lines=len(lines)), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(comparison.base.GateError):
                    self.run_stream(directory, lines)
                path = Path(directory) / "request.sse"
                self.assertTrue(path.read_bytes())
                self.assertFalse(json.loads(path.with_suffix(".json").read_text())["passed"])

    def test_frame_after_done_and_http_error_are_retained(self):
        for lines, status in ((sse_lines() + sse_lines(tokens=1), 200), ([], 500)):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(comparison.base.GateError):
                    self.run_stream(directory, lines, status=status)
                path = Path(directory) / "request.sse"
                self.assertFalse(json.loads(path.with_suffix(".json").read_text())["passed"])

    def test_finish_must_be_last_token_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(comparison.base.GateError, "after finish"):
                self.run_stream(directory, sse_lines(early_finish=True))
            self.assertFalse(json.loads((Path(directory) / "request.json").read_text())["passed"])

    def test_stopped_or_zombie_compiler_is_not_active_contention(self):
        with mock.patch.object(comparison.base, "run_checked", return_value="11 T cc1plus\n12 Z ninja\n13 S python"):
            comparison.reject_build_contention()
        with mock.patch.object(comparison.base, "run_checked", return_value="11 R cc1plus"):
            with self.assertRaisesRegex(comparison.base.GateError, "compilation"):
                comparison.reject_build_contention()

    def test_primary_failure_survives_cleanup_failure_and_log_is_closed(self):
        import contextlib
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            server = mock.Mock(returncode=7)
            log = mock.Mock()
            for owner, name, kwargs in (
                (comparison, "reject_build_contention", {"return_value": None}),
                (comparison.paper, "admit", {"return_value": {"safety": {}}}),
                (comparison, "runtime_inventory", {"return_value": {}}),
                (comparison.paper, "interrupt_warnings", {"return_value": []}),
                (comparison.paper, "launch", {"return_value": (server, log)}),
                (comparison.base, "wait_ready", {"side_effect": ValueError("original startup failure")}),
                (comparison.base, "stop_owned_process_group", {"side_effect": RuntimeError("cleanup failure")}),
                (comparison.base, "wait_for_post_server_safety", {"return_value": {}}),
                (comparison.base, "validate_log", {"return_value": None}),
            ):
                stack.enter_context(mock.patch.object(owner, name, **kwargs))
            output = Path(directory) / "cell"
            with self.assertRaisesRegex(ValueError, "original startup failure"):
                comparison.run_cell("paper-bpf", output, 18230, list(range(1, 9)), Path(directory))
            result = json.loads((output / "result.json").read_text())
            self.assertFalse(result["passed"])
            self.assertEqual(result["error"], "original startup failure")
            self.assertIn("cleanup failure", " ".join(result["cleanup_errors"]))
            log.close.assert_called_once()

    def test_raw_artifact_durability_syncs_every_file_and_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # These are test fixtures, not experiment records.
            (root / "request.sse").write_bytes(b"data: [DONE]\n\n")
            (root / "gpu-telemetry.csv").write_text("sample\n")
            with mock.patch.object(comparison.os, "fsync", wraps=comparison.os.fsync) as sync:
                comparison.sync_cell_artifacts(root)
                self.assertEqual(sync.call_count, 3)

    def test_durability_refuses_aliased_or_nested_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "unexpected-directory").mkdir()
            with self.assertRaises(comparison.base.GateError):
                comparison.sync_cell_artifacts(root)


if __name__ == "__main__":
    unittest.main()
