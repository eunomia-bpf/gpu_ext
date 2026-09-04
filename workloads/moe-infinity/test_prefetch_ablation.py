from copy import deepcopy
import tempfile
import unittest
from unittest import mock

import audit_prefetch_ablation as audit
import run_prefetch_ablation as ablation


def activation_pair(arm):
    mode, prefetch = ablation.ARM_CONFIG[arm]
    backend = 2 if mode == "paper-bpf" else 1
    base_dispatcher = {key: 0 for key in ablation.DELTA_FIELDS}
    base_dispatcher.update({
        "mode": backend,
        "prefetch_enabled": int(prefetch),
        "cache_budget_bytes": 1024,
        "temporary_slot_enabled": 0,
        "prefetch_unused_resident": 0,
        "prefetch_unused_resident_bytes": 0,
        "prefetch_protected_candidates": 0,
    })
    before = {
        "mode": mode, "prefetch_enabled": prefetch,
        "controller": {
            "completed_requests": 10, "matched_predictions": 20,
            "prefetch_candidates_selected": 30, "rank_calls": 0,
            "bpf_match_calls": 0, "aborted_requests": 0,
        },
        "dispatcher": deepcopy(base_dispatcher),
    }
    after = deepcopy(before)
    after["controller"].update({
        "completed_requests": 10 + ablation.REQUESTS_PER_CELL,
        "matched_predictions": 40,
        "prefetch_candidates_selected": 60,
        "rank_calls": 30 if mode == "paper-bpf" else 0,
        "bpf_match_calls": 30 if mode == "paper-bpf" else 0,
    })
    measured = {
        "eviction_selections": 14 if prefetch else 11,
        "bpf_eviction_calls": (14 if prefetch else 11) if mode == "paper-bpf" else 0,
        "bpf_demand_eviction_calls": 11 if mode == "paper-bpf" else 0,
        "bpf_prefetch_eviction_calls": 3 if prefetch and mode == "paper-bpf" else 0,
        "evictions": 10 if prefetch else 7,
        "demand_evictions": 7,
        "prefetch_evictions": 3 if prefetch else 0,
        "demand_prefill_accesses": 30,
        "demand_prefill_hits": 20,
        "demand_prefill_misses": 10,
        "demand_decode_accesses": 60,
        "demand_decode_hits": 45,
        "demand_decode_misses": 15,
        "demand_copy_started": 25,
        "demand_bytes": 2500,
    }
    if prefetch:
        measured.update({
            "prefetch_submitted": 20, "prefetch_completed": 12,
            "prefetch_copy_started": 12, "prefetch_copy_waits": 12,
            "prefetch_copy_wait_ns": 100, "prefetch_bytes": 1200,
            "prefetch_hits": 9, "prefetch_hit_bytes": 900,
            "prefetch_wasted": 2, "prefetch_wasted_bytes": 200,
            "prefetch_prediction_epoch": 40,
            "prefetch_protected_resident_skips": 50,
        })
        after["dispatcher"].update(
            prefetch_unused_resident=1,
            prefetch_unused_resident_bytes=100,
        )
    else:
        measured["prefetch_prediction_epoch"] = 40
    for key, value in measured.items():
        after["dispatcher"][key] = before["dispatcher"].get(key, 0) + value
    return before, after


class ProtocolTests(unittest.TestCase):
    def test_schedule_is_five_randomized_complete_paired_blocks(self):
        schedule = ablation.schedule()
        self.assertEqual(len(schedule), 5)
        self.assertEqual(len({tuple(block["arms"]) for block in schedule}), 5)
        for block in schedule:
            self.assertEqual(set(block["arms"]), set(ablation.ARMS))
            self.assertEqual(set(block["prompts"]), set(range(1, 7)))
        manifest = ablation.protocol_manifest()
        self.assertEqual(manifest["planned_cells"], 20)
        self.assertEqual(manifest["planned_measured_requests"], 120)

    def test_all_four_arms_pass_complete_accounting(self):
        for arm in ablation.ARMS:
            with self.subTest(arm=arm):
                before, after = activation_pair(arm)
                delta = ablation.activation_delta(arm, before, after)
                self.assertEqual(delta["controller"]["completed_requests"], 6)
                self.assertEqual(delta["dispatcher"]["temporary_slot_uses"], 0)

    def test_temporary_slot_contamination_fails_closed(self):
        before, after = activation_pair("native-prefetch-off")
        after["dispatcher"]["temporary_slot_uses"] = 1
        with self.assertRaisesRegex(ablation.base.GateError, "temporary overload"):
            ablation.activation_delta("native-prefetch-off", before, after)

    def test_prefetch_off_rejects_any_speculative_copy(self):
        before, after = activation_pair("bpf-prefetch-off")
        after["dispatcher"]["prefetch_copy_started"] = 1
        with self.assertRaisesRegex(ablation.base.GateError, "prefetch-off"):
            ablation.activation_delta("bpf-prefetch-off", before, after)

    def test_bpf_arm_requires_measured_demand_eviction_calls(self):
        before, after = activation_pair("bpf-prefetch-on")
        after["dispatcher"]["bpf_demand_eviction_calls"] = 0
        with self.assertRaisesRegex(ablation.base.GateError, "real BPF"):
            ablation.activation_delta("bpf-prefetch-on", before, after)

    def test_prefetch_byte_conservation_is_mandatory(self):
        before, after = activation_pair("native-prefetch-on")
        after["dispatcher"]["prefetch_hit_bytes"] += 1
        with self.assertRaisesRegex(ablation.base.GateError, "byte accounting"):
            ablation.activation_delta("native-prefetch-on", before, after)

    def test_analysis_requires_exact_twenty_cells_and_120_requests(self):
        blocks = []
        for item in ablation.schedule():
            cells = []
            for index, arm in enumerate(item["arms"]):
                counters = {key: 0 for key in ablation.DELTA_FIELDS}
                counters.update(demand_prefill_accesses=10, demand_prefill_hits=8,
                                demand_prefill_misses=2, demand_decode_accesses=20,
                                demand_decode_hits=15, demand_decode_misses=5,
                                prefetch_unused_resident=0,
                                prefetch_unused_resident_bytes=0)
                cells.append({
                    "arm": arm, "passed": True, "shadow_verification": False,
                    "verified_requests": 6, "verified_output_tokens": 384,
                    "requests": [{"passed": True}] * 6,
                    "prompt_order": item["prompts"],
                    "activation_delta": {"cache_budget_bytes": 1024,
                                         "dispatcher": counters},
                    "output_throughput_tokens_per_s": 10.0 + index,
                    "first_text_ttft_median_ms": 1.0,
                    "e2e_median_ms": 2.0, "final_drain_s": 0.1,
                })
            blocks.append({**item, "passed": True, "cells": cells})
        result = ablation.analyze(blocks)
        self.assertTrue(result["complete"])
        self.assertEqual(result["valid_cells"], 20)
        self.assertEqual(result["verified_measured_requests"], 120)
        blocks[0]["cells"][0]["requests"].pop()
        self.assertFalse(ablation.analyze(blocks)["complete"])


class SourceWiringTests(unittest.TestCase):
    def test_runtime_inventory_rejects_old_four_argument_store_abi(self):
        old = ("0000 T ExpertDispatcher::ConfigureActivationPolicy(int, "
               "std::string const&, std::string const&, bool)")
        with (mock.patch.object(ablation.Path, "glob", return_value=[
                  ablation.Path("/tmp/_store.cpython-312-x86_64-linux-gnu.so")]),
              mock.patch.object(ablation.base, "run_checked", return_value=old)):
            with self.assertRaisesRegex(ablation.base.GateError, "five-argument"):
                ablation.runtime_inventory({})

    def test_runtime_inventory_accepts_five_argument_store_abi(self):
        new = ("0000 T ExpertDispatcher::ConfigureActivationPolicy(int, "
               "std::string const&, std::string const&, bool, bool)")
        expected = {"files": []}
        with (mock.patch.object(ablation.Path, "glob", return_value=[
                  ablation.Path("/tmp/_store.cpython-312-x86_64-linux-gnu.so")]),
              mock.patch.object(ablation.base, "run_checked", return_value=new),
              mock.patch.object(ablation.prior, "runtime_inventory",
                                return_value=expected),
              mock.patch.object(ablation.base, "file_metadata",
                                side_effect=lambda path: {"path": str(path)})):
            observed = ablation.runtime_inventory({})
        self.assertEqual(observed["files"][-2:], [
            {"path": str(ablation.Path(ablation.__file__))},
            {"path": str(ablation.Path(ablation.__file__).with_name(
                "audit_prefetch_ablation.py"))},
        ])

    def test_dispatcher_has_real_toggle_and_required_observability(self):
        source = (ablation.Path(__file__).resolve().parent /
                  "deps/MoE-Infinity/core/parallel/expert_dispatcher.cpp").read_text()
        self.assertIn("if (!activation_prefetch_enabled_)", source)
        self.assertIn('"demand_prefill_misses"', source)
        self.assertIn('"demand_decode_misses"', source)
        self.assertIn('"temporary_slot_uses"', source)
        self.assertIn('"demand_bytes"', source)
        self.assertIn('"prefetch_copy_wait_ns"', source)
        self.assertIn('"bpf_demand_eviction_calls"', source)

    def test_additive_source_patch_matches_the_staged_tree(self):
        here = ablation.Path(__file__).resolve().parent
        ablation.base.run_checked([
            "git", "apply", "--check", "--reverse",
            str(here / "predictive-prefetch-ablation.patch"),
        ], here / "deps/MoE-Infinity")

    def test_raw_auditor_requires_exact_prefetch_launch_toggle(self):
        with tempfile.TemporaryDirectory() as directory:
            cell = ablation.Path(directory)
            with mock.patch.object(ablation.paper.subprocess, "Popen", return_value=object()):
                _, log = ablation.paper.launch(
                    "paper-bpf", cell, 18230, verify=False, prefetch=False)
                log.close()
            audit._launch(cell, "bpf-prefetch-off")
            launch = __import__("json").loads((cell / "launch.json").read_text())
            launch["env"]["MOE_REVISION_PREFETCH"] = "1"
            (cell / "launch.json").write_text(__import__("json").dumps(launch))
            with self.assertRaisesRegex(ablation.base.GateError, "launch differs"):
                audit._launch(cell, "bpf-prefetch-off")


if __name__ == "__main__":
    unittest.main()
