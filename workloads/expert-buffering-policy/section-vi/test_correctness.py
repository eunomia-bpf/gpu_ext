"""Small CPU parser/environment fixtures, never model or live GPU evidence."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import correctness as gate


def fixture(arm, numerical=True):
    # Deliberately synthetic accounting records for rejection tests, not raw runs.
    data = {"model": {"fixture": True}, "warmup": [{"question_id": 0, "input_ids": [10]}],
            "evaluation": [{"question_id": i, "input_ids": [10 + i]} for i in range(1, 9)]}
    records = []
    for row in data["warmup"] + data["evaluation"]:
        offset = row["question_id"] * 20
        records.append({**row, "generated_ids": list(range(16)), "begin_ns": offset + 1,
                        "token_ready_ns": list(range(offset + 2, offset + 18)), "verified_ready_ns": offset + 18,
                        "ttft_ns": 1, "tpot_ns": 1.0,
                        "correctness": {"exact_token_match": True, "checked_generated_tokens": 16,
                                        "logits_checked": True, "compared_logits": 16 * 151936,
                                        "max_abs_error": 0.0}})
    gold = {"model": data["model"], "runtime_versions": {}, "absolute_tolerance": 0.0,
            "requests": {str(r["question_id"]): copy.deepcopy(r) for r in records}}
    before = {"decisions": 5, "jit_calls": 5 if arm == "bpf" else 0, "admissions": 2, "evictions": 0}
    after = {"decisions": 8, "jit_calls": 8 if arm == "bpf" else 0, "admissions": 4, "evictions": 1,
             "registered_layers": 24, "resident_sparse_bytes": 2 * gate.EXPERT_BYTES}
    result = {"status": "passed", "arm": arm, "capacity": 16, "correctness_only": True,
              "check_logits": True, "engine_cleanup_returned": True, "model": data["model"],
              "runtime_versions": {}, "golden_absolute_tolerance": 0.0, "decoding_configuration": {},
              "warmup": records[:1], "requests": records[1:], "evaluation_generated_tokens": 128,
              "correctness_generated_tokens": 144,
              "begin_ns": 20, "end_ns": 179, "clock": "perf_counter_ns",
              "drain_begin_ns": 180, "drain_end_ns": 181,
              "eb_before": before, "eb_after": after,
              "shadow": {"checks": 8, "jit_calls": 8, "mismatches": 0} if arm == "bpf" else None,
              "selector_library": str(gate.HERE / "build" / ("libeb_shadow.so" if arm == "bpf" else "libeb_policy.so")),
              "real_selector_library": str(gate.HERE / "build/libeb_policy.so") if arm == "bpf" else None,
              "application_native_begin_ns": 1, "application_native_end_ns": 8,
              "native_drained_ns": 9, "application_clock": "steady_clock",
              "before": {"pool_capacity_bytes": gate.POOL_BYTES},
              "after": {"drained": True, "copy_fields": gate.common.FIELDS, "clock": "steady_clock",
                        "copies": [[1, 1, gate.EXPERT_BYTES, 0, 2, 3, 0, 0],
                                   [2, 2, gate.EXPERT_BYTES, 0, 4, 5, 0, 0]],
                        "counters": {"demand_copy_started": 2, "demand_copy_completed": 2,
                                     "demand_copy_bytes": 2 * gate.EXPERT_BYTES,
                                     "compute_release_sync_errors": 0, "expert_demand_uses": 3,
                                     "expert_demand_cache_hits": 1, "expert_demand_cache_misses": 2,
                                     "compute_release_syncs": 3, "peak_pool_resident_bytes": 3 * gate.EXPERT_BYTES},
                        "pool_capacity_bytes": gate.POOL_BYTES, "pool_resident_bytes": 3 * gate.EXPERT_BYTES,
                        "resident_sparse_bytes": 2 * gate.EXPERT_BYTES, "resident_dense_bytes": gate.EXPERT_BYTES,
                        "sparse_budget_bytes": 4 * gate.EXPERT_BYTES}}
    if not numerical:
        result.update(correctness_only=False, check_logits=False, shadow=None, real_selector_library=None,
                      selector_library=str(gate.HERE / "build/libeb_policy.so"),
                      elapsed_seconds=7e-9, tokens_per_second=128e9 / 7,
                      elapsed_seconds_including_drain=8e-9, drain_seconds=1e-9,
                      cpu_seconds=1.0, cpu_seconds_including_drain=1.1)
        for request in records:
            request["correctness"] = {"exact_token_match": True, "checked_generated_tokens": 16,
                                      "logits_checked": False}
    return result, data, gold


class CorrectnessTests(unittest.TestCase):
    def setUp(self):
        # Synthetic model metadata is not a real checkpoint. Actual decoding is
        # independently read from generation_config.json, never a golden field.
        self.decoding = mock.patch.object(gate, "expected_decoding", return_value={})
        self.decoding.start()
        self.addCleanup(self.decoding.stop)

    def test_sources_parse_without_importing_gpu_worker(self):
        for name in ("correctness.py", "inference_eb.py", "analyze_results.py", "test_shadow_bridge.py"):
            ast.parse((gate.HERE / name).read_text(), filename=name)

    def test_environment_is_explicit_and_drops_inherited_shadow(self):
        inherited = {"EB_SECTION_VI_UNTIMED_SHADOW": "1", "EB_SECTION_VI_REAL_LIBRARY": "/wrong",
                     "EB_SECTION_VI_LIBRARY": "/wrong", "FINEMOE_POLICY": "wrong"}
        for arm in gate.ARMS:
            env, changes, removed = gate.environment(arm, 16, inherited)
            self.assertEqual(env["EB_SECTION_VI_ARM"], arm)
            self.assertEqual(env["EB_SECTION_VI_CAPACITY"], "16")
            self.assertEqual(env["EB_SECTION_VI_CORRECTNESS_ONLY"], "1")
            self.assertIn("FINEMOE_POLICY", removed)
            self.assertNotIn("FINEMOE_POLICY", env)
            self.assertEqual("EB_SECTION_VI_UNTIMED_SHADOW" in env, arm == "bpf")
            expected = "libeb_shadow.so" if arm == "bpf" else "libeb_policy.so"
            self.assertEqual(Path(changes["EB_SECTION_VI_LIBRARY"]).name, expected)
            timed, _, _ = gate.environment(arm, 16, inherited, numerical=False)
            self.assertNotIn("EB_SECTION_VI_CORRECTNESS_ONLY", timed)
            self.assertNotIn("EB_SECTION_VI_UNTIMED_SHADOW", timed)
            self.assertNotIn("EB_SECTION_VI_REAL_LIBRARY", timed)
            self.assertEqual(Path(timed["EB_SECTION_VI_LIBRARY"]).name, "libeb_policy.so")

    def test_command_uses_private_runtime_and_no_history(self):
        args = SimpleNamespace(source=Path("/private"), data=Path("/data"), golden=Path("/golden"),
                               capacity=16, mode="preflight")
        cmd = gate.command(args, "cell", Path("/output"), "bpf")
        self.assertIn(str(gate.HERE / "inference_eb.py"), cmd)
        self.assertEqual(cmd[cmd.index("--source") + 1], "/private")
        self.assertNotIn("--history", cmd)
        self.assertIn("--check-logits", cmd)
        args.mode = "full"
        self.assertNotIn("--check-logits", gate.command(args, "cell", Path("/output"), "bpf"))

    def test_accounting_accepts_all_three_fixture_arms(self):
        for arm in gate.ARMS:
            result, data, gold = fixture(arm)
            self.assertEqual(gate.audit_cell(result, data, arm, True, gold)["correctness_generated_tokens"], 144)

    def test_rejects_missing_shadow_numerics_eviction_or_pool(self):
        mutations = (
            lambda r: r["shadow"].update(checks=7),
            lambda r: r["shadow"].update(mismatches=1),
            lambda r: r["eb_after"].update(evictions=0),
            lambda r: r["eb_after"].update(decisions=True),
            lambda r: r["after"].update(pool_capacity_bytes=gate.POOL_BYTES + 1),
            lambda r: r["requests"][0]["correctness"].update(max_abs_error=1e-9),
            lambda r: r["requests"][0]["correctness"].update(compared_logits=1),
            lambda r: r["after"]["copies"][0].__setitem__(2, 1),
            lambda r: r["after"]["counters"].update(prefetch_copy_started=1),
            lambda r: r.update(correctness_only=False),
        )
        for mutation in mutations:
            result, data, gold = fixture("bpf")
            mutation(result)
            with self.assertRaises(gate.common.base.GateError):
                gate.audit_cell(result, data, "bpf", True, gold)

    def test_timed_has_real_tokens_and_engagement_but_no_numerical_claim(self):
        for arm in gate.ARMS:
            result, data, gold = fixture(arm, False)
            metrics = gate.audit_cell(result, data, arm, False, gold)
            self.assertEqual(metrics["tokens_per_second"], 128e9 / 7)
            self.assertNotIn("compared_logits", metrics)
            self.assertNotIn("maximum_absolute_error", metrics)
        mutations = (
            lambda r: r.update(shadow={"checks": 8, "mismatches": 0, "jit_calls": 8}),
            lambda r: r["requests"][0]["correctness"].update(compared_logits=0),
            lambda r: r["requests"][0]["correctness"].update(logits_checked=True),
            lambda r: r["requests"][0].update(logits_file="fake.npy"),
            lambda r: r["requests"][0]["generated_ids"].__setitem__(0, 99),
            lambda r: r.update(tokens_per_second=1),
            lambda r: r["eb_after"].update(jit_calls=7),
            lambda r: r.update(selector_library=str(gate.HERE / "build/libeb_shadow.so")),
        )
        for mutation in mutations:
            result, data, gold = fixture("bpf", False)
            mutation(result)
            with self.assertRaises(gate.common.base.GateError):
                gate.audit_cell(result, data, "bpf", False, gold)

    def test_five_complete_randomized_three_arm_blocks(self):
        rows = gate.orders()
        self.assertEqual(gate.SEED, 20260903)
        self.assertEqual(rows, gate.orders())
        self.assertEqual(len({tuple(row) for row in rows}), 5)
        self.assertTrue(all(sorted(row) == sorted(gate.ARMS) for row in rows))

    def test_incomplete_or_changed_preflight_is_rejected_before_any_launch(self):
        expected = dict.fromkeys(("source", "capacity", "runtime", "data", "model_files", "reference_files", "golden"), {})
        saved = {**expected, "complete": True, "mode": "correctness_only", "valid_cells": 3,
                 "valid_blocks": 1, "arms": list(gate.ARMS), "orders": [list(gate.ARMS)],
                 "preflight": None, "numerical_audits": dict.fromkeys(gate.ARMS, {})}
        directory = mock.MagicMock()
        directory.iterdir.return_value = [SimpleNamespace(name=arm, is_dir=lambda: True) for arm in gate.ARMS]
        directory.rglob.return_value = []
        mutations = (lambda row: row.update(valid_cells=2),
                     lambda row: row.update(arms=["fifo", "native", "native"]),
                     lambda row: row.update(orders=[["bpf", "native", "fifo"]]),
                     lambda row: row.update(runtime={"changed": True}),
                     lambda row: row.update(capacity=17),
                     lambda row: row.update(preflight="another-attempt"))
        for mutate in mutations:
            invalid = copy.deepcopy(saved)
            mutate(invalid)
            with mock.patch.object(gate, "read_json", return_value=invalid), \
                    mock.patch.object(gate, "audit_saved_cell") as audit:
                with self.assertRaises(gate.common.base.GateError):
                    gate.validate_preflight(directory, expected, {})
                audit.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
