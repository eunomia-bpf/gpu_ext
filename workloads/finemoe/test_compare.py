"""CPU-only protocol/accounting tests; fixtures are never performance results."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import compare


def fixture(arm="finemoe-bpf", numerical=True):
    data = json.loads((Path(__file__).parent / "dataset-mtbench-v1.json").read_text())
    gold = {"absolute_tolerance": 0., "requests": {}, "runtime_versions": {"fixture": "not an experiment"}}
    records = []
    for i, row in enumerate(data["warmup"] + data["evaluation"]):
        start = 1 + i * 100
        record = {"question_id": row["question_id"], "input_ids": row["input_ids"],
                  "generated_ids": list(range(16)), "begin_ns": start,
                  "verified_ready_ns": start + 80, "token_ready_ns": list(range(start + 1, start + 17)),
                  "ttft_ns": 1, "tpot_ns": 1,
                  "correctness": {"exact_token_match": True, "checked_generated_tokens": 16,
                                  "logits_checked": numerical, "compared_logits": 16, "max_abs_error": 0.}}
        records.append(record)
        gold["requests"][str(row["question_id"])] = copy.deepcopy(record)
    demand = arm == "demand-only"
    counts = {key: 0 for key in (
        "prefetch_copy_started", "prefetch_copy_completed", "prefetch_copy_bytes",
        "demand_copy_started", "demand_copy_completed", "demand_copy_bytes",
        "prefetch_first_use_copies", "prefetch_first_use_bytes", "prefetch_evicted_unused_copies",
        "prefetch_evicted_unused_bytes", "prefetch_resident_unused_copies", "prefetch_resident_unused_bytes",
        "prefetch_queue_enqueued", "prefetch_queue_canceled", "prefetch_queue_dequeued",
        "prefetch_enqueue_resident_skip", "prefetch_copy_errors", "compute_release_sync_errors")}
    counts.update(expert_demand_uses=1, expert_demand_cache_hits=0, expert_demand_cache_misses=1,
                  compute_release_syncs=1, peak_pool_resident_bytes=400)
    prefix = "demand" if demand else "prefetch"
    counts.update({f"{prefix}_copy_started": 1, f"{prefix}_copy_completed": 1, f"{prefix}_copy_bytes": 256})
    if not demand:
        counts.update(prefetch_first_use_copies=1, prefetch_first_use_bytes=256,
                      prefetch_queue_enqueued=4, prefetch_queue_canceled=3, prefetch_queue_dequeued=1)
    stats = {"arm": arm, "shadow": numerical, "prediction_maps": 1, "selector_rows": 1,
             "selected_candidates": 0 if demand else 4, "engine_admitted_candidates": 0 if demand else 4,
             "engine_enqueue_calls": 0 if demand else 4, "jit_calls": int(arm == "finemoe-bpf")}
    stats[f"cardinality_{0 if demand else 4}"] = 1
    if arm in ("finemoe-c", "finemoe-bpf"):
        stats.update(policy_calls=1, oracle_checks=int(numerical), cardinality_4=1)
    events = [{"event": "selector", "layer_start": 1, "layer_end": 2, "delta": .5,
               "probabilities": [[.7, .1, .1, .1] + [0.] * 56], "masks": [0 if demand else 15]}]
    if not demand:
        events += [{"event": "engine_candidates", "tensor_ids": [1, 2, 3, 4]}]
        events += [{"event": "engine_enqueue", "tensor_id": i, "device": 0, "probability": .1} for i in range(1, 5)]
    result = {"status": "passed", "arm": arm, "check_logits": numerical, "model": data["model"],
              "runtime_versions": gold["runtime_versions"],
              "requests": records[1:], "warmup": records[:1], "begin_ns": 100, "end_ns": 900,
              "elapsed_seconds": 8e-7, "generated_tokens": 128, "tokens_per_second": 128 / 8e-7,
              "cpu_seconds": 0.1, "golden_absolute_tolerance": 0., "policy_before": {},
              "cpu_seconds_including_drain": .11, "application_native_begin_ns": 100,
              "application_native_end_ns": 900, "native_drained_ns": 1000,
              "application_clock": "steady_clock", "drain_begin_ns": 910, "drain_end_ns": 990,
              "drain_seconds": 8e-8, "elapsed_seconds_including_drain": 9e-7,
              "policy_after": {"stats": stats, "events": events if numerical else []},
              "after": {"drained": True, "copy_fields": compare.FIELDS, "clock": "steady_clock",
                        "counters": counts, "copies": [[1, 3, 256, int(not demand), 110, 120, 0 if demand else 130, 0]],
                        "pool_resident_bytes": 300, "pool_capacity_bytes": 1000, "resident_sparse_bytes": 256,
                        "resident_dense_bytes": 44, "sparse_budget_bytes": 956}}
    return result, data, gold


class CompareTests(unittest.TestCase):
    def test_all_four_arms_and_both_protocols(self):
        for arm in compare.ARMS:
            for numerical in (False, True):
                result, data, gold = fixture(arm, numerical)
                self.assertGreater(compare.audit_cell(result, data, arm, numerical, gold)["tokens_per_second"], 0)

    def reject(self, mutate):
        result, data, gold = fixture()
        mutate(result)
        with self.assertRaises(compare.base.GateError):
            compare.audit_cell(result, data, "finemoe-bpf", True, gold)

    def test_incomplete_copy_rejected(self):
        self.reject(lambda r: r["after"]["copies"][0].__setitem__(5, 0))

    def test_resident_unused_not_relabelled_waste(self):
        self.reject(lambda r: r["after"]["copies"][0].__setitem__(6, 0))

    def test_partial_enqueue_cannot_pass(self):
        self.reject(lambda r: r["after"]["counters"].__setitem__("prefetch_queue_enqueued", 3))

    def test_pool_overcommit_rejected(self):
        self.reject(lambda r: r["after"]["counters"].__setitem__("peak_pool_resident_bytes", 1001))

    def test_bpf_fallback_rejected(self):
        self.reject(lambda r: r["policy_after"]["stats"].__setitem__("jit_calls", 0))

    def test_false_correctness_flag_rejected_by_raw_tokens(self):
        self.reject(lambda r: r["requests"][0]["generated_ids"].__setitem__(0, 999))

    def test_actual_input_mask_rechecked_independently(self):
        self.reject(lambda r: r["policy_after"]["events"][0]["masks"].__setitem__(0, 31))

    def test_all_sixty_is_a_valid_negative_result_not_a_failure(self):
        result, data, gold = fixture(numerical=False)
        stats = result["policy_after"]["stats"]
        stats.pop("cardinality_4")
        stats["cardinality_60"] = 1
        stats.update(selected_candidates=60, engine_admitted_candidates=60, engine_enqueue_calls=60)
        result["after"]["counters"].update(prefetch_queue_enqueued=60, prefetch_queue_canceled=59)
        # Cardinality is reported, not conditioned on observing a favorable set reduction.
        metrics = compare.audit_cell(result, data, "finemoe-bpf", False, gold)
        self.assertFalse(metrics["observed_dynamic_set_reduction"])

    def test_tail_copies_are_not_reported_in_application_window(self):
        result, data, gold = fixture()
        result["after"]["copies"][0][5:7] = [950, 0]
        counts = result["after"]["counters"]
        counts.update(prefetch_first_use_copies=0, prefetch_first_use_bytes=0,
                      prefetch_resident_unused_copies=1, prefetch_resident_unused_bytes=256)
        metrics = compare.audit_cell(result, data, "finemoe-bpf", True, gold)
        self.assertEqual(metrics["copies_completed_after_application_window"]["prefetch_copy_bytes"], 256)
        self.assertNotIn("prefetch_copy_bytes", metrics["copies_completed_in_application_window"])
        self.assertLess(metrics["tokens_per_second_including_drain"], metrics["tokens_per_second"])

    def test_zero_prefetch_copies_is_not_a_forced_positive_gate(self):
        result, data, gold = fixture()
        result["after"]["copies"] = []
        counts = result["after"]["counters"]
        counts.update(prefetch_copy_started=0, prefetch_copy_completed=0, prefetch_copy_bytes=0,
                      prefetch_first_use_copies=0, prefetch_first_use_bytes=0,
                      expert_demand_cache_hits=1, expert_demand_cache_misses=0)
        metrics = compare.audit_cell(result, data, "finemoe-bpf", True, gold)
        self.assertFalse(metrics["observed_actual_prefetch_copy"])

    def test_formal_orders_have_five_complete_blocks(self):
        orders = compare.orders("full")
        self.assertEqual(len(orders), 5)
        for order in orders:
            self.assertEqual(sorted(order), sorted(compare.ARMS))
        self.assertEqual(orders, compare.orders("full"))

    def test_commands_call_real_worker_and_numerics_only_in_preflight(self):
        args = SimpleNamespace(data=Path("data.json"), mode="preflight", golden=Path("gold"), history=Path("history"))
        command = compare.command(args, "cell", Path("out"), "finemoe-bpf")
        self.assertIn(str(compare.HERE / "inference.py"), command)
        self.assertIn("--check-logits", command)
        args.mode = "full"
        self.assertNotIn("--check-logits", compare.command(args, "cell", Path("out"), "finemoe-bpf"))

    def test_common_allocator_environment_suppresses_legacy_alias(self):
        inherited = {"PYTORCH_CUDA_ALLOC_CONF": "legacy-fixture-do-not-record",
                     "PYTORCH_ALLOC_CONF": "old-fixture", "PATH": "/fixture/bin"}
        child, recorded, removed = compare.child_environment(inherited)
        self.assertEqual(child["PYTORCH_ALLOC_CONF"], "expandable_segments:True")
        self.assertEqual(recorded["PYTORCH_ALLOC_CONF"], "expandable_segments:True")
        self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", child)
        self.assertEqual(removed, ["PYTORCH_CUDA_ALLOC_CONF"])
        self.assertEqual(child["PATH"], inherited["PATH"])
        self.assertEqual(inherited["PYTORCH_CUDA_ALLOC_CONF"], "legacy-fixture-do-not-record")
        self.assertNotIn("legacy-fixture-do-not-record", str(recorded))
        self.assertEqual(compare.child_environment({})[2], [])


if __name__ == "__main__":
    unittest.main()
