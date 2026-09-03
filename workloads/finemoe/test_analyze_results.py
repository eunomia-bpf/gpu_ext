"""Synthetic CPU-only fixtures, never experimental performance results."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import analyze_results as analysis
import compare
from test_compare import fixture


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def safety():
    return {"power_limit_service": "active", "power_limit_w": 400,
            "dmesg_abnormal": [], "journal_abnormal": [], "xids": [], "uvm_refcount": 0,
            "struct_ops": {"maps": [], "links": []},
            "gpu": {"compute_apps": [], "memory_used_mib": 2, "utilization_gpu_percent": 0}}


def campaign(root):
    """Build complete disposable logs with synthetic values, not CUDA results."""
    _, data, golden = fixture(numerical=False)
    data_path = root / "data.json"
    write(data_path, data)
    full = root / "full"
    full.mkdir()
    runtime = {"synthetic_fixture_only": {"bytes": 1, "mtime_ns": 1}}
    gold_dir, history_dir = root / "golden/stage", root / "history/stage"
    golden.update(status="passed", model=data["model"], data=str(data_path), same_arm_repeat_max_abs_error=0.)
    for row in data["history"]:
        golden["requests"][str(row["question_id"])] = {
            "question_id": row["question_id"], "input_ids": row["input_ids"],
            "generated_ids": list(range(16)), "correctness": {
                "exact_token_match": True, "checked_generated_tokens": 16, "logits_checked": False}}
    config = compare.read_json(Path(data["model"]["snapshot"]) / "generation_config.json")
    decoding = {"checkpoint_fields": {key: config[key] for key in
                ("do_sample", "repetition_penalty", "eos_token_id", "pad_token_id", "temperature", "top_k", "top_p")},
                "explicit_overrides": {"do_sample": False, "min_new_tokens": 16,
                                       "max_new_tokens": 16, "pad_token_id": 151643}}
    history = {"status": "passed", "model": data["model"], "data": str(data_path),
               "runtime_versions": golden["runtime_versions"], "store_capacity": 1000,
               "store_data_size": 1000, "question_ids": [r["question_id"] for r in data["history"]],
               "decoding_configuration": decoding,
               "requests": [copy.deepcopy(golden["requests"][str(row["question_id"])]) for row in data["history"]]}
    gold_dir.mkdir(parents=True)
    for row in data["evaluation"] + data["warmup"]:
        filename = f"question-{row['question_id']}-logits.npy"
        golden["requests"][str(row["question_id"])]["logits_file"] = filename
        np.save(gold_dir / filename, np.zeros((16, 1, 3), dtype=np.float32), allow_pickle=False)
    write(gold_dir / "golden.json", golden)
    write(history_dir / "history.json", history)
    # Metadata-only fixtures: these are deliberately not real NPY/model content.
    for name in ("store~embed~1000.npy", "store~traj~1000.npy"):
        (history_dir / name).write_bytes(b"synthetic fixture, not an array")
    model = root / "fixture-model-not-real"
    model.write_text("fixture")
    references = {str(path): compare.metadata(path) for directory in (gold_dir, history_dir)
                  for path in directory.iterdir()}
    for directory, mode in ((gold_dir, "golden"), (history_dir, "history")):
        write(directory.parent / "campaign.json", {"mode": mode, "complete": True, "diagnostic": False,
              "data": data, "golden": str(gold_dir), "reference_files": {
                  key: item for key, item in references.items() if Path(key).parent == gold_dir}})
        write(directory / "result.json", {"status": "passed", "diagnostic": False})
    manifest = {"schema": "finemoe_dynamic_set_v1", "mode": "full", "complete": True,
                "diagnostic": False, "valid_blocks": 5, "seed": analysis.SEED,
                "orders": analysis.expected_orders(), "data": data, "runtime": runtime,
                "golden": str(gold_dir), "history": str(history_dir), "reference_files": references,
                "model_files": [compare.metadata(model)], "source_revision": "synthetic fixture only",
                "preflight": str(root / "preflight")}
    write(full / "campaign.json", manifest)
    preflight = {**manifest, "mode": "preflight", "valid_blocks": 1,
                 "orders": manifest["orders"][:1], "preflight": None}
    write(root / "preflight/campaign.json", preflight)
    position = 0
    ordered_blocks = [(root / "preflight", 0, preflight["orders"][0], True)]
    ordered_blocks += [(full, block, order, False) for block, order in enumerate(manifest["orders"])]
    for campaign_dir, block, order, numerical in ordered_blocks:
        for arm in order:
            directory = campaign_dir / f"block-{block:02d}" / arm
            directory.mkdir(parents=True)
            worker, _, _ = fixture(arm, numerical=numerical)
            worker.update(data=str(data_path), clock="perf_counter_ns", decoding_configuration=decoding)
            if numerical:
                for row in worker["warmup"] + worker["requests"]:
                    row["logits_file"] = f"question-{row['question_id']}-logits.npy"
                    row["correctness"]["compared_logits"] = 48
                    np.save(directory / row["logits_file"], np.zeros((16, 1, 3), dtype=np.float32), allow_pickle=False)
                    write(directory / f"question-{row['question_id']}-result.json", {
                        "status": "passed", "request": copy.deepcopy(row),
                        "expected_generated_ids": row["generated_ids"], "golden_absolute_tolerance": 0.})
                    row["verified_ready_ns"] += 1  # Expected post-check update, no other changed fields.
            offset = position * 2000
            for key in ("application_native_begin_ns", "application_native_end_ns", "native_drained_ns"):
                worker[key] += offset
            for row in worker["after"]["copies"]:
                for index in (4, 5, 6, 7):
                    if row[index]:
                        row[index] += offset
            args = SimpleNamespace(mode="preflight" if numerical else "full", data=data_path,
                                   golden=gold_dir, history=history_dir)
            launch = {"status": "running", "stage": "cell", "arm": arm,
                      "command": compare.command(args, "cell", directory, arm), "diagnostic": False,
                      "environment": compare.child_environment({})[1], "environment_removed_names": [],
                      "safety_before": safety(), "runtime_before": runtime}
            (directory / "worker.log").write_text("SYNTHETIC TEST FIXTURE ONLY\n")
            telemetry = directory / "gpu-telemetry.csv"
            telemetry.write_text("timestamp,memory,temp,power,clock,util,reason\n"
                                 "fixture,2,30,10,300,0,Not Active\n")
            result = {**launch, "status": "passed", "returncode": 0,
                      "runtime_after": runtime, "safety_after": safety(),
                      "telemetry": compare.base.validate_gpu_telemetry(telemetry, allow_fixed_power_cap=True),
                      "metrics": compare.audit_cell(worker, data, arm, numerical, golden)}
            write(directory / "launch.json", launch)
            write(directory / "worker-result.json", worker)
            write(directory / "result.json", result)
            position += 1
    return full, manifest


class AnalysisTests(unittest.TestCase):
    def test_raw_times_not_summary_values(self):
        worker, data, golden = fixture(numerical=False)
        audited = compare.audit_cell(worker, data, worker["arm"], False, golden)
        worker["requests"][0]["ttft_ns"] = 999
        metrics, requests = analysis.reconstruct(worker)
        self.assertEqual(requests[0]["ttft_ms"], 1e-6)
        analysis.check_derived(metrics, audited)
        with self.assertRaises(compare.base.GateError):
            compare.audit_cell(worker, data, worker["arm"], False, golden)

    def test_lifecycles_at_cutoff_and_after_drain(self):
        worker, _, _ = fixture(numerical=False)
        worker["after"]["copies"] = [
            [1, 1, 10, 1, 110, 120, 130, 200],  # First-used, then evicted.
            [2, 2, 20, 1, 110, 120, 0, 950],  # Resident at cutoff, evicted in tail.
            [3, 3, 30, 1, 800, 900, 0, 0],  # Exact cutoff belongs in window.
            [4, 4, 40, 1, 800, 950, 0, 960],  # Inflight at cutoff; tail eviction.
            [5, 5, 50, 1, 910, 950, 0, 0],  # Started in tail, right-censored.
            [6, 6, 60, 0, 110, 120, 0, 0],
        ]
        metrics, _ = analysis.reconstruct(worker)
        self.assertEqual(metrics["in_window.prefetch_copy_bytes"], 60)
        self.assertEqual(metrics["in_window.prefetch_resident_unused_bytes"], 50)
        self.assertEqual(metrics["drained.prefetch_evicted_unused_bytes"], 60)
        self.assertEqual(metrics["tail.prefetch_evicted_unused_bytes"], 40)
        self.assertEqual(metrics["tail.prefetch_resident_unused_bytes"], 50)
        self.assertEqual(metrics["tail.prefetch_inflight_at_deadline_bytes"], 40)
        self.assertEqual(metrics["tail.prefetch_started_after_window_bytes"], 50)
        self.assertEqual(metrics["in_window.demand_copy_bytes"], 60)

    def test_positive_and_zero_ratio_semantics(self):
        draws = [tuple(range(5))] * 20
        positive = analysis.paired([3.] * 5, [2.] * 5, draws)
        self.assertAlmostEqual(positive["geometric_mean_ratio"], 1.5)
        self.assertEqual(positive["absolute_difference_ci95"], [1., 1.])
        zero = analysis.paired([0.] * 5, [2.] * 5, draws)
        self.assertFalse(zero["ratio_defined"])
        self.assertIsNone(zero["ratio_ci95"])
        self.assertEqual(zero["paired_ratios"], [0.] * 5)
        self.assertEqual(zero["absolute_difference_ci95"], [-2., -2.])
        both_zero = analysis.paired([0.] * 5, [0.] * 5, draws)
        self.assertEqual(both_zero["paired_ratios"], [None] * 5)
        self.assertEqual(both_zero["absolute_difference_ci95"], [0., 0.])

    def test_negative_and_partial_values_not_dropped(self):
        for a, b in (([-1.] + [1.] * 4, [1.] * 5), ([1.] * 4, [1.] * 5)):
            with self.assertRaises(compare.base.GateError):
                analysis.paired(a, b, [tuple(range(5))])

    def test_complete_campaign_all_twenty_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            full, _ = campaign(Path(temporary))
            # Short draws only for this integration fixture; production is always 10,000.
            with mock.patch.object(analysis, "DRAWS", 20):
                result = analysis.analyze(full)
            self.assertTrue(result["complete"])
            self.assertEqual(len(result["cells"]), 20)
            self.assertEqual(len(result["comparisons"]), 6)
            self.assertEqual(result["preflight"]["raw_logit_arrays"], 36)
            self.assertEqual(result["history_raw_tokens_checked"], 1024)
            self.assertEqual({row["arm"] for row in result["cells"]}, set(analysis.ARMS))
            comparison = result["comparisons"]["finemoe-bpf_over_demand-only"]
            self.assertFalse(comparison["drained.prefetch_copy_bytes"]["ratio_defined"])

    def test_matrix_rejects_incomplete_failure_and_extra_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            full, manifest = campaign(Path(temporary))
            incomplete = {**manifest, "complete": False}
            with self.assertRaises(compare.base.GateError):
                analysis.validate_matrix(full, incomplete)
            (full / "block-00" / "retry").mkdir()
            with self.assertRaises(compare.base.GateError):
                analysis.validate_matrix(full, manifest)

    def test_gate_rejects_runtime_data_and_false_engagement(self):
        with tempfile.TemporaryDirectory() as temporary:
            full, manifest = campaign(Path(temporary))
            directory = full / "block-00" / "finemoe-bpf"
            original = compare.read_json(directory / "result.json")
            worker = compare.read_json(directory / "worker-result.json")
            golden = compare.read_json(Path(manifest["golden"]) / "golden.json")
            mutations = (
                lambda row: row.update(status="rejected"),
                lambda row: row.update(runtime_after={"different": {}}),
                lambda row: row.update(cleanup_error="fixture failure"),
                lambda row: row["environment"].update(OMP_NUM_THREADS="8"),
            )
            for mutate in mutations:
                result = copy.deepcopy(original)
                mutate(result)
                write(directory / "result.json", result)
                with self.assertRaises(compare.base.GateError):
                    analysis.saved_cell(directory, manifest, golden)
            write(directory / "result.json", original)
            worker["policy_after"]["stats"]["jit_calls"] = 0
            write(directory / "worker-result.json", worker)
            with self.assertRaises(compare.base.GateError):
                analysis.saved_cell(directory, manifest, golden)

    def test_reference_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, manifest = campaign(Path(temporary))
            analysis.validate_references(manifest)
            path = Path(manifest["history"]) / "store~embed~1000.npy"
            path.write_bytes(b"changed synthetic fixture")
            with self.assertRaises(compare.base.GateError):
                analysis.validate_references(manifest)

    def test_saved_arrays_not_scalar_claims(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, manifest = campaign(Path(temporary))
            golden = analysis.validate_references(manifest)
            analysis.audit_preflight(manifest, golden)
            directory = Path(manifest["preflight"]) / "block-00/finemoe-bpf"
            qid = manifest["data"]["warmup"][0]["question_id"]
            filename = directory / f"question-{qid}-logits.npy"
            for array in (np.ones((16, 1, 3), dtype=np.float32),
                          np.full((16, 1, 3), np.nan, dtype=np.float32),
                          np.zeros((16, 1, 2), dtype=np.float32)):
                np.save(filename, array, allow_pickle=False)
                with self.assertRaises(compare.base.GateError):
                    analysis.audit_preflight(manifest, golden)

    def test_preflight_path_and_consistency_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, manifest = campaign(Path(temporary))
            golden = analysis.validate_references(manifest)
            with self.assertRaises(compare.base.GateError):
                analysis.audit_preflight({**manifest, "preflight": None}, golden)
            path = Path(manifest["preflight"]) / "campaign.json"
            original = compare.read_json(path)
            for altered in ({**original, "complete": False}, {**original, "runtime": {"changed": {}}},
                            {**original, "reference_files": {}}, {**original, "data": {}}):
                write(path, altered)
                with self.assertRaises(compare.base.GateError):
                    analysis.audit_preflight(manifest, golden)

    def test_retained_preflight_token_forgery_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, manifest = campaign(Path(temporary))
            golden = analysis.validate_references(manifest)
            directory = Path(manifest["preflight"]) / "block-00/demand-only"
            qid = manifest["data"]["evaluation"][0]["question_id"]
            path = directory / f"question-{qid}-result.json"
            record = compare.read_json(path)
            record["request"]["generated_ids"][0] = 999
            write(path, record)
            with self.assertRaises(compare.base.GateError):
                analysis.audit_preflight(manifest, golden)

    def test_history_raw_tokens_not_success_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, manifest = campaign(Path(temporary))
            path = Path(manifest["history"]) / "history.json"
            history = compare.read_json(path)
            history["requests"][0]["generated_ids"][0] = 999
            write(path, history)
            manifest["reference_files"][str(path)] = compare.metadata(path)
            with self.assertRaisesRegex(compare.base.GateError, "history raw tokens"):
                analysis.validate_references(manifest)

    def test_cell_decoding_must_match_frozen_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            full, manifest = campaign(Path(temporary))
            path = full / "block-00/finemoe-c/worker-result.json"
            worker = compare.read_json(path)
            worker["decoding_configuration"]["checkpoint_fields"]["repetition_penalty"] = 1.
            write(path, worker)
            golden = analysis.validate_references(manifest)
            with self.assertRaisesRegex(compare.base.GateError, "decoding"):
                analysis.saved_cell(path.parent, manifest, golden)

    def test_bootstrap_is_paired_and_frozen(self):
        rng = analysis.random.Random(analysis.SEED)
        draws = [tuple(rng.randrange(5) for _ in range(5)) for _ in range(analysis.DRAWS)]
        self.assertEqual(len(draws), 10000)
        # Same block-dependent scale must not acquire variation from unpaired sampling.
        result = analysis.paired([2., 4., 8., 16., 32.], [1., 2., 4., 8., 16.], draws)
        for endpoint in result["ratio_ci95"]:
            self.assertAlmostEqual(endpoint, 2.)
        self.assertEqual(analysis.expected_orders(), compare.orders("full"))


if __name__ == "__main__":
    unittest.main()
