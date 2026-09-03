"""CPU-only validation of the real inference worker's input/token protocol."""
import copy
import json
from pathlib import Path
import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import inference


class InferenceProtocolTests(unittest.TestCase):
    def test_offload_loads_the_same_checkpoint_generation_config(self):
        data = json.loads((Path(__file__).parent / "dataset-mtbench-v1.json").read_text())
        from transformers import AutoConfig, GenerationConfig
        default = GenerationConfig.from_model_config(
            AutoConfig.from_pretrained(data["model"]["snapshot"], local_files_only=True))
        self.assertEqual(default.repetition_penalty, 1.)
        wrapped = SimpleNamespace(model=SimpleNamespace(generation_config=default))
        fake_module = SimpleNamespace(MoE=mock.Mock(return_value=wrapped))
        with mock.patch.dict("sys.modules", {"finemoe": fake_module}), \
                mock.patch.object(inference.importlib.util, "find_spec", return_value=None), \
                mock.patch.object(inference.GenerationConfig, "from_pretrained",
                                  wraps=GenerationConfig.from_pretrained) as load:
            result = inference.create_finemoe(data, Path("unused-test-cache"), True)
        load.assert_called_once_with(data["model"]["snapshot"], local_files_only=True)
        self.assertIs(result, wrapped)
        self.assertEqual(result.model.generation_config.repetition_penalty, 1.05)
        self.assertEqual(result.model.generation_config.eos_token_id, [151645, 151643])
        self.assertEqual(result.model.generation_config.top_p, .8)
        self.assertFalse(inference.torch.cuda.is_initialized())

    def test_recorded_decoding_fields_distinguish_checkpoint_from_overrides(self):
        config = SimpleNamespace(do_sample=True, repetition_penalty=1.05,
                                 eos_token_id=[151645, 151643], pad_token_id=151643,
                                 temperature=.7, top_k=20, top_p=.8,
                                 _private_identifier="must not be recorded")
        recorded = inference.decoding_configuration(config)
        self.assertEqual(recorded["checkpoint_fields"]["repetition_penalty"], 1.05)
        self.assertTrue(recorded["checkpoint_fields"]["do_sample"])
        self.assertEqual(recorded["explicit_overrides"],
                         {"do_sample": False, "min_new_tokens": 16, "max_new_tokens": 16,
                          "pad_token_id": 151643})
        self.assertNotIn("_private_identifier", recorded["checkpoint_fields"])

    def retained_fixture(self):
        record = {"question_id": 141, "input_ids": [1, 2], "generated_ids": list(range(16)),
                  "begin_ns": 1, "verified_ready_ns": 20, "token_ready_ns": list(range(2, 18))}
        golden = {"requests": {"141": {**record, "logits_file": "reference.npy"}}}
        logits = inference.np.arange(32, dtype=inference.np.float32).reshape(2, 1, 16)
        return record, golden, logits

    def test_history_token_failure_retains_expected_and_actual_before_raising(self):
        record, golden, _ = self.retained_fixture()
        record["generated_ids"] = [999] * 16
        with TemporaryDirectory(prefix="finemoe-retention-test-") as directory:
            path = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "token mismatch"):
                inference.retain_and_check_result(record, None, golden, path, 0., path)
            retained = json.loads((path / "question-141-result.json").read_text())
            self.assertEqual(retained["status"], "failed")
            self.assertEqual(retained["request"], record)
            self.assertEqual(retained["expected_generated_ids"], list(range(16)))
            self.assertEqual(retained["golden_absolute_tolerance"], 0.)
            self.assertEqual(list(path.glob("*.npy")), [])

    def test_preflight_logit_failure_keeps_actual_array_and_frozen_gate(self):
        record, golden, logits = self.retained_fixture()
        with TemporaryDirectory(prefix="finemoe-retention-test-") as directory:
            path = Path(directory)
            inference.np.save(path / "reference.npy", logits, allow_pickle=False)
            actual = logits + 1
            with self.assertRaisesRegex(RuntimeError, "max_abs_error=1.0 > frozen 0.0"):
                inference.retain_and_check_result(record, actual, golden, path, 0., path)
            retained = json.loads((path / "question-141-result.json").read_text())
            self.assertEqual(retained["status"], "failed")
            self.assertEqual(retained["request"]["logits_file"], "question-141-logits.npy")
            inference.np.testing.assert_array_equal(
                inference.np.load(path / retained["request"]["logits_file"], allow_pickle=False), actual)
            self.assertEqual(retained["golden_absolute_tolerance"], 0.)

    def test_preflight_pass_retains_recomputable_array_and_check(self):
        record, golden, logits = self.retained_fixture()
        with TemporaryDirectory(prefix="finemoe-retention-test-") as directory:
            path = Path(directory)
            inference.np.save(path / "reference.npy", logits, allow_pickle=False)
            check = inference.retain_and_check_result(record, logits, golden, path, 0., path)
            retained = json.loads((path / "question-141-result.json").read_text())
            self.assertEqual(retained["status"], "passed")
            self.assertEqual(retained["request"]["correctness"], check)
            self.assertEqual(check["max_abs_error"], 0.)
            inference.np.testing.assert_array_equal(
                inference.np.load(path / record["logits_file"], allow_pickle=False), logits)

    def test_formal_path_adds_no_files_or_record_fields(self):
        record, golden, _ = self.retained_fixture()
        before = copy.deepcopy(record)
        with mock.patch.object(inference, "atomic_write_json") as write, \
                mock.patch.object(inference.np, "save") as save:
            check = inference.retain_and_check_result(record, None, golden, Path("unused"), 0.)
        write.assert_not_called()
        save.assert_not_called()
        self.assertEqual(record, before)
        self.assertFalse(check["logits_checked"])

    def test_repeat_array_is_persisted_without_changing_values(self):
        # Storage-only fixture, not a model or performance experiment.
        array = inference.np.arange(32, dtype=inference.np.float32).reshape(2, 1, 16)
        with TemporaryDirectory(prefix="finemoe-repeat-test-") as directory:
            filename = inference.save_repeat_logits(Path(directory), 137, array)
            self.assertEqual(filename, "question-137-repeat-logits.npy")
            restored = inference.np.load(Path(directory) / filename, allow_pickle=False)
            self.assertEqual(restored.dtype, array.dtype)
            inference.np.testing.assert_array_equal(restored, array)

    def test_exact_full_model_and_64_8_1_cohort(self):
        data = json.loads((Path(__file__).parent / "dataset-mtbench-v1.json").read_text())
        inference.validate_data(data)
        for mutation in (lambda d: d["history"].pop(),
                         lambda d: d["evaluation"][0].update(input_ids=d["history"][0]["input_ids"]),
                         lambda d: d["model"].update(source_revision="wrong-model-revision")):
            bad = copy.deepcopy(data)
            mutation(bad)
            with self.assertRaises(ValueError):
                inference.validate_data(bad)

    def test_streamer_skips_prompt_and_records_actual_generated_ids(self):
        recorder = inference.TokenRecorder()
        recorder.put(inference.torch.tensor([[7, 8, 9]]))
        for token in range(16):
            recorder.put(inference.torch.tensor([token]))
        self.assertEqual(recorder.tokens, list(range(16)))
        self.assertEqual(len(recorder.ready_ns), 16)
        self.assertEqual(sorted(recorder.ready_ns), recorder.ready_ns)
        with self.assertRaises(RuntimeError):
            recorder.put(inference.torch.tensor([1, 2]))
        self.assertFalse(inference.torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
