"""CPU-only validation of the real inference worker's input/token protocol."""
import copy
import json
from pathlib import Path
import unittest
from tempfile import TemporaryDirectory

import inference


class InferenceProtocolTests(unittest.TestCase):
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
