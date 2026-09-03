"""CPU-only validation of the real inference worker's input/token protocol."""
import copy
import json
from pathlib import Path
import unittest

import inference


class InferenceProtocolTests(unittest.TestCase):
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
