import copy
import unittest

import run_paper_policy as runner


class CanaryAccountingTests(unittest.TestCase):
    def setUp(self):
        self.before = {"revision": {"engine_generated_tokens": 128},
                       "metrics": {"moe_tokens_generated_total": 128.0}}
        self.after = {"revision": {"engine_generated_tokens": 192},
                      "metrics": {"moe_tokens_generated_total": 192.0}}
        self.stream = {"frames": [{}] * 64 + [{"done": True}], "finish_reason": "length"}

    def test_complete_stream_uses_two_independent_actual_token_counters(self):
        result = runner.validate_stream_accounting(self.before, self.after, self.stream)
        self.assertEqual(result["engine_generated_tokens"], 64)
        self.assertEqual(result["metric_generated_tokens"], 64)

    def test_engine_or_metrics_mismatch_rejected(self):
        for source, key in (("revision", "engine_generated_tokens"),
                            ("metrics", "moe_tokens_generated_total")):
            with self.subTest(source=source):
                after = copy.deepcopy(self.after)
                after[source][key] -= 1
                with self.assertRaises(runner.base.GateError):
                    runner.validate_stream_accounting(self.before, after, self.stream)

    def test_missing_token_frame_and_early_finish_rejected(self):
        for stream in ({"frames": [{}] * 64, "finish_reason": "length"},
                       {"frames": [{}] * 65, "finish_reason": "stop"}):
            with self.assertRaises(runner.base.GateError):
                runner.validate_stream_accounting(self.before, self.after, stream)


if __name__ == "__main__":
    unittest.main()
