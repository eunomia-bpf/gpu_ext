"""CPU-only serving integration regressions; all model execution is mocked."""
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from paper_policy import ActivationPolicy
from paper_server import PaperEngine, ContinuousBatchingEngine


class PaperServerTests(unittest.TestCase):
    def make_engine(self):
        runtime = object.__new__(PaperEngine)
        runtime.revision_activation = ActivationPolicy(2, 2)
        runtime._revision_abort_lock = threading.Lock()
        runtime._revision_aborts = set()
        runtime._request_to_seq_ids = {"request": [3]}
        return runtime

    def test_api_abort_defers_native_removal_until_execution_owner_step(self):
        runtime = self.make_engine()
        policy = runtime.revision_activation
        policy.begin_iteration(3, True)
        policy.observe(0, [3, 1])
        with mock.patch.object(ContinuousBatchingEngine, "abort_request") as remove:
            runtime.abort_request("request")
            remove.assert_not_called()
            self.assertIn(3, policy.requests)
            policy.observe(1, [1, 3])
            policy.end_iteration()
            self.assertNotIn(3, policy.requests)
            self.assertEqual(policy.collection.entries, [])
            with mock.patch.object(ContinuousBatchingEngine, "step", return_value=[]):
                self.assertEqual(runtime.step(), [])
            remove.assert_called_once_with("request")

    def test_actual_execute_seam_counts_one_complete_iteration(self):
        runtime = self.make_engine()
        batch = SimpleNamespace(seq_ids=[3], is_prefill=[True])
        marker = object()

        def forward(_batch):
            runtime.revision_activation.observe(0, [3, 1])
            runtime.revision_activation.observe(1, [2, 2])
            return marker

        with mock.patch.object(ContinuousBatchingEngine, "_execute_batch", side_effect=forward):
            self.assertIs(runtime._execute_batch(batch), marker)
        self.assertEqual(runtime.revision_activation.stats["iterations"], 1)
        self.assertIsNone(runtime.revision_activation.current_iteration)

    def test_execution_error_abandons_partial_iteration_without_training(self):
        runtime = self.make_engine()
        batch = SimpleNamespace(seq_ids=[3], is_prefill=[True])
        with mock.patch.object(ContinuousBatchingEngine, "_execute_batch", side_effect=RuntimeError("model error")):
            with self.assertRaisesRegex(RuntimeError, "model error"):
                runtime._execute_batch(batch)
        self.assertIsNone(runtime.revision_activation.current_iteration)
        self.assertEqual(runtime.revision_activation.stats["iterations"], 0)
        self.assertEqual(runtime.revision_activation.collection.entries, [])

    def test_multi_sequence_execution_is_explicitly_rejected(self):
        runtime = self.make_engine()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            runtime._execute_batch(SimpleNamespace(seq_ids=[1, 2], is_prefill=[True, True]))


if __name__ == "__main__":
    unittest.main()
