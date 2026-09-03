import pathlib
import unittest

import numpy as np

from paper_policy import ActivationPolicy, EAMCollection, EPSILON, JitMatcher, JitRanker, native_rank


class PaperPolicyTests(unittest.TestCase):
    def test_fig5_probability_and_two_distinct_layer_decays(self):
        policy = ActivationPolicy(3, 2)
        policy.collection.insert(np.array([[4, 0], [0, 4], [0, 4]]), "prefill")
        policy.collection.insert(np.array([[0, 4], [4, 0], [4, 0]]), "decode")
        policy.begin_iteration(17, True)
        prediction = policy.observe(0, [4, 0])
        self.assertEqual(prediction.matched_entries, [0])
        self.assertEqual(prediction.prefetch_identities, [(1 << 32) | 1, (2 << 32) | 1])
        expected = np.array([[1 + EPSILON, EPSILON],
                             [EPSILON * 2 / 3, (1 + EPSILON) * 2 / 3],
                             [EPSILON / 3, (1 + EPSILON) / 3]])
        np.testing.assert_allclose(prediction.reuse_scores.reshape(3, 2), expected)

    def test_request_phase_boundary_and_iteration_reset(self):
        policy = ActivationPolicy(2, 2)
        policy.begin_iteration(4, True)
        policy.observe(0, [3, 1])
        policy.observe(1, [2, 2])
        policy.end_iteration()
        self.assertEqual(len(policy.collection.entries), 0)
        policy.begin_iteration(4, False)
        np.testing.assert_array_equal(policy.current_iteration, np.zeros((2, 2)))
        policy.observe(0, [1, 0])
        policy.observe(1, [0, 1])
        policy.end_iteration()
        policy.finish_request(4)
        self.assertEqual(policy.collection.phases, ["prefill", "decode"])
        np.testing.assert_array_equal(policy.collection.entries[0], [[3, 1], [2, 2]])
        np.testing.assert_array_equal(policy.collection.entries[1], [[1, 0], [0, 1]])
        self.assertEqual(policy.snapshot_stats()["completed_requests"], 1)

    def test_eamc_replaces_closest_to_preserve_diversity(self):
        collection = EAMCollection(2, 2, capacity=2)
        a = np.array([[3, 0], [3, 0]])
        b = np.array([[0, 3], [0, 3]])
        collection.insert(a, "prefill")
        collection.insert(b, "decode")
        self.assertEqual(collection.insert(a * 2, "decode"), 0)
        np.testing.assert_array_equal(collection.entries[1], b)
        self.assertEqual(collection.replacements, 1)

    def test_cached_rows_preserve_original_cosine_bits_and_ties(self):
        rng = np.random.default_rng(173)
        collection = EAMCollection(5, 7, capacity=11)
        for i in range(25):
            collection.insert(rng.integers(0, 512, (5, 7)), "decode")
            query = rng.integers(0, 512, (5, 7)).astype(np.float64).reshape(-1)
            rows = np.stack(collection.entries).reshape(len(collection.entries), -1)
            denominator = np.linalg.norm(rows, axis=1) * np.linalg.norm(query)
            expected = np.divide(rows @ query, denominator,
                                 out=np.zeros(len(rows)), where=denominator > 0)
            np.testing.assert_array_equal(collection.similarities(query.reshape(5, 7)), expected)

    def test_equal_cosine_matches_aggregate_then_row_normalize(self):
        collection = EAMCollection(2, 2)
        collection.insert(np.array([[2, 0], [2, 0]]), "prefill")
        collection.insert(np.array([[2, 0], [0, 2]]), "decode")
        prediction, matched = collection.predict(np.array([[1, 0], [0, 0]]))
        self.assertEqual(matched, [0, 1])
        np.testing.assert_array_equal(prediction, [[1, 0], [.5, .5]])

    def test_zero_history_has_neutral_reuse_but_no_fake_prefetch(self):
        policy = ActivationPolicy(2, 2)
        policy.begin_iteration(1, True)
        prediction = policy.observe(0, [1, 1])
        self.assertEqual(prediction.matched_entries, [])
        self.assertEqual(prediction.prefetch_identities, [])
        self.assertTrue(np.all(np.isfinite(prediction.reuse_scores)))

    def test_invalid_layer_and_incomplete_iteration_fail(self):
        policy = ActivationPolicy(2, 2)
        policy.begin_iteration(1, False)
        with self.assertRaises(RuntimeError):
            policy.observe(1, [1, 0])
        with self.assertRaises(RuntimeError):
            policy.end_iteration()
        self.assertIsNone(policy.current_iteration)
        policy.finish_request(1, aborted=True)
        self.assertEqual(len(policy.collection.entries), 0)

    def test_abort_during_forward_defers_trace_removal_and_never_trains(self):
        policy = ActivationPolicy(2, 2)
        policy.begin_iteration(3, True)
        policy.observe(0, [3, 1])
        policy.mark_aborted(3)
        policy.drain_aborted()
        self.assertIn(3, policy.requests)
        policy.observe(1, [1, 3])
        policy.end_iteration()
        self.assertNotIn(3, policy.requests)
        self.assertEqual(len(policy.collection.entries), 0)
        self.assertEqual(policy.stats["aborted_requests"], 1)

    def test_rank_ties_and_ieee_edges(self):
        values = np.array([0, -0., 2, np.nan, np.inf, 2, -np.inf, 1.])
        self.assertEqual(native_rank(values), [4, 2, 5, 7])

    def test_actual_jit_rank_same_bits_and_layer_major_order(self):
        output = pathlib.Path(__file__).resolve().parents[2] / "extension/.output"
        if not (output / "moe_expert_policy_rank.bin").exists():
            self.skipTest("build real BPF rank artifacts first")
        ranker = JitRanker(str(output / "libmoe_expert_policy.so"),
                           str(output / "moe_expert_policy_rank.bin"))
        values = np.array([0, 3., 3., np.inf, np.nan, np.nextafter(3., 4.)])
        self.assertEqual(ranker(list(range(6)), values), native_rank(values))
        policy = ActivationPolicy(3, 2, ranker=ranker, verify_rank=True)
        policy.collection.insert(np.array([[1, 0], [1, 1], [1, 1]]), "prefill")
        policy.begin_iteration(2, False)
        result = policy.observe(0, [1, 0])
        self.assertEqual(result.prefetch_identities, [1 << 32, (1 << 32) | 1,
                                                    2 << 32, (2 << 32) | 1])
        self.assertEqual(policy.stats["rank_calls"], 1)

    def test_actual_jit_eamc_matching_and_diversity_replacement(self):
        output = pathlib.Path(__file__).resolve().parents[2] / "extension/.output"
        if not (output / "moe_expert_policy_match.bin").exists():
            self.skipTest("build real BPF match artifacts first")
        matcher = JitMatcher(str(output / "libmoe_expert_policy.so"),
                             str(output / "moe_expert_policy_match.bin"))
        self.assertEqual(matcher(np.array([-np.inf, -np.inf, np.nan])), [0, 1])
        collection = EAMCollection(2, 2, capacity=2, matcher=matcher, verify=True)
        collection.insert(np.array([[2, 0], [2, 0]]), "prefill")
        collection.insert(np.array([[2, 0], [0, 2]]), "decode")
        prediction, matches = collection.predict(np.array([[1, 0], [0, 0]]))
        self.assertEqual(matches, [0, 1])
        np.testing.assert_array_equal(prediction, [[1, 0], [.5, .5]])
        self.assertEqual(collection.insert(np.array([[4, 0], [0, 4]]), "decode"), 1)
        self.assertEqual(collection.match_calls, 2)
        self.assertEqual(collection.match_mismatches, 0)


if __name__ == "__main__":
    unittest.main()
