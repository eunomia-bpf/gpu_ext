"""CPU-only real-selector/downstream-method integration; no inference claim."""
from types import SimpleNamespace
import unittest

import torch

from policy_runtime import FineMoePolicy, python_oracle
from test_dynamic_set import matcher, prefetcher


class RuntimePolicyTests(unittest.TestCase):
    def execute(self, arm, probabilities, delta_score=.5, top_k=1):
        policy = FineMoePolicy(arm, shadow=True, capture=True)
        try:
            match = matcher(layers=2, experts=4, top_k=top_k)
            consumer = prefetcher(layers=2, experts=4)
            engine = SimpleNamespace(expert_map_matcher=match, expert_prefetcher=consumer)
            policy.install(engine)
            priorities, values = match.process_expert_map(
                0, 2, torch.tensor(delta_score), torch.tensor([probabilities] * 2))
            consumer.prefetch_experts(priorities, values)
            return policy.snapshot(), consumer.archer_engine.target.enqueued
        finally:
            policy.close()

    def test_four_arm_downstream_admission(self):
        counts = {}
        for arm in ("demand-only", "all-positive", "finemoe-c", "finemoe-bpf"):
            snap, calls = self.execute(arm, [.5, .25, .125, .125])
            counts[arm] = len(calls)
            self.assertEqual(snap["stats"].get("engine_enqueue_calls", 0), len(calls))
            if arm in ("finemoe-c", "finemoe-bpf"):
                self.assertEqual(snap["stats"]["oracle_checks"], 2)
            self.assertEqual(snap["stats"]["jit_calls"], 2 if arm == "finemoe-bpf" else 0)
        self.assertEqual(counts, {"demand-only": 0, "all-positive": 8,
                                  "finemoe-c": 2, "finemoe-bpf": 2})

    def test_zero_probability_selected_to_meet_k_is_not_dropped(self):
        for arm in ("finemoe-c", "finemoe-bpf"):
            snap, calls = self.execute(arm, [1., 0., 0., 0.], delta_score=1., top_k=4)
            self.assertEqual(len(calls), 8)
            self.assertEqual(snap["stats"]["selected_candidates"], 8)
            masks = next(e["masks"] for e in snap["events"] if e["event"] == "selector")
            self.assertEqual(masks, [15, 15])

    def test_binary64_prefix_uses_independent_oracle(self):
        values = torch.tensor([.4, .3, .2, .1], dtype=torch.float32).tolist()
        delta = torch.tensor(.8, dtype=torch.float32).item()
        expected = python_oracle(values, delta, 1)[0]
        self.assertEqual(expected, 7)
        for arm in ("finemoe-c", "finemoe-bpf"):
            policy = FineMoePolicy(arm, shadow=True)
            try:
                self.assertEqual(policy.select(values, delta, 1), expected)
            finally:
                policy.close()

    def test_invalid_input_fails_without_native_fallback(self):
        policy = FineMoePolicy("finemoe-bpf", shadow=True)
        try:
            with self.assertRaises(ValueError):
                policy.select([float("nan"), .5], .5, 1)
            with self.assertRaises(ValueError):
                policy.select([.5, .5], float("inf"), 1)
            self.assertEqual(policy.snapshot()["stats"]["jit_calls"], 0)
        finally:
            policy.close()


if __name__ == "__main__":
    unittest.main()
