"""CPU regression of actual FineMoE methods, without importing/building CUDA ops.

The class AST is loaded verbatim from the pinned, patched author source. A capture
engine observes the *real* Python prefetcher's calls; this is not a GPU performance
experiment and cannot establish that a copy completed.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "deps" / "FineMoE-EuroSys26" / "finemoe"


def load_author_class(relative_path, name):
    path = SOURCE / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    namespace = {"torch": torch, "PretrainedConfig": object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


Matcher = load_author_class("runtime/model_offload.py", "ExpertMapMatcher")
Prefetcher = load_author_class("memory/expert_prefetcher.py", "ExpertPrefetcher")


def load_author_forward():
    path = SOURCE / "models/modeling_qwen/modeling_qwen2_moe.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "SyncQwen2MoeSparseMoeBlock")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    namespace = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["forward"]


class CaptureEngine:
    def __init__(self):
        self.candidates = []
        self.enqueued = []

    def replace_cache_candidates(self, tids):
        self.candidates.append(list(tids))

    def get_node_default_device(self, tids):
        return 0

    def enqueue_prefetch(self, tid, gpu_id, probability):
        self.enqueued.append((tid, gpu_id, probability))


def matcher(layers=2, experts=4, top_k=1):
    store = SimpleNamespace(num_layers=layers, num_experts=experts,
                            embed_dim=2, device=torch.device("cpu"))
    return Matcher(SimpleNamespace(top_k=top_k), store, None, 1)


def prefetcher(layers=2, experts=4):
    value = Prefetcher.__new__(Prefetcher)
    value.num_layers, value.num_experts = layers, experts
    value.archer_engine = CaptureEngine()
    value._tensor_id_grid = None
    value.set_expert_tensor_map({(layer, expert): layer * experts + expert
                                for layer in range(layers) for expert in range(experts)})
    return value


class DynamicSetTests(unittest.TestCase):
    def test_crossing_expert_is_included(self):
        result = matcher()._select_by_cumsum(torch.tensor([[.4, .3, .2, .1]]),
                                            torch.tensor(.8), 1)
        self.assertEqual(torch.nonzero(result[0]).flatten().tolist(), [0, 1, 2])
        self.assertGreaterEqual(result.sum().item(), .8)

    def test_exact_threshold_stops_at_equality(self):
        result = matcher()._select_by_cumsum(torch.tensor([[.5, .25, .125, .125]]),
                                            torch.tensor(.75), 1)
        self.assertEqual(torch.nonzero(result[0]).flatten().tolist(), [0, 1])

    def test_top_k_minimum_and_stable_ties(self):
        result = matcher()._select_by_cumsum(torch.full((1, 4), .25), torch.tensor(0.), 2)
        self.assertEqual(torch.nonzero(result[0]).flatten().tolist(), [0, 1])

    def test_delta_one_and_zero_probability_rows(self):
        probs = torch.tensor([[.5, .25, .125, .125], [0., 0., 0., 0.]])
        result = matcher()._select_by_cumsum(probs, torch.tensor([1., 1.]), 1)
        torch.testing.assert_close(result, probs)

    def test_target_band_does_not_reenable_excluded_experts(self):
        probs = torch.tensor([[.5, .25, .125, .125]] * 3)
        result, returned_probs = matcher(layers=3).process_expert_map(1, 2, torch.tensor(.5), probs)
        self.assertEqual(torch.nonzero(result).tolist(), [[1, 0]])
        self.assertEqual(returned_probs[1, 0].item(), .5)
        self.assertEqual(probs[0, 0].item(), .5)  # Original input is not zeroed.

    def test_real_downstream_prefetcher_enqueues_only_selected_set(self):
        probs = torch.tensor([[.5, .25, .125, .125]] * 2)
        result, returned_probs = matcher().process_expert_map(0, 2, torch.tensor(.5), probs)
        consumer = prefetcher()
        consumer.prefetch_experts(result, returned_probs)
        self.assertEqual(consumer.archer_engine.candidates, [[0, 4]])
        self.assertEqual(consumer.archer_engine.enqueued, [(0, 0, .5), (4, 0, .5)])

    def test_lower_confidence_expands_actual_queued_set(self):
        probs = torch.tensor([[.4, .3, .2, .1]] * 2)
        counts = []
        for confidence in (.9, .5, .1):
            priorities, probabilities = matcher().process_expert_map(
                0, 2, torch.tensor(confidence), probs)
            consumer = prefetcher()
            consumer.prefetch_experts(priorities, probabilities)
            counts.append(len(consumer.archer_engine.enqueued))
        self.assertEqual(counts, [2, 4, 6])

    def test_empty_rows_do_not_enqueue(self):
        priorities, probabilities = matcher().process_expert_map(
            0, 2, torch.tensor(.5), torch.zeros((2, 4)))
        consumer = prefetcher()
        consumer.prefetch_experts(priorities, probabilities)
        self.assertEqual(consumer.archer_engine.candidates, [])
        self.assertEqual(consumer.archer_engine.enqueued, [])

    def test_qwen_k_four_is_not_artificial_top_one(self):
        probs = torch.full((24, 60), 1. / 60.)
        priorities, probabilities = matcher(24, 60, 4).process_expert_map(
            0, 6, torch.tensor(1.), probs)
        consumer = prefetcher(24, 60)
        consumer.prefetch_experts(priorities, probabilities)
        self.assertEqual(len(consumer.archer_engine.enqueued), 6 * 4)
        self.assertEqual({tid % 60 for tid, _, _ in consumer.archer_engine.enqueued}, {0, 1, 2, 3})


class CommonRuntimeTests(unittest.TestCase):
    def run_forward(self, seq_ids):
        calls = []
        observed = []
        trace = {"retained-old": SimpleNamespace(iters=[{"probs": torch.tensor([[.9, .1]])}])}
        for seq_id in seq_ids:
            trace[seq_id] = SimpleNamespace(iters=[{"probs": torch.zeros((1, 2))}])

        def update_entry(seq_id, expert_probs, **unused):
            trace[seq_id].iters[-1]["probs"][0] = expert_probs[0]

        def expert(index):
            def invoke(states):
                calls.append((index, states.shape[0]))
                return states * (index + 1)
            return invoke

        model = SimpleNamespace(
            num_experts=2, top_k=1, norm_topk_prob=True, layer_id=0,
            device=torch.device("cpu"), seq_id_list=seq_ids,
            gate=lambda states: states,
            experts=[expert(0), expert(1)],
            shared_expert=lambda states: torch.zeros_like(states),
            shared_expert_gate=lambda states: torch.zeros((states.shape[0], 1)),
            expert_tracer=SimpleNamespace(trace=trace, update_entry=update_entry),
            expert_map_matcher=SimpleNamespace(
                traj_prefetch=lambda seq_id, values: observed.append((seq_id, values.clone()))),
        )
        values = torch.tensor([[[1., 2.]]]).expand(len(seq_ids), -1, -1).clone()
        output, _ = load_author_forward()(model, values)
        return output, values, calls, observed

    def test_retained_history_does_not_replace_current_trajectory(self):
        output, inputs, calls, observed = self.run_forward(["current"])
        self.assertEqual([entry[0] for entry in observed], ["current"])
        torch.testing.assert_close(observed[0][1][0, 0], torch.softmax(inputs[0, 0], 0))
        torch.testing.assert_close(output, inputs * 2)  # Expert 1 is the sole route.

    def test_empty_experts_do_not_execute_offload_hooks(self):
        output, inputs, calls, _ = self.run_forward(["current"])
        self.assertEqual(calls, [(1, 1)])
        torch.testing.assert_close(output, inputs * 2)

    def test_active_batch_order_not_trace_archive_order(self):
        _, _, calls, observed = self.run_forward(["current-b", "current-a"])
        self.assertEqual([entry[0] for entry in observed], ["current-b", "current-a"])
        self.assertEqual(calls, [(1, 2)])


if __name__ == "__main__":
    unittest.main()
