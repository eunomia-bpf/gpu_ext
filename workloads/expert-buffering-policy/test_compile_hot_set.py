#!/usr/bin/env python3

import unittest

from compile_hot_set import compile_hot_set


def layout(layer: int, kind: str, base: int, *, tgid: int = 17) -> dict:
    return {
        "event": "layout",
        "tgid": tgid,
        "name": f"blk.{layer}.ffn_{kind}_exps.weight",
        "base": base,
        "n_experts": 8,
    }


def route(base: int, expert: int, *, graph: int = 1, tgid: int = 17) -> dict:
    return {
        "event": "route",
        "tgid": tgid,
        "graph": graph,
        "tensor_base": base,
        "expert_id": expert,
    }


class CompileHotSetTests(unittest.TestCase):
    def complete_events(self) -> list[dict]:
        events: list[dict] = []
        for layer_number in range(2):
            bases = {
                "gate": 1000 + layer_number * 100,
                "down": 1010 + layer_number * 100,
                "up": 1020 + layer_number * 100,
            }
            for kind, base in bases.items():
                events.append(layout(layer_number, kind, base))
                for expert in (1, 3, 5):
                    events.append(route(base, expert))
        return events

    def test_triplicate_weight_routes_count_once_per_graph(self) -> None:
        compiled = compile_hot_set(
            self.complete_events(), expected_layers=2, expected_experts=8, top_k=2
        )
        self.assertEqual(compiled.graphs, 1)
        self.assertEqual(compiled.route_events, 18)
        self.assertEqual(compiled.selections[0], (0, (1, 3)))
        self.assertEqual(compiled.counts[0][1][1], 1)

    def test_ties_use_ascending_expert_id(self) -> None:
        compiled = compile_hot_set(
            self.complete_events(), expected_layers=2, expected_experts=8, top_k=3
        )
        self.assertEqual(compiled.selections[1], (1, (1, 3, 5)))

    def test_missing_weight_kind_is_rejected(self) -> None:
        events = self.complete_events()
        events = [
            event
            for event in events
            if not (
                event.get("event") == "route"
                and event.get("tensor_base") == 1020
            )
        ]
        with self.assertRaisesRegex(ValueError, "route kinds"):
            compile_hot_set(events, expected_layers=2, expected_experts=8, top_k=2)

    def test_disagreement_across_weight_kinds_is_rejected(self) -> None:
        events = self.complete_events()
        events.append(route(1000, 7))
        with self.assertRaisesRegex(ValueError, "disagrees"):
            compile_hot_set(events, expected_layers=2, expected_experts=8, top_k=2)

    def test_unknown_route_base_is_rejected(self) -> None:
        events = self.complete_events()
        events.append(route(9999, 1))
        with self.assertRaisesRegex(ValueError, "no source layout"):
            compile_hot_set(events, expected_layers=2, expected_experts=8, top_k=2)


if __name__ == "__main__":
    unittest.main()
