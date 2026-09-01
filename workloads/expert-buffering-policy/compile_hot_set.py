#!/usr/bin/env python3
"""Compile exact-model expert route observations into a ranked hot set."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

WEIGHT_PATTERN = re.compile(
    r"^blk\.(?P<layer>\d+)\.ffn_(?P<kind>gate|down|up)_exps\.weight$"
)
EXPECTED_KINDS = frozenset({"gate", "down", "up"})


@dataclass(frozen=True)
class WeightLayout:
    layer: int
    kind: str
    n_experts: int


@dataclass(frozen=True)
class HotSet:
    graphs: int
    route_events: int
    selections: tuple[tuple[int, tuple[int, ...]], ...]
    counts: tuple[tuple[int, tuple[int, ...]], ...]


def load_events(paths: list[Path]) -> list[dict]:
    events: list[dict] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON in {path} line {line_number}: {error}") from error
                if event.get("event") in {"layout", "route"}:
                    events.append(event)
    if not events:
        raise ValueError("no layout or route events")
    return events


def compile_hot_set(
    events: list[dict], *, expected_layers: int, expected_experts: int, top_k: int
) -> HotSet:
    layouts: dict[tuple[int, int], WeightLayout] = {}
    routes: list[dict] = []

    for event in events:
        if event.get("event") == "layout":
            match = WEIGHT_PATTERN.fullmatch(str(event.get("name", "")))
            if not match:
                continue
            tgid = int(event["tgid"])
            base = int(event["base"])
            n_experts = int(event["n_experts"])
            layout = WeightLayout(
                layer=int(match.group("layer")),
                kind=match.group("kind"),
                n_experts=n_experts,
            )
            key = (tgid, base)
            previous = layouts.get(key)
            if previous is not None and previous != layout:
                raise ValueError(f"conflicting source layout for TGID/base {key}")
            layouts[key] = layout
        elif event.get("event") == "route":
            routes.append(event)

    if not layouts:
        raise ValueError("no source expert weight layouts")
    if not routes:
        raise ValueError("no route events")

    observed_layers = {layout.layer for layout in layouts.values()}
    required_layers = set(range(expected_layers))
    if observed_layers != required_layers:
        raise ValueError(
            f"expected source layers 0..{expected_layers - 1}, found {sorted(observed_layers)}"
        )
    for layout in layouts.values():
        if layout.n_experts != expected_experts:
            raise ValueError(
                f"layer {layout.layer} {layout.kind} has {layout.n_experts} experts, "
                f"expected {expected_experts}"
            )

    per_operation: dict[tuple[int, int, int, str], set[int]] = defaultdict(set)
    graphs: set[tuple[int, int]] = set()
    for event in routes:
        tgid = int(event["tgid"])
        graph = int(event["graph"])
        base = int(event["tensor_base"])
        expert = int(event["expert_id"])
        if graph <= 0:
            raise ValueError("route event has no positive graph ordinal")
        layout = layouts.get((tgid, base))
        if layout is None:
            raise ValueError(f"route base {base} for TGID {tgid} has no source layout")
        if expert < 0 or expert >= layout.n_experts:
            raise ValueError(f"route expert {expert} is out of range")
        graphs.add((tgid, graph))
        per_operation[(tgid, graph, layout.layer, layout.kind)].add(expert)

    selection_counts = [defaultdict(int) for _ in range(expected_layers)]
    for tgid, graph in sorted(graphs):
        for layer in range(expected_layers):
            by_kind = {
                kind: per_operation.get((tgid, graph, layer, kind), set())
                for kind in EXPECTED_KINDS
            }
            present = {kind for kind, selected in by_kind.items() if selected}
            if present != EXPECTED_KINDS:
                raise ValueError(
                    f"TGID {tgid} graph {graph} layer {layer} has route kinds {sorted(present)}"
                )
            selected_sets = list(by_kind.values())
            if not all(selected == selected_sets[0] for selected in selected_sets[1:]):
                raise ValueError(
                    f"TGID {tgid} graph {graph} layer {layer} disagrees across weight kinds"
                )
            for expert in selected_sets[0]:
                selection_counts[layer][expert] += 1

    selections: list[tuple[int, tuple[int, ...]]] = []
    dense_counts: list[tuple[int, tuple[int, ...]]] = []
    for layer, sparse in enumerate(selection_counts):
        if len(sparse) < top_k:
            raise ValueError(
                f"layer {layer} observed only {len(sparse)} experts, fewer than top-k {top_k}"
            )
        ranking = sorted(sparse, key=lambda expert: (-sparse[expert], expert))
        selections.append((layer, tuple(ranking[:top_k])))
        dense_counts.append(
            (layer, tuple(sparse.get(expert, 0) for expert in range(expected_experts)))
        )

    return HotSet(
        graphs=len(graphs),
        route_events=len(routes),
        selections=tuple(selections),
        counts=tuple(dense_counts),
    )


def write_hot_set(path: Path, hot_set: HotSet) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    directory = path.parent if path.parent != Path("") else Path(".")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(
                f"# graphs {hot_set.graphs} route_events {hot_set.route_events}\n"
            )
            for layer, experts in hot_set.selections:
                for expert in experts:
                    output.write(f"{layer} {expert}\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-layers", type=int, default=36)
    parser.add_argument("--expected-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.expected_layers <= 0 or args.expected_experts <= 1:
        raise ValueError("expected dimensions must be positive")
    if args.top_k <= 0 or args.top_k > args.expected_experts:
        raise ValueError("top-k is outside the expert range")

    hot_set = compile_hot_set(
        load_events(args.input),
        expected_layers=args.expected_layers,
        expected_experts=args.expected_experts,
        top_k=args.top_k,
    )
    write_hot_set(args.output, hot_set)
    summary = {
        "event": "hot_set_compiled",
        "graphs": hot_set.graphs,
        "route_events": hot_set.route_events,
        "layers": len(hot_set.selections),
        "top_k": args.top_k,
        "output": str(args.output),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
