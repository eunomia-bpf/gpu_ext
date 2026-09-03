"""Read-only analysis of exactly five complete Section VI three-arm blocks.

Reaudits the actual preflight and all timed cells; emits JSON on stdout only.
Run after the timing window, using the FineMoE Python environment for NumPy.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import sys

import correctness as gate

# Reuse the existing whole-block bootstrap arithmetic, not its four-arm/history
# campaign parser. Importing this analysis helper does not import torch or CUDA.
sys.path.insert(0, str(gate.FINE))
spec = importlib.util.spec_from_file_location("finemoe_paired_statistics", gate.FINE / "analyze_results.py")
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)


def analyze(directory):
    directory = directory.resolve()
    manifest = gate.read_json(directory / "campaign.json")
    require, read = gate.require, gate.read_json
    require(manifest["mode"] == "full" and manifest["complete"] is True and
            manifest["valid_blocks"] == 5 and manifest["valid_cells"] == 15 and
            manifest["arms"] == list(gate.ARMS) and manifest["seed"] == gate.SEED and
            manifest["orders"] == gate.orders() and manifest["numerical_audits"] == {} and
            not list(directory.rglob("*failure*.json")), "not a complete five-block timed campaign")
    paths = [directory / f"block-{block:02d}" / arm
             for block, order in enumerate(manifest["orders"]) for arm in order]
    require({path for path in directory.iterdir() if path.is_dir()} == {path.parent for path in paths} and
            {path for block in {path.parent for path in paths} for path in block.iterdir() if path.is_dir()} == set(paths),
            "unexpected/missing blocks or arm attempts")
    for name in ("launch.json", "result.json", "worker-result.json"):
        require({path.parent for path in directory.rglob(name)} == set(paths), "unexpected/missing raw cells")
    golden_dir = Path(manifest["golden"])
    gate.common.validate_reference(golden_dir, "golden")
    require(manifest["data"] == read(gate.FINE / "dataset-mtbench-v1.json") and
            manifest["runtime"] == gate.inventory(Path(manifest["source"])) and
            manifest["model_files"] == gate.common.model_inventory() and
            manifest["reference_files"] == gate.reference_inventory(golden_dir),
            "runtime/model/original reference changed")
    golden = read(golden_dir / "golden.json")
    previous = gate.validate_preflight(Path(manifest["preflight"]), manifest, golden)
    cells, workers = [], {}
    for index, path in enumerate(paths):
        worker, audited = gate.audit_saved_cell(path, manifest, golden, False)
        require(previous <= worker["application_native_begin_ns"], "real randomized cell intervals overlap")
        previous = worker["native_drained_ns"]
        begin, end = worker["application_native_begin_ns"], worker["application_native_end_ns"]
        requests = worker["requests"]
        metrics = {
            "tokens_per_second": sum(len(row["generated_ids"]) for row in requests) * 1e9 / (end - begin),
            "tokens_per_second_including_drain": 128e9 / (previous - begin),
            "median_ttft_ms": statistics.median((r["token_ready_ns"][0] - r["begin_ns"]) / 1e6 for r in requests),
            "median_tpot_ms": statistics.median((r["token_ready_ns"][-1] - r["token_ready_ns"][0]) / 15e6
                                               for r in requests),
            "drain_seconds": (worker["drain_end_ns"] - worker["drain_begin_ns"]) / 1e9,
            "cpu_seconds": worker["cpu_seconds"],
        }
        require(all(math.isfinite(value) and value >= 0 and
                    math.isclose(value, audited[key], rel_tol=1e-12, abs_tol=1e-12)
                    for key, value in metrics.items()), "independent raw timing arithmetic differs")
        cells.append({"block": index // 3, "arm": path.name, "path": str(path), "metrics": metrics,
                      "eb_delta": audited["eb_delta"], "executor_counters": worker["after"]["counters"]})
        workers[path.name] = worker
        if index % 3 == 2:
            gate.matching_decisions(workers)
            workers = {}
    rng = random.Random(gate.SEED)
    draws = [tuple(rng.randrange(5) for _ in range(5)) for _ in range(10000)]
    by_arm = {arm: [cell for cell in cells if cell["arm"] == arm] for arm in gate.ARMS}
    keys = tuple(cells[0]["metrics"])
    medians = {arm: {key: statistics.median(cell["metrics"][key] for cell in rows) for key in keys}
               for arm, rows in by_arm.items()}
    effects = {f"{candidate}_over_{reference}": {
        key: stats.paired([cell["metrics"][key] for cell in by_arm[candidate]],
                          [cell["metrics"][key] for cell in by_arm[reference]], draws) for key in keys}
        for candidate, reference in (("native", "fifo"), ("bpf", "fifo"), ("bpf", "native"))}
    return {"complete": True, "campaign": str(directory), "valid_blocks": 5, "valid_cells": 15,
            "capacity": manifest["capacity"], "preflight": manifest["preflight"], "medians": medians,
            "paired_effects": effects, "cells": cells,
            "bootstrap": {"seed": gate.SEED, "draws": 10000, "unit": "complete paired block",
                          "ci": "95% percentile: mean absolute difference / geometric mean ratio"},
            "scope": "Section VI policy port on Qwen; not original-paper end-to-end reproduction"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    print(json.dumps(analyze(parser.parse_args().campaign), indent=2, allow_nan=False))
