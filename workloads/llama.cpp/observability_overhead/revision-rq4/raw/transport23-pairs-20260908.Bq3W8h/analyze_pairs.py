#!/usr/bin/env python3
"""Describe every recorded pair; never run, reject or retry an experiment."""
import json
import random
import statistics
from pathlib import Path


def describe(values):
    if not values:
        return None
    return {"n": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "min": min(values),
            "max": max(values)}


def main():
    directory = Path(__file__).resolve().parent
    rows = json.loads((directory / "cells.json").read_text())
    arms = {}
    for name in ("baseline", "transport2", "transport3"):
        arm = [r for r in rows if r["arm"] == name]
        arms[name] = {
            "recorded_cells": len(arm),
            "throughput_tok_s": describe([r["throughput_tok_s"] for r in arm
                                           if r.get("throughput_tok_s") is not None]),
            "benchmark_zero_exits": sum(r.get("returncode") == 0 for r in arm),
            "probe_teardown_errors": sum(bool(r.get("probe_teardown_error")) for r in arm),
        }
    indexed = {(r["block"], r["arm"]): r for r in rows}
    pairs = []
    for block in sorted({r["block"] for r in rows}):
        baseline, off, on = [indexed.get((block, name), {}).get("throughput_tok_s")
                             for name in ("baseline", "transport2", "transport3")]
        if off is None or on is None:
            continue
        pairs.append({"block": block, "transport2_tok_s": off, "transport3_tok_s": on,
                      "transport3_over_transport2": on / off,
                      "throughput_change_pct": 100 * (on / off - 1),
                      "prefill_time_change_pct": 100 * (off / on - 1),
                      "off_loss_vs_earlier_baseline_pct": None if baseline is None else 100 * (1 - off / baseline),
                      "on_loss_vs_earlier_baseline_pct": None if baseline is None else 100 * (1 - on / baseline)})
    ratios = [p["transport3_over_transport2"] for p in pairs]
    rng = random.Random(1797)
    boot = sorted(statistics.median(rng.choices(ratios, k=len(ratios)))
                  for _ in range(10000)) if ratios else []
    result = {"scope": "Original-object transport 3 versus transport 2; not a new NVBit H2H",
              "baseline_order": "No baseline replay; both transport arms measured fresh in five rotating pairs",
              "arms": arms, "pairs": pairs,
              "paired_ratio": describe(ratios),
              "paired_ratio_median_bootstrap_95pct": [boot[250], boot[9749]] if boot else None,
              "bootstrap_replicates": 10000, "bootstrap_seed": 1797,
              "all_ten_attached_throughputs_recorded": len(pairs) == 5,
              "measurement_errors_retained": True}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
