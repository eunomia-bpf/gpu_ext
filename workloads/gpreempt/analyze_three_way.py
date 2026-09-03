"""Offline paired-block estimates from audited original request samples.

Never runs CUDA, changes drivers, or infers drops from nominal request counts.
"""
import argparse
import json
import math
from pathlib import Path
import random
import statistics

import run_three_way as run


def estimate_ratios(values, draws=10000):
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("positive finite paired ratios required")
    logs = [math.log(x) for x in values]
    result = {"geometric_ratio": math.exp(statistics.mean(logs)), "block_ratios": values,
              "paired_block_bootstrap_ci95": None}
    if len(values) > 1:
        rng = random.Random(20260903)
        samples = sorted(math.exp(statistics.mean(logs[rng.randrange(len(logs))] for _ in logs))
                         for _ in range(draws))
        result["paired_block_bootstrap_ci95"] = [samples[int(.025 * draws)], samples[int(.975 * draws)]]
    return result


def summarize_blocks(blocks):
    ids = [item["block"] for item in blocks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate block; do not silently select one")
    for block in blocks:
        if set(block["cells"]) != set(run.ARMS):
            raise ValueError("paired estimate requires all three cells")
    result = {"valid_paired_blocks": len(blocks), "required_blocks": 5,
              "complete": sorted(ids) == list(range(5)), "arms": {}, "paired": {},
              "latency_definition": "nearest-rank p99 of the original six recorded service stages, not arrival-to-completion",
              "ci_method": "percentile bootstrap of paired blocks, geometric ratios, 10000 draws",
              "equivalence_claimed": False, "arrival_or_drop_counts_inferred": False}
    if not blocks:
        return result
    for arm in run.ARMS:
        result["arms"][arm] = {role: {
            "median_p99_latency_us": statistics.median(b["cells"][arm][role]["p99_latency_us"] for b in blocks),
            "median_throughput_rps": statistics.median(b["cells"][arm][role]["throughput_rps"] for b in blocks),
            "min_completed_requests": min(b["cells"][arm][role]["completed_requests"] for b in blocks),
            "max_completed_requests": max(b["cells"][arm][role]["completed_requests"] for b in blocks),
        } for role in run.TASKS}
    for numerator, denominator in (("original_gpreempt", "native"), ("bpf_gpreempt", "native"),
                                   ("bpf_gpreempt", "original_gpreempt")):
        for role, metric, direction in ((run.TASKS[0], "p99_latency_us", "lower is better"),
                                         (run.TASKS[1], "throughput_rps", "higher is better")):
            ratios = [b["cells"][numerator][role][metric] / b["cells"][denominator][role][metric] for b in blocks]
            result["paired"][f"{numerator}/{denominator}:{role}:{metric}"] = {
                **estimate_ratios(ratios), "direction": direction}
    return result


def audit_cell(directory, arm, plan, campaign):
    result = json.loads((directory / "result.json").read_text())
    if result.get("status") != "passed" or result.get("error") or result.get("cleanup_errors") or result["returncode"] != 0:
        raise ValueError("cell did not finish cleanly")
    if result["arm"] != arm or result["command"] != run.client_command(arm, campaign / "config-A.json", plan["flag_transport"]):
        raise ValueError("recorded arm or actual command changed")
    expected_transport = "not_used" if arm == "native" else plan["flag_transport"]
    if result["flag_transport"] != expected_transport:
        raise ValueError("mixed transports cannot form a pair")
    pin = Path(result["environment"].get("GPREEMPT_BPF_MAPS", "/unused"))
    if result["environment"] != run.environment(arm, pin, Path(plan["gdrcopy_directory"])):
        raise ValueError("actual runtime environment changed")
    client = (directory / "client.log").read_text()
    parsed = run.parse_report(client)
    if parsed["report"] != json.loads((directory / "request-report.json").read_text()) or parsed["metrics"] != result["metrics"]:
        raise ValueError("raw request samples disagree with reported metrics")
    loader = (directory / "loader.log").read_text() if arm == "bpf_gpreempt" else ""
    if run.check_engagement(arm, client, loader, plan["flag_transport"]) != result["engagement"]:
        raise ValueError("raw policy engagement differs")
    run.safety.validate_pre_server_safety(result["safety_before"])
    run.safety.validate_post_server_safety(result["safety_before"], result["safety_after"])
    if run.safety.validate_gpu_telemetry(directory / "gpu-telemetry.csv", allow_fixed_power_cap=True) != result["telemetry"]:
        raise ValueError("raw GPU telemetry differs")
    return parsed["metrics"]


def analyze(campaign):
    plan = json.loads((campaign / "plan.json").read_text())
    run.validate_config(plan["config"])
    if (len(plan["orders"]) != 5 or plan["orders"] != run.orders(5, plan["seed"]) or
            json.loads((campaign / "config-A.json").read_text()) != plan["config"]):
        raise ValueError("not the fixed five-block original config-A comparison")
    valid, rejected, incomplete = [], [], []
    for number, order in enumerate(plan["orders"]):
        block = {"block": number, "cells": {}}
        for arm in order:
            directory = campaign / f"block-{number:02d}" / arm
            if not (directory / "result.json").exists():
                incomplete.append({"block": number, "arm": arm})
                continue
            try:
                block["cells"][arm] = audit_cell(directory, arm, plan, campaign)
            except (OSError, ValueError, KeyError, TypeError, run.safety.GateError) as exc:
                rejected.append({"block": number, "arm": arm, "error": str(exc)})
        if len(block["cells"]) == 3:
            valid.append(block)
    return {**summarize_blocks(valid), "rejected_cells": rejected, "incomplete_cells": incomplete,
            "flag_transport": plan["flag_transport"], "comparison_variant": plan["comparison_variant"],
            "original_gdr_transport": plan["flag_transport"] == "gdr"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.campaign.resolve())
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        run.safety.atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))
