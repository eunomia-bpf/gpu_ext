#!/usr/bin/env python3
"""Summarize enhanced gpu_ext prefetch callback and final-decision traces."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def number(value: str | None) -> int:
    try:
        return int(value or 0, 0)
    except ValueError:
        return 0


def histogram_percentile(histogram: Counter[int], fraction: float) -> float | str:
    count = sum(histogram.values())
    if not count:
        return "UNAVAILABLE"
    target = max(1, int(count * fraction + 0.999999))
    observed = 0
    for value, occurrences in sorted(histogram.items()):
        observed += occurrences
        if observed >= target:
            return float(value)
    raise AssertionError("histogram percentile target was not reached")


def analyze(root: Path) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for path in sorted((root / "results" / "stage3").glob("**/prefetch_decision_trace.csv")):
        run = path.parent
        manifest_path = run / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        callback_count = 0
        decision_count = 0
        matched_call_ids = 0
        decision_without_callback = 0
        pending_callbacks: set[int] = set()
        final_pages = Counter()
        final_pages_sum = 0
        candidate_pages_sum = 0
        policy_pages_sum = 0
        actions = {name: 0 for name in ("DEFAULT", "BYPASS", "ENTER_LOOP", "UNKNOWN")}
        equal_policy_final = 0
        equal_candidate_final = 0
        with path.open(newline="", errors="replace") as source:
            for row in csv.DictReader(source):
                event_type = row.get("event_type")
                call_id = number(row.get("call_id"))
                if event_type == "CALLBACK":
                    callback_count += 1
                    if call_id:
                        pending_callbacks.add(call_id)
                    continue
                if event_type != "DECISION":
                    continue
                decision_count += 1
                if call_id and call_id in pending_callbacks:
                    pending_callbacks.remove(call_id)
                    matched_call_ids += 1
                else:
                    decision_without_callback += 1
                action = row.get("action_name")
                actions[action if action in actions else "UNKNOWN"] += 1
                pages = number(row.get("final_pages"))
                final_pages[pages] += 1
                final_pages_sum += pages
                candidate_pages_sum += max(
                    0, number(row.get("max_candidate_outer")) - number(row.get("max_candidate_first")))
                policy_pages_sum += max(
                    0, number(row.get("policy_result_outer")) - number(row.get("policy_result_first")))
                equal_policy_final += (
                    number(row.get("policy_result_first")) == number(row.get("final_effective_first"))
                    and number(row.get("policy_result_outer")) == number(row.get("final_effective_outer")))
                equal_candidate_final += (
                    number(row.get("max_candidate_first")) == number(row.get("final_effective_first"))
                    and number(row.get("max_candidate_outer")) == number(row.get("final_effective_outer")))
        output.append({
            "experiment": manifest.get("experiment", "UNAVAILABLE"),
            "policy": manifest.get("policy", "UNAVAILABLE"),
            "ratio": manifest.get("ratio", "na"),
            "run_id": run.name,
            "evidence_class": "GPU_EXT_PREFETCH_DECISION_TRACE",
            "callback_count": callback_count,
            "decision_count": decision_count,
            "matched_call_ids": matched_call_ids,
            "callback_without_decision": len(pending_callbacks),
            "decision_without_callback": decision_without_callback,
            "default_count": actions["DEFAULT"],
            "bypass_count": actions["BYPASS"],
            "enter_loop_count": actions["ENTER_LOOP"],
            "unknown_action_count": actions["UNKNOWN"],
            "policy_final_equal_count": equal_policy_final,
            "candidate_final_equal_count": equal_candidate_final,
            "final_pages_mean": final_pages_sum / decision_count if decision_count else "UNAVAILABLE",
            "final_pages_median": histogram_percentile(final_pages, 0.5),
            "final_pages_p95": histogram_percentile(final_pages, 0.95),
            "final_pages_min": min(final_pages) if final_pages else "UNAVAILABLE",
            "final_pages_max": max(final_pages) if final_pages else "UNAVAILABLE",
            "candidate_pages_mean": candidate_pages_sum / decision_count if decision_count else "UNAVAILABLE",
            "policy_pages_mean": policy_pages_sum / decision_count if decision_count else "UNAVAILABLE",
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_dir.resolve()
    rows = analyze(root)
    fields = [
        "experiment", "policy", "ratio", "run_id", "evidence_class", "callback_count",
        "decision_count", "matched_call_ids", "callback_without_decision",
        "decision_without_callback", "default_count", "bypass_count", "enter_loop_count",
        "unknown_action_count", "policy_final_equal_count", "candidate_final_equal_count",
        "final_pages_mean", "final_pages_median", "final_pages_p95", "final_pages_min",
        "final_pages_max", "candidate_pages_mean", "policy_pages_mean",
    ]
    target = root / "results" / "stage3_trace_summary.csv"
    with target.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {target} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
