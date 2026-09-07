#!/usr/bin/env python3
"""Offline stage decomposition of the 2026-09-07 five-block write-workers campaign.

Reads the 30 already-published result.json files under one campaign directory
and prints a JSON stage attribution summary on stdout.  Read-only: no cells
are launched, rerun, deleted, or rewritten, and no device access happens.

Stage definitions (monotonic seconds relative to the cell t0):
  reads:  pre_dispatch   = offer_s - scheduled_offer_s
          dispatch_gap   = submitted_s - offer_s   (zero by stamping)
          post_dispatch  = completed_s - submitted_s
  writes: pre_dispatch   = offer_s - scheduled_offer_s
          release_admit  = submitted_s - offer_s   (deferral wait, release,
                          and save-coroutine dispatch to the timing stamp)
          save_exec      = completed_s - submitted_s
For each request the three components add to the scheduled total; quantiles
of different requests never add and are never summed here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import median


ARM_NAMES = {
    ("gds_fifo", 0): "FIFO default",
    ("gds_native", 0): "Native default",
    ("gds_bpf", 0): "BPF default",
    ("gds_fifo", 4): "FIFO 4",
    ("gds_native", 4): "Native 4",
    ("gds_bpf", 4): "BPF 4",
}

ARM_ORDER = [
    "FIFO default",
    "Native default",
    "BPF default",
    "FIFO 4",
    "Native 4",
    "BPF 4",
]

PAIRS = [
    ("BPF 4 / BPF default", "BPF 4", "BPF default"),
    ("Native 4 / Native default", "Native 4", "Native default"),
    ("FIFO 4 / FIFO default", "FIFO 4", "FIFO default"),
    ("BPF default / Native default", "BPF default", "Native default"),
    ("BPF 4 / Native 4", "BPF 4", "Native 4"),
    ("Native 4 / FIFO 4", "Native 4", "FIFO 4"),
    ("BPF 4 / FIFO 4", "BPF 4", "FIFO 4"),
]

READ_STAGES = ("pre_dispatch", "dispatch_gap", "post_dispatch", "scheduled_total")
WRITE_STAGES = (
    "write_pre_dispatch",
    "write_release_admit",
    "write_save_exec",
    "write_scheduled_total",
)


def nearest_rank(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(pct / 100.0 * len(ordered)) - 1)
    return ordered[index]


def med(values):
    if not values:
        return None
    return median(values)


def quantile_block(values):
    return {
        "n": len(values),
        "p50_nearest_rank": r6(nearest_rank(values, 50)),
        "p99_nearest_rank": r6(nearest_rank(values, 99)),
    }


def r6(value):
    if value is None:
        return None
    return round(value, 6)


def stage_ms(record, total_key):
    scheduled = record.get("scheduled_offer_s")
    offer = record.get("offer_s")
    submitted = record.get("submitted_s")
    completed = record.get("completed_s")
    if None in (scheduled, offer, submitted, completed):
        return None
    stages = {
        "pre_dispatch": (offer - scheduled) * 1000.0,
        "dispatch_gap": (submitted - offer) * 1000.0,
        "post_dispatch": (completed - submitted) * 1000.0,
    }
    stages["scheduled_total"] = (completed - scheduled) * 1000.0
    if total_key == "write":
        stages["release_admit"] = stages.pop("dispatch_gap")
        stages["save_exec"] = stages.pop("post_dispatch")
        ordered = (
            "pre_dispatch",
            "release_admit",
            "save_exec",
        )
    else:
        ordered = ("pre_dispatch", "dispatch_gap", "post_dispatch")
    stages["component_sum"] = sum(stages[name] for name in ordered)
    return stages


def worst_read(reads):
    best = None
    for record in reads:
        stages = stage_ms(record, "read")
        if stages is None or record.get("status") != "completed":
            continue
        if best is None or stages["scheduled_total"] > best[1]["scheduled_total"]:
            best = (record, stages)
    return best


def feedback_summary(feedback_records):
    defer = [r for r in feedback_records if r.get("decision") == 1]
    submit_now = [r for r in feedback_records if r.get("decision") == 0]
    other = [
        r for r in feedback_records if r.get("decision") not in (0, 1)
    ]
    pending_of = lambda rows: [r.get("pending_demand_reads") for r in rows]

    def class_summary(rows, wait_field=False):
        pending = pending_of(rows)
        known = [p for p in pending if p is not None]
        summary = {
            "records": len(rows),
            "with_pending_demand_reads": sum(1 for p in known if p > 0),
            "pending_demand_reads_median": r6(med(known)),
            "pending_demand_reads_max": max(known) if known else None,
        }
        if wait_field:
            waits = [r.get("requested_wait_ns") for r in rows]
            waits_ms = [w / 1e6 for w in waits if w is not None]
            summary["requested_wait_ms_median"] = r6(med(waits_ms))
            summary["requested_wait_ms_max"] = r6(max(waits_ms)) if waits_ms else None
            budgets = [r.get("remaining_budget_ns") for r in rows]
            budgets_ms = [b / 1e6 for b in budgets if b is not None]
            summary["remaining_budget_ms_median"] = r6(med(budgets_ms))
            summary["remaining_budget_ms_min"] = r6(min(budgets_ms)) if budgets_ms else None
        return summary

    return {
        "records_total": len(feedback_records),
        "defer_records": class_summary(defer, wait_field=True),
        "submit_now_records": class_summary(submit_now),
        "other_decision_records": len(other),
    }


def analyze_cell(path):
    with open(path) as handle:
        data = json.load(handle)
    requests = data["requests"]
    reads = [r for r in requests if r.get("role") == "read_demand"]
    writes = [r for r in requests if r.get("role") == "write_background"]

    read_stages = [(r, stage_ms(r, "read")) for r in reads]
    read_ok = [
        (r, s) for r, s in read_stages if s is not None and r.get("status") == "completed"
    ]
    read_missing = [r.get("id") for r, s in read_stages if s is None]
    read_not_completed = [
        r.get("id")
        for r, s in read_stages
        if s is not None and r.get("status") != "completed"
    ]
    write_stages = [(r, stage_ms(r, "write")) for r in writes]
    write_ok = [
        (r, s) for r, s in write_stages if s is not None and r.get("status") == "completed"
    ]
    write_missing = [r.get("id") for r, s in write_stages if s is None]
    write_not_completed = [
        r.get("id")
        for r, s in write_stages
        if s is not None and r.get("status") != "completed"
    ]

    read_quantiles = {
        name: quantile_block([s[name] for _, s in read_ok])
        for name in READ_STAGES
    }
    write_quantiles = {
        "write_" + name: quantile_block([s[name] for _, s in write_ok])
        for name in ("pre_dispatch", "release_admit", "save_exec", "scheduled_total")
    }

    record, stages = worst_read([r for r, _ in read_ok])
    worst = None
    if record is not None:
        total = stages["scheduled_total"]
        worst = {
            "id": record.get("id"),
            "object": record.get("object"),
            "scheduled_total_ms": r6(total),
            "pre_dispatch_ms": r6(stages["pre_dispatch"]),
            "dispatch_gap_ms": r6(stages["dispatch_gap"]),
            "post_dispatch_ms": r6(stages["post_dispatch"]),
            "post_fraction_of_total": stages["post_dispatch"] / total,
            "component_sum_equals_total": abs(
                stages["component_sum"] - stages["scheduled_total"]
            )
            < 1e-6,
        }

    read_gap_stamps = [
        r["submitted_s"] - r["offer_s"]
        for r, _ in read_ok
        if r.get("submitted_s") is not None and r.get("offer_s") is not None
    ]

    decision_counts = data.get("decision_counts", {})
    return {
        "block": data.get("block"),
        "position": data.get("position"),
        "arm": ARM_NAMES[(data["config"], data["write_io_workers"])],
        "config": data.get("config"),
        "write_io_workers": data.get("write_io_workers"),
        "decision_counts": {
            "decisions": decision_counts.get("decisions"),
            "submit_now": decision_counts.get("submit_now"),
            "defer": decision_counts.get("defer"),
            "recompute": decision_counts.get("recompute"),
            "blocking_defer_miss": decision_counts.get("blocking_defer_miss"),
        },
        "reads": {
            "completed": len(read_ok),
            "missing_stage_fields": read_missing,
            "not_completed": read_not_completed,
            "submitted_minus_offer_max_s": r6(max(read_gap_stamps)) if read_gap_stamps else None,
            "stages_ms": read_quantiles,
        },
        "writes": {
            "completed": len(write_ok),
            "missing_stage_fields": write_missing,
            "not_completed": write_not_completed,
            "stages_ms": write_quantiles,
        },
        "worst_scheduled_latency_read": worst,
        "feedback": feedback_summary(data.get("feedback_records", [])),
    }


def paired_changes(cells):
    by_block = {}
    for cell in cells:
        by_block.setdefault(cell["block"], []).append(cell)
    out = []
    for label, num, den in PAIRS:
        changes = {"worst_read_total": [], "read_post_median": [], "write_release_median": []}
        for block, block_cells in sorted(by_block.items()):
            arms = {c["arm"]: c for c in block_cells}
            if num not in arms or den not in arms:
                continue
            worst_num = arms[num]["worst_scheduled_latency_read"]["scheduled_total_ms"]
            worst_den = arms[den]["worst_scheduled_latency_read"]["scheduled_total_ms"]
            post_num = arms[num]["reads"]["stages_ms"]["post_dispatch"]["p50_nearest_rank"]
            post_den = arms[den]["reads"]["stages_ms"]["post_dispatch"]["p50_nearest_rank"]
            rel_num = arms[num]["writes"]["stages_ms"]["write_release_admit"]["p50_nearest_rank"]
            rel_den = arms[den]["writes"]["stages_ms"]["write_release_admit"]["p50_nearest_rank"]
            changes["worst_read_total"].append(r6(100.0 * (worst_num / worst_den - 1.0)))
            changes["read_post_median"].append(r6(100.0 * (post_num / post_den - 1.0)))
            changes["write_release_median"].append(
                r6(100.0 * (rel_num / rel_den - 1.0))
            )
        out.append(
            {
                "comparison": label,
                "numerator_arm": num,
                "denominator_arm": den,
                "blocks": sorted(by_block),
                "paired_per_block": changes,
                "paired_median": {
                    key: r6(med(values)) for key, values in changes.items()
                },
            }
        )
    return out


def arm_summary(cells):
    arms = {}
    for arm in ARM_ORDER:
        arm_cells = [c for c in cells if c["arm"] == arm]
        if not arm_cells:
            continue
        summary = {"cells": len(arm_cells), "blocks": sorted(c["block"] for c in arm_cells)}
        summary["read_stage_ms_median_of_cell_medians"] = {
            name: r6(med([c["reads"]["stages_ms"][name]["p50_nearest_rank"] for c in arm_cells]))
            for name in READ_STAGES
        }
        summary["write_stage_ms_median_of_cell_medians"] = {
            name: r6(med([c["writes"]["stages_ms"][name]["p50_nearest_rank"] for c in arm_cells]))
            for name in WRITE_STAGES
        }
        summary["worst_read_total_ms_max_over_cells"] = r6(
            max(c["worst_scheduled_latency_read"]["scheduled_total_ms"] for c in arm_cells)
        )
        summary["defers_total"] = sum(c["decision_counts"]["defer"] or 0 for c in arm_cells)
        summary["feedback_records_total"] = sum(
            c["feedback"]["records_total"] for c in arm_cells
        )
        arms[arm] = summary
    return arms


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Offline stage attribution for one existing campaign directory."
    )
    parser.add_argument("campaign", type=Path, help="campaign directory with result.json cells")
    args = parser.parse_args(argv)

    paths = sorted(args.campaign.glob("block-*/position-*/result.json"))
    if not paths:
        parser.error(f"no result.json cells under {args.campaign}")

    cells = [analyze_cell(path) for path in paths]
    worst_pre = [c["worst_scheduled_latency_read"]["pre_dispatch_ms"] for c in cells]
    worst_post = [c["worst_scheduled_latency_read"]["post_dispatch_ms"] for c in cells]
    worst_frac = [
        100.0 * c["worst_scheduled_latency_read"]["post_fraction_of_total"] for c in cells
    ]

    missing = [
        {"block": c["block"], "position": c["position"], "reads": c["reads"]["missing_stage_fields"],
         "writes": c["writes"]["missing_stage_fields"], "not_completed_reads": c["reads"]["not_completed"],
         "not_completed_writes": c["writes"]["not_completed"]}
        for c in cells
        if c["reads"]["missing_stage_fields"] or c["writes"]["missing_stage_fields"]
        or c["reads"]["not_completed"] or c["writes"]["not_completed"]
    ]

    report = {
        "schema": 1,
        "kind": "gds_write_workers_stage_analysis",
        "script": "gds-control/analyze_worker_stages.py",
        "campaign_path": str(args.campaign),
        "cells": cells,
        "arms": arm_summary(cells),
        "same_block_paired_changes": paired_changes(cells),
        "worst_read_cross_check": {
            "pre_dispatch_ms": {"min": r6(min(worst_pre)), "max": r6(max(worst_pre))},
            "post_dispatch_ms": {"min": r6(min(worst_post)), "max": r6(max(worst_post))},
            "post_fraction_percent": {"min": r6(min(worst_frac)), "max": r6(max(worst_frac))},
        },
        "cells_with_missing_or_incomplete_records": missing,
        "limitations": [
            "feedback_records carry no timestamps and no stable per-write id: request_id is unique per admission event, so release (submit_now) records cannot be joined to a specific write request; reported decision state is distribution-level only",
            "no wake-reason field exists; pending_demand_reads on submit_now records does not identify demand-drain versus budget-expiry wakeups",
            "read submitted_s equals offer_s by stamping before get_blocking, so read post_dispatch includes adapter admission, the blocking blocking-get call, scheduler and lock waits, and completion bookkeeping; it is not pure device service time",
            "write submitted_s is stamped at save-coroutine entry, so write_release_admit includes deferral wait, release decisions, and save-coroutine dispatch; write_save_exec is the coroutine body through completion",
            "with 64 reads the nearest-rank p99 equals the maximum",
            "attribution analysis only: no causal effect of worker count is identified",
        ],
    }
    json.dump(report, sys.stdout, indent=2, sort_keys=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
