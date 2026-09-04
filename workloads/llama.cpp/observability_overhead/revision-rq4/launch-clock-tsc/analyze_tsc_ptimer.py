#!/usr/bin/env python3
"""Independently replay the stock RM TSC/PTIMER control JSONL."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

PTIMER_ALLOWANCE_NS = 32


def ceil_cycles_ns(cycles: int, hz: int) -> int:
    return (cycles * 1_000_000_000 + hz - 1) // hz


def analyze(path: Path) -> dict[str, object]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    summaries = [row for row in rows if row.get("record") == "summary"]
    samples = [row for row in rows if row.get("record") == "sample"]
    if len(summaries) != 1:
        raise ValueError(f"expected one summary, found {len(summaries)}")
    summary = summaries[0]
    hz = int(summary["tsc_hz"])
    if hz <= 0 or summary.get("method") != "stock_rm_tsc_ptimer_v1":
        raise ValueError("unexpected clock method or TSC frequency")
    if any(row.get("valid") is not True for row in samples):
        raise ValueError("invalid sample in completed control")

    widths: list[int] = []
    previous_tsc = 0
    previous_ptimer = 0
    for row in samples:
        low = int(row["tsc_low"])
        mid = int(row["tsc_mid"])
        high = int(row["tsc_high"])
        ptimer = int(row["ptimer_ns"])
        if not 0 < low <= mid <= high or ptimer <= 0:
            raise ValueError("malformed TSC/PTIMER interval")
        width = ceil_cycles_ns(high - low, hz) + 2 * PTIMER_ALLOWANCE_NS
        if width != int(row["bracket_width_ns"]):
            raise ValueError("stored precision width does not replay")
        if previous_tsc and (mid <= previous_tsc or ptimer <= previous_ptimer):
            raise ValueError("clock regression")
        previous_tsc, previous_ptimer = mid, ptimer
        widths.append(width)

    accepted = len(samples)
    attempted = int(summary["attempted"])
    median_width = int(statistics.median(widths)) if widths else 0
    max_width = max(widths, default=0)
    if accepted != int(summary["accepted"]):
        raise ValueError("accepted count mismatch")
    if attempted != accepted + int(summary["rejected"]):
        raise ValueError("attempt accounting mismatch")
    if median_width != int(summary["median_bracket_width_ns"]):
        raise ValueError("median precision mismatch")
    if max_width != int(summary["max_bracket_width_ns"]):
        raise ValueError("maximum precision mismatch")

    first = samples[0]
    last = samples[-1]
    delta_tsc = int(last["tsc_mid"]) - int(first["tsc_mid"])
    delta_ptimer = int(last["ptimer_ns"]) - int(first["ptimer_ns"])
    if delta_tsc <= 0 or delta_ptimer <= 0:
        raise ValueError("invalid rate endpoints")
    predicted_num = delta_tsc * 1_000_000_000
    observed_num = delta_ptimer * hz
    rate_error_ppb = abs(predicted_num - observed_num) * 1_000_000_000 // predicted_num
    if rate_error_ppb != int(summary["rate_error_ppb"]):
        raise ValueError("clock-rate result does not replay")

    expected_gate = (
        accepted >= 200
        and int(summary["rejected"]) == 0
        and int(summary["regressions"]) == 0
        and int(summary["migration_errors"]) == 0
        and summary["cleanup_complete"] is True
        and median_width <= int(summary["precision_limit_ns"])
        and rate_error_ppb <= int(summary["rate_limit_ppb"])
    )
    if expected_gate is not bool(summary["gate_pass"]):
        raise ValueError("summary gate does not replay")
    return {
        "run_status": "valid_control",
        "attempted": attempted,
        "accepted": accepted,
        "median_bracket_width_ns": median_width,
        "max_bracket_width_ns": max_width,
        "precision_limit_ns": int(summary["precision_limit_ns"]),
        "rate_error_ppb": rate_error_ppb,
        "rate_limit_ppb": int(summary["rate_limit_ppb"]),
        "cleanup_complete": bool(summary["cleanup_complete"]),
        "gate_pass": expected_gate,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.records), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
