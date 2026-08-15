#!/usr/bin/env python3
"""Compare normalized Stage 4 reduced- and natural-capacity trends."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def number(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "", "UNAVAILABLE") else None
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.summary.open()))
    output = []
    for row in rows:
        if not row.get("capacity_model"):
            continue
        working_set = number(row.get("effective_capacity_bytes"))
        ratio = number(row.get("ratio"))
        gib = working_set * ratio / (1 << 30) if working_set and ratio else None
        result = {
            "capacity_model": row.get("capacity_model"),
            "policy": row.get("policy"),
            "ratio": row.get("ratio"),
            "working_set_gib": gib,
        }
        for phase in ("phase_A_first", "phase_B_first", "phase_A_reuse", "phase_B_reuse"):
            value = number(row.get(f"{phase}_mean"))
            result[f"{phase}_ms_per_gib"] = value / gib if value is not None and gib else None
        output.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(output[0]) if output else ["status"]
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output or [{"status": "NO_COMPARABLE_RESULTS"}])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
