#!/usr/bin/env python3
"""Offline analyzer for a completed write-io-workers campaign.

Reads the supplied campaign directory's existing ``run-plan.json`` (cell
fields ``block``, ``position``, ``arm``, ``workers``, ``dir``) and each
cell's existing ``result.json``, then prints JSON to stdout with:

- every plan row retained, including cells whose ``result.json`` is
  missing, unreadable, not a JSON object, or carries an error;
- each cell resolved under the supplied campaign directory via the stored
  ``block-XX/position-P-arm`` layout (or the validated relative suffix of
  the recorded ``dir``), so relocated artifacts are analyzed from the
  supplied directory, never from the old checkout; the recorded source
  path stays in the row;
- per-arm medians of the four existing headline metrics;
- within-block paired changes (100*(numerator/denominator - 1)) for the
  same four metrics across the six arms, with the block IDs listed
  alongside the values so their order is explicit, missing or
  zero-denominator blocks kept as null, and negative values retained.

The analyzer launches nothing, deletes nothing, writes no files, adds no
hashes or gates, and runs no new experiment.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

METRICS = (
    "read_scheduled_offer_to_completion_p50_ms",
    "read_scheduled_offer_to_completion_p99_ms",
    "write_completion_throughput_mib_s",
    "total_storage_bandwidth_mib_s",
)
PAIRS = (
    ("fifo4", "fifo_default"),
    ("native4", "native_default"),
    ("bpf4", "bpf_default"),
    ("bpf_default", "native_default"),
    ("bpf4", "native4"),
    ("native_default", "fifo_default"),
    ("bpf_default", "fifo_default"),
    ("native4", "fifo4"),
    ("bpf4", "fifo4"),
)


def cell_suffix(plan_cell: dict) -> tuple[str, str] | None:
    """Relative cell location under a campaign directory.

    Prefers the stored layout built from the plan's own fields,
    ``block-XX/position-P-arm``; falls back to the trailing two
    components of the recorded absolute dir when they validate against
    the same schema.
    """
    block = plan_cell.get("block")
    position = plan_cell.get("position")
    arm = plan_cell.get("arm")
    parts = [part for part in Path(str(plan_cell.get("dir", ""))).parts
             if part not in ("", ".")]
    tail = tuple(parts[-2:]) if len(parts) >= 2 else ()
    if isinstance(block, int) and isinstance(position, int) and arm:
        schema = (f"block-{block:02d}", f"position-{position}-{arm}")
        if tail == schema:
            return schema
    if (len(tail) == 2
            and re.fullmatch(r"block-\d{2}", tail[0])
            and re.fullmatch(r"position-\d+-\S+", tail[1])):
        return tail
    return None


def load_cell(plan_cell: dict, campaign_dir: Path) -> dict:
    row = {key: plan_cell.get(key)
           for key in ("block", "position", "arm", "config", "workers")}
    row["dir"] = plan_cell.get("dir")
    suffix = cell_suffix(plan_cell)
    if suffix is None:
        row["error"] = "cell dir not resolvable under campaign directory"
        row["cell_dir"] = None
        row["metrics"] = {name: None for name in METRICS}
        return row
    cell_dir = campaign_dir.joinpath(*suffix)
    row["cell_dir"] = str(cell_dir)
    result_path = cell_dir / "result.json"
    record = None
    if result_path.is_file():
        try:
            record = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            row["error"] = f"result.json unreadable: {error}"
    else:
        row["error"] = "result.json missing"
    metrics = {}
    if isinstance(record, dict):
        if record.get("error") and not row.get("error"):
            row["error"] = str(record["error"])
        if record.get("cleanup_errors"):
            row["cleanup_errors"] = record["cleanup_errors"]
        raw_metrics = record.get("metrics")
        if isinstance(raw_metrics, dict):
            metrics = raw_metrics
        elif "metrics" in record:
            row["error"] = row.get("error") or "metrics block is not an object"
    elif record is not None and not row.get("error"):
        row["error"] = "result.json is not a JSON object"
    row["metrics"] = {name: metrics.get(name) for name in METRICS}
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path,
                        help="directory containing run-plan.json")
    args = parser.parse_args()
    plan_path = args.campaign_dir / "run-plan.json"
    if not plan_path.is_file():
        print(f"no run-plan.json in {args.campaign_dir}", file=sys.stderr)
        return 2
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    rows = [load_cell(cell, args.campaign_dir)
            for cell in plan.get("cells", [])]

    arms: list[str] = []
    for row in rows:
        if row.get("arm") and row["arm"] not in arms:
            arms.append(row["arm"])
    per_arm = {}
    for arm in arms:
        values = {name: [] for name in METRICS}
        for row in rows:
            if row.get("arm") != arm:
                continue
            for name in METRICS:
                value = row["metrics"].get(name)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    values[name].append(float(value))
        per_arm[arm] = {
            "cells": sum(1 for row in rows if row.get("arm") == arm),
            "medians": {name: (statistics.median(vals) if vals else None)
                        for name, vals in values.items()},
        }

    blocks = sorted({row["block"] for row in rows
                     if row.get("block") is not None})
    by_arm_block = {(row.get("arm"), row.get("block")): row for row in rows}
    paired = {}
    for numerator, denominator in PAIRS:
        per_metric = {}
        for name in METRICS:
            values = []
            for block in blocks:
                value_a = by_arm_block.get((numerator, block), {}).get(
                    "metrics", {}).get(name)
                value_b = by_arm_block.get((denominator, block), {}).get(
                    "metrics", {}).get(name)
                if (isinstance(value_a, (int, float))
                        and isinstance(value_b, (int, float))
                        and float(value_b) != 0.0):
                    values.append(
                        100.0 * (float(value_a) / float(value_b) - 1.0))
                else:
                    values.append(None)
            present = [value for value in values if value is not None]
            per_metric[name] = {
                "blocks": list(blocks),
                "values": values,
                "median": statistics.median(present) if present else None,
                "min": min(present) if present else None,
                "max": max(present) if present else None,
            }
        paired[f"{numerator}/{denominator}"] = per_metric

    output = {
        "campaign_dir": str(args.campaign_dir),
        "source": plan.get("source"),
        "cells_attempted": len(rows),
        "cells_with_metrics": sum(
            1 for row in rows if any(v is not None
                                     for v in row["metrics"].values())),
        "cells": rows,
        "per_arm": per_arm,
        "paired": paired,
        "paired_definition": ("within each block: 100*(numerator/denominator "
                              "- 1) on the existing per-cell metrics; "
                              "missing or zero-denominator blocks stay "
                              "null; negative values are retained"),
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
