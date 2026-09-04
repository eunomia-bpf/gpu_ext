#!/usr/bin/env python3
"""Revalidate a retained raw-map campaign and summarize direct evidence."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import protocol


def require_clean_safety(snapshot: dict[str, Any], label: str) -> None:
    gpu = snapshot.get("gpu", {})
    struct_ops = snapshot.get("struct_ops", {})
    expected_empty = {
        "compute_apps": gpu.get("compute_apps"),
        "xids": snapshot.get("xids"),
        "dmesg_abnormal": snapshot.get("dmesg_abnormal"),
        "journal_abnormal": snapshot.get("journal_abnormal"),
        "struct_ops.links": struct_ops.get("links"),
        "struct_ops.maps": struct_ops.get("maps"),
    }
    dirty = {key: value for key, value in expected_empty.items() if value != []}
    if snapshot.get("uvm_refcount") != 0 or dirty:
        raise protocol.EvidenceError(
            f"{label} safety state is not clean: uvm={snapshot.get('uvm_refcount')}, "
            f"nonempty={dirty}"
        )


def summarize(path: Path) -> dict[str, Any]:
    manifest = protocol.validate_campaign_manifest(path, "full")
    require_clean_safety(manifest["safety_before"], "campaign.before")
    require_clean_safety(manifest["safety_after"], "campaign.after")

    totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for recorded in manifest["cells"]:
        directory = path / recorded["directory"]
        cell = json.loads((directory / "cell.json").read_text())
        require_clean_safety(cell["safety_after"], f"{recorded['directory']}.after")
        for field in ("native_truth_records", "instrumented_truth_records",
                      "raw_records", "aggregate_callbacks", "full_drops"):
            totals[recorded["arm"]][field] += recorded[field]

    return {
        "schema": 1,
        "protocol": protocol.PROTOCOL,
        "source": str(path.resolve()),
        "status": "passed",
        "blocks": protocol.blocks_for("full"),
        "cells": manifest["cell_count"],
        "positive_cells": manifest["positive_cells"],
        "negative_cells": manifest["negative_drop_gates"],
        "totals_by_arm": {name: dict(totals[name]) for name in protocol.ARM_BY_NAME},
        "positive_exact_raw_records": (
            totals["small"]["raw_records"] + totals["large"]["raw_records"]
        ),
        "negative_accounted_drops": totals["overflow_negative"]["full_drops"],
        "claim_boundary": manifest["claim_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.campaign), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
