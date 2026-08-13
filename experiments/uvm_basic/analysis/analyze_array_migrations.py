#!/usr/bin/env python3
"""Classify Nsight UVM migration and GPU-fault rows by recorded A/B/C ranges."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path


def load_program(path: Path) -> tuple[dict[str, tuple[int, int]], str, float | str]:
    ranges: dict[str, tuple[int, int]] = {}
    mode = "UNAVAILABLE"
    latency: float | str = "UNAVAILABLE"
    if not path.exists():
        return ranges, mode, latency
    for line in path.read_text(errors="replace").splitlines():
        try: row = json.loads(line)
        except json.JSONDecodeError: continue
        mode = str(row.get("kernel_mode", mode))
        if row.get("phase") == "allocation_addresses":
            for name in "abc":
                ranges[name.upper()] = (int(row[f"{name}_base_u64"]), int(row[f"{name}_end_u64"]))
        if row.get("phase") == "kernel_1_demand":
            latency = float(row["elapsed_ms"])
    return ranges, mode, latency


def classify(address: int | None, ranges: dict[str, tuple[int, int]]) -> str:
    if address is None:
        return "unknown"
    for name, (first, outer) in ranges.items():
        if first <= address < outer:
            return name
    return "unknown"


def analyze(run: Path) -> list[dict[str, object]]:
    ranges, mode, latency = load_program(run / "program.jsonl")
    manifest = json.loads((run / "manifest.json").read_text())
    database = run / "representative.sqlite"
    totals = {name: {"h2d": 0, "d2h": 0, "faults": 0} for name in (*ranges, "unknown")}
    schema = "UNAVAILABLE"
    if database.exists() and ranges:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        tables = {row[0] for row in connection.execute("select name from sqlite_master where type='table'")}
        if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
            columns = {row[1] for row in connection.execute("pragma table_info(CUPTI_ACTIVITY_KIND_MEMCPY)")}
            if {"copyKind", "bytes", "virtualAddress"} <= columns:
                schema = "ADDRESS_BEARING_UVM_MEMCPY"
                for kind, size, address in connection.execute(
                        "select copyKind, bytes, virtualAddress from CUPTI_ACTIVITY_KIND_MEMCPY "
                        "where copyKind in (11, 12)"):
                    bucket = classify(address, ranges)
                    totals[bucket]["h2d" if kind == 11 else "d2h"] += int(size)
        if "CUDA_UM_GPU_PAGE_FAULT_EVENTS" in tables:
            for address, count in connection.execute(
                    "select address, numberOfPageFaults from CUDA_UM_GPU_PAGE_FAULT_EVENTS"):
                totals[classify(address, ranges)]["faults"] += int(count)
        connection.close()
    rows = []
    for allocation in (*ranges, "unknown"):
        rows.append({
            "policy": manifest.get("policy"), "run_id": run.name, "kernel_mode": mode,
            "allocation": allocation, "h2d_bytes": totals[allocation]["h2d"] if ranges else "UNAVAILABLE",
            "d2h_bytes": totals[allocation]["d2h"] if ranges else "UNAVAILABLE",
            "gpu_faults": totals[allocation]["faults"] if ranges else "UNAVAILABLE",
            "kernel_1_ms": latency, "nsight_schema": schema,
            "evidence_class": "NSIGHT_UVM+PROGRAM_ALLOCATION_RANGE",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_dir.resolve()
    rows = []
    for manifest in sorted((root / "results" / "stage3" / "array_migration").glob("**/manifest.json")):
        if json.loads(manifest.read_text()).get("run_kind") == "nsys":
            rows.extend(analyze(manifest.parent))
    fields = ["policy", "run_id", "kernel_mode", "allocation", "h2d_bytes", "d2h_bytes",
              "gpu_faults", "kernel_1_ms", "nsight_schema", "evidence_class"]
    target = root / "results" / "stage3_array_migration_summary.csv"
    with target.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {target} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
