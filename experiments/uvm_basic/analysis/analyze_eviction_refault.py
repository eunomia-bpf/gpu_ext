#!/usr/bin/env python3
"""Correlate selected eviction victims with later A-reuse prefetch decisions."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def integer(value: str | int | None) -> int:
    try:
        return int(str(value or "0"), 0)
    except ValueError:
        return 0


def iter_csv(path: Path):
    if not path.exists() or not path.stat().st_size:
        return
    with path.open(newline="", errors="replace") as source:
        for row in csv.DictReader(source):
            if row:
                yield row


def phase_windows(program: Path) -> dict[str, tuple[int, int]]:
    windows: dict[str, tuple[int, int]] = {}
    if not program.exists():
        return windows
    for line in program.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        phase = str(row.get("phase", ""))
        if phase.startswith(("phase_A_first", "phase_B_first", "phase_A_reuse", "phase_B_reuse")):
            windows[phase] = (integer(row.get("monotonic_start_ns")), integer(row.get("monotonic_end_ns")))
    return windows


def in_window(timestamp: int, window: tuple[int, int] | None) -> bool:
    return bool(window and window[0] <= timestamp <= window[1])


def analyze_run(run: Path) -> dict[str, object]:
    manifest = json.loads((run / "manifest.json").read_text())
    windows = phase_windows(run / "program.jsonl")
    a_first = next((value for key, value in windows.items() if key.startswith("phase_A_first")), None)
    b_first = next((value for key, value in windows.items() if key.startswith("phase_B_first")), None)
    a_reuse = next((value for key, value in windows.items() if key.startswith("phase_A_reuse")), None)

    evictions: dict[int, int] = {}
    eviction_selected_count = 0
    for row in iter_csv(run / "chunk_trace.csv"):
        if row.get("hook_type") != "EVICTION_SELECTED":
            continue
        eviction_selected_count += 1
        block = integer(row.get("va_start"))
        timestamp = integer(row.get("timestamp_ns"))
        if block and in_window(timestamp, b_first):
            evictions[block] = timestamp
    a_blocks: set[int] = set()
    refaults: dict[int, int] = {}
    sizes: dict[int, int] = {}
    for row in iter_csv(run / "prefetch_decision_trace.csv"):
        if row.get("event_type") != "DECISION":
            continue
        timestamp = integer(row.get("timestamp_ns"))
        block = integer(row.get("va_start"))
        if block and in_window(timestamp, a_first):
            a_blocks.add(block)
        if block and in_window(timestamp, a_reuse):
            refaults.setdefault(block, timestamp)
            end = integer(row.get("va_end"))
            sizes[block] = end - block + 1 if end >= block else 0
    identities = a_blocks & set(evictions) & set(refaults)
    latencies = [(refaults[block] - evictions[block]) / 1000.0 for block in identities
                 if refaults[block] >= evictions[block]]
    identity_available = bool(eviction_selected_count) and bool(windows)
    return {
        "experiment": manifest.get("experiment"), "policy": manifest.get("policy"),
        "ratio": manifest.get("ratio"), "run_id": run.name,
        "evidence_class": "GPU_EXT_EVICTION_TRACE+GPU_EXT_PREFETCH_DECISION_TRACE",
        "identity_available": identity_available,
        "eviction_selected_count": eviction_selected_count,
        "refaulted_block_count": len(identities) if identity_available else "UNAVAILABLE",
        "refaulted_bytes": sum(sizes.get(block, 0) for block in identities) if identity_available else "UNAVAILABLE",
        "eviction_to_refault_us_mean": statistics.fmean(latencies) if latencies else "UNAVAILABLE",
        "interpretation": "SAME_USER_VA_BLOCK_IN_A_FIRST_B_EVICTION_A_REUSE" if identities
                          else "NO_PROVEN_REFAULT" if identity_available else "UNAVAILABLE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_dir.resolve()
    rows = []
    for manifest in sorted((root / "results" / "stage3").glob("**/manifest.json")):
        if json.loads(manifest.read_text()).get("experiment") in {"oversub", "joint_policy"}:
            rows.append(analyze_run(manifest.parent))
    fields = ["experiment", "policy", "ratio", "run_id", "evidence_class",
              "identity_available", "eviction_selected_count", "refaulted_block_count",
              "refaulted_bytes", "eviction_to_refault_us_mean", "interpretation"]
    target = root / "results" / "stage3_eviction_refault_summary.csv"
    with target.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {target} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
