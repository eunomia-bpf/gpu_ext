#!/usr/bin/env python3
"""Emit per-repeat timings and median/min/max from saved client logs."""
import csv
import re
import statistics
import sys
from pathlib import Path

directory = Path(sys.argv[1])
arms = {
    "baseline_gpu": "t_ns", "register": "t_ns",
    "offload1": "t_complete_ns", "cpu_restore": "t_ns",
    "offload2_release": "t_complete_ns", "gpu_restore": "t_ns",
    "steady_gpu": "t_ns",
}
rows = []
for log in sorted(directory.glob("repeat-*/client.log")):
    lines = log.read_text().splitlines()
    for arm, field in arms.items():
        line, = [line for line in lines if line.startswith("arm=" + arm + " ")]
        duration_ns = int(re.search(r"\b" + field + r"=(\d+)", line)[1])
        rows.append({"repeat": log.parent.name, "arm": arm,
                     "elapsed_ms": duration_ns / 1e6,
                     "source": str(log.relative_to(directory))})
with (directory / "timings.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print("arm,n,median_ms,min_ms,max_ms")
for arm in arms:
    values = [row["elapsed_ms"] for row in rows if row["arm"] == arm]
    print(f"{arm},{len(values)},{statistics.median(values):.6f},{min(values):.6f},{max(values):.6f}")
