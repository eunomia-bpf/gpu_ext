#!/usr/bin/env python3
"""Summarize the completed geometry/work campaign from original CUDA logs."""
import csv
import random
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
print("setting,native_ms,off_ms,on_ms,on_off_paired_pct,ci95_low,ci95_high,slower_pairs")
for sweep in ("blocks-current-abi", "work-current-abi"):
    directory = root / sweep
    with (directory / "cells.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    groups = {}
    for row in rows:
        lines = (directory / row["source"] / "application.log").read_text().splitlines()
        measurement, = [line.split("\t") for line in lines
                        if line.startswith("FIG15_MEASUREMENT\t")]
        # The original CSV mislabeled microseconds as milliseconds. Preserve
        # that file and derive milliseconds directly from the application log.
        elapsed_ms = float(measurement[3])
        if abs(float(row["elapsed_ms"]) / 1000 - elapsed_ms) > 1e-8:
            raise ValueError(f"Unexpected original CSV unit: {row['source']}")
        if row["application_exit"] != "0" or row["loader_exit"] not in ("", "0"):
            raise ValueError(f"Recorded process failure: {row['source']}")
        row["elapsed_ms"] = f"{elapsed_ms:.9f}"
        groups.setdefault(row["setting"], {}).setdefault(int(row["block"]), {})[row["arm"]] = elapsed_ms
    with (directory / "cells-derived-ms.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for setting, pairs in groups.items():
        values = list(pairs.values())
        medians = [statistics.median(p[arm] for p in values) for arm in ("native", "off", "on")]
        effects = [100 * (p["on"] / p["off"] - 1) for p in values]
        rng = random.Random(1797)
        boot = sorted(statistics.median(rng.choices(effects, k=len(effects))) for _ in range(10000))
        result = [*medians, statistics.median(effects), boot[249], boot[9749]]
        print(setting + "," + ",".join(f"{v:.9f}" for v in result) + f",{sum(v > 0 for v in effects)}/{len(effects)}")
