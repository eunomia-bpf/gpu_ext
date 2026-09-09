#!/usr/bin/env python3
"""Summarize Native/Off pairs using original CUDA event times."""
import csv
import random
import statistics as st
from pathlib import Path
root = Path(__file__).resolve().parent
summaries = []
for work in (0, 128, 1024):
    directory = root / f"work-{work}"
    rows = list(csv.DictReader((directory / "cells.csv").open()))
    assert len(rows) == 80
    groups = {}
    for row in rows:
        assert row["arm"] in ("native", "off")
        assert row["application_exit"] == "0"
        assert row["loader_exit"] in ("", "0")
        log = (directory / row["source"] / "application.log").read_text()
        values = [line.split("\t") for line in log.splitlines() if line.startswith("FIG15_MEASUREMENT\t")]
        assert len(values) == 1
        ms = float(values[0][3])
        assert abs(ms - float(row["elapsed_ms"])) < 1e-8
        if row["arm"] == "off":
            assert row["map_key0"] == "true"
            assert row["automatic_admitted"] == "false"
        groups.setdefault(int(row["cta_blocks"]), {}).setdefault(int(row["block"]), {})[row["arm"]] = ms
    for blocks, pairs in groups.items():
        assert len(pairs) == 10
        assert all(set(p) == {"native", "off"} for p in pairs.values())
        effects = [100 * (p["off"] / p["native"] - 1) for p in pairs.values()]
        delta = [(p["off"] - p["native"]) * 1000 / 128 for p in pairs.values()]
        rng = random.Random(1797)
        boot = sorted(st.median(rng.choices(effects, k=10)) for _ in range(10000))
        summaries.append(dict(blocks=blocks, threads_per_block=128, work=work,
            pairs=10, native_ms=st.median(p["native"] for p in pairs.values()),
            off_ms=st.median(p["off"] for p in pairs.values()),
            overhead_pct=st.median(effects), ci95_low=boot[249], ci95_high=boot[9749],
            added_us_per_launch=st.median(delta)))
with (root / "summary.csv").open("x", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
    writer.writeheader()
    writer.writerows(summaries)
for row in summaries:
    print(row)
