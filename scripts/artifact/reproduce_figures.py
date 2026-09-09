#!/usr/bin/env python3
"""Reproduce the paper-selected observability figure without touching the paper.

The obs figure script
(docs/paper/tex-revision/img/results-raw/revision/plot_obs_with_array.py)
expects the GPU-array campaign to hold exactly the five blocks the paper
selected. The raw campaign was later extended to ten paired blocks; the
first ten cells.json entries (blocks 1..5, both arms) are unchanged, and the
ten-block extension is retained in the raw. This wrapper therefore:

  1. reads the current raw cells.json (read-only),
  2. reconstructs the paper-selected blocks 1..5 plus the summary fields
     those cells imply into a fresh output directory outside docs/paper,
  3. invokes the unchanged obs figure script with --new-cells /
     --new-summary / --data-output / --output-prefix all pointing at that
     output directory, and
  4. invokes the current seven-panel policy plot script with its
     --output-prefix redirected to the same directory.

It is CPU-only, uses no GPU, runs no workloads, imposes no gates, records
no hashes, and writes nothing into docs/paper or into the raw campaign
directories. The five-block subset is a selected subset of the current
ten-block cells.json, not an independent run; the ten-block extension
stays preserved in the raw.

Default output directory: <repo-root>/reproduction. All outputs are
created exclusively and never overwrite anything; when a name is already
taken, choose a new --out-dir instead of deleting previous outputs.
Pass --out-dir to move it.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

REVISION = "docs/paper/tex-revision/img/results-raw/revision"
OBS_PLOT = Path(REVISION) / "plot_obs_with_array.py"
PANEL_PLOT = Path(REVISION) / "plot_port_panels.py"
NEW_CAMPAIGN = ("workloads/llama.cpp/observability_overhead/revision-rq4/"
                "results-onevalue-array-bootstrap-575-20260907")
NEW_CELLS_REL = NEW_CAMPAIGN + "/cells.json"
NEW_SUMMARY_REL = NEW_CAMPAIGN + "/summary.json"
PAPER_SELECTED_BLOCKS = range(1, 6)
TOOL_ARM = "gpubpf_kernelretsnoop"
REQUIRED_STORAGE = "gpu-array-onevalue"


def die(msg: str) -> None:
    print(f"reproduce_figures: error: {msg}", file=sys.stderr)
    sys.exit(1)


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value)


def exclusive_write(path: Path, text: str) -> None:
    try:
        with path.open("x", encoding="utf-8") as fh:
            fh.write(text)
    except FileExistsError:
        die(f"output already exists: {path} (choose a new --out-dir; "
            "previous outputs are never deleted or overwritten)")


def load_cells(root: Path) -> list:
    path = root / NEW_CELLS_REL
    if not path.is_file():
        die(f"missing raw cells file: {NEW_CELLS_REL}")
    cells = json.loads(path.read_text())
    if not isinstance(cells, list):
        die(f"{NEW_CELLS_REL} is not a JSON list")
    return cells


def select_paper_blocks(cells: list) -> list:
    subset = [c for c in cells
              if isinstance(c, dict) and c.get("block") in PAPER_SELECTED_BLOCKS]
    baseline = [c for c in subset if c.get("arm") == "baseline"]
    tool = [c for c in subset if c.get("arm") == TOOL_ARM]
    if len(baseline) != 5 or sorted(c["block"] for c in baseline) != [1, 2, 3, 4, 5]:
        die("blocks 1..5 do not contain five complete baseline cells; "
            "refusing to reconstruct the paper subset")
    if len(tool) != 5 or sorted(c["block"] for c in tool) != [1, 2, 3, 4, 5]:
        die(f"blocks 1..5 do not contain five complete {TOOL_ARM} cells")
    for cell in tool:
        if cell.get("storage") != REQUIRED_STORAGE:
            die(f"{TOOL_ARM} block {cell.get('block')} storage is "
                f"{cell.get('storage')!r}, expected {REQUIRED_STORAGE!r}")
    for cell in baseline + tool:
        if not finite(cell.get("throughput_tok_s")):
            die(f"{cell.get('arm')} block {cell.get('block')} lacks numeric throughput")
    return subset


def derive_summary(subset: list) -> dict:
    baseline = {c["block"]: c["throughput_tok_s"]
                for c in subset if c["arm"] == "baseline"}
    tool = {c["block"]: c["throughput_tok_s"]
            for c in subset if c["arm"] == TOOL_ARM}
    derived = [100.0 * (baseline[b] - tool[b]) / baseline[b]
               for b in PAPER_SELECTED_BLOCKS]
    return {
        "arms": [
            {"arm": "baseline", "cells": 5,
             "throughput_tok_s_mean": sum(baseline.values()) / 5,
             "mean_overhead_pct": None},
            {"arm": TOOL_ARM, "cells": 5,
             "throughput_tok_s_mean": sum(tool.values()) / 5,
             "mean_overhead_pct": sum(derived) / 5},
        ]
    }


def guarded_out_dir(out: Path, root: Path) -> Path:
    target = out.resolve()
    paper = (root / "docs" / "paper").resolve()
    zones = [paper, (root / NEW_CAMPAIGN).resolve(),
             (root / "workloads/llama.cpp/observability_overhead").resolve()]
    for zone in zones:
        if target == zone or zone in target.parents:
            die(f"--out-dir must stay outside {zone}")
    if paper in target.parents:
        die("--out-dir must stay outside docs/paper")
    return target


def run_plot(root: Path, script_rel: Path, extra: list[str]) -> None:
    script = root / script_rel
    if not script.is_file():
        die(f"missing plot script: {script_rel}")
    proc = subprocess.run([sys.executable, str(script), *extra], text=True)
    if proc.returncode != 0:
        sys.exit(f"reproduce_figures: error: {script.name} failed "
                 f"(exit {proc.returncode})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output directory (default: <repo-root>/reproduction)")
    args = parser.parse_args()

    root = (args.repo_root or Path(__file__).resolve().parents[2]).resolve()
    if not (root / "workloads").is_dir():
        die(f"no workloads/ under {root}; pass --repo-root")
    out_dir = guarded_out_dir(args.out_dir or root / "reproduction", root)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = load_cells(root)
    subset = select_paper_blocks(cells)
    derived = derive_summary(subset)

    cells_path = out_dir / "obs-blocks1-5-cells.json"
    summary_path = out_dir / "obs-blocks1-5-summary.json"
    exclusive_write(cells_path, json.dumps(subset, indent=2) + "\n")
    exclusive_write(summary_path, json.dumps(derived, indent=2) + "\n")

    current = json.loads((root / NEW_SUMMARY_REL).read_text())
    current_arms = {e["arm"]: e for e in current["arms"]}
    note = {
        "purpose": "reconstruct the paper-selected five GPU-array blocks "
                   "outside the paper and reproduce the figure scripts",
        "selected_subset": {
            "source_cells": NEW_CELLS_REL,
            "blocks": [1, 2, 3, 4, 5],
            "arms": ["baseline", TOOL_ARM],
            "storage": REQUIRED_STORAGE,
            "cells_json_entries": len(subset),
            "note": "strict subset of the current ten-block cells.json; the "
                    "first ten entries are unchanged; this is the paper-"
                    "selected subset, not an independent run",
        },
        "derived_five_block_summary": derived,
        "retained_ten_block_extension": {
            "source_summary": NEW_SUMMARY_REL,
            "cells": current_arms.get(TOOL_ARM, {}).get("cells"),
            "mean_overhead_pct":
                current_arms.get(TOOL_ARM, {}).get("mean_overhead_pct"),
            "note": "ten-block extension preserved in the raw; supplemental",
        },
    }
    exclusive_write(out_dir / "reproduction-note.json",
                    json.dumps(note, indent=2) + "\n")

    print("\n".join([
        "Reproduce paper-selected obs figure (CPU-only, read-only raw inputs)",
        f"repo root: {root}",
        f"output dir: {out_dir}",
        f"selected subset: blocks {list(PAPER_SELECTED_BLOCKS)} of "
        f"{NEW_CELLS_REL} ({len(subset)} cells)",
        f"derived five-block mean overhead: "
        f"{derived['arms'][1]['mean_overhead_pct']:.6f}% "
        f"(baseline mean {derived['arms'][0]['throughput_tok_s_mean']:.6f} tok/s)",
        f"retained ten-block extension: "
        f"{note['retained_ten_block_extension']['mean_overhead_pct']:.6f}% "
        f"mean over {note['retained_ten_block_extension']['cells']} cells",
    ]))

    run_plot(root, OBS_PLOT, [
        "--new-cells", str(cells_path),
        "--new-summary", str(summary_path),
        "--data-output", str(out_dir / "obs-with-array-data.json"),
        "--output-prefix", str(out_dir / "obs-overhead-with-array"),
    ])
    run_plot(root, PANEL_PLOT, [
        "--output-prefix", str(out_dir / "matched-policy-panels"),
    ])
    print("wrote (obs figure): obs-with-array-data.json, "
          "obs-overhead-with-array.pdf/.png")
    print("wrote (seven panels): matched-policy-panels.pdf/.png")
    print("all outputs are outside docs/paper and the raw campaign dirs; "
          "raw cells.json/summary.json were not modified")


if __name__ == "__main__":
    main()
