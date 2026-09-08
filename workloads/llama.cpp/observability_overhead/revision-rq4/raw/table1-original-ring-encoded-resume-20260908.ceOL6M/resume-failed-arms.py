#!/usr/bin/env python3
"""Use existing runner helpers to finish only the failed attached cells."""
import json
import os
import sys
from pathlib import Path

here = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(here))
import run_table1_perf as run

output = Path(__file__).resolve().parent
original = here / "raw/table1-original-ring-encoded-20260908.iVdbES"
args = run.parse_args([
    "--auto-warp-three-arm", "--auto-warp-task", "kernelretsnoop",
    "--auto-warp-transport", "2", "--blocks", "10",
    "--bpftime-root", "/home/yunwei37/workspace/gpu/bpftime-auto-warp",
    "--bpftime-build-dir", "/home/yunwei37/workspace/gpu/bpftime-auto-warp/build-auto-warp-575",
    "--output-dir", str(output),
])
args.nvbit_tool = None
os.environ["BPFTIME_GPU_WARP_HOOK_CALL_COUNT"] = "0"
cells = []
for row in json.loads((original / "cells.json").read_text()):
    if row["arm"] == "baseline":
        row["reused_from"] = str(original)
        if row.get("log"):
            row["log"] = str(original / row["log"])
        cells.append(row)
tools = {"kernelretsnoop": original / "gpubpf_tool_build/kernelretsnoop"}
for block in range(1, 11):
    for arm in run.block_schedule(block, run.AUTO_WARP_ARMS):
        if arm == "baseline":
            continue
        print(f"block={block} arm={arm}", flush=True)
        row = {"block": block, "arm": arm, "throughput_tok_s": None,
               "returncode": None}
        try:
            row.update(run.run_arm_cell(arm, block, args, output, tools, None))
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            cells.append(row)
            run.write_records(output, cells, run.AUTO_WARP_ARMS, "auto_warp_three_arm")
            raise
        cells.append(row)
        run.write_records(output, cells, run.AUTO_WARP_ARMS, "auto_warp_three_arm")
print("Completed 20 repaired attached cells; retained the original 10 measured baselines", flush=True)
