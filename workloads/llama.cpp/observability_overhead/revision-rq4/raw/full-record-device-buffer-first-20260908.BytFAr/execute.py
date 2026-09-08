#!/usr/bin/env python3
"""First actual pp512 run of the built thread-major full-record buffer."""
import json
import os
import sys
from pathlib import Path

output = Path(__file__).resolve().parent
here = output.parents[1]
sys.path.insert(0, str(here))
import run_table1_perf as run

args = run.parse_args([
    "--bpftime-root", "/home/yunwei37/workspace/gpu/bpftime-auto-warp",
    "--bpftime-build-dir", "/home/yunwei37/workspace/gpu/bpftime-auto-warp/build-auto-warp-575",
])
args.wait_for_probe_exit = True
args.timeout_s = None
args.auto_warp_transport = "3"
os.environ["BPFTIME_GPU_WARP_HOOK_CALL_COUNT"] = "0"
tools = {"kernelretsnoop": here / "raw/full-record-device-buffer-build-20260908.sLXIqf"}
row = run.run_arm_cell("auto_warp_off", 1, args, output, tools, None)
row.update(source_commit="743ef571", layout="thread-major GPU-local full records",
           scope="single initial prefill measurement; no paired overhead claim")
(output / "result.json").write_text(json.dumps(row, indent=2) + "\n")
print(json.dumps(row, indent=2), flush=True)
