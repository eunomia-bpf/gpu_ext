#!/usr/bin/env python3
"""Invoke the existing prefill runner once with the opt-in SoA collector."""
import json
import os
import sys
from pathlib import Path

output = Path(__file__).resolve().parent
sys.path.insert(0, str(output.parents[1]))
import run_table1_perf as run

args = run.parse_args([
    "--bpftime-root", "/home/yunwei37/workspace/gpu/bpftime-auto-warp",
    "--bpftime-build-dir", "/home/yunwei37/workspace/gpu/bpftime-auto-warp/build-auto-warp-575",
])
args.wait_for_probe_exit = True
args.timeout_s = None
args.auto_warp_transport = "3"
os.environ["BPFTIME_GPU_WARP_HOOK_CALL_COUNT"] = "0"
row = run.run_arm_cell("auto_warp_off", 1, args, output,
                       {"kernelretsnoop": output}, None)
row.update(source_commit="e6b049bc", layout="32-bank field-major SoA full records",
           scope="first SoA prefill; no paired overhead claim")
(output / "result.json").write_text(json.dumps(row, indent=2) + "\n")
print(json.dumps({key: row.get(key) for key in (
    "returncode", "throughput_tok_s", "elapsed_s", "bench_error")}), flush=True)
