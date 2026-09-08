#!/usr/bin/env python3
"""Execute five new transport pairs through the existing Table 1 helpers."""
import os
import sys
from pathlib import Path

output = Path(__file__).resolve().parent
here = output.parents[1]
sys.path.insert(0, str(here))
import run_table1_perf as run

args = run.parse_args([
    "--auto-warp-three-arm", "--auto-warp-task", "kernelretsnoop",
    "--auto-warp-transport", "3", "--blocks", "5",
    "--bpftime-root", "/home/yunwei37/workspace/gpu/bpftime-auto-warp",
    "--bpftime-build-dir", "/home/yunwei37/workspace/gpu/bpftime-auto-warp/build-auto-warp-575",
    "--output-dir", str(output),
])
args.nvbit_tool = None
args.wait_for_probe_exit = True
args.timeout_s = None
os.environ["BPFTIME_GPU_WARP_HOOK_CALL_COUNT"] = "0"
tools = {"kernelretsnoop": here / "raw/table1-original-ring-encoded-20260908.iVdbES/gpubpf_tool_build/kernelretsnoop"}
arms = ("transport2", "transport3")
run.SOURCE_CONTRACTS["transport_pairs"] = "Same original per-thread probe: encoded AoS versus transposed SoA transport; both auto-execution env ON; no baseline replay."
cells = []
for block in range(1, 6):
    for mode in ((2, 3) if block % 2 else (3, 2)):
        args.auto_warp_transport = mode
        cell_output = output / f"pair-{block:02d}" / f"transport{mode}"
        cell_output.mkdir(parents=True, exist_ok=True)
        print(f"START block={block} transport={mode}", flush=True)
        row = {"block": block, "arm": f"transport{mode}", "cell_root": str(cell_output.relative_to(output))}
        row.update(run.run_arm_cell("auto_warp_on", block, args, cell_output, tools, None))
        cells.append(row)
        run.write_records(output, cells, arms, "transport_pairs")
        print(f"DONE block={block} transport={mode} returncode={row.get('returncode')} tok_s={row.get('throughput_tok_s')}", flush=True)
