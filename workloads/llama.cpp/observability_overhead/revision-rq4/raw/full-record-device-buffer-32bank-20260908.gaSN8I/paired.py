#!/usr/bin/env python3
"""Complete five rotating comparisons, retaining the successful first cell."""
import json
import os
import sys
from pathlib import Path

output = Path(__file__).resolve().parent
here = output.parents[1]
sys.path.insert(0, str(here))
import run_table1_perf as run

if (output / "paired_cells.json").exists():
    raise SystemExit("Comparison already started; resume its unfinished cells, do not replay it")
args = run.parse_args([
    "--bpftime-root", "/home/yunwei37/workspace/gpu/bpftime-auto-warp",
    "--bpftime-build-dir", "/home/yunwei37/workspace/gpu/bpftime-auto-warp/build-auto-warp-575",
])
args.wait_for_probe_exit = True
args.timeout_s = None
args.auto_warp_transport = "3"
os.environ["BPFTIME_GPU_WARP_HOOK_CALL_COUNT"] = "0"
ring = here / "raw/table1-original-ring-encoded-20260908.iVdbES/gpubpf_tool_build/kernelretsnoop"
arms = ("gpu_local", "baseline", "transport3")
first = json.loads((output / "result.json").read_text())
first.update(block=1, arm="gpu_local", retained_initial_cell=True, cell_root=".")
cells = [first]
(output / "paired_cells.json").write_text(json.dumps(cells, indent=2) + "\n")
for block in range(1, 6):
    offset = (block - 1) % len(arms)
    for arm in arms[offset:] + arms[:offset]:
        if block == 1 and arm == "gpu_local":
            continue
        target = output / f"pair-{block:02d}-{arm}"
        target.mkdir(exist_ok=False)
        mode = {"gpu_local": "auto_warp_off", "baseline": "baseline", "transport3": "auto_warp_on"}[arm]
        tools = {"kernelretsnoop": output if arm == "gpu_local" else ring}
        print(f"START block={block} arm={arm}", flush=True)
        row = run.run_arm_cell(mode, block, args, target, tools, None)
        row.update(block=block, arm=arm, cell_root=target.name)
        cells.append(row)
        (output / "paired_cells.json").write_text(json.dumps(cells, indent=2) + "\n")
        print(f"DONE block={block} arm={arm} rc={row.get('returncode')} tok_s={row.get('throughput_tok_s')}", flush=True)
