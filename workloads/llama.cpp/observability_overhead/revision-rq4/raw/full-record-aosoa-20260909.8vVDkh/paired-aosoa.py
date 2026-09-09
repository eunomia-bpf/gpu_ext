#!/usr/bin/env python3
"""Five rotating AoSoA/SoA/baseline comparisons for the 32-slot grouped SoA
(full-record-device-buffer LAYOUT=aosoa) layout.

Copy this file into the new campaign raw directory
(revision-rq4/raw/full-record-aosoa-*), next to the copied grouped-SoA
collector binary renamed kernelretsnoop, then run it from there. All 15
cells are new: no completed cell of the older SoA/AoS campaigns is
overwritten or rerun, and the frozen SoA arm binary is reused, not
rebuilt.

Arms:
  aosoa    - 32-bank 32-slot grouped SoA (AoSoA) GPU-local full records
             (the kernelretsnoop binary in this campaign directory),
             auto_warp off, transport 3.
  soa      - measured 32-bank field-major SoA GPU-local full records,
             frozen binary from raw/full-record-soa-20260908.HWfuRS,
             auto_warp off, transport 3.
  baseline - uninstrumented pp512 control.
"""
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
if not (output / "kernelretsnoop").is_file():
    raise SystemExit(f"missing grouped-SoA kernelretsnoop binary under {output}")
args = run.parse_args([
    "--bpftime-root", "/home/yunwei37/workspace/gpu/bpftime-auto-warp",
    "--bpftime-build-dir", "/home/yunwei37/workspace/gpu/bpftime-auto-warp/build-auto-warp-575",
])
args.wait_for_probe_exit = True
args.timeout_s = None
args.auto_warp_transport = "3"
os.environ["BPFTIME_GPU_WARP_HOOK_CALL_COUNT"] = "0"
soa_dir = here / "raw/full-record-soa-20260908.HWfuRS"
if not (soa_dir / "kernelretsnoop").is_file():
    raise SystemExit(f"missing frozen SoA kernelretsnoop under {soa_dir}")
arms = ("aosoa", "soa", "baseline")
cells = []
(output / "paired_cells.json").write_text(json.dumps(cells, indent=2) + "\n")
for block in range(1, 6):
    offset = (block - 1) % len(arms)
    for arm in arms[offset:] + arms[:offset]:
        target = output / f"pair-{block:02d}-{arm}"
        target.mkdir(exist_ok=False)
        mode = {"aosoa": "auto_warp_off", "soa": "auto_warp_off",
                "baseline": "baseline"}[arm]
        tools = {"kernelretsnoop": output if arm == "aosoa" else soa_dir}
        print(f"START block={block} arm={arm}", flush=True)
        row = run.run_arm_cell(mode, block, args, target, tools, None)
        row.update(block=block, arm=arm, cell_root=target.name)
        cells.append(row)
        (output / "paired_cells.json").write_text(json.dumps(cells, indent=2) + "\n")
        print(f"DONE block={block} arm={arm} rc={row.get('returncode')} tok_s={row.get('throughput_tok_s')}", flush=True)
