#!/usr/bin/env python3
"""Continue five rotating SoA/baseline/AoS comparisons, retaining the
successful first SoA cell from raw/full-record-soa-20260908.HWfuRS.

Arms:
  soa      - 32-bank field-major SoA GPU-local full records
             (raw/full-record-soa-20260908.HWfuRS/kernelretsnoop),
             auto_warp off, transport 3.
  baseline - uninstrumented pp512 control.
  aos      - measured 32-bank record-major AoS GPU-local full records
             (raw/full-record-device-buffer-32bank-20260908.gaSN8I/kernelretsnoop),
             auto_warp off, transport 3.

The original full-record ring (transport 3) is not rerun.
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
args = run.parse_args([
    "--bpftime-root", "/home/yunwei37/workspace/gpu/bpftime-auto-warp",
    "--bpftime-build-dir", "/home/yunwei37/workspace/gpu/bpftime-auto-warp/build-auto-warp-575",
])
args.wait_for_probe_exit = True
args.timeout_s = None
args.auto_warp_transport = "3"
os.environ["BPFTIME_GPU_WARP_HOOK_CALL_COUNT"] = "0"
soa_dir = here / "raw/full-record-soa-20260908.HWfuRS"
aos_dir = here / "raw/full-record-device-buffer-32bank-20260908.gaSN8I"
if not (soa_dir / "kernelretsnoop").is_file():
    raise SystemExit(f"missing SoA kernelretsnoop under {soa_dir}")
if not (aos_dir / "kernelretsnoop").is_file():
    raise SystemExit(f"missing AoS kernelretsnoop under {aos_dir}")
arms = ("soa", "baseline", "aos")
first = json.loads((soa_dir / "result.json").read_text())
first.update(block=1, arm="soa", retained_initial_cell=True,
             cell_root="../" + soa_dir.name)
cells = [first]
(output / "paired_cells.json").write_text(json.dumps(cells, indent=2) + "\n")
for block in range(1, 6):
    offset = (block - 1) % len(arms)
    for arm in arms[offset:] + arms[:offset]:
        if block == 1 and arm == "soa":
            continue
        target = output / f"pair-{block:02d}-{arm}"
        target.mkdir(exist_ok=False)
        mode = {"soa": "auto_warp_off", "baseline": "baseline",
                "aos": "auto_warp_off"}[arm]
        tools = {"kernelretsnoop": soa_dir if arm == "soa" else aos_dir}
        print(f"START block={block} arm={arm}", flush=True)
        row = run.run_arm_cell(mode, block, args, target, tools, None)
        row.update(block=block, arm=arm, cell_root=target.name)
        cells.append(row)
        (output / "paired_cells.json").write_text(json.dumps(cells, indent=2) + "\n")
        print(f"DONE block={block} arm={arm} rc={row.get('returncode')} tok_s={row.get('throughput_tok_s')}", flush=True)
