#!/usr/bin/env python3
"""Existing Off/Native runner, expanded input grid, isolated benchmark build."""
import subprocess
import sys
from pathlib import Path
here = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(here))
import run_strict_warp_map_scaling as runner
out = Path(__file__).resolve().parent
runner.APPLICATION = out / "warp-map-bench"
runner.PTX = out / "warp-map-bench.ptx"
runner.SWEEP_ARMS = ("native", "off")
runner.SWEEP_LOADER_MODES = {"off": "shared_update"}
with runner.ReadOnlyLeases():
    with (out / "build.log").open("x") as log:
        common = ["/usr/local/cuda-12.9/bin/nvcc", "-std=c++17", "-O2", "-lineinfo"]
        commands = [common + ["-cudart", "shared", "-gencode", "arch=compute_120,code=sm_120", "-gencode", "arch=compute_120,code=compute_120", "-Xcompiler=-Wall,-Wextra", "-Xlinker", "--build-id=none", str(out / "warp_map_bench.cu"), "-o", str(runner.APPLICATION)],
                    common + ["--ptx", "-gencode", "arch=compute_120,code=compute_120", str(out / "warp_map_bench.cu"), "-o", str(runner.PTX)]]
        for command in commands:
            print(command, file=log, flush=True)
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
for work in (0, 128, 1024):
    runner.sweep_settings = lambda kind, work=work: [{"blocks": b, "work": work} for b in (128, 512, 2048, 8192)]
    sys.argv = [__file__, "--sweep", "blocks", "--output", str(out / f"work-{work}"),
        "--bpftime-root", "/home/yunwei37/workspace/gpu/bpftime-auto-warp",
        "--bpftime-build", "/home/yunwei37/workspace/gpu/bpftime-auto-warp/build-auto-warp-575"]
    runner.main()
