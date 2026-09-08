#!/usr/bin/env python3
"""Reuse the existing runner at 16/32/64 CTAs and at fixed 64 CTAs."""
import sys
from pathlib import Path
here = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(here))
import run_strict_warp_map_scaling as runner
output = Path(__file__).resolve().parent
runner.SWEEP_CTA_BLOCKS = (16, 32, 64)
original_settings = runner.sweep_settings
def settings(kind):
    if kind == "work":
        return [{"blocks": 64, "work": value} for value in (8, 32, 128)]
    return original_settings(kind)
runner.sweep_settings = settings
for kind in ("blocks", "work"):
    sys.argv = [__file__, "--sweep", kind, "--output", str(output / kind),
        "--bpftime-root", "/home/yunwei37/workspace/gpu/bpftime-auto-warp",
        "--bpftime-build", "/home/yunwei37/workspace/gpu/bpftime-auto-warp/build-auto-warp-575"]
    runner.main()
