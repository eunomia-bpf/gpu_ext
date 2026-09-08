#!/usr/bin/env python3
"""Separate invocation-count observation; not a performance campaign."""
import json
import re
import sys
from pathlib import Path

here = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(here))
import run_strict_warp_map_scaling as runner

output = Path(__file__).resolve().parent
build = Path("/home/yunwei37/workspace/gpu/bpftime-auto-warp/build-auto-warp-575")
original_environment = runner.attached_environment

def diagnostic_environment(*args, **kwargs):
    loader, agent = original_environment(*args, **kwargs)
    loader["BPFTIME_GPU_WARP_HOOK_CALL_COUNT"] = "1"
    agent["BPFTIME_GPU_WARP_HOOK_CALL_COUNT"] = "1"
    return loader, agent

runner.attached_environment = diagnostic_environment
records = []
with runner.ReadOnlyLeases():
    for blocks, work in ((2, 0), (4, 0), (8, 0), (1, 8), (1, 32), (1, 128)):
        for mode in ("off", "on"):
            directory = output / f"b{blocks}-w{work}-{mode}"
            runner.run_attached("shared_update", directory, build, 128,
                                0, 2, 93000 + len(records), blocks, work, mode)
            text = (directory / "application.log").read_text() + (directory / "agent.log").read_text()
            counts = [int(value) for value in re.findall(r"warp hook call count: (\d+)", text)]
            record = {"blocks": blocks, "work": work, "auto": mode,
                      "warmup": 0, "launches": 2,
                      "observed_counter_records": counts,
                      "source": directory.name,
                      "scope": "diagnostic, not performance"}
            records.append(record)
            print(json.dumps(record), flush=True)
            if not counts:
                raise RuntimeError("Application completed, but no actual counter readback was emitted")
