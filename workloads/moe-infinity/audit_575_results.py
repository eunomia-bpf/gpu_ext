#!/usr/bin/env python3
"""Read-only raw-evidence audit of completed 575 timing blocks (no CUDA)."""

import argparse
import json
import math
from pathlib import Path

import run_575_head_to_head as current
import run_moe_head_to_head as base


def require(condition, message):
    if not condition:
        raise base.GateError(message)


def audit_stream(path, request, golden, config):
    raw = path.read_bytes()
    texts, finishes, usages, done = [], [], [], 0
    frames = 0
    for line in raw.splitlines():
        if not line.startswith(b"data: "):
            continue
        frames += 1
        if line == b"data: [DONE]":
            done += 1
            continue
        payload = json.loads(line[6:])
        if isinstance(payload.get("usage"), dict):
            usages.append(payload["usage"])
        for choice in payload.get("choices", []):
            texts.append(choice.get("text") or "")
            if choice.get("finish_reason") is not None:
                finishes.append(choice["finish_reason"])
    require(done == 1 and finishes[-1:] == ["length"], f"incomplete SSE: {path}")
    require("".join(texts) == request["text"] == golden, f"SSE output mismatch: {path}")
    require(len(raw) == request["raw_sse_bytes"] and frames == len(request["frames"]),
            f"SSE byte/frame count mismatch: {path}")
    if config == "moe_infinity_075":
        require(frames == 65, f"MoE stream lost one of its 64 token frames: {path}")
    if config != "moe_infinity_075":
        require(bool(usages) and usages[-1] == request["usage"], f"SSE usage mismatch: {path}")
        require(usages[-1]["prompt_tokens"] == 512 and usages[-1]["completion_tokens"] == 64,
                f"SSE token count mismatch: {path}")
    times = [request[key] for key in ("start_ns", "first_text_ns", "done_ns", "eof_ns")]
    require(times == sorted(times), f"non-monotonic request timestamps: {path}")
    require(request["ttft_ms"] == (times[1] - times[0]) / 1e6
            and request["e2e_ms"] == (times[3] - times[0]) / 1e6,
            f"request duration mismatch: {path}")


def audit(preflight, timing):
    checked = current.load_preflight(preflight)
    moe = checked["results"]["moe_infinity_075"]
    parity = moe["stream_parity"]
    moe_path = preflight / checked.get("cell_directories", {}).get("moe_infinity_075", "moe_infinity_075")
    for sequence, (prompt, request) in enumerate(zip(parity["prompt_order"], parity["requests"]), 1):
        audit_stream(moe_path / f"parity-{sequence:02d}-prompt-{prompt}.sse", request,
                     moe["goldens"][prompt - 1], "moe_infinity_075")
    session = current.read_json(timing / "session.json")
    require(session["runtime_files"] == checked["runtime_files"], "session runtime differs")
    require(session["schedule"] == current.read_json(base.SCHEDULE), "frozen schedule differs")
    blocks = []
    for scheduled in session["schedule"]["attempts"]:
        directory = timing / f"attempt-{scheduled['attempt']:02d}"
        if not (directory / "block.json").is_file():
            continue
        block = current.read_json(directory / "block.json")
        if not block.get("valid"):
            continue
        require(block["configuration_order"] == scheduled["configuration_order"], "order differs")
        require(set(block["results"]) == set(base.CONFIGS), "incomplete four-cell block")
        for config, result in block["results"].items():
            cell = directory / config
            require(current.read_json(cell / "result.json") == result, f"result differs: {cell}")
            safety = current.read_json(cell / "safety.json")
            require(safety["passed"] is True, f"cleanup failed: {cell}")
            base.validate_post_server_safety(safety["before"], safety["after"])
            base.validate_log(cell / "server.log")
            require(base.validate_gpu_telemetry(cell / "gpu-telemetry.csv", allow_fixed_power_cap=True)
                    == result["gpu_telemetry"], f"telemetry differs: {cell}")
            require(base.validate_measured_engagement(config, result["engagement_before"],
                                                      result["engagement_after"], current_deployment=True)
                    == result["engagement_delta"], f"engagement differs: {cell}")
            require(result["prompt_order"] == scheduled["prompt_order"]
                    and len(result["requests"]) == 8 and result["verified_output_tokens"] == 512,
                    f"request workload differs: {cell}")
            for sequence, (prompt, request) in enumerate(zip(result["prompt_order"], result["requests"]), 1):
                audit_stream(cell / f"request-{sequence:02d}-prompt-{prompt}.sse", request,
                             checked["results"][config]["goldens"][prompt - 1], config)
            duration = (result["block_end_ns"] - result["block_start_ns"]) / 1e9
            require(duration == result["duration_s"] and math.isclose(
                512 / duration, result["output_throughput_tokens_per_s"], rel_tol=1e-12),
                f"throughput calculation differs: {cell}")
        blocks.append(block)
    return {"protocol": current.PROTOCOL, "audited_complete_blocks": len(blocks),
            "audited_cells": len(blocks) * 4, "audited_streams": len(blocks) * 32,
            "verified_output_tokens": len(blocks) * 2048,
            "full_five_block_experiment": len(blocks) == 5,
            "descriptive": current.descriptive_summary(blocks),
            "analysis": base.analyze_valid_blocks(blocks)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, default=current.DEFAULT_OUTPUT / "preflight")
    parser.add_argument("--timing", type=Path, default=current.DEFAULT_OUTPUT / "timing")
    args = parser.parse_args()
    print(json.dumps(audit(args.preflight, args.timing), indent=2))
