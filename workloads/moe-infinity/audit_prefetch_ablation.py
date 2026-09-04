"""Independent raw audit for one predictive-prefetch factorial block."""
from __future__ import annotations

import json
import math
from pathlib import Path

import paper_result_audit as raw
import run_moe_head_to_head as base
import run_paper_policy as paper
import run_prefetch_ablation as ablation


def require(condition, message):
    if not condition:
        raise base.GateError(message)


def _launch(cell, arm):
    mode, enabled = ablation.ARM_CONFIG[arm]
    observed = json.loads((cell / "launch.json").read_text())
    argv = observed["argv"]
    require(isinstance(argv, list) and argv.count("--port") == 1,
            "ambiguous launch port")
    port = int(argv[argv.index("--port") + 1])
    expected, cwd = base.server_command(ablation.CONFIG, port, cell, paper.STORE)
    expected[4:6] = [str(paper.HERE / "paper_server.py")]
    env = base.controlled_environment(ablation.CONFIG, cuda129_triton=True)
    artifacts = base.EXTENSION / ".output"
    env.update(
        MOE_REVISION_POLICY=mode, MOE_REVISION_VERIFY="0",
        MOE_REVISION_PREFETCH="1" if enabled else "0",
        MOE_EXPERT_POLICY_LIBRARY=str(artifacts / "libmoe_expert_policy.so"),
        MOE_EXPERT_RANK_CODE=str(artifacts / "moe_expert_policy_rank.bin"),
        MOE_EXPERT_SCORED_CODE=str(artifacts / "moe_expert_policy_scored.bin"),
        MOE_EXPERT_MATCH_CODE=str(artifacts / "moe_expert_policy_match.bin"),
    )
    require(observed == {"argv": expected, "cwd": str(cwd), "env": env},
            "launch differs from exact factorial arm")


def audit_block(attempt, planned, runtime_inventory):
    attempt = Path(attempt)
    block = json.loads((attempt / "result.json").read_text())
    require(block.get("passed") is True, "producer block did not pass")
    require(block.get("block") == planned["block"] and
            block.get("arms") == planned["arms"] and
            block.get("prompts") == planned["prompts"],
            "block differs from fixed randomized schedule")
    cells = block.get("cells", [])
    require(len(cells) == 4 and {cell.get("arm") for cell in cells} == set(ablation.ARMS),
            "block does not contain exactly four factorial cells")
    prompts = json.loads(base.PROMPTS.read_text())
    goldens = json.loads((paper.OLD_CORRECTNESS / "result.json").read_text())
    budgets = set()
    for embedded in cells:
        arm = embedded["arm"]
        cell = attempt / arm
        stored = json.loads((cell / "result.json").read_text())
        require(stored == embedded and stored.get("passed") is True,
                f"embedded/stored result mismatch for {arm}")
        admission = json.loads((cell / "admission.json").read_text())
        require(admission.get("runtime_inventory") == runtime_inventory,
                f"runtime inventory mismatch for {arm}")
        _launch(cell, arm)
        warmup = base.validate_completion_response(
            json.loads((cell / "warmup.json").read_text()), 512)
        require(all(stored["warmup"][key] == value for key, value in warmup.items()) and
                stored["warmup"].get("text") == goldens["warmup"]["text"],
                f"excluded warmup correctness failure for {arm}")
        requests = stored.get("requests", [])
        require(len(requests) == ablation.REQUESTS_PER_CELL,
                f"wrong measured request count for {arm}")
        expected_names = {
            f"request-{seq:02d}-prompt-{number}.{suffix}"
            for seq, number in enumerate(planned["prompts"], 1)
            for suffix in ("sse", "json")
        }
        require({path.name for path in cell.glob("request-*")} == expected_names,
                f"request artifacts differ from schedule for {arm}")
        for seq, number in enumerate(planned["prompts"], 1):
            raw._request(cell, seq, number,
                         prompts["records"][number]["prompt_token_ids"],
                         goldens["goldens"][number - 1], requests[seq - 1])
        delta = ablation.activation_delta(
            arm, stored["activation_before"], stored["activation_after"])
        require(delta == stored["activation_delta"],
                f"activation delta was not independently reproducible for {arm}")
        raw._engagement(stored)
        raw._log(cell, ablation.ARM_CONFIG[arm][0], stored)
        telemetry = base.validate_gpu_telemetry(
            cell / "gpu-telemetry.csv", allow_fixed_power_cap=True)
        require(telemetry == stored["gpu_telemetry"],
                f"raw GPU telemetry differs for {arm}")
        budgets.add(delta["cache_budget_bytes"])
        require(stored.get("verified_requests") == 6 and
                stored.get("verified_output_tokens") == 384,
                f"exact correctness total mismatch for {arm}")
        duration = (stored["block_end_ns"] - stored["block_start_ns"]) / 1e9
        require(duration > 0 and math.isclose(duration, stored["duration_s"], rel_tol=0, abs_tol=1e-9),
                f"wall duration mismatch for {arm}")
        require(math.isclose(384 / duration, stored["output_throughput_tokens_per_s"],
                             rel_tol=1e-12, abs_tol=0),
                f"throughput mismatch for {arm}")
        require(stored.get("cleanup_errors") == [] and stored.get("server_exit_code") == 0,
                f"cleanup failure for {arm}")
        base.validate_pre_server_safety(admission["safety"])
        base.validate_post_server_safety(admission["safety"], stored["safety_after"])
        require(stored["interrupt_warnings_before"] == stored["interrupt_warnings_after"],
                f"new RM interrupt warning for {arm}")
    require(len(budgets) == 1, "cache budget differs across factorial cells")
    return block
