"""Thin same-frontend launcher; reuse frozen requests, telemetry and safety."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import run_moe_head_to_head as base

HERE = Path(__file__).resolve().parent
MODES = ("native-off", "paper-native", "paper-bpf")
STORE = HERE / "raw/head-to-head-575/preflight/expert-store"
OLD_CORRECTNESS = HERE / "raw/head-to-head-575-lossless/preflight/moe_infinity_075"
DRIVER_STAGE = Path("/opt/gpubpf/modules/575.57.08/gpreempt-e7d46fa5-6.15.11")


def emit(message):
    print(json.dumps({"utc_ns": time.time_ns(), "progress": message}), flush=True)


def admit(port):
    safety = base.safety_snapshot()
    base.validate_pre_server_safety(safety)
    if safety["gpu"]["driver"] != "575.57.08" or os.uname().release != "6.15.11-061511-generic":
        raise base.GateError("paper run requires the coordinated 575/6.15 stack")
    if not base.port_is_free(port):
        raise base.GateError("server port is already in use")
    source = base.git_revision(base.MOE_SOURCE, base.EXPECTED_MOE_COMMIT,
                               allow_instrumentation=True, paper_activation=True)
    files = [HERE / name for name in ("paper_policy.py", "paper_server.py",
             "paper-activation.patch", "run_paper_policy.py", "prompts.json")]
    files += list(base.MOE_SOURCE.glob("moe_infinity/_store*.so"))
    files += [base.EXTENSION / ".output" / name for name in (
        "libmoe_expert_policy.so", "moe_expert_policy_rank.bin",
        "moe_expert_policy_scored.bin", "moe_expert_policy_match.bin")]
    return {"safety": safety, "source": source,
            "files": [base.file_metadata(path) for path in files],
            "driver_stage_declared_by_coordinator": str(DRIVER_STAGE),
            "driver_stage_module": base.file_metadata(DRIVER_STAGE / "nvidia-uvm.ko"),
            "models": base.verify_model_artifacts()}


def interrupt_warnings():
    return [line for line in base.run_checked(["sudo", "-n", "dmesg", "--color=never"]).splitlines()
            if "Going over RM unhandled interrupt threshold" in line]


def launch(mode, output, port, verify):
    argv, cwd = base.server_command("moe_infinity_075", port, output, STORE)
    argv[4:6] = [str(HERE / "paper_server.py")]
    env = base.controlled_environment("moe_infinity_075", cuda129_triton=True)
    artifacts = base.EXTENSION / ".output"
    env.update(MOE_REVISION_POLICY=mode, MOE_REVISION_VERIFY="1" if verify else "0",
               MOE_EXPERT_POLICY_LIBRARY=str(artifacts / "libmoe_expert_policy.so"),
               MOE_EXPERT_RANK_CODE=str(artifacts / "moe_expert_policy_rank.bin"),
               MOE_EXPERT_SCORED_CODE=str(artifacts / "moe_expert_policy_scored.bin"),
               MOE_EXPERT_MATCH_CODE=str(artifacts / "moe_expert_policy_match.bin"))
    base.atomic_write_json(output / "launch.json", {"argv": argv, "cwd": str(cwd), "env": env})
    log = (output / "server.log").open("x")
    process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    return process, log


def validate_activation(mode, state):
    if mode == "native-off":
        if state["dispatcher"]["mode"] != 0:
            raise base.GateError("native-off unexpectedly enabled policy")
        return
    controller, dispatcher = state["controller"], state["dispatcher"]
    for key in ("matched_predictions", "completed_requests", "prefetch_candidates_selected"):
        if controller[key] <= 0:
            raise base.GateError(f"paper controller did not engage {key}")
    for key in ("prefetch_completed", "prefetch_bytes", "eviction_selections"):
        if dispatcher[key] <= 0:
            raise base.GateError(f"paper dispatcher did not engage {key}")
    if controller["rank_mismatches"] or controller["match_mismatches"] or dispatcher["eviction_mismatches"]:
        raise base.GateError("same-snapshot native/BPF policy mismatch")
    if mode == "paper-bpf" and not (
        controller["rank_calls"] > 0 and controller["bpf_match_calls"] > 0
        and dispatcher["bpf_eviction_calls"] > 0):
        raise base.GateError("not all three real BPF programs engaged")


def canary(mode, output, port):
    output.mkdir(parents=True, exist_ok=False)
    lease = base.LeaseSet.acquire()
    before = None
    process = log = telemetry = telemetry_log = None
    result = {"protocol": "paper-v3-same-frontend-canary-1", "mode": mode,
              "execution_domain": "host-ubpf-jit", "performance_result": False}
    try:
        admission = admit(port)
        before = admission["safety"]
        base.atomic_write_json(output / "admission.json", admission)
        result["interrupt_warnings_before"] = interrupt_warnings()
        emit("finite MoEMLP row/accumulation numerical canary")
        result["numerical"] = base.run_row_chunking_numerical_gate()
        base.atomic_write_json(output / "numerical.json", result["numerical"])
        process, log = launch(mode, output, port, verify=True)
        emit(f"{mode}: model loading PID {process.pid}")
        base.wait_ready(process, port, output / "server.log", 900)
        telemetry, telemetry_log, telemetry_path = base.start_gpu_telemetry(output)
        records = json.loads((HERE / "prompts.json").read_text())["records"]
        old = json.loads((OLD_CORRECTNESS / "result.json").read_text())
        responses = []
        for index in (0, 1):
            emit(f"{mode}: full 512+64 canary request {index + 1}/2")
            response = base.nonstream_completion("moe_infinity_075", port,
                       records[index]["prompt_token_ids"], output / f"request-{index}.json", timeout=600)
            golden = old["warmup"]["text"] if index == 0 else old["goldens"][0]
            if response["text"] != golden:
                raise base.GateError(f"same-frontend exact output mismatch at canary request {index}")
            state = base.http_json(port, "/revision/activation/drain", {}, timeout=600)
            base.atomic_write_json(output / f"activation-{index}.json", state)
            responses.append(response)
        validate_activation(mode, state)
        result.update(responses=responses, activation=state, exact_old_frontend_outputs=True)
        base.stop_exact_process(telemetry)
        telemetry_log.close()
        telemetry = telemetry_log = None
        result["telemetry"] = base.validate_gpu_telemetry(telemetry_path, allow_fixed_power_cap=True)
        result["passed"] = True
    except BaseException as exc:
        result.update(passed=False, error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        try:
            if telemetry is not None:
                base.stop_exact_process(telemetry)
            if telemetry_log is not None:
                telemetry_log.close()
            if process is not None:
                base.stop_owned_process_group(process)
                result["server_exit_code"] = process.returncode
            if log is not None:
                log.close()
            if before is not None:
                after = base.wait_for_post_server_safety(before)
                base.atomic_write_json(output / "safety.json", {"before": before, "after": after, "passed": True})
                result["interrupt_warnings_after"] = interrupt_warnings()
                if result["interrupt_warnings_after"] != result["interrupt_warnings_before"]:
                    raise base.GateError("new RM unhandled interrupt warning during canary")
            if process is not None and process.returncode != 0:
                raise base.GateError(f"server teardown exited {process.returncode}")
        except BaseException as exc:
            result.update(passed=False, cleanup_error=str(exc))
            raise
        finally:
            base.atomic_write_json(output / "result.json", result)
            lease.close()
    emit(f"{mode}: canary passed with real policy engagement and clean teardown")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["canary"])
    parser.add_argument("--mode", choices=MODES, default="paper-bpf")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18230)
    args = parser.parse_args()
    canary(args.mode, args.output.resolve(), args.port)
