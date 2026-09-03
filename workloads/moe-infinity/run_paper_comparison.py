"""Five paired blocks: current frontend, paper-native, identical paper-BPF.

GPU timing is serialized; the coordinator must keep heavy CPU builds stopped.
Every cell starts a fresh server, performs the same excluded warmup, then eight
frozen 512+64 requests. The primary wall window includes the final prefetch drain.
No shadow selector/oracle runs in timed cells. Failed attempts remain on disk.
"""
from __future__ import annotations

import argparse
import http.client
import itertools
import json
import math
from pathlib import Path
import random
import signal
import statistics
import time

import run_moe_head_to_head as base
import run_paper_policy as paper

PROTOCOL = "paper-v3-same-frontend-paired-1"
BLOCKS = 5
SEED = 20260903
CONFIG = "moe_infinity_075"


def schedule(seed=SEED):
    rng = random.Random(seed)
    orders = list(itertools.permutations(paper.MODES))
    rng.shuffle(orders)
    result = []
    for number, modes in enumerate(orders[:BLOCKS], 1):
        prompts = list(range(1, 9))
        rng.shuffle(prompts)
        result.append({"block": number, "modes": list(modes), "prompts": prompts})
    return result


def reject_build_contention():
    processes = base.run_checked(["ps", "-eo", "pid=,stat=,comm="]).splitlines()
    builds = {"cc1", "cc1plus", "nvcc", "cicc", "ptxas", "ninja", "ld", "ld.gold", "ld.lld"}
    active = [line.strip() for line in processes if line.split()[-1] in builds
              and line.split()[1][0] not in "TZt"]
    if active:
        raise base.GateError(f"heavy compilation overlaps GPU timing: {active}")


def runtime_inventory(admission):
    """Ordinary revisions and file metadata; never a content digest."""
    files = list(admission["files"])
    paths = [Path(__file__), Path(base.__file__),
             base.MOE_PYTHON, base.MOE_SOURCE / "moe_infinity/entrypoints/openai/revision_server.py",
             base.MOE_SOURCE / "moe_infinity/entrypoints/openai/api_server_v2.py"]
    files.extend(base.file_metadata(path) for path in paths)
    return {"files": files, "source": admission["source"], "models": admission["models"],
            "driver_stage": admission["driver_stage_declared_by_coordinator"],
            "driver_stage_module": admission["driver_stage_module"],
            "prompts": base.file_metadata(base.PROMPTS),
            "goldens": base.file_metadata(paper.OLD_CORRECTNESS / "result.json")}


def stream_request(port, token_ids, golden, raw_path):
    """Strict SSE lifecycle, preserving even a partial/failed response."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    payload = base.completion_payload(CONFIG, token_ids, True)
    result = {"frames": [], "request_payload": payload,
              "start_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)}
    fragments = []
    first = done = finish = None
    try:
        with raw_path.open("xb") as raw:
            connection.request("POST", "/v1/completions",
                body=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            result["http_status"] = response.status
            if response.status != 200:
                raw.write(response.read())
                raise base.GateError(f"SSE HTTP {response.status}")
            while line := response.readline():
                now = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                raw.write(line)
                raw.flush()
                stripped = line.strip()
                if not stripped.startswith(b"data: "):
                    continue
                if done is not None:
                    raise base.GateError("SSE frame after DONE")
                if stripped == b"data: [DONE]":
                    done = now
                    result["frames"].append({"timestamp_ns": now, "done": True})
                    continue
                if finish is not None:
                    raise base.GateError("SSE token frame after finish")
                value = json.loads(stripped[6:])
                choices = value.get("choices", [])
                if len(choices) != 1 or choices[0].get("index") != 0:
                    raise base.GateError("SSE expected exactly one choice at index 0")
                choice = choices[0]
                piece = choice.get("text") or ""
                if piece:
                    first = now if first is None else first
                    fragments.append(piece)
                if choice.get("finish_reason") is not None:
                    if finish is not None:
                        raise base.GateError("duplicate SSE finish")
                    finish = choice["finish_reason"]
                result["frames"].append({"timestamp_ns": now, "payload_bytes": len(stripped[6:])})
        eof = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        text = "".join(fragments)
        result.update(first_text_ns=first, done_ns=done, eof_ns=eof,
                      text=text, finish_reason=finish, raw_sse_bytes=raw_path.stat().st_size)
        if first is None or done is None or finish != "length" or len(result["frames"]) != 65:
            raise base.GateError("SSE requires 64 token frames, one DONE and length finish")
        if text != golden:
            raise base.GateError("SSE exact same-frontend golden mismatch")
        result.update(ttft_ms=(first - result["start_ns"]) / 1e6,
                      e2e_ms=(eof - result["start_ns"]) / 1e6, passed=True)
        return result
    except BaseException as exc:
        result.update(passed=False, error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        connection.close()
        base.atomic_write_json(raw_path.with_suffix(".json"), result)


def activation_delta(mode, before, after):
    paper.validate_activation(mode, after)
    if before["mode"] != mode or after["mode"] != mode:
        raise base.GateError("activation mode changed")
    if mode == "native-off":
        if before["controller"] or after["controller"]:
            raise base.GateError("native baseline unexpectedly has EAMC state")
        if any(after["dispatcher"][key] for key in
               ("bpf_eviction_calls", "eviction_selections", "prefetch_submitted")):
            raise base.GateError("native baseline unexpectedly activated paper decisions")
        return {}
    expected_mode = 2 if mode == "paper-bpf" else 1
    if before["dispatcher"]["mode"] != expected_mode or after["dispatcher"]["mode"] != expected_mode:
        raise base.GateError("native/BPF dispatcher mismatch")
    controller = base.counter_delta(before["controller"], after["controller"],
        ("completed_requests", "matched_predictions", "prefetch_candidates_selected",
         "rank_calls", "bpf_match_calls", "aborted_requests"))
    dispatcher = base.counter_delta(before["dispatcher"], after["dispatcher"],
        ("prefetch_completed", "prefetch_bytes", "eviction_selections", "bpf_eviction_calls",
         "prefetch_hits", "prefetch_wasted", "prefetch_wasted_bytes"))
    if controller["completed_requests"] != 8 or controller["aborted_requests"] != 0:
        raise base.GateError(f"paper request accounting mismatch: {controller}")
    if any(controller[k] <= 0 for k in ("matched_predictions", "prefetch_candidates_selected")):
        raise base.GateError("no measured-window EAMC prediction/prefetch selection")
    if any(dispatcher[k] <= 0 for k in ("prefetch_completed", "prefetch_bytes", "eviction_selections")):
        raise base.GateError("no measured-window prefetch/eviction engagement")
    if mode == "paper-bpf":
        if not (controller["rank_calls"] > 0 and controller["bpf_match_calls"] > 0
                and dispatcher["bpf_eviction_calls"] > 0):
            raise base.GateError("one of three BPF programs did not engage during timing")
    elif controller["rank_calls"] or controller["bpf_match_calls"] or dispatcher["bpf_eviction_calls"]:
        raise base.GateError("native policy unexpectedly called BPF")
    return {"controller": controller, "dispatcher": dispatcher}


def run_cell(mode, output, port, prompt_order, driver_stage, expected_runtime=None):
    output.mkdir(parents=True, exist_ok=False)
    result = {"protocol": PROTOCOL, "mode": mode, "passed": False,
              "prompt_order": prompt_order, "shadow_verification": False,
              "execution_domain": "host-ubpf-jit" if mode == "paper-bpf" else "native",
              "requests": []}
    server = log = telemetry = telemetry_log = None
    admission = None
    failure = None
    try:
        reject_build_contention()
        admission = paper.admit(port, driver_stage)
        admission["runtime_inventory"] = runtime_inventory(admission)
        base.atomic_write_json(output / "admission.json", admission)
        if expected_runtime is not None and admission["runtime_inventory"] != expected_runtime:
            raise base.GateError("runtime changed within the paired campaign")
        result["interrupt_warnings_before"] = paper.interrupt_warnings()
        prompts = json.loads(base.PROMPTS.read_text())
        old = json.loads((paper.OLD_CORRECTNESS / "result.json").read_text())
        server, log = paper.launch(mode, output, port, verify=False)
        paper.emit(f"{mode}: cold model loading PID {server.pid}")
        base.wait_ready(server, port, output / "server.log", 900)
        result["identity"] = base.check_server_identity(CONFIG, port, prompts, output / "server.log")
        cold = base.http_json(port, "/revision/activation")
        if cold["controller"].get("eamc_entries", 0) or cold["controller"].get("completed_requests", 0):
            raise base.GateError("fresh server inherited EAMC state")
        result["activation_cold"] = cold
        result["warmup"] = base.nonstream_completion(CONFIG, port,
            prompts["records"][0]["prompt_token_ids"], output / "warmup.json", timeout=600)
        if result["warmup"]["text"] != old["warmup"]["text"]:
            raise base.GateError("excluded warmup differs from same-frontend golden")
        activation_before = base.http_json(port, "/revision/activation/drain", {}, timeout=600)
        before = base.engagement_snapshot(CONFIG, port, output, server.pid, current_deployment=True)
        if any(member["affinity"] != list(range(8)) for member in before["process_io"]["members"]):
            raise base.GateError("server process tree escaped CPU 0-7")
        reject_build_contention()
        telemetry, telemetry_log, telemetry_path = base.start_gpu_telemetry(output)
        started = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        for seq, number in enumerate(prompt_order, 1):
            reject_build_contention()
            paper.emit(f"{mode}: measured request {seq}/8, prompt {number}")
            request = stream_request(port, prompts["records"][number]["prompt_token_ids"],
                old["goldens"][number - 1], output / f"request-{seq:02d}-prompt-{number}.sse")
            result["requests"].append(request)
        last_eof = result["requests"][-1]["eof_ns"]
        activation_after = base.http_json(port, "/revision/activation/drain", {}, timeout=600)
        ended = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        after = base.engagement_snapshot(CONFIG, port, output, server.pid, current_deployment=True)
        if any(member["affinity"] != list(range(8)) for member in after["process_io"]["members"]):
            raise base.GateError("server process tree escaped CPU 0-7 during timing")
        reject_build_contention()
        base.stop_exact_process(telemetry)
        telemetry_log.close()
        telemetry = telemetry_log = None
        result.update(activation_before=activation_before, activation_after=activation_after,
            activation_delta=activation_delta(mode, activation_before, activation_after),
            engagement_before=before, engagement_after=after,
            engagement_delta=base.validate_measured_engagement(CONFIG, before, after, current_deployment=True),
            gpu_telemetry=base.validate_gpu_telemetry(telemetry_path, allow_fixed_power_cap=True))
        duration = (ended - started) / 1e9
        result.update(block_start_ns=started, block_end_ns=ended, duration_s=duration,
            final_drain_s=(ended - last_eof) / 1e9, verified_output_tokens=512,
            output_throughput_tokens_per_s=512 / duration,
            request_only_tokens_per_s=512000 / sum(r["e2e_ms"] for r in result["requests"]),
            first_text_ttft_median_ms=statistics.median(r["ttft_ms"] for r in result["requests"]),
            first_text_ttft_max_ms=max(r["ttft_ms"] for r in result["requests"]),
            e2e_median_ms=statistics.median(r["e2e_ms"] for r in result["requests"]),
            e2e_max_ms=max(r["e2e_ms"] for r in result["requests"]))
    except BaseException as exc:
        failure = exc
        result.update(error_type=type(exc).__name__, error=str(exc))
    finally:
        cleanup_errors = []
        # Finish every cleanup even if an earlier one fails; retain primary error.
        for resource, action in ((telemetry, base.stop_exact_process), (telemetry_log, lambda x: x.close()),
                                 (server, base.stop_owned_process_group), (log, lambda x: x.close())):
            if resource is not None:
                try:
                    action(resource)
                except BaseException as exc:
                    cleanup_errors.append(f"{type(exc).__name__}: {exc}")
        if server is not None:
            result["server_exit_code"] = server.returncode
            if server.returncode != 0:
                cleanup_errors.append(f"server exited {server.returncode}")
        if admission is not None:
            try:
                result["safety_after"] = base.wait_for_post_server_safety(admission["safety"])
                result["interrupt_warnings_after"] = paper.interrupt_warnings()
                if result["interrupt_warnings_after"] != result.get("interrupt_warnings_before"):
                    raise base.GateError("new RM unhandled interrupt warning")
            except BaseException as exc:
                cleanup_errors.append(f"{type(exc).__name__}: {exc}")
        if log is not None:
            try:
                base.validate_log(output / "server.log")
            except BaseException as exc:
                cleanup_errors.append(f"{type(exc).__name__}: {exc}")
        result["cleanup_errors"] = cleanup_errors
        result["passed"] = failure is None and not cleanup_errors
        base.atomic_write_json(output / "result.json", result)
    if failure is not None:
        raise failure
    if cleanup_errors:
        raise base.GateError(f"cell cleanup failed: {cleanup_errors}")
    return result


def analyze(blocks):
    valid = []
    invalid = []
    counts = {b["block"]: sum(x["block"] == b["block"] for x in blocks) for b in blocks}
    for block in blocks:
        cells = block.get("cells", [])
        ok = (block.get("passed") is True and counts[block["block"]] == 1
              and len(cells) == 3 and {c.get("mode") for c in cells} == set(paper.MODES)
              and all(c.get("passed") is True and c.get("shadow_verification") is False
                      and c.get("verified_output_tokens") == 512 and len(c.get("requests", [])) == 8
                      and c.get("prompt_order") == block["prompts"]
                      and math.isfinite(c.get("output_throughput_tokens_per_s", 0))
                      and c.get("output_throughput_tokens_per_s", 0) > 0 for c in cells))
        (valid if ok else invalid).append(block)
    result = {"protocol": PROTOCOL, "required_blocks": BLOCKS, "valid_blocks": len(valid),
              "invalid_block_numbers": [b["block"] for b in invalid],
              "complete": len(valid) == BLOCKS and {b["block"] for b in valid} == set(range(1, BLOCKS + 1)),
              "primary": "512 tokens / full eight-request wall window including final prefetch drain",
              "ttft_definition": "first nonempty text, not first model token", "modes": {}, "paired": {}}
    if not valid:
        return result
    by_mode = {mode: [next(c for c in b["cells"] if c["mode"] == mode) for b in valid] for mode in paper.MODES}
    for mode, rows in by_mode.items():
        result["modes"][mode] = {key: statistics.median(r[key] for r in rows) for key in
            ("output_throughput_tokens_per_s", "first_text_ttft_median_ms", "e2e_median_ms", "final_drain_s")}
    rng = random.Random(SEED + 1)
    samples = [[rng.randrange(len(valid)) for _ in valid] for _ in range(10000)]
    for numerator, denominator in (("paper-bpf", "paper-native"), ("paper-native", "native-off"), ("paper-bpf", "native-off")):
        logs = [math.log(a["output_throughput_tokens_per_s"] / b["output_throughput_tokens_per_s"])
                for a, b in zip(by_mode[numerator], by_mode[denominator])]
        ratio = math.exp(statistics.mean(logs))
        ci = None
        if len(valid) >= 2:
            boot = sorted(math.exp(statistics.mean(logs[i] for i in sample)) for sample in samples)
            ci = [boot[249], boot[9749]]
        result["paired"][f"{numerator}/{denominator}"] = {"geometric_throughput_ratio": ratio,
            "paired_block_bootstrap_ci95": ci, "interpretation": "ratio > 1 favors numerator"}
    return result


def run(output, port, driver_stage, seed=SEED, max_new_blocks=BLOCKS):
    lease = base.LeaseSet.acquire()
    try:
        output.mkdir(parents=True, exist_ok=True)
        manifest_path = output / "manifest.json"
        inventory = runtime_inventory(paper.admit(port, driver_stage))
        manifest = {"protocol": PROTOCOL, "schedule": schedule(seed), "seed": seed,
                    "driver_stage": str(driver_stage.resolve()), "required_blocks": BLOCKS,
                    "runtime_inventory": inventory,
                    "warmup_prompt": 0, "measured_input_output_tokens": [512, 64],
                    "memory_budget": 0.75, "kv_blocks": 128, "timing_shadow_verification": False,
                    "cooldown": "no extra sleep; fresh model load + identical excluded warmup per cell"}
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise base.GateError("resume protocol/driver/schedule differs; use a new output directory")
        else:
            if any(output.iterdir()):
                raise base.GateError("nonempty output has no matching manifest")
            base.atomic_write_json(manifest_path, manifest)
        completed = []
        new_blocks = 0
        for item in manifest["schedule"]:
            attempts = sorted(output.glob(f"block-{item['block']:02d}-attempt-*"))
            passed = []
            for attempt in attempts:
                path = attempt / "result.json"
                if path.exists():
                    previous = json.loads(path.read_text())
                    if previous.get("passed"):
                        from paper_result_audit import audit_block
                        passed.append(audit_block(attempt, item, inventory))
            if len(passed) > 1:
                raise base.GateError("duplicate successful block; refusing selective resume")
            if passed:
                completed.extend(passed)
                continue
            if new_blocks >= max_new_blocks:
                continue
            attempt = output / f"block-{item['block']:02d}-attempt-{len(attempts) + 1:02d}"
            attempt.mkdir(exist_ok=False)
            block = {**item, "passed": False, "cells": []}
            try:
                for mode in item["modes"]:
                    paper.emit(f"block {item['block']}/{BLOCKS}, {mode}")
                    block["cells"].append(run_cell(mode, attempt / mode, port, item["prompts"], driver_stage, inventory))
                block["passed"] = True
                completed.append(block)
                new_blocks += 1
            except BaseException as exc:
                block.update(error_type=type(exc).__name__, error=str(exc))
                raise
            finally:
                base.atomic_write_json(attempt / "result.json", block)
                base.atomic_write_json(output / "analysis.json", analyze(completed))
        summary = analyze(completed)
        base.atomic_write_json(output / "analysis.json", summary)
        paper.emit(f"valid paired blocks: {summary['valid_blocks']}/{BLOCKS}; complete={summary['complete']}")
    finally:
        lease.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18230)
    parser.add_argument("--driver-stage", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-new-blocks", type=int, default=BLOCKS)
    args = parser.parse_args()
    if not 1 <= args.max_new_blocks <= BLOCKS:
        parser.error("--max-new-blocks must be 1..5; the full protocol always requires five")
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    run(args.output.resolve(), args.port, args.driver_stage, args.seed, args.max_new_blocks)


if __name__ == "__main__":
    main()
