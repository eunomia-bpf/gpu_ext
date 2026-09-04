"""Read-only, fail-closed resume audit of a complete paired paper-policy block.

This checks the retained observations, not merely producer ``passed`` flags.
Runtime identity uses ordinary file metadata, source revisions and inventories.
It is not cryptographic attestation: coordinated fabrication of raw observations
cannot be detected from those observations alone. No GPU or subprocess is used.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import statistics

import run_moe_head_to_head as base
import run_paper_policy as paper


class AuditError(base.GateError):
    """A previously successful attempt cannot safely be reused."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise AuditError(f"nonfinite JSON constant: {value}")


def _loads(data):
    return json.loads(data, object_pairs_hook=_pairs, parse_constant=_invalid_constant)


def _artifact(root, name):
    path = root / name
    require(path.is_file() and not path.is_symlink(), f"missing/nonregular artifact: {path}")
    require(path.resolve().is_relative_to(root.resolve()), f"artifact outside cell: {path}")
    return path


def _json(root, name):
    return _loads(_artifact(root, name).read_text(encoding="utf-8"))


def _number(value, name, *, integer=False, positive=False):
    require(type(value) in (int, float) and math.isfinite(value), f"invalid number {name}")
    require(value >= 0 and (not positive or value > 0), f"negative/zero {name}")
    require(not integer or int(value) == value, f"noninteger {name}")
    return value


def _ns(value, name):
    require(type(value) is int, f"timestamp is not integer: {name}")
    return _number(value, name, positive=True)


def _equal_number(actual, expected, name):
    _number(actual, name)
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9),
            f"recomputed {name} differs: {actual} != {expected}")


def _success(result, label):
    require(result.get("passed") is True, f"{label}: not successful")
    require(not any(key in result for key in ("error", "error_type", "cleanup_error")),
            f"{label}: retained failure despite passed flag")


def _metadata(entry):
    require(isinstance(entry, dict) and Path(entry["path"]).is_absolute(), "invalid runtime path")
    require(base.file_metadata(Path(entry["path"])) == entry,
            f"runtime file metadata changed: {entry['path']}")


def _runtime(expected):
    require(isinstance(expected["files"], list) and expected["files"], "empty runtime inventory")
    paths = [entry["path"] for entry in expected["files"]]
    require(len(paths) == len(set(paths)), "duplicate runtime file path")
    for entry in expected["files"]:
        _metadata(entry)
    for name in ("driver_stage_module", "prompts", "goldens"):
        _metadata(expected[name])
    models = expected["models"]
    view = Path(models["model_view"])
    require(sorted(p.name for p in view.iterdir()) == sorted(models["view_members"]),
            "model view inventory changed")
    for name, size in models["all_sizes"].items():
        require(Path(name).name == name, "model inventory contains nonlocal name")
        for directory in (view, Path(models["hf_snapshot"])):
            require((directory / name).stat().st_size == size, f"model size changed: {directory / name}")
    prompts = _loads(Path(expected["prompts"]["path"]).read_text(encoding="utf-8"))
    goldens = _loads(Path(expected["goldens"]["path"]).read_text(encoding="utf-8"))
    require(len(prompts["records"]) == 9 and len(goldens["goldens"]) == 8,
            "frozen corpus must contain one warmup plus eight measured prompts")
    for record in prompts["records"]:
        ids = record["prompt_token_ids"]
        require(isinstance(ids, list) and len(ids) == 512 and
                all(type(value) is int and value >= 0 for value in ids), "invalid 512-token prompt")
    require(all(isinstance(text, str) and text for text in goldens["goldens"]), "invalid goldens")
    return prompts, goldens


def _launch(cell, mode):
    launch = _json(cell, "launch.json")
    argv = launch["argv"]
    require(isinstance(argv, list) and argv.count("--port") == 1, "ambiguous launch port")
    port = int(argv[argv.index("--port") + 1])
    require(1 <= port <= 65535, "invalid launch port")
    expected, cwd = base.server_command("moe_infinity_075", port, cell, paper.STORE)
    expected[4:6] = [str(paper.HERE / "paper_server.py")]
    env = base.controlled_environment("moe_infinity_075", cuda129_triton=True)
    artifacts = base.EXTENSION / ".output"
    env.update(MOE_REVISION_POLICY=mode, MOE_REVISION_VERIFY="0",
               MOE_EXPERT_POLICY_LIBRARY=str(artifacts / "libmoe_expert_policy.so"),
               MOE_EXPERT_RANK_CODE=str(artifacts / "moe_expert_policy_rank.bin"),
               MOE_EXPERT_SCORED_CODE=str(artifacts / "moe_expert_policy_scored.bin"),
               MOE_EXPERT_MATCH_CODE=str(artifacts / "moe_expert_policy_match.bin"))
    require(launch == {"argv": expected, "cwd": str(cwd), "env": env},
            "launch differs from exact timed same-frontend/no-shadow configuration")


def _request(cell, seq, number, ids, golden, embedded):
    name = f"request-{seq:02d}-prompt-{number}"
    result = _json(cell, name + ".json")
    require(result == embedded, f"embedded request differs: {name}")
    _success(result, name)
    require(result["request_payload"] == base.completion_payload("moe_infinity_075", ids, True),
            f"actual request payload differs from frozen prompt: {name}")
    require(result["http_status"] == 200, "non-200 SSE")
    raw = _artifact(cell, name + ".sse").read_bytes()
    require(len(raw) == result["raw_sse_bytes"], "raw SSE byte size differs")
    frames = result["frames"]
    require(isinstance(frames, list) and len(frames) == 65, "SSE metadata must contain 64 frames and DONE")
    start = _ns(result["start_ns"], "request.start")
    previous = start
    fragments, first, done, finish, frame_index = [], None, None, None, 0
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith(b":"):
            continue
        require(line.startswith(b"data: "), "unexpected SSE line")
        require(done is None and frame_index < 65, "SSE data after DONE/extra frames")
        frame = frames[frame_index]
        now = _ns(frame["timestamp_ns"], "frame.timestamp")
        require(now >= previous, "SSE timestamps moved backwards")
        previous = now
        frame_index += 1
        data = line[6:]
        if data == b"[DONE]":
            require(frame_index == 65 and finish == "length", "early/missing finish or DONE")
            require(frame == {"timestamp_ns": now, "done": True}, "DONE metadata differs")
            done = now
            continue
        require(finish is None, "SSE token frame after finish")
        require(frame == {"timestamp_ns": now, "payload_bytes": len(data)}, "SSE frame size differs")
        value = _loads(data)
        choices = value.get("choices")
        require(isinstance(choices, list) and len(choices) == 1 and
                choices[0].get("index") == 0, "SSE choice/index differs")
        choice = choices[0]
        piece = choice.get("text")
        require(piece is None or isinstance(piece, str), "non-string SSE text")
        if piece:
            first = now if first is None else first
            fragments.append(piece)
        if choice.get("finish_reason") is not None:
            require(frame_index == 64 and choice["finish_reason"] == "length", "invalid SSE finish")
            finish = choice["finish_reason"]
    require(frame_index == 65 and first is not None and done is not None, "incomplete SSE")
    require("".join(fragments) == result["text"] == golden, "raw SSE/golden text differs")
    require(result["finish_reason"] == finish and result["first_text_ns"] == first and
            result["done_ns"] == done, "SSE lifecycle metadata differs")
    eof = _ns(result["eof_ns"], "request.eof")
    require(eof >= done > start, "SSE EOF/window differs")
    _equal_number(result["ttft_ms"], (first - start) / 1e6, "request.ttft_ms")
    _equal_number(result["e2e_ms"], (eof - start) / 1e6, "request.e2e_ms")
    return result


def _activation(result, mode, runner):
    cold, before, after = (result[f"activation_{phase}"] for phase in ("cold", "before", "after"))
    for state in (cold, before, after):
        require(state["mode"] == mode, "activation mode changed")
        require(state["algorithm"] == "arxiv-2401.14361v3-reimplementation" and
                state["features"] == "shared-float64-EAMC-cosine-and-probability", "algorithm/features differ")
        for group in ("controller", "dispatcher"):
            for key, value in state[group].items():
                _number(value, f"activation.{group}.{key}", integer=True)
        require(state["dispatcher"]["mode"] == paper.MODES.index(mode), "actual dispatcher arm differs")
        for key in ("rank_mismatches", "match_mismatches", "aborted_requests", "active_request_traces"):
            require(state["controller"].get(key, 0) == 0, f"nonzero controller {key}")
        require(state["dispatcher"]["eviction_mismatches"] == 0, "eviction mismatch")
        dispatcher = state["dispatcher"]
        require(dispatcher["prefetch_completed"] == dispatcher["prefetch_hits"] +
                dispatcher["prefetch_wasted"] + dispatcher["prefetch_unused_resident"],
                "prefetch completion conservation failed")
    require(not cold["controller"].get("eamc_entries", 0) and
            not cold["controller"].get("completed_requests", 0), "noncold EAMC")
    if mode != "native-off":
        require(before["controller"]["completed_requests"] == 1 and
                after["controller"]["completed_requests"] == 9, "warmup/measured request boundary differs")
    else:
        for state in (cold, before, after):
            require(not state["controller"] and not any(value for key, value in state["dispatcher"].items()
                    if key != "mode"), "baseline contains paper-policy state/counters")
    delta = runner.activation_delta(mode, before, after)
    require(result["activation_delta"] == delta, "activation delta differs from raw snapshots")


def _engagement(result, *, expected_generated_tokens=512):
    before, after = result["engagement_before"], result["engagement_after"]
    for state in (before, after):
        members = state["process_io"]["members"]
        require(members and all(member["affinity"] == list(range(8)) for member in members),
                "missing process tree/CPU affinity differs")
        for key in ("read_bytes", "cpu_time_s"):
            _number(state["process_io"][key], key, integer=key == "read_bytes")
        for group in ("revision", "metrics"):
            for key, value in state["moe"][group].items():
                _number(value, f"{group}.{key}", integer=key != "exposed_fetch_seconds_total")
        require(state["moe"]["revision"]["kv_cache_num_blocks"] == 128 and
                state["moe"]["metrics"]["moe_kv_cache_total_blocks"] == 128, "KV capacity differs")
    for key in ("read_bytes", "cpu_time_s"):
        require(after["process_io"][key] >= before["process_io"][key], "process accounting decreased")
    delta = base.validate_measured_engagement(
        "moe_infinity_075", before, after, current_deployment=True,
        expected_generated_tokens=expected_generated_tokens)
    require(result["engagement_delta"] == delta, "engine/metrics delta differs from snapshots")


def _log(cell, mode, result):
    path = _artifact(cell, "server.log")
    base.validate_log(path)
    text = path.read_text(encoding="utf-8")
    ready = re.findall(r"moe_expert_policy_ready: backend=ubpf-jit kind=(\w+) abi=1 instructions=(\d+)", text)
    stats = re.findall(r"moe_expert_policy_stats: backend=ubpf-jit kind=(\w+) calls=(\d+) "
                       r"candidates=(\d+) selected=(\d+) no_victim=(\d+) errors=(\d+)", text)
    if mode != "paper-bpf":
        require(not ready and not stats, "native cell unexpectedly loaded BPF policy")
        return
    kinds = {"paper_rank", "paper_match", "paper_scored"}
    require(len(ready) == 3 and {kind for kind, _ in ready} == kinds and
            all(int(count) > 0 for _, count in ready), "three actual JIT programs not ready")
    require(len(stats) == 3 and {row[0] for row in stats} == kinds, "missing final JIT program stats")
    after = result["activation_after"]
    calls = {"paper_rank": after["controller"]["rank_calls"],
             "paper_match": after["controller"]["bpf_match_calls"],
             "paper_scored": after["dispatcher"]["bpf_eviction_calls"]}
    for kind, count, candidates, selected, no_victim, errors in stats:
        require(int(count) == calls[kind] and int(count) > 0 and int(candidates) > 0 and
                int(selected) > 0 and int(errors) == 0, f"JIT {kind} shutdown counters disagree")


def _cell(cell, mode, order, embedded, expected, prompts, goldens, runner):
    require(cell.is_dir() and not cell.is_symlink(), f"missing/aliased cell: {cell}")
    result = _json(cell, "result.json")
    require(result == embedded, "block/cell result disagree")
    _success(result, mode)
    require(result["protocol"] == runner.PROTOCOL and result["mode"] == mode and
            result["prompt_order"] == order, "cell protocol/mode/order differs")
    require(result["shadow_verification"] is False and result["execution_domain"] ==
            ("host-ubpf-jit" if mode == "paper-bpf" else "native"), "execution/shadow mode differs")
    admission = _json(cell, "admission.json")
    require(admission["runtime_inventory"] == expected, "cell runtime inventory differs")
    for key in ("source", "models", "driver_stage_module"):
        require(admission[key] == expected[key], f"admission {key} differs")
    require(admission["driver_stage_declared_by_coordinator"] == expected["driver_stage"], "driver stage differs")
    require(admission["files"] and len({entry["path"] for entry in admission["files"]}) == len(admission["files"])
            and all(entry in expected["files"] for entry in admission["files"]), "admission files differ")
    _launch(cell, mode)
    ids = {model.get("id") for model in result["identity"]["models"]["data"]}
    require("gpt-oss-120b" in ids or base.HF_REVISION in ids, "served model identity differs")
    warmup = result["warmup"]
    validated = base.validate_completion_response(_json(cell, "warmup.json"), 512)
    require(all(warmup[key] == value for key, value in validated.items()), "warmup response differs")
    require(warmup["text"] == goldens["warmup"]["text"], "warmup golden differs")
    for key, expected_value in (("prompt_tokens", 512), ("completion_tokens", 64)):
        _number(warmup["usage"][key], key, integer=True)
        require(warmup["usage"][key] == expected_value, "warmup usage differs")
    warm_start, warm_end = _ns(warmup["start_ns"], "warmup.start"), _ns(warmup["end_ns"], "warmup.end")
    require(warm_end > warm_start, "invalid warmup window")
    _equal_number(warmup["e2e_ms"], (warm_end - warm_start) / 1e6, "warmup.e2e_ms")
    expected_names = {f"request-{seq:02d}-prompt-{number}.{suffix}" for seq, number in enumerate(order, 1)
                      for suffix in ("sse", "json")}
    require({path.name for path in cell.glob("request-*")} == expected_names, "request artifacts differ from schedule")
    require(len(result["requests"]) == 8, "not eight measured requests")
    requests = [_request(cell, seq, number, prompts["records"][number]["prompt_token_ids"],
                         goldens["goldens"][number - 1], result["requests"][seq - 1])
                for seq, number in enumerate(order, 1)]
    started, ended = _ns(result["block_start_ns"], "block.start"), _ns(result["block_end_ns"], "block.end")
    previous = started
    require(warm_end <= started, "warmup overlaps measured interval")
    for request in requests:
        require(request["start_ns"] >= previous, "requests overlap/reorder or precede measurement")
        previous = request["eof_ns"]
    require(ended >= previous and ended > started, "invalid block/drain window")
    require(result["verified_output_tokens"] == 512, "not 512 verified output tokens")
    duration = (ended - started) / 1e9
    values = {"duration_s": duration, "final_drain_s": (ended - previous) / 1e9,
              "output_throughput_tokens_per_s": 512 / duration,
              "request_only_tokens_per_s": 512000 / sum(r["e2e_ms"] for r in requests),
              "first_text_ttft_median_ms": statistics.median(r["ttft_ms"] for r in requests),
              "first_text_ttft_max_ms": max(r["ttft_ms"] for r in requests),
              "e2e_median_ms": statistics.median(r["e2e_ms"] for r in requests),
              "e2e_max_ms": max(r["e2e_ms"] for r in requests)}
    for key, value in values.items():
        _equal_number(result[key], value, key)
    _activation(result, mode, runner)
    _engagement(result)
    telemetry_path = _artifact(cell, "gpu-telemetry.csv")
    rows = [line.split(",") for line in telemetry_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(rows) >= 2 and len(rows[0]) >= 7 and len(set(rows[0])) == len(rows[0]) and
            all(len(row) == len(rows[0]) for row in rows[1:]), "incomplete GPU telemetry rows/header")
    for row in rows[1:]:
        for value in row[1:6]:
            _number(float(value.strip().split()[0]), "GPU telemetry sample")
    telemetry = base.validate_gpu_telemetry(telemetry_path, allow_fixed_power_cap=True)
    require(telemetry == result["gpu_telemetry"], "raw GPU telemetry differs")
    require(result["cleanup_errors"] == [] and type(result["server_exit_code"]) is int and
            result["server_exit_code"] == 0, "unclean server teardown")
    base.validate_pre_server_safety(admission["safety"])
    base.validate_post_server_safety(admission["safety"], result["safety_after"])
    require(admission["safety"]["gpu"]["driver"] == result["safety_after"]["gpu"]["driver"] == "575.57.08",
            "safety driver differs")
    require(result["interrupt_warnings_before"] == result["interrupt_warnings_after"], "new RM interrupt warning")
    _log(cell, mode, result)
    return result


def audit_block(attempt_path, expected_schedule_item, expected_runtime):
    """Re-read and return one complete block, or raise AuditError without writes."""
    # The runner imports this function only during resume; importing its pure
    # accounting helpers here avoids a module-level dependency cycle.
    import run_paper_comparison as runner

    attempt = Path(attempt_path)
    try:
        require(attempt.is_dir() and not attempt.is_symlink(), "missing/aliased attempt")
        item = expected_schedule_item
        require(type(item["block"]) is int and 1 <= item["block"] <= runner.BLOCKS and
                len(item["modes"]) == 3 and set(item["modes"]) == set(paper.MODES) and
                sorted(item["prompts"]) == list(range(1, 9)), "invalid expected schedule")
        prompts, goldens = _runtime(expected_runtime)
        block = _json(attempt, "result.json")
        _success(block, "block")
        require(all(block[key] == item[key] for key in ("block", "modes", "prompts")), "block schedule differs")
        require(len(block["cells"]) == 3, "block lacks all three cells")
        previous = 0
        for index, mode in enumerate(item["modes"]):
            cell = _cell(attempt / mode, mode, item["prompts"], block["cells"][index],
                         expected_runtime, prompts, goldens, runner)
            require(cell["warmup"]["start_ns"] >= previous, "three modes ran out of order/overlapped")
            previous = cell["block_end_ns"]
        return block
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError,
            ArithmeticError, base.GateError) as exc:
        raise AuditError(f"resume audit refused {attempt}: {exc}") from exc
