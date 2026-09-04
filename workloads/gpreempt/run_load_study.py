#!/usr/bin/env python3
"""Frozen GPreempt load profiles; never builds or changes the driver."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import statistics
import subprocess
import time

import run_three_way as original

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build/load-study"
ARMS, TASKS = original.ARMS, original.TASKS
SCENARIOS = ("be100", "be200", "be_continuous")
LC_KNEE_SCENARIOS = ("lc500", "lc625", "lc800")
STUDIES = ("load", "lc-knee")
SEED = 20260903
safety = original.safety
stop_owned = original.stop_owned


def scenario_loads(study: str, scenario: str) -> tuple[dict, dict]:
    if study == "load" and scenario in SCENARIOS:
        foreground = {"type": "periodic_fifo", "frequency": 100, "priority": 0}
        if scenario == "be_continuous":
            background = {"type": "continuous"}
        else:
            background = {"type": "periodic_fifo",
                          "frequency": 200 if scenario == "be200" else 100, "priority": 0}
        return foreground, background
    if study == "lc-knee" and scenario in LC_KNEE_SCENARIOS:
        return ({"type": "periodic_fifo", "frequency": int(scenario[2:]), "priority": 0},
                {"type": "continuous"})
    raise ValueError("unknown frozen study/scenario")


def make_config(scenario: str, seconds: int, study: str = "load") -> dict:
    if type(seconds) is not int or seconds not in (10, 60):
        raise ValueError("only the frozen 10-second preflight or 60-second study is supported")
    loads = scenario_loads(study, scenario)
    tasks = []
    for role, name, model in zip((0, 1), TASKS, ("vgg", "resnet152")):
        tasks.append({"id": name, "load": loads[role], "client": {
            "name": name, "model_name": model, "priority": role,
            "batch_size": 1, "use_cuda_graph": True, "preprocess_time": 200}})
    return {"time": seconds, "tasks": tasks}


def validate_config(config: dict, scenario: str | None = None, study: str = "load") -> str:
    def same(left, right):
        if type(left) is not type(right):
            return False
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
        if isinstance(left, list):
            return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
        return left == right
    candidates = SCENARIOS if study == "load" else LC_KNEE_SCENARIOS
    for candidate in candidates if scenario is None else (scenario,):
        if same(config, make_config(candidate, config.get("time"), study)):
            return candidate
    raise ValueError("configuration differs from frozen models, roles, rates, graph or preprocessing")


def balanced_orders(items, count: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    permutations = list(itertools.permutations(items))
    rng.shuffle(permutations)
    return [list(item) for item in permutations[:count]]


def latin_orders(items, count: int, seed: int) -> list[list[str]]:
    """Seeded rotations give exact position balance for the three-block knee sweep."""
    if count != len(items):
        raise ValueError("Latin ordering requires one block per item")
    row = list(items)
    random.Random(seed).shuffle(row)
    return [row[index:] + row[:index] for index in range(count)]


def make_plan(mode: str, timeout: int = 240, cooldown: int = 10,
              gdrcopy: Path = original.DEFAULT_GDRCOPY, study: str = "load",
              preflight: Path | None = None) -> dict:
    if (study not in STUDIES or mode not in ("full", "preflight")
            or not 90 <= timeout <= 3500 or not 0 <= cooldown <= 60):
        raise ValueError("invalid mode, timeout or cooldown")
    full = mode == "full"
    knee = study == "lc-knee"
    if preflight is not None and not (knee and full):
        raise ValueError("preflight evidence is only valid for an LC-knee full plan")
    blocks, seconds = ((3 if knee else 5), 60) if full else (1, 10)
    all_scenarios = LC_KNEE_SCENARIOS if knee else SCENARIOS
    scenarios = list(all_scenarios) if full else ["lc800" if knee else "be_continuous"]
    if full:
        scenario_orders = (latin_orders(scenarios, blocks, SEED)
                           if knee else balanced_orders(scenarios, blocks, SEED))
    else:
        scenario_orders = [scenarios]
    arm_orders = {scenario: (latin_orders(ARMS, blocks, SEED + index + 1)
                             if knee and full else balanced_orders(ARMS, blocks, SEED + index + 1))
                  for index, scenario in enumerate(all_scenarios)}
    orders = []
    for block, scenario_order in enumerate(scenario_orders):
        for scenario in scenario_order:
            for arm in arm_orders[scenario][block]:
                orders.append({"cell": len(orders), "block": block, "scenario": scenario, "arm": arm})
    plan = {"schema": "gpreempt_lc_knee_v1" if knee else "gpreempt_load_study_v1",
            "mode": mode, "seed": SEED,
            "blocks": blocks, "scenarios": scenarios, "scenario_orders": scenario_orders,
            "orders": orders, "configs": {s: make_config(s, seconds, study) for s in scenarios},
            "timed_seconds_per_cell": seconds, "required_cells": len(orders),
            "formal_required_blocks": 3 if knee else 5, "cell_timeout_seconds": timeout,
            "cooldown_seconds": cooldown, "gdrcopy_directory": str(gdrcopy.resolve()),
            "flag_transport": "host_mapped", "comparison_variant": "host_mapped_compatibility",
            "clock": "steady_clock", "arrival_phase_ns": 0,
            "p99_definition": "nearest rank, scheduled arrival to synchronized and numerically verified output",
            "p99_population": "all_started_and_verified_including_after_window",
            "window": "[begin_ns,end_ns); no new start at or after end",
            "goodput_definition": "verified_ready_ns < end_ns, divided by configured seconds",
            "incomplete_rule": "report backlog/coverage and conditional completed-request p99; never discard overload",
            "policy_changes": False, "kernel_repetition": 1,
            "statistics": {"estimator": "paired geometric-mean ratios",
                           "interval": "percentile paired-block bootstrap", "draws": 10000,
                           "seed": SEED, "confidence": 0.95, "equivalence_claimed": False},
            "privilege": "all clients and loader run as root without permission changes"}
    if knee:
        plan.update(study=study, evidence_role="supporting",
                    prespecified_lc_rates_rps=[500, 625, 800],
                    background_load="continuous_closed_loop",
                    post_hoc_rate_additions_allowed=False,
                    scope_note="supporting knee evidence only; no rates may be appended after execution",
                    preflight_required=True,
                    preflight_campaign=(str(preflight.resolve()) if preflight is not None else None))
    return plan


def independent_campaign_paths(preflight: Path, full_output: Path) -> bool:
    preflight, full_output = preflight.resolve(), full_output.resolve()
    return (preflight != full_output and preflight not in full_output.parents
            and full_output not in preflight.parents)


def validate_completed_preflight(preflight: Path, full_output: Path | None = None) -> dict:
    """Fail closed unless an independent raw audit accepts the frozen real preflight."""
    preflight = preflight.resolve()
    if full_output is not None and not independent_campaign_paths(preflight, full_output):
        raise ValueError("LC-knee preflight and full output must be independent campaign directories")
    import analyze_load_study as independent
    return independent.validate_completed_preflight(preflight)


def integer(value, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"invalid integer {label}")
    return value


def p99(values: list[int]) -> float | None:
    return sorted(values)[math.ceil(len(values) * .99) - 1] / 1000 if values else None


def unique_rows(log: str, prefix: str) -> dict:
    rows = [json.loads(line[len(prefix):]) for line in log.splitlines() if line.startswith(prefix)]
    if len(rows) != 2 or {row.get("task") for row in rows} != set(TASKS):
        raise ValueError(f"missing, duplicate or unexpected {prefix} task records")
    return {row["task"]: row for row in rows}


def native_priorities(log: str, arm: str) -> dict:
    rows = [dict(re.findall(r"(\w+)=([^\s]+)", line)) for line in log.splitlines()
            if line.startswith("GPREEMPT_LOAD_PRIORITY ")]
    if arm != "native":
        if rows:
            raise ValueError("policy arm unexpectedly used native priority instrumentation")
        return {}
    if len(rows) != 2 or {row.get("task") for row in rows} != set(TASKS):
        raise ValueError("native arm must record both actual stream priorities")
    result = {row["task"]: row for row in rows}
    ranges = set()
    for role, name in enumerate(TASKS):
        row = result[name]
        actual, least, greatest = (int(row[key]) for key in ("actual", "least", "greatest"))
        if int(row["role"]) != role or greatest >= least or actual != (least if role else greatest):
            raise ValueError("native actual stream priority does not match the role/range")
        ranges.add((least, greatest))
    if len(ranges) != 1:
        raise ValueError("native workers disagree on the GPU priority range")
    return result


def parse_report(log: str, config: dict, arm: str, study: str = "load") -> dict:
    validate_config(config, study=study)
    if arm not in ARMS:
        raise ValueError("unknown arm")
    seconds = config["time"]
    decoder, candidates = json.JSONDecoder(), []
    for match in re.finditer(r"(?m)^\{", log):
        try:
            item, _ = decoder.raw_decode(log[match.start():])
        except ValueError:
            continue
        if isinstance(item, dict) and "benchmarkTime(s)" in item:
            candidates.append(item)
    if len(candidates) != 1 or candidates[0].get("benchmarkTime(s)") != seconds:
        raise ValueError("missing unique DISB report with the configured duration")
    report = candidates[0]
    begin = integer(report.get("loadStudyBeginNs"), "suite begin", 1)
    end = integer(report.get("loadStudyEndNs"), "suite end", 1)
    if end - begin != seconds * 1_000_000_000 or report.get("loadStudyClock") != "steady_clock":
        raise ValueError("suite does not expose the frozen common monotonic window")
    checks = unique_rows(log, "GPREEMPT_VALIDATION ")
    loads = unique_rows(log, "GPREEMPT_LOAD_STUDY ")
    priorities = native_priorities(log, arm)
    results = report.get("results", [])
    if len(results) != 2 or {r.get("clientName") for r in results} != set(TASKS):
        raise ValueError("missing both distinct DISB workload results")
    results = {row["clientName"]: row for row in results}
    metrics = {}
    for task in config["tasks"]:
        name, expected = task["id"], task["load"]
        row, check = loads[name], checks[name]
        periodic = expected["type"] == "periodic_fifo"
        interval = 1_000_000_000 // expected["frequency"] if periodic else 0
        offered = (end - begin + interval - 1) // interval if periodic else None
        if (row.get("clock") != "steady_clock" or row.get("phase_ns") != 0
                or row.get("begin_ns") != begin or row.get("end_ns") != end
                or row.get("interval_ns") != interval or row.get("offered") != offered
                or row.get("mode") != ("periodic_fifo" if periodic else "continuous_closed_loop")
                or row.get("request_fields") != "id,scheduled_ns,started_ns,verified_ready_ns"):
            raise ValueError(f"{name}: actual load mode/rate/phase/window differs from config")
        count = integer(row.get("started"), "started")
        integer(row.get("phase_ns"), "arrival phase")
        integer(row.get("interval_ns"), "arrival interval")
        if periodic:
            integer(row.get("offered"), "offered")
        requests = row.get("requests", [])
        if len(requests) != count or (periodic and count > offered):
            raise ValueError("started requests disagree with raw records or offered slots")
        previous = begin
        responses, window_responses = [], []
        for index, request in enumerate(requests):
            if not isinstance(request, list) or len(request) != 4:
                raise ValueError("malformed request row")
            request_id, scheduled, started, verified = [integer(x, "request timestamp/id") for x in request]
            if (request_id != index or scheduled != (begin + index * interval if periodic else started)
                    or not begin <= scheduled <= started < end or verified < started
                    or started < previous):
                raise ValueError("request IDs/order/arrival/admission/completion violate FIFO and common cutoff")
            response = verified - scheduled
            if response <= 0:
                raise ValueError("response must include real positive measured work")
            responses.append(response)
            if verified < end:
                window_responses.append(response)
            previous = verified
        completed, after = len(window_responses), count - len(window_responses)
        if after > 1:
            raise ValueError("serial worker completed more than one request after cutoff")
        analyzers = [a for a in results[name].get("analyzers", []) if a.get("type") == "basic"]
        if len(analyzers) != 1:
            raise ValueError("missing unique basic analyzer")
        analyzer = analyzers[0]
        samples = analyzer.get("requestLatencyNs", [])
        if (integer(analyzer.get("completedRequests"), "service count") != count
                or len(samples) != count or any(type(x) is not int or x <= 0 for x in samples)
                or analyzer.get("latencyDefinition") != "sum_of_original_six_recorded_stages"):
            raise ValueError("six-stage raw count/values differ from measured started requests")
        legacy_rate = analyzer.get("avgThroughput(req/s)")
        if not isinstance(legacy_rate, (int, float)) or not math.isfinite(legacy_rate) or abs(legacy_rate - count / seconds) > 1e-6:
            raise ValueError("legacy report rate is inconsistent; it is not the new window goodput")
        if (integer(check.get("timed_checked"), "timed numerical count") != count
                or integer(check.get("checked"), "all numerical count") != count + 110
                or check.get("atol") != 1e-6 or check.get("rtol") != 1e-4
                or not isinstance(check.get("max_absolute_error"), (float, int))
                or not math.isfinite(check["max_absolute_error"]) or check["max_absolute_error"] < 0):
            raise ValueError("every warmup/calibration/timed output must pass common numerical checks")
        metrics[name] = {
            "offered_requests": offered, "started_requests": count,
            "completed_in_window": completed, "completed_after_window": after,
            "started_unfinished": 0, "never_started_backlog": offered - count if periodic else None,
            "unfinished_offered_at_deadline": offered - completed if periodic else None,
            "window_completion_fraction": completed / offered if periodic else None,
            "completion_coverage": count / offered if periodic else None,
            "goodput_rps": completed / seconds,
            "response_p99_us": p99(responses),
            "mean_response_us": statistics.mean(responses) / 1000 if responses else None,
            "in_window_response_p99_us": p99(window_responses),
            "response_p99_conditional": periodic and count < offered,
            "response_p99_population": "all_started_and_verified_including_after_window",
            "p99_sample_count": count, "p99_less_than_100_samples": count < 100,
            "service_p99_us": p99(samples), "numerics": check}
    return {"metrics": metrics, "report": report, "load_reports": loads, "native_priorities": priorities}


def environment(arm: str, pin: Path, gdrcopy: Path = original.DEFAULT_GDRCOPY) -> dict:
    env = original.environment(arm, pin, gdrcopy)
    env["LD_LIBRARY_PATH"] = env["LD_LIBRARY_PATH"].replace(str(HERE / "build/ninja"), str(BUILD))
    return env


def client_command(arm: str, config: Path) -> list[str]:
    command = original.client_command(arm, config, "host_mapped")
    command[0] = str(BUILD / Path(command[0]).name)
    return command


def runtime_inventory() -> dict:
    paths = [BUILD / name for name in ("baseclient", "gpreemptclient", "libgpreempt.so", "block.cubin")]
    paths += [original.EXTENSION / name for name in
              ("gpreempt_policy", "libgpreempt_bridge.so", "gpreempt_hint.bin")]
    result = {}
    for path in paths:
        info = path.stat()
        if not path.is_file() or info.st_size <= 0:
            raise ValueError(f"missing runtime artifact: {path}")
        result[str(path)] = {"path": str(path), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}
    return result


def build_inventory() -> dict:
    cache = {}
    wanted = {"GPREEMPT_LOAD_STUDY", "GPREEMPT_CUDA_ARCH", "CMAKE_CUDA_ARCHITECTURES",
              "CMAKE_CXX_COMPILER", "CMAKE_C_COMPILER", "CMAKE_CUDA_COMPILER"}
    for line in (BUILD / "CMakeCache.txt").read_text().splitlines():
        if ":" in line and "=" in line:
            key = line.split(":", 1)[0]
            if key in wanted:
                cache[key] = line.split("=", 1)[1]
    if cache.get("GPREEMPT_LOAD_STUDY") != "ON" or cache.get("GPREEMPT_CUDA_ARCH") != "120":
        raise ValueError("load-study build must explicitly enable new instrumentation and sm_120")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE.parents[1], text=True).strip()
    return {"files": runtime_inventory(), "cmake": cache,
            "source_revisions": {"gpu_ext_head": revision, "upstream": "249ee3e",
                                 "expected_driver_port": "849ea75d"},
            "driver_port_note": "expected source revision from coordinated preparation, not inferred from runtime counters"}


def run_cell(directory: Path, specification: dict, config_path: Path, timeout: int,
             gdrcopy: Path = original.DEFAULT_GDRCOPY, expected_inventory: dict | None = None,
             study: str = "load") -> dict:
    config = json.loads(config_path.read_text())
    validate_config(config, specification["scenario"], study)
    arm = specification["arm"]
    if arm not in ARMS:
        raise ValueError("unknown arm")
    directory.mkdir(parents=True, exist_ok=False)
    result = {"status": "failed", **specification, "config": config, "config_path": str(config_path),
              "timed_seconds": config["time"], "flag_transport": "not_used" if arm == "native" else "host_mapped",
              "comparison_variant": "host_mapped_compatibility"}
    before = client = loader = telemetry = None
    streams = []
    pin = Path(f"/sys/fs/bpf/gpreempt-load-{os.getpid()}-{time.monotonic_ns()}")
    try:
        result["runtime_before"] = runtime_inventory()
        if expected_inventory is not None and result["runtime_before"] != expected_inventory:
            raise RuntimeError("runtime artifact inventory changed since campaign preparation")
        before = safety.safety_snapshot()
        safety.validate_pre_server_safety(before)
        if before["gpu"]["driver"] != "575.57.08":
            raise RuntimeError("separately prepared 575 scheduling driver required")
        result["safety_before"] = before
        telemetry, telemetry_stream, telemetry_path = safety.start_gpu_telemetry(directory)
        streams.append(telemetry_stream)
        env = environment(arm, pin, gdrcopy)
        if arm == "bpf_gpreempt":
            loader_stream = (directory / "loader.log").open("x")
            streams.append(loader_stream)
            loader_command = [str(original.EXTENSION / "gpreempt_policy"), "--library",
                              str(original.EXTENSION / "libgpreempt_bridge.so"), "--pin-dir", str(pin),
                              "--duration", str(timeout + 30)]
            result["loader_command"] = loader_command
            loader = subprocess.Popen(loader_command, stdout=loader_stream, stderr=subprocess.STDOUT,
                                      start_new_session=True, env=env)
            deadline = time.monotonic() + 15
            while "gpreempt_policy_ready:" not in (directory / "loader.log").read_text():
                if loader.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("BPF loader did not become ready")
                time.sleep(.1)
        command = client_command(arm, config_path)
        result.update(command=command, environment=env, timeout_seconds=timeout)
        client_stream = (directory / "client.log").open("x")
        streams.append(client_stream)
        start = time.monotonic()
        client = subprocess.Popen(command, stdout=client_stream, stderr=subprocess.STDOUT,
                                  start_new_session=True, env=env)
        while client.poll() is None:
            if time.monotonic() - start >= timeout:
                raise TimeoutError("owned load-study client exceeded its bound")
            if loader is not None and loader.poll() is not None:
                raise RuntimeError("BPF policy exited before client")
            time.sleep(.2)
        result.update(returncode=client.returncode, process_wall_seconds=time.monotonic() - start)
        stop_owned(client)
        stop_owned(loader)
        if client.returncode != 0 or (loader is not None and loader.returncode != 0):
            raise RuntimeError("client or attached policy exited unsuccessfully")
        log = (directory / "client.log").read_text(errors="replace")
        parsed = parse_report(log, config, arm, study)
        safety.atomic_write_json(directory / "request-report.json", parsed.pop("report"))
        safety.atomic_write_json(directory / "load-study-report.json", parsed.pop("load_reports"))
        result.update(parsed)
        loader_log = (directory / "loader.log").read_text(errors="replace") if loader else ""
        result["engagement"] = original.check_engagement(arm, log, loader_log, "host_mapped")
        result["status"] = "passed"
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        errors = []
        for process in (client, loader, telemetry):
            try:
                stop_owned(process)
            except BaseException as error:
                errors.append(str(error))
        for stream in streams:
            stream.close()
        try:
            result["runtime_after"] = runtime_inventory()
            if result.get("runtime_before") != result["runtime_after"]:
                raise RuntimeError("runtime artifact inventory changed during this cell")
            if before is not None:
                result["safety_after"] = safety.wait_for_post_server_safety(before)
            if loader is not None and pin.exists():
                raise RuntimeError(f"owned pins survived loader cleanup: {pin}")
            if telemetry is not None:
                result["telemetry"] = safety.validate_gpu_telemetry(telemetry_path, allow_fixed_power_cap=True)
        except BaseException as error:
            errors.append(str(error))
        if errors:
            result.update(status="failed", cleanup_errors=errors)
        safety.atomic_write_json(directory / "result.json", result)
        if errors:
            raise RuntimeError("; ".join(errors))
    return result


def summarize(results: list[dict], plan: dict) -> dict:
    keys = [(r["block"], r["scenario"], r["arm"]) for r in results]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate cells cannot silently replace an attempt")
    expected = {(r["block"], r["scenario"], r["arm"]) for r in plan["orders"]}
    if not set(keys) <= expected:
        raise ValueError("unexpected cell outside the frozen execution plan")
    passed = {key: row for key, row in zip(keys, results) if row["status"] == "passed"}
    groups = []
    for block, scenario_order in enumerate(plan["scenario_orders"]):
        for scenario in scenario_order:
            if all((block, scenario, arm) in passed for arm in ARMS):
                groups.append({"block": block, "scenario": scenario,
                               "cells": {arm: passed[block, scenario, arm]["metrics"] for arm in ARMS}})
    return {"completed_cells": len(passed), "required_cells": len(expected),
            "valid_paired_groups": len(groups), "paired_groups": groups,
            "complete": set(passed) == expected,
            "formal_complete": (plan["mode"] == "full"
                                and len(expected) == (27 if plan.get("study", "load") == "lc-knee" else 45)
                                and set(passed) == expected),
            "mode": plan["mode"], "independently_audited": False,
            "note": "progress only; final paired intervals require independent raw audit"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "full"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cell-timeout", type=int, default=240)
    parser.add_argument("--cooldown-seconds", type=int, default=10)
    parser.add_argument("--gdrcopy-dir", type=Path, default=original.DEFAULT_GDRCOPY)
    parser.add_argument("--study", choices=STUDIES, default="load")
    parser.add_argument("--preflight", type=Path,
                        help="completed independent lc-knee preflight required by an actual lc-knee full run")
    parser.add_argument("--plan", action="store_true", help="print frozen plan without GPU access or writes")
    args = parser.parse_args()
    try:
        if args.preflight is not None and (args.study != "lc-knee" or args.mode != "full"):
            raise ValueError("--preflight is only valid for an lc-knee full run")
        plan = make_plan(args.mode, args.cell_timeout, args.cooldown_seconds,
                         args.gdrcopy_dir, args.study, args.preflight)
    except ValueError as error:
        parser.error(str(error))
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    if args.output is None or os.geteuid() != 0:
        parser.error("actual runs need --output and the same root privilege for all arms")
    output = args.output.resolve()
    if args.study == "lc-knee" and args.mode == "full":
        if args.preflight is None:
            parser.error("an actual lc-knee full run requires --preflight COMPLETED_CAMPAIGN")
        try:
            validate_completed_preflight(args.preflight, output)
        except (OSError, ValueError, KeyError, TypeError, OverflowError,
                original.safety.GateError) as error:
            parser.error(f"LC-knee preflight gate failed: {error}")
    output.mkdir(parents=True, exist_ok=False)
    safety.atomic_write_json(output / "plan.json", plan)
    (output / "configs").mkdir()
    for scenario, config in plan["configs"].items():
        safety.atomic_write_json(output / "configs" / f"{scenario}.json", config)
    results, lease, run_error = [], None, None
    def interrupted(signum, _frame):
        raise InterruptedError(f"signal {signum}; cleaning up owned study processes")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        inventory = build_inventory()
        safety.atomic_write_json(output / "build-inventory.json", inventory)
        safety.atomic_write_json(output / "model-assets.json", original.model_assets())
        lease = original.Leases()
        for specification in plan["orders"]:
            block, scenario, arm = (specification[key] for key in ("block", "scenario", "arm"))
            print(f"START cell={specification['cell']} block={block} scenario={scenario} arm={arm}", flush=True)
            directory = output / f"block-{block:02d}" / scenario / arm
            result = run_cell(directory, specification, output / "configs" / f"{scenario}.json",
                              args.cell_timeout, args.gdrcopy_dir.resolve(), inventory["files"], args.study)
            results.append(result)
            safety.atomic_write_json(output / "progress.json", {"completed_cells": len(results), "last": specification})
            safety.atomic_write_json(output / "summary.json", summarize(results, plan))
            values = result["metrics"]
            print(f"PASS cell={specification['cell']} lc_response_p99_us={values[TASKS[0]]['response_p99_us']} "
                  f"be_goodput_rps={values[TASKS[1]]['goodput_rps']}", flush=True)
            if len(results) < plan["required_cells"]:
                time.sleep(args.cooldown_seconds)
    except BaseException as error:
        run_error = f"{type(error).__name__}: {error}"
        raise
    finally:
        summary = summarize(results, plan)
        summary.update(status="failed" if run_error else "completed", error=run_error)
        safety.atomic_write_json(output / "summary.json", summary)
        if lease is not None:
            lease.close()


if __name__ == "__main__":
    main()
