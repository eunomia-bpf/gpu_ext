#!/usr/bin/env python3
"""Read-only audit of the fixed GPreempt FIFO/continuous contention study.

Raw request timestamps, not producer summaries, define response and goodput.
No CUDA, driver changes, or content digests are used by this module.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import re
import random
import statistics

from analyze_three_way import estimate_ratios
import run_three_way as base

SCENARIOS = ("be100", "be200", "be_continuous")
NS = 1_000_000_000


def integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def p99(values):
    return sorted(values)[math.ceil(len(values) * .99) - 1] if values else None


def json_records(log, prefix):
    return [json.loads(line[len(prefix):]) for line in log.splitlines()
            if line.startswith(prefix)]


def parse_client(log, scenario, seconds, arm):
    """Independently validate one process's raw reports and return derived metrics."""
    if scenario not in SCENARIOS or arm not in base.ARMS:
        raise ValueError("unknown scenario or arm")
    integer(seconds, "duration", 1)
    decoder = json.JSONDecoder()
    reports = []
    for match in re.finditer(r"(?m)^\{", log):
        try:
            value, _ = decoder.raw_decode(log[match.start():])
        except ValueError:
            continue
        if isinstance(value, dict) and "benchmarkTime(s)" in value:
            reports.append(value)
    if len(reports) != 1 or reports[0].get("benchmarkTime(s)") != seconds:
        raise ValueError("missing unique DISB report with the declared duration")
    report = reports[0]
    rows = json_records(log, "GPREEMPT_LOAD_STUDY ")
    checks = json_records(log, "GPREEMPT_VALIDATION ")
    for values, label in ((rows, "load reports"), (checks, "numerical reports")):
        if len(values) != 2 or {row.get("task") for row in values} != set(base.TASKS):
            raise ValueError(f"missing unique reports for both roles: {label}")
    rows = {row["task"]: row for row in rows}
    checks = {row["task"]: row for row in checks}
    results = report.get("results", [])
    if len(results) != 2 or {item.get("clientName") for item in results} != set(base.TASKS):
        raise ValueError("missing unique DISB results for both roles")
    analyzers = {}
    for item in results:
        basic = [value for value in item["analyzers"] if value.get("type") == "basic"]
        if len(basic) != 1:
            raise ValueError("missing unique six-stage analyzer")
        analyzers[item["clientName"]] = basic[0]
    begin = integer(report.get("loadStudyBeginNs"), "common begin")
    end = integer(report.get("loadStudyEndNs"), "common end")
    if end - begin != seconds * NS or report.get("loadStudyClock") != "steady_clock":
        raise ValueError("common monotonic measurement window changed")
    metrics = {}
    for role, task in enumerate(base.TASKS):
        row, analyzer, check = rows[task], analyzers[task], checks[task]
        continuous = role == 1 and scenario == "be_continuous"
        interval = 0 if continuous else NS // (200 if role == 1 and scenario == "be200" else 100)
        mode = "continuous_closed_loop" if continuous else "periodic_fifo"
        if (row.get("clock") != "steady_clock" or row.get("mode") != mode
                or row.get("begin_ns") != begin or row.get("end_ns") != end
                or row.get("phase_ns") != 0 or row.get("interval_ns") != interval
                or row.get("request_fields") != "id,scheduled_ns,started_ns,verified_ready_ns"):
            raise ValueError(f"{task}: arrival mode, shared window, phase, or raw schema changed")
        for field in ("begin_ns", "end_ns", "phase_ns", "interval_ns"):
            integer(row.get(field), field)
        offered = None if continuous else (end - begin + interval - 1) // interval
        if continuous:
            if row.get("offered") is not None:
                raise ValueError("continuous load must not invent periodic offered requests")
        elif integer(row.get("offered"), "offered") != offered:
            raise ValueError("offered count does not match the fixed arrival schedule")
        requests = row.get("requests")
        started = integer(row.get("started"), "started")
        if not isinstance(requests, list) or len(requests) != started or (offered is not None and started > offered):
            raise ValueError("raw rows, started count, or offered bound disagree")
        responses, window_responses = [], []
        previous_finish = begin
        for identifier, request in enumerate(requests):
            if not isinstance(request, list) or len(request) != 4:
                raise ValueError("request row must contain exactly four integers")
            rid, scheduled, actual_start, finish = [integer(value, "request timestamp/id") for value in request]
            if rid != identifier:
                raise ValueError("started request IDs are not a consecutive FIFO prefix")
            if scheduled != (actual_start if continuous else begin + rid * interval):
                raise ValueError("scheduled arrival was shifted, skipped, or reconstructed from completion")
            if not (begin <= scheduled <= actual_start < end and previous_finish <= actual_start <= finish):
                raise ValueError("request timing violates arrival, cutoff, or serial-worker ordering")
            previous_finish = finish
            if finish <= scheduled:
                raise ValueError("response must include positive measured work")
            responses.append(finish - scheduled)
            if finish < end:
                window_responses.append(finish - scheduled)
        after = started - len(window_responses)
        if after > 1:
            raise ValueError("a serial worker cannot complete two requests after its admission cutoff")
        samples = analyzer.get("requestLatencyNs")
        if (integer(analyzer.get("completedRequests"), "six-stage count") != started
                or not isinstance(samples, list) or len(samples) != started
                or any(type(value) is not int or value <= 0 for value in samples)
                or analyzer.get("latencyDefinition") != "sum_of_original_six_recorded_stages"):
            raise ValueError("six-stage samples do not match every started/verified request")
        old_rate = analyzer.get("avgThroughput(req/s)")
        if (not isinstance(old_rate, (int, float)) or not math.isfinite(old_rate)
                or abs(old_rate - started / seconds) > 1e-6):
            raise ValueError("auxiliary original throughput disagrees with its recorded request count")
        maximum_error = check.get("max_absolute_error")
        if (integer(check.get("checked"), "checked") != started + 110
                or integer(check.get("timed_checked"), "timed_checked") != started
                or check.get("atol") != 1e-6 or check.get("rtol") != 1e-4
                or not isinstance(maximum_error, (int, float))
                or not math.isfinite(maximum_error) or maximum_error < 0):
            raise ValueError("full-output numerical counts or tolerances disagree")
        backlog = None if continuous else offered - started
        metrics[task] = {
            "offered_requests": offered, "started_requests": started,
            "completed_in_window": len(window_responses), "completed_after_window": after,
            "started_unfinished": 0, "never_started_backlog": backlog,
            "unfinished_offered_at_deadline": None if continuous else offered - len(window_responses),
            "window_completion_fraction": None if continuous else len(window_responses) / offered,
            "completion_coverage": None if continuous else started / offered,
            "goodput_rps": len(window_responses) / seconds,
            "response_p99_us": p99(responses) / 1000 if responses else None,
            "mean_response_us": statistics.mean(responses) / 1000 if responses else None,
            "in_window_response_p99_us": p99(window_responses) / 1000 if window_responses else None,
            "response_p99_conditional": not continuous and backlog > 0,
            "response_p99_population": "all_started_and_verified_including_after_window",
            "service_p99_us": p99(samples) / 1000 if samples else None,
            "numerics": check, "p99_sample_count": started,
            "p99_less_than_100_samples": started < 100,
        }
    priority_lines = [line for line in log.splitlines() if line.startswith("GPREEMPT_LOAD_PRIORITY ")]
    priorities = [dict(re.findall(r"(\w+)=([^\s]+)", line)) for line in priority_lines]
    if arm == "native":
        if len(priorities) != 2 or {row.get("task") for row in priorities} != set(base.TASKS):
            raise ValueError("missing native stream-priority evidence for both roles")
        ranges = set()
        for row in priorities:
            role = base.TASKS.index(row["task"])
            least, greatest, actual = (int(row[key]) for key in ("least", "greatest", "actual"))
            if int(row["role"]) != role or greatest >= least or actual != (greatest if role == 0 else least):
                raise ValueError("native actual stream priorities do not match foreground/background roles")
            ranges.add((least, greatest))
        if len(ranges) != 1:
            raise ValueError("native roles report different device priority ranges")
    elif priorities:
        raise ValueError("policy client unexpectedly emitted native stream-priority evidence")
    return {"report": report, "load_reports": rows, "metrics": metrics,
            "native_priorities": {row["task"]: row for row in priorities}}


def plotting_metrics(metrics):
    result = {}
    for task, values in metrics.items():
        offered = values["offered_requests"]
        coverage = None if offered is None else values["started_requests"] / offered
        result[task] = {**values, "p99_response_us": values["response_p99_us"],
                       "p99_is_conditional": values["response_p99_conditional"],
                       "completion_coverage": coverage,
                       "all_offered_p99_response_us": values["response_p99_us"] if coverage == 1 else None}
    return result


def paired_estimate(values):
    if any(value is None or not math.isfinite(value) or value <= 0 for value in values):
        return {"geometric_ratio": None, "block_ratios": values,
                "paired_block_bootstrap_ci95": None,
                "unavailable_reason": "zero goodput or absent latency; adverse blocks are not dropped"}
    return estimate_ratios(values, draws=10000)


def summarize_scenario(blocks, points, required=5):
    identifiers = [block["block"] for block in blocks]
    if len(set(identifiers)) != len(identifiers) or any(set(block["cells"]) != set(base.ARMS) for block in blocks):
        raise ValueError("duplicate or incomplete paired block")
    result = {"valid_paired_blocks": len(blocks), "required_blocks": required,
              "complete": sorted(identifiers) == list(range(required)),
              "arms": {}, "paired": {}, "per_cell_points": points}
    if not blocks:
        return result
    for arm in base.ARMS:
        result["arms"][arm] = {}
        for task in base.TASKS:
            cells = [block["cells"][arm][task] for block in blocks]
            def median(field):
                values = [cell[field] for cell in cells]
                return statistics.median(values) if all(value is not None for value in values) else None
            result["arms"][arm][task] = {
                "median_p99_response_us": median("p99_response_us"),
                "median_goodput_rps": median("goodput_rps"),
                "median_completion_coverage": median("completion_coverage"),
                "max_never_started_backlog": max((cell["never_started_backlog"] for cell in cells
                                                  if cell["never_started_backlog"] is not None), default=None),
                "any_p99_conditional": any(cell["p99_is_conditional"] for cell in cells),
                "paired_cell_count": len(cells),
            }
    for numerator, denominator in (("original_gpreempt", "native"), ("bpf_gpreempt", "native"),
                                   ("bpf_gpreempt", "original_gpreempt")):
        for task, metric, direction in ((base.TASKS[0], "p99_response_us", "lower is better"),
                                        (base.TASKS[1], "goodput_rps", "higher is better")):
            values, conditional = [], False
            for block in blocks:
                above, below = block["cells"][numerator][task], block["cells"][denominator][task]
                values.append(above[metric] / below[metric] if above[metric] is not None and below[metric] else None)
                conditional |= metric == "p99_response_us" and (above["p99_is_conditional"] or below["p99_is_conditional"])
            result["paired"][f"{numerator}/{denominator}:{task}:{metric}"] = {
                **paired_estimate(values), "direction": direction, "conditional_response_population": conditional,
                "all_offered_latency_comparison": metric == "p99_response_us" and not conditional,
                "block_ids": identifiers,
            }
    return result


def expected_config(scenario, seconds):
    tasks = []
    for role, task, model in zip((0, 1), base.TASKS, ("vgg", "resnet152")):
        load = {"type": "periodic_fifo", "frequency": 200 if role and scenario == "be200" else 100,
                "priority": 0}
        if role and scenario == "be_continuous":
            load = {"type": "continuous"}
        tasks.append({"id": task, "load": load, "client": {"name": task, "model_name": model,
                      "priority": role, "batch_size": 1, "use_cuda_graph": True, "preprocess_time": 200}})
    return {"time": seconds, "tasks": tasks}


def validate_plan(plan, campaign):
    mode = plan.get("mode")
    if mode not in ("full", "preflight"):
        raise ValueError("unknown campaign mode")
    full = mode == "full"
    blocks, seconds = (5, 60) if full else (1, 10)
    scenarios = list(SCENARIOS) if full else ["be_continuous"]
    checks = {"schema": "gpreempt_load_study_v1", "seed": 20260903, "blocks": blocks,
              "scenarios": scenarios, "timed_seconds_per_cell": seconds,
              "required_cells": 45 if full else 3, "formal_required_blocks": 5,
              "flag_transport": "host_mapped", "comparison_variant": "host_mapped_compatibility",
              "clock": "steady_clock", "arrival_phase_ns": 0,
              "policy_changes": False, "kernel_repetition": 1,
              "statistics": {"estimator": "paired geometric-mean ratios",
                             "interval": "percentile paired-block bootstrap", "draws": 10000,
                             "seed": 20260903, "confidence": .95, "equivalence_claimed": False}}
    for key, value in checks.items():
        require_equal(plan.get(key), value, f"plan.{key}")
    if (not 90 <= integer(plan.get("cell_timeout_seconds"), "cell timeout") <= 3500
            or not 0 <= integer(plan.get("cooldown_seconds"), "cooldown") <= 60):
        raise ValueError("timeout or cooldown lies outside the frozen bounds")
    if not Path(plan.get("gdrcopy_directory", "")).is_absolute():
        raise ValueError("missing explicit GDRCopy library directory")
    if full:
        permutations = list(itertools.permutations(SCENARIOS))
        random.Random(20260903).shuffle(permutations)
        scenario_orders = [list(row) for row in permutations[:5]]
    else:
        scenario_orders = [scenarios]
    require_equal(plan.get("scenario_orders"), scenario_orders, "scenario order")
    orders = []
    arm_orders = {scenario: base.orders(blocks, 20260903 + index + 1)
                  for index, scenario in enumerate(SCENARIOS)}
    for block, scenario_order in enumerate(scenario_orders):
        for scenario in scenario_order:
            for arm in arm_orders[scenario][block]:
                orders.append({"cell": len(orders), "block": block, "scenario": scenario, "arm": arm})
    require_equal(plan.get("orders"), orders, "full seeded execution order")
    configs = {scenario: expected_config(scenario, seconds) for scenario in scenarios}
    require_equal(plan.get("configs"), configs, "frozen workloads")
    for scenario, config in configs.items():
        require_equal(read_json(campaign / "configs" / f"{scenario}.json"), config, "saved scenario config")
    return full


def require_equal(actual, expected, label):
    """Strict structural/integer equality; tolerate only insignificant float rounding."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{label}: fields disagree")
        for key, value in expected.items():
            require_equal(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{label}: list length disagrees")
        for index, (left, right) in enumerate(zip(actual, expected)):
            require_equal(left, right, f"{label}[{index}]")
    elif isinstance(expected, float):
        if (type(actual) not in (int, float) or not math.isfinite(actual)
                or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9)):
            raise ValueError(f"{label}: floating-point metric disagrees")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{label}: value disagrees")


def read_json(path):
    return json.loads(path.read_text())


def validate_models(assets):
    if set(assets) != {"vgg", "resnet152"}:
        raise ValueError("missing recorded model assets")
    for name, layers in (("vgg", 19), ("resnet152", 152)):
        specification = assets[name]["specification"]
        expected = {"model": name, "layers": layers, "architecture": "sm_120", "dtype": "float32",
                    "input_shape": [1, 3, 224, 224], "output_shape": [1, 1000], "parameter_seed": 0,
                    "input_formula": "((element_index % 257) - 128) / 128.0"}
        for key, value in expected.items():
            require_equal(specification.get(key), value, f"model {name}.{key}")
        inventory = assets[name]["inventory"]
        if set(inventory) != {"mod.cu", "mod.cubin", "mod.json", "host.json", "mod.params", "reference.f32"}:
            raise ValueError("recorded model inventory is incomplete")
        for filename, metadata in inventory.items():
            if (not Path(metadata["path"]).is_absolute() or Path(metadata["path"]).name != filename
                    or integer(metadata["bytes"], "model bytes", 1) <= 0
                    or (filename == "reference.f32" and metadata["bytes"] != 4000)):
                raise ValueError("invalid model asset path or size")


def validate_inventory(inventory):
    expected = {str(base.HERE / "build/load-study" / name) for name in
                ("baseclient", "gpreemptclient", "libgpreempt.so", "block.cubin")}
    expected |= {str(base.EXTENSION / name) for name in
                 ("gpreempt_policy", "libgpreempt_bridge.so", "gpreempt_hint.bin")}
    files = inventory["files"]
    if set(files) != expected:
        raise ValueError("frozen runtime inventory does not contain the exact seven artifacts")
    for path, metadata in files.items():
        if metadata.get("path") != path:
            raise ValueError("runtime inventory path/key mismatch")
        integer(metadata.get("bytes"), "runtime bytes", 1)
        integer(metadata.get("mtime_ns"), "runtime mtime", 1)
    cache = inventory["cmake"]
    for key, expected_value in (("GPREEMPT_LOAD_STUDY", "ON"), ("GPREEMPT_CUDA_ARCH", "120"),
                                ("CMAKE_CUDA_ARCHITECTURES", "120")):
        if cache.get(key) != expected_value:
            raise ValueError(f"build configuration changed: {key}")
    for key in ("CMAKE_CXX_COMPILER", "CMAKE_C_COMPILER", "CMAKE_CUDA_COMPILER"):
        if not cache.get(key) or not Path(cache[key]).is_absolute():
            raise ValueError("missing explicit build compiler")
    revisions = inventory["source_revisions"]
    if (not revisions.get("gpu_ext_head") or revisions.get("upstream") != "249ee3e"
            or revisions.get("expected_driver_port") != "849ea75d"):
        raise ValueError("missing recorded implementation/upstream/expected driver revisions")
    return files


def policy_action_coverage(arm, engagement, metrics):
    if arm == "native":
        return {"applicable": False}
    fields = engagement["bridge"]
    foreground = metrics[base.TASKS[0]]["started_requests"]
    all_timed = sum(metrics[task]["started_requests"] for task in base.TASKS)
    # Both roles execute 10 warmup + 100 calibration requests before initialized=1.
    # They call preprocess/infer, but only initialized LC requests yield actions.
    expected = {"preprocess": all_timed + 220, "infer": all_timed + 220,
                "reset": foreground, "hint": foreground, "block": foreground, "release": foreground}
    for key, count in expected.items():
        if int(fields.get(key, -1)) != count:
            raise ValueError(f"policy {key} count does not cover every real request")
    if int(fields.get("due", -1)) < foreground:
        raise ValueError("daemon due decisions do not cover every foreground hint")
    if arm == "original_gpreempt" and any(int(fields.get(key, -1)) != 0 for key in ("scopes", "registered", "ended")):
        raise ValueError("original-C arm unexpectedly used BPF context lifecycle")
    return {"applicable": True, "foreground_started": foreground, "all_timed_started": all_timed,
            "warmup_calibration_requests": 220, "verified_counts": expected,
            "due_decisions": int(fields["due"])}


def audit_cell(directory, specification, plan, campaign, inventory):
    result = read_json(directory / "result.json")
    if (result.get("status") != "passed" or result.get("error") or result.get("cleanup_errors")
            or result.get("returncode") != 0):
        raise ValueError("cell did not pass execution and cleanup")
    for key, value in specification.items():
        require_equal(result.get(key), value, f"cell.{key}")
    arm, scenario = specification["arm"], specification["scenario"]
    config = campaign / "configs" / f"{scenario}.json"
    require_equal(result.get("config_path"), str(config), "actual config path")
    require_equal(result.get("config"), plan["configs"][scenario], "actual config")
    require_equal(result.get("timed_seconds"), plan["timed_seconds_per_cell"], "actual duration")
    require_equal(result.get("timeout_seconds"), plan["cell_timeout_seconds"], "actual timeout")
    require_equal(result.get("flag_transport"), "not_used" if arm == "native" else "host_mapped", "actual flag transport")
    require_equal(result.get("comparison_variant"), "host_mapped_compatibility", "actual comparison variant")
    command = base.client_command(arm, config, "host_mapped")
    command[0] = str(base.HERE / "build/load-study" / Path(command[0]).name)
    require_equal(result.get("command"), command, "actual client command")
    pin = Path(result.get("environment", {}).get("GPREEMPT_BPF_MAPS", "/unused"))
    environment = base.environment(arm, pin, Path(plan["gdrcopy_directory"]))
    environment["LD_LIBRARY_PATH"] = environment["LD_LIBRARY_PATH"].replace(str(base.HERE / "build/ninja"),
                                                                            str(base.HERE / "build/load-study"))
    require_equal(result.get("environment"), environment, "actual isolated environment")
    if arm == "bpf_gpreempt":
        if pin.parent != Path("/sys/fs/bpf") or not pin.name.startswith("gpreempt-load-"):
            raise ValueError("BPF maps are not in the owned load-study pin directory")
        expected_loader = [str(base.EXTENSION / "gpreempt_policy"), "--library",
                           str(base.EXTENSION / "libgpreempt_bridge.so"), "--pin-dir", str(pin),
                           "--duration", str(plan["cell_timeout_seconds"] + 30)]
        require_equal(result.get("loader_command"), expected_loader, "actual BPF loader command")
    elif result.get("loader_command"):
        raise ValueError("non-BPF arm unexpectedly launched a BPF loader")
    require_equal(result.get("runtime_before"), inventory, "runtime before")
    require_equal(result.get("runtime_after"), inventory, "runtime after")
    client = (directory / "client.log").read_text()
    parsed = parse_client(client, scenario, plan["timed_seconds_per_cell"], arm)
    require_equal(read_json(directory / "request-report.json"), parsed["report"], "saved DISB raw report")
    require_equal(read_json(directory / "load-study-report.json"), parsed["load_reports"], "saved request timestamp report")
    require_equal(result.get("metrics"), parsed["metrics"], "producer metrics vs raw audit")
    require_equal(result.get("native_priorities"), parsed["native_priorities"], "recorded native priorities")
    loader_path = directory / "loader.log"
    loader = loader_path.read_text() if loader_path.exists() else ""
    if arm == "bpf_gpreempt" and not loader:
        raise ValueError("missing actual BPF loader log")
    engagement = base.check_engagement(arm, client, loader, "host_mapped")
    require_equal(result.get("engagement"), engagement, "engagement")
    action_coverage = policy_action_coverage(arm, engagement, parsed["metrics"])
    base.safety.validate_pre_server_safety(result["safety_before"])
    base.safety.validate_post_server_safety(result["safety_before"], result["safety_after"])
    if result["safety_before"]["gpu"]["driver"] != "575.57.08":
        raise ValueError("cell did not run on the common 575 driver")
    telemetry = base.safety.validate_gpu_telemetry(directory / "gpu-telemetry.csv", allow_fixed_power_cap=True)
    require_equal(result.get("telemetry"), telemetry, "raw GPU telemetry")
    latest = max((request[3] for row in parsed["load_reports"].values() for request in row["requests"]),
                 default=parsed["report"]["loadStudyEndNs"])
    return {**specification, "path": str(directory), "metrics": plotting_metrics(parsed["metrics"]),
            "begin_ns": parsed["report"]["loadStudyBeginNs"], "end_ns": parsed["report"]["loadStudyEndNs"],
            "last_verified_ready_ns": latest, "native_priorities": parsed["native_priorities"],
            "policy_action_coverage": action_coverage}


def analyze(campaign):
    campaign = Path(campaign).resolve()
    plan = read_json(campaign / "plan.json")
    formal = validate_plan(plan, campaign)
    inventory = read_json(campaign / "build-inventory.json")
    files = validate_inventory(inventory)
    validate_models(read_json(campaign / "model-assets.json"))
    accepted, rejected, incomplete, unexpected = [], [], [], []
    expected_paths = set()
    previous_finish = None
    for specification in plan["orders"]:
        directory = campaign / f"block-{specification['block']:02d}" / specification["scenario"] / specification["arm"]
        expected_paths.add(directory)
        if not (directory / "result.json").is_file():
            incomplete.append({**specification, "path": str(directory),
                               "state": "partial" if directory.exists() else "not_started"})
            continue
        try:
            point = audit_cell(directory, specification, plan, campaign, files)
            if previous_finish is not None and point["begin_ns"] <= previous_finish:
                raise ValueError("actual raw windows overlap or contradict the frozen cell execution order")
            previous_finish = max(point["end_ns"], point["last_verified_ready_ns"])
            accepted.append(point)
        except (OSError, ValueError, KeyError, TypeError, OverflowError, base.safety.GateError) as error:
            rejected.append({**specification, "path": str(directory), "error": str(error)})
    for filename in ("result.json", "client.log"):
        for path in campaign.rglob(filename):
            if path.parent not in expected_paths and str(path.parent) not in unexpected:
                unexpected.append(str(path.parent))
    scenarios = {}
    for scenario in plan["scenarios"]:
        points = [point for point in accepted if point["scenario"] == scenario]
        blocks = []
        for block in range(plan["blocks"]):
            cells = {point["arm"]: point["metrics"] for point in points if point["block"] == block}
            if set(cells) == set(base.ARMS):
                blocks.append({"block": block, "cells": cells})
        scenarios[scenario] = summarize_scenario(blocks, points, required=plan["blocks"])
    complete = (len(accepted) == plan["required_cells"] and not rejected and not incomplete and not unexpected)
    return {"schema": "gpreempt_load_study_audit_v1", "campaign": str(campaign), "mode": plan["mode"],
            "complete": complete, "formal_eligible": formal, "formal_complete": complete and formal,
            "valid_cells": len(accepted), "required_cells": plan["required_cells"],
            "rejected_cells": rejected, "incomplete_cells": incomplete, "unexpected_cells": sorted(unexpected),
            "scenarios": scenarios, "flag_transport": "host_mapped", "original_gdr_transport": False,
            "statistics": plan["statistics"], "equivalence_claimed": False,
            "latency_definition": "nearest-rank scheduled-arrival to synchronized/numerically-verified output ready",
            "goodput_definition": "verified completions strictly before common cutoff divided by configured seconds",
            "continuous_offered_or_miss_rate_defined": False,
            "source_revisions": inventory["source_revisions"],
            "driver_revision_evidence": "declared expected source revision, not inferred from runtime counters"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.campaign)
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        base.safety.atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
