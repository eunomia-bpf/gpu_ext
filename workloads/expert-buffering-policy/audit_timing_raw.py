#!/usr/bin/env python3
"""Read-only, small-file audit of the retained historical timing campaign.

Does not import either launcher or the summary analyzer, run a workload, write
reports, or read the five large framework route traces. Successful arithmetic
checks do not close the evidence limitations listed in raw-audit.md.
"""

from __future__ import annotations

import argparse
import ast
import csv
import itertools
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
CONFIGS = ("plain_uvm", "gpubpf_observe", "gpubpf_profile_protect", "llama_ncmoe32")
BLOCK_BYTES = 2 * 1024 * 1024
SMALL_LIMIT = 2_000_000


def small_text(path: Path) -> str:
    if path.stat().st_size > SMALL_LIMIT:
        raise ValueError(f"Refusing large-file read: {path.relative_to(HERE)}")
    return path.read_text(encoding="utf-8", errors="strict")


def document(path: Path):
    return json.loads(small_text(path))


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9), (actual, expected)


def completion(path: Path):
    value = document(path)
    assert value["object"] == "text_completion" and value["model"] == "gpt-oss-120b"
    assert len(value["choices"]) == 1
    choice = value["choices"][0]
    assert choice["finish_reason"] == "length" and isinstance(choice["text"], str)
    choice["text"].encode("utf-8", errors="strict")
    assert value["usage"] == {"prompt_tokens": 512, "completion_tokens": 64, "total_tokens": 576}
    assert (value["timings"]["prompt_n"], value["timings"]["predicted_n"], value["timings"]["cache_n"]) == (512, 64, 0)
    return choice["text"], value["created"]


def activation(cell: Path, result, summary, layout):
    lines = small_text(cell / "class-table.txt").splitlines()
    header = lines[0].split()
    metadata = dict(zip(header[::2], map(int, header[1::2])))
    classes = dict(tuple(map(int, line.split())) for line in lines[1:])
    assert len(classes) == len(lines) - 1
    hot = {index for index, kind in classes.items() if kind == 2}
    assert len(hot) == layout["hot_blocks"]
    assert Counter(classes.values())[3] == layout["shared_blocks"]
    events = [json.loads(line) for line in small_text(cell / "policy.jsonl").splitlines()]
    stats = [event for event in events if event["event"] == "policy_stats"]
    assert stats and all(event["setter_failure"] == event["cold_head"] == 0 for event in stats)
    ready = [event for event in events if event["event"] == "policy_ready"]
    assert len(ready) == 1
    ready = ready[0]
    assert ready["mode"] == ("observe" if cell.name == "gpubpf_observe" else "protect")
    for key in ("struct_ops_map_id", "activate_program_id", "activation_counts_map_id", "pid"):
        assert ready[key] > 0
    assert ready["layout_base"] == metadata["base"] == layout["base"]
    assert ready["layout_blocks"] == metadata["blocks"] == layout["blocks"]
    assert ready["hot_bytes"] == metadata["hot_bytes"] == layout["hot_bytes"] <= 8 * 1024**3
    snapshots, preceding_stats = [], []
    for ordinal in (1, 2):
        begins = [i for i, event in enumerate(events) if event["event"] == "block_snapshot_begin" and event["ordinal"] == ordinal]
        ends = [i for i, event in enumerate(events) if event["event"] == "block_snapshot_end" and event["ordinal"] == ordinal]
        assert len(begins) == len(ends) == 1 and begins[0] < ends[0]
        begin, end = begins[0], ends[0]
        assert events[begin]["base"] == layout["base"] and events[begin]["blocks"] == layout["blocks"]
        values = events[begin + 1:end]
        assert all(event["event"] == "hot_block_activation" and event["ordinal"] == ordinal and event["class"] == 2 for event in values)
        counts = {event["index"]: event["count"] for event in values}
        assert len(values) == len(counts) == events[end]["records"] == len(hot)
        assert set(counts) == hot and all(isinstance(value, int) and value >= 0 for value in counts.values())
        snapshots.append(counts)
        preceding_stats.append(next(event for event in reversed(events[:begin]) if event["event"] == "policy_stats"))
    metric = result["hot_activation_metric"]
    for field, snapshot in zip(("counts_before", "counts_after"), snapshots):
        assert {int(key): value for key, value in metric[field].items()} == snapshot
    deltas = [snapshots[1][index] - snapshots[0][index] for index in hot]
    assert min(deltas) >= 0
    assert metric["block_bytes"] == BLOCK_BYTES and metric["hot_blocks"] == len(hot)
    assert metric["full_activation_bytes"] == BLOCK_BYTES * sum(deltas) == summary["full_hot_activation_bytes"]
    assert metric["repeated_activation_bytes"] == BLOCK_BYTES * sum(max(0, value - 1) for value in deltas) == summary["repeated_hot_activation_bytes"]
    delta = result["policy_delta"]
    assert all(preceding_stats[1][key] - preceding_stats[0][key] == value >= 0 for key, value in delta.items())
    for key, value in summary.items():
        if key in delta:
            assert delta[key] == value
    assert delta["setter_failure"] == delta["cold_head"] == 0
    if cell.name == "gpubpf_observe":
        assert delta["observe_activate"] > 0 and delta["observe_access"] > 0
        assert sum(delta[key] for key in ("hot_tail", "shared_tail", "hot_access_tail", "shared_access_tail")) == 0
    else:
        assert all(delta[key] > 0 for key in ("mapped", "hot_tail", "cold_native", "hot_access_tail"))
    return metric["repeated_activation_bytes"]


def interval(values, geometric=True):
    def estimate(sample):
        return math.exp(sum(map(math.log, sample)) / len(sample)) if geometric else sum(sample) / len(sample)
    samples = sorted(estimate([values[i] for i in indices]) for indices in itertools.product(range(len(values)), repeat=len(values)))
    def percentile(p):
        position = (len(samples) - 1) * p
        low = int(position)
        return samples[low] + (samples[min(low + 1, len(samples) - 1)] - samples[low]) * (position - low)
    return estimate(values), percentile(.025), percentile(.975)


def audit(inventory=False):
    schedule = document(HERE / "timing-schedule.json")["blocks"]
    recorded = document(HERE / "timing-results.json")
    summaries = recorded["blocks"]
    assert recorded["status"] == "passed" and recorded["completed_blocks"] == recorded["planned_blocks"] == 5
    assert recorded["measurement"]["output_tokens_per_cell"] == 512 and recorded["measurement"]["requests_per_cell"] == 8
    assert [entry["block"] for entry in schedule] == [1, 2, 3, 4, 5]
    assert [entry["block"] for entry in summaries] == [1, 2, 3, 4, 5]
    rates, hot_counts, differences = [], [], []
    measured, completions, telemetry_samples = 0, 0, 0
    repeats, cross_text, inventory_groups = Counter(), Counter(), {}
    timings, creation_compatible = [], 0
    print("Historical raw-file audit; no large route-trace reads")
    for entry, summary in zip(schedule, summaries):
        block = HERE / "raw/timing" / f"block-{entry['block']:02d}"
        status, admitted = document(block / "status.json"), document(block / "admission.json")
        assert status["status"] == summary["status"] == "passed"
        for key in ("configuration_order", "prompt_order"):
            assert status[key] == summary[key] == entry[key]
        assert admitted["driver"] == "610.43.02" and admitted["compute_apps"] == []
        launches = {config: document(block / config / "launch.json") for config in CONFIGS}
        assert launches[CONFIGS[0]] == launches[CONFIGS[1]] == launches[CONFIGS[2]]
        assert launches[CONFIGS[3]]["argv"] == launches[CONFIGS[0]]["argv"] + ["--n-cpu-moe", "32"]
        env = dict(launches[CONFIGS[0]]["environment_overrides"])
        assert env.pop("GGML_CUDA_ENABLE_UNIFIED_MEMORY") == "1"
        env["GPUBPF_EXPERT_ROUTE_TRACE"] = "1"
        assert launches[CONFIGS[3]]["environment_overrides"] == env
        block_rates, block_hot, block_texts = {}, {}, {}
        for config in entry["configuration_order"]:
            cell = block / config
            result, layout = document(cell / "result.json"), document(cell / "layout-report.json")
            assert result["config"] == config and result["prompt_order"] == entry["prompt_order"]
            assert len(result["requests"]) == 8 and result["verified_output_tokens"] == 512
            expected_files = {f"measured-request-{sequence:02d}-prompt-{prompt}.json" for sequence, prompt in enumerate(entry["prompt_order"], 1)}
            assert {path.name for path in cell.glob("measured-request-*.json")} == expected_files
            warmup, _ = completion(cell / "warmup.json")
            assert warmup == result["warmup"]["text"]
            untimed = [[completion(cell / f"untimed-pass-{number}-request-{sequence:02d}-prompt-{prompt}.json")[0] for sequence, prompt in enumerate(entry["prompt_order"], 1)] for number in (1, 2)]
            assert len(list(cell.glob("untimed-pass-*.json"))) == 16
            matching = sum(a == b for a, b in zip(*untimed))
            assert matching == result["untimed_repeated_matching_prompts"]
            repeats[config] += matching
            completions += 25
            previous = result["block_start_ns"]
            texts, created = [], []
            for sequence, (prompt, request) in enumerate(zip(entry["prompt_order"], result["requests"]), 1):
                text, timestamp = completion(cell / f"measured-request-{sequence:02d}-prompt-{prompt}.json")
                assert text == request["text"] and previous <= request["start_ns"] < request["end_ns"] <= result["block_end_ns"]
                close(request["e2e_ms"], (request["end_ns"] - request["start_ns"]) / 1e6)
                previous = request["end_ns"]
                texts.append(text)
                created.append(timestamp)
                measured += 1
            duration = (result["block_end_ns"] - result["block_start_ns"]) / 1e9
            close(result["duration_s"], duration)
            rate = 512 / duration
            close(rate, result["output_throughput_tokens_per_s"])
            close(rate, status["throughput_tokens_per_s"][config])
            close(rate, summary["throughput_tokens_per_s"][config])
            block_rates[config], block_texts[config] = rate, texts
            timings.append((result["block_start_ns"], result["block_end_ns"]))
            rows = [[value.strip() for value in row] for row in csv.reader(small_text(cell / "gpu-telemetry.csv").splitlines())]
            assert len(rows) >= 2 and all(len(row) == 8 for row in rows)
            assert all(value.lower() == "not active" for row in rows for value in row[5:])
            telem = result["gpu_telemetry"]
            assert telem["samples"] == len(rows) and not telem["thermal_or_power_brake_throttled"]
            for key, column, operation in (("peak_memory_mib", 1, max), ("peak_temperature_c", 2, max), ("min_sm_clock_mhz", 4, min), ("max_sm_clock_mhz", 4, max)):
                close(telem[key], operation(float(row[column]) for row in rows))
            close(telem["mean_power_w"], sum(float(row[3]) for row in rows) / len(rows))
            wall = [datetime.strptime(row[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=ZoneInfo("America/Vancouver")).timestamp() for row in rows]
            assert all(a < b for a, b in zip(wall, wall[1:]))
            creation_compatible += sum(wall[0] - 1 <= timestamp <= wall[-1] + 1 for timestamp in created)
            telemetry_samples += len(rows)
            server = small_text(cell / "server.log")
            assert not re.search(r"CUDA error|illegal memory access|out of memory|failed to load|Traceback", server, re.I)
            assert server.count("request: POST /v1/completions 127.0.0.1 200") == 25
            assert "cleaning up before exit" in server
            assert layout["registrations"] == 216 and layout["hot_blocks"] > 0
            if config in CONFIGS[1:3]:
                block_hot[config] = activation(cell, result, summary[config], layout)
            else:
                assert not (cell / "policy.jsonl").exists() and result["policy_delta"] is None
            if config == "llama_ncmoe32":
                route = document(cell / "route-diagnostic-report.json")
                assert [layer["layer"] for layer in route["layers"]] == list(range(32))
                assert route["graphs"] == 1105 and route["incomplete_graphs"] == 0
                assert all(layer["observed_graphs"] == 1105 for layer in route["layers"])
                assert route["route_events"] == result["trace_final"]["routes"] == summary["llama_ncmoe32_route_coverage"]["route_events"]
                assert result["trace_final"]["dropped"] == 0
                print(f"  stat only: {cell.relative_to(HERE)}/trace.jsonl { (cell / 'trace.jsonl').stat().st_size} bytes")
            else:
                events = [json.loads(line) for line in small_text(cell / "trace.jsonl").splitlines()]
                layouts = [event for event in events if event["event"] == "layout"]
                assert len(layouts) == 216 and Counter(event["is_bias"] for event in layouts) == {0: 108, 1: 108}
                assert len({event["tgid"] for event in layouts}) == 1
                assert {int(event["name"].split('.')[1]) for event in layouts} == set(range(36))
                assert events[-1] == {"event": "final", "graphs": 0, "layouts": 216, "routes": 0, "dropped": 0}
            print(f"  block {entry['block']} {config}: {rate:.9f} token/s, 8 saved measured replies match result, untimed equal={matching}/8")
        for label, a, b in (("O/U", CONFIGS[1], CONFIGS[0]), ("E/O", CONFIGS[2], CONFIGS[1]), ("F/U", CONFIGS[3], CONFIGS[0])):
            cross_text[label] += sum(x == y for x, y in zip(block_texts[a], block_texts[b]))
            key = {"O/U": "observe_over_plain", "E/O": "protect_over_observe", "F/U": "llama_ncmoe32_over_plain"}[label]
            close(summary["paired_effect_percent"][key], 100 * (block_rates[a] / block_rates[b] - 1))
        assert summary["thermal_or_power_brake_throttled"] is False
        rates.append(block_rates)
        hot_counts.append(block_hot[CONFIGS[2]] / block_hot[CONFIGS[1]])
        differences.append(block_hot[CONFIGS[2]] - block_hot[CONFIGS[1]])
    assert all(a[1] <= b[0] for a, b in zip(timings, timings[1:]))
    for start, description in ((0, "all five historical blocks"), (1, "blocks 2-5 sensitivity only, not a replacement fifth block")):
        for label, a, b in (("O/U", CONFIGS[1], CONFIGS[0]), ("E/O", CONFIGS[2], CONFIGS[1]), ("F/U", CONFIGS[3], CONFIGS[0])):
            values = interval([row[a] / row[b] for row in rates[start:]])
            if start == 0:
                key = {"O/U": "mechanism_observe_over_plain", "E/O": "policy_protect_over_observe", "F/U": "context_llama_ncmoe32_over_plain"}[label]
                aggregate = recorded["aggregate"][key]
                close(values[0], aggregate["geometric_mean_ratio"])
                for actual, expected in zip(values, [aggregate["effect_percent"], *aggregate["ci95_effect_percent"]]):
                    close(100 * (actual - 1), expected)
            print(f"{description}: throughput {label} effect/CI percent = " + ", ".join(f"{100*(value-1):+.9f}" for value in values))
    hot_interval, difference_interval = interval(hot_counts), interval(differences, False)
    aggregate = recorded["aggregate"]["repeated_hot_activation_protect_over_observe"]
    close(hot_interval[0], aggregate["geometric_mean_ratio"])
    for actual, expected in zip(hot_interval, [aggregate["effect_percent"], *aggregate["ci95_effect_percent"]]):
        close(100 * (actual - 1), expected)
    aggregate = recorded["aggregate"]["repeated_hot_activation_protect_minus_observe"]
    for actual, expected in zip(difference_interval, [aggregate["mean_difference_bytes"], *aggregate["ci95_difference_bytes"]]):
        close(actual, expected)
    print("All-five repeated-hot E/O ratio / CI:", hot_interval)
    print("All-five repeated-hot E-O bytes / CI:", difference_interval)
    failed = HERE / "raw/timing/block-02-failed-attempt-01"
    failure = document(failed / "status.json")
    assert failure["status"] == "failed" and failure["error_type"] == "GateError"
    value = ast.literal_eval(failure["error"].split(": ", 1)[1])
    assert value["usage"]["prompt_tokens"] == 512 and value["usage"]["completion_tokens"] == 1
    assert value["choices"][0]["finish_reason"] == "stop"
    failed_cell = failed / "gpubpf_observe"
    assert len(list(failed_cell.glob("untimed-pass-*.json"))) == 6
    assert not list(failed_cell.glob("measured-request-*.json")) and not (failed_cell / "result.json").exists()
    for path in failed_cell.glob("untimed-pass-*.json"):
        completion(path)
    _, sixth_created = completion(failed_cell / "untimed-pass-1-request-06-prompt-3.json")
    assert sixth_created < value["created"]
    print("Failed attempt: six valid saved untimed replies, including request 6 / prompt 3; one-token stop follows. Failed request identity was not saved; frozen next slot is request 7 / prompt 7.")
    print(f"Checked {measured} measured replies / {measured*64} output-token counts; {completions} total saved replies including excluded checks; {telemetry_samples} telemetry rows.")
    print(f"Untimed exact text agreement out of 40/config: {dict(repeats)}; measured cross-arm agreement out of 40/pair: {dict(cross_text)}")
    print(f"Saved response creation seconds compatible with telemetry under America/Vancouver interpretation: {creation_compatible}/{measured}; no saved wall/monotonic clock bridge.")
    for path in sorted((HERE / "raw/timing").rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            category = "failed-attempt" if "failed-attempt" in str(path) else ("measured-request" if path.name.startswith("measured-request") else "untimed-request" if path.name.startswith("untimed-pass") else path.name)
            counts = inventory_groups.setdefault(category, [0, 0])
            counts[0] += 1
            counts[1] += size
            if inventory:
                print(f"FILE {size} {path.relative_to(HERE)}")
    print("Retained timing file inventory (category: files, bytes):")
    for category, counts in sorted(inventory_groups.items()):
        print(category, *counts)
    print("Arithmetic audit completed; protocol, raw route, clock and lifecycle limits remain open in raw-audit.md.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", action="store_true", help="print every retained path and byte size, without reading its contents")
    audit(parser.parse_args().inventory)
