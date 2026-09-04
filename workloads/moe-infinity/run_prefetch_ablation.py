"""Five-block factorial ablation of predictive prefetch and eviction executor.

Each block contains all four arms in randomized order.  Every arm runs the same
paper-policy executor with the strict cache budget and no temporary overload
slot; only the native/BPF eviction selector and prefetch issue toggle differ.
GPU execution is intentionally available only through the ``preflight`` and
``run`` subcommands; ``dry-run`` is CPU-only.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import random
import signal
import statistics
import time

import run_moe_head_to_head as base
import run_paper_comparison as prior
import run_paper_policy as paper

PROTOCOL = "moe-predictive-prefetch-factorial-1"
BLOCKS = 5
REQUESTS_PER_CELL = 6
SEED = 20260904
CONFIG = "moe_infinity_075"
ARMS = (
    "native-prefetch-off",
    "native-prefetch-on",
    "bpf-prefetch-off",
    "bpf-prefetch-on",
)
ARM_CONFIG = {
    "native-prefetch-off": ("paper-native", False),
    "native-prefetch-on": ("paper-native", True),
    "bpf-prefetch-off": ("paper-bpf", False),
    "bpf-prefetch-on": ("paper-bpf", True),
}

STATIC_FIELDS = ("mode", "prefetch_enabled", "cache_budget_bytes",
                 "temporary_slot_enabled")
DELTA_FIELDS = (
    "eviction_selections", "bpf_eviction_calls",
    "bpf_demand_eviction_calls", "bpf_prefetch_eviction_calls",
    "eviction_mismatches", "evictions", "demand_evictions",
    "prefetch_evictions", "prefetch_submitted", "prefetch_completed",
    "prefetch_hits", "prefetch_hit_bytes", "prefetch_wasted",
    "prefetch_wasted_bytes", "prefetch_bytes",
    "prefetch_prediction_epoch", "prefetch_protected_resident_skips",
    "prefetch_stale_discarded", "prefetch_no_victim",
    "prefetch_copy_started", "prefetch_victim_recheck_rejected",
    "prefetch_copy_waits", "prefetch_copy_wait_ns",
    "demand_prefill_accesses", "demand_prefill_hits", "demand_prefill_misses",
    "demand_decode_accesses", "demand_decode_hits", "demand_decode_misses",
    "demand_copy_started", "demand_bytes", "demand_prefetch_waits",
    "demand_prefetch_wait_ns", "demand_cache_waits", "demand_cache_wait_ns",
    "temporary_slot_uses", "temporary_slot_bytes", "temporary_slot_waits",
    "temporary_slot_wait_ns",
)
PREFETCH_ZERO_FIELDS = (
    "prefetch_submitted", "prefetch_completed", "prefetch_hits",
    "prefetch_hit_bytes", "prefetch_wasted", "prefetch_wasted_bytes",
    "prefetch_bytes", "prefetch_evictions", "prefetch_copy_started",
    "prefetch_copy_waits", "prefetch_copy_wait_ns",
    "prefetch_protected_resident_skips", "bpf_prefetch_eviction_calls",
)


def schedule(seed=SEED):
    rng = random.Random(seed)
    orders = list(itertools.permutations(ARMS))
    rng.shuffle(orders)
    result = []
    for number, arms in enumerate(orders[:BLOCKS], 1):
        prompts = list(range(1, REQUESTS_PER_CELL + 1))
        rng.shuffle(prompts)
        result.append({"block": number, "arms": list(arms), "prompts": prompts})
    return result


def protocol_manifest(seed=SEED):
    planned = schedule(seed)
    return {
        "protocol": PROTOCOL,
        "seed": seed,
        "schedule": planned,
        "required_blocks": BLOCKS,
        "arms_per_block": len(ARMS),
        "requests_per_cell": REQUESTS_PER_CELL,
        "planned_cells": BLOCKS * len(ARMS),
        "planned_measured_requests": BLOCKS * len(ARMS) * REQUESTS_PER_CELL,
        "planned_verified_output_tokens": (
            BLOCKS * len(ARMS) * REQUESTS_PER_CELL * 64
        ),
        "input_output_tokens_per_request": [512, 64],
        "memory_budget": 0.75,
        "kv_blocks": 128,
        "temporary_overload": False,
        "executor": "paper-activation-dispatcher",
        "timing_shadow_verification": False,
        "correctness": "every measured SSE response exactly matches retained same-frontend golden",
    }


def _require_counter_map(dispatcher):
    for key in STATIC_FIELDS + DELTA_FIELDS + (
            "prefetch_unused_resident", "prefetch_unused_resident_bytes"):
        value = dispatcher.get(key)
        if type(value) is not int or value < 0:
            raise base.GateError(f"missing/invalid ablation counter {key}")


def activation_delta(arm, before, after, expected_requests=REQUESTS_PER_CELL):
    if arm not in ARM_CONFIG:
        raise base.GateError(f"unknown ablation arm: {arm}")
    mode, prefetch = ARM_CONFIG[arm]
    if before.get("mode") != mode or after.get("mode") != mode:
        raise base.GateError("policy backend changed within ablation cell")
    if before.get("prefetch_enabled") is not prefetch or after.get("prefetch_enabled") is not prefetch:
        raise base.GateError("runtime prefetch toggle differs from requested arm")
    for state in (before, after):
        _require_counter_map(state.get("dispatcher", {}))
        dispatcher = state["dispatcher"]
        expected_mode = 2 if mode == "paper-bpf" else 1
        if dispatcher["mode"] != expected_mode or dispatcher["prefetch_enabled"] != int(prefetch):
            raise base.GateError("dispatcher backend/prefetch state differs from arm")
        if dispatcher["cache_budget_bytes"] <= 0 or dispatcher["temporary_slot_enabled"] != 0:
            raise base.GateError("strict common cache budget/overload condition is not active")
    if before["dispatcher"]["cache_budget_bytes"] != after["dispatcher"]["cache_budget_bytes"]:
        raise base.GateError("cache budget changed within ablation cell")

    controller = base.counter_delta(before["controller"], after["controller"], (
        "completed_requests", "matched_predictions", "prefetch_candidates_selected",
        "rank_calls", "bpf_match_calls", "aborted_requests",
    ))
    dispatcher = base.counter_delta(before["dispatcher"], after["dispatcher"], DELTA_FIELDS)
    dispatcher["prefetch_unused_resident"] = after["dispatcher"]["prefetch_unused_resident"]
    dispatcher["prefetch_unused_resident_bytes"] = after["dispatcher"]["prefetch_unused_resident_bytes"]

    if controller["completed_requests"] != expected_requests or controller["aborted_requests"] != 0:
        raise base.GateError(f"measured request accounting mismatch: {controller}")
    if controller["matched_predictions"] <= 0 or controller["prefetch_candidates_selected"] <= 0:
        raise base.GateError("predictor/ranker did not execute in measured window")
    for phase in ("prefill", "decode"):
        accesses = dispatcher[f"demand_{phase}_accesses"]
        hits = dispatcher[f"demand_{phase}_hits"]
        misses = dispatcher[f"demand_{phase}_misses"]
        if accesses <= 0 or accesses != hits + misses:
            raise base.GateError(f"invalid {phase} cache hit/miss accounting")
    if dispatcher["demand_copy_started"] <= 0 or dispatcher["demand_bytes"] <= 0:
        raise base.GateError("demand-copy path did not engage")
    if dispatcher["demand_evictions"] <= 0 or dispatcher["eviction_selections"] <= 0:
        raise base.GateError("demand eviction did not engage under the common budget")
    if dispatcher["evictions"] != dispatcher["demand_evictions"] + dispatcher["prefetch_evictions"]:
        raise base.GateError("demand/prefetch eviction accounting does not conserve")
    if any(dispatcher[key] for key in (
            "temporary_slot_uses", "temporary_slot_bytes",
            "temporary_slot_waits", "temporary_slot_wait_ns")):
        raise base.GateError("temporary overload slot contaminated the causal ablation")

    if mode == "paper-bpf":
        if (controller["rank_calls"] <= 0 or controller["bpf_match_calls"] <= 0 or
                dispatcher["bpf_demand_eviction_calls"] <= 0):
            raise base.GateError("real BPF match/rank/demand-eviction did not all engage")
        if dispatcher["bpf_eviction_calls"] != (
                dispatcher["bpf_demand_eviction_calls"] +
                dispatcher["bpf_prefetch_eviction_calls"]):
            raise base.GateError("BPF demand/prefetch call accounting does not conserve")
    elif (controller["rank_calls"] or controller["bpf_match_calls"] or
          dispatcher["bpf_eviction_calls"] or dispatcher["bpf_demand_eviction_calls"] or
          dispatcher["bpf_prefetch_eviction_calls"]):
        raise base.GateError("native arm unexpectedly executed BPF")

    if prefetch:
        if (dispatcher["prefetch_submitted"] <= 0 or dispatcher["prefetch_completed"] <= 0 or
                dispatcher["prefetch_copy_started"] <= 0 or dispatcher["prefetch_bytes"] <= 0):
            raise base.GateError("prefetch-on arm did not issue and complete real copies")
        if dispatcher["prefetch_copy_started"] != dispatcher["prefetch_completed"]:
            raise base.GateError("prefetch issue/completion mismatch after drain")
        if dispatcher["prefetch_completed"] != (
                dispatcher["prefetch_hits"] + dispatcher["prefetch_wasted"] +
                dispatcher["prefetch_unused_resident"]):
            raise base.GateError("unused-prefetch count accounting does not conserve")
        if dispatcher["prefetch_bytes"] != (
                dispatcher["prefetch_hit_bytes"] + dispatcher["prefetch_wasted_bytes"] +
                dispatcher["prefetch_unused_resident_bytes"]):
            raise base.GateError("unused-prefetch byte accounting does not conserve")
        if dispatcher["prefetch_copy_waits"] != dispatcher["prefetch_completed"]:
            raise base.GateError("prefetch synchronization accounting does not conserve")
    elif any(dispatcher[key] for key in PREFETCH_ZERO_FIELDS) or any(
            after["dispatcher"][key] for key in (
                "prefetch_unused_resident", "prefetch_unused_resident_bytes",
                "prefetch_protected_candidates")):
        raise base.GateError("prefetch-off arm performed or retained speculative work")
    if after["dispatcher"]["prefetch_protected_candidates"] != 0:
        raise base.GateError("activation snapshot was not taken after prediction drain")
    if dispatcher["eviction_mismatches"]:
        raise base.GateError("native/BPF eviction shadow mismatch")
    return {"controller": controller, "dispatcher": dispatcher,
            "cache_budget_bytes": after["dispatcher"]["cache_budget_bytes"],
            "temporary_slot_enabled": False}


def runtime_inventory(admission):
    stores = sorted(base.MOE_PACKAGE.glob("_store.cpython-312-*.so"))
    if len(stores) != 1:
        raise base.GateError(
            f"expected one active Python 3.12 _store extension, found {stores}")
    symbols = base.run_checked(["nm", "-D", "-C", str(stores[0])])
    configure = [line for line in symbols.splitlines()
                 if "ExpertDispatcher::ConfigureActivationPolicy(" in line]
    if len(configure) != 1 or not configure[0].rstrip().endswith(", bool, bool)"):
        raise base.GateError(
            "active _store lacks the five-argument predictive-prefetch toggle ABI; "
            "rebuild the patched extension before real preflight")
    inventory = prior.runtime_inventory(admission)
    inventory["files"].extend(base.file_metadata(path) for path in (
        Path(__file__), Path(__file__).with_name("audit_prefetch_ablation.py"),
    ))
    return inventory


def run_cell(arm, output, port, prompt_order, driver_stage, expected_runtime=None):
    mode, prefetch = ARM_CONFIG[arm]
    output.mkdir(parents=True, exist_ok=False)
    result = {"protocol": PROTOCOL, "arm": arm, "mode": mode,
              "prefetch_enabled": prefetch, "passed": False,
              "prompt_order": prompt_order, "shadow_verification": False,
              "execution_domain": "host-ubpf-jit" if mode == "paper-bpf" else "native",
              "requests": []}
    server = log = telemetry = telemetry_log = None
    admission = failure = None
    try:
        prior.reject_build_contention()
        admission = paper.admit(port, driver_stage)
        admission["runtime_inventory"] = runtime_inventory(admission)
        base.atomic_write_json(output / "admission.json", admission)
        if expected_runtime is not None and admission["runtime_inventory"] != expected_runtime:
            raise base.GateError("runtime changed within paired campaign")
        result["interrupt_warnings_before"] = paper.interrupt_warnings()
        prompts = json.loads(base.PROMPTS.read_text())
        goldens = json.loads((paper.OLD_CORRECTNESS / "result.json").read_text())
        server, log = paper.launch(mode, output, port, verify=False, prefetch=prefetch)
        paper.emit(f"{arm}: cold model loading PID {server.pid}")
        base.wait_ready(server, port, output / "server.log", 900)
        result["identity"] = base.check_server_identity(CONFIG, port, prompts, output / "server.log")
        cold = base.http_json(port, "/revision/activation")
        if cold["controller"].get("eamc_entries", 0) or cold["controller"].get("completed_requests", 0):
            raise base.GateError("fresh server inherited EAMC state")
        result["activation_cold"] = cold
        result["warmup"] = base.nonstream_completion(
            CONFIG, port, prompts["records"][0]["prompt_token_ids"],
            output / "warmup.json", timeout=600)
        if result["warmup"]["text"] != goldens["warmup"]["text"]:
            raise base.GateError("excluded warmup differs from same-frontend golden")
        activation_before = base.http_json(port, "/revision/activation/drain", {}, timeout=600)
        before = base.engagement_snapshot(CONFIG, port, output, server.pid, current_deployment=True)
        if any(member["affinity"] != list(range(8)) for member in before["process_io"]["members"]):
            raise base.GateError("server process tree escaped CPU 0-7")
        prior.reject_build_contention()
        telemetry, telemetry_log, telemetry_path = base.start_gpu_telemetry(output)
        started = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        for seq, number in enumerate(prompt_order, 1):
            prior.reject_build_contention()
            paper.emit(f"{arm}: measured request {seq}/{len(prompt_order)}, prompt {number}")
            result["requests"].append(prior.stream_request(
                port, prompts["records"][number]["prompt_token_ids"],
                goldens["goldens"][number - 1],
                output / f"request-{seq:02d}-prompt-{number}.sse"))
        last_eof = result["requests"][-1]["eof_ns"]
        activation_after = base.http_json(port, "/revision/activation/drain", {}, timeout=600)
        ended = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        after = base.engagement_snapshot(CONFIG, port, output, server.pid, current_deployment=True)
        if any(member["affinity"] != list(range(8)) for member in after["process_io"]["members"]):
            raise base.GateError("server process tree escaped CPU 0-7 during timing")
        prior.reject_build_contention()
        base.stop_exact_process(telemetry)
        telemetry_log.close()
        telemetry = telemetry_log = None
        delta = activation_delta(arm, activation_before, activation_after,
                                 expected_requests=len(prompt_order))
        result.update(
            activation_before=activation_before, activation_after=activation_after,
            activation_delta=delta, engagement_before=before, engagement_after=after,
            engagement_delta=base.validate_measured_engagement(
                CONFIG, before, after, current_deployment=True,
                expected_generated_tokens=len(prompt_order) * 64),
            gpu_telemetry=base.validate_gpu_telemetry(telemetry_path, allow_fixed_power_cap=True),
        )
        duration = (ended - started) / 1e9
        result.update(
            block_start_ns=started, block_end_ns=ended, duration_s=duration,
            final_drain_s=(ended - last_eof) / 1e9,
            verified_requests=len(prompt_order),
            verified_output_tokens=len(prompt_order) * 64,
            output_throughput_tokens_per_s=len(prompt_order) * 64 / duration,
            first_text_ttft_median_ms=statistics.median(r["ttft_ms"] for r in result["requests"]),
            e2e_median_ms=statistics.median(r["e2e_ms"] for r in result["requests"]),
        )
    except BaseException as exc:
        failure = exc
        result.update(error_type=type(exc).__name__, error=str(exc))
    finally:
        cleanup_errors = []
        for resource, action in ((telemetry, base.stop_exact_process),
                                 (telemetry_log, lambda x: x.close()),
                                 (server, base.stop_owned_process_group),
                                 (log, lambda x: x.close())):
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
        try:
            prior.sync_cell_artifacts(output)
        except BaseException as exc:
            cleanup_errors.append(f"raw artifact durability: {type(exc).__name__}: {exc}")
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
    seen = {}
    for block in blocks:
        seen[block.get("block")] = seen.get(block.get("block"), 0) + 1
    for block in blocks:
        cells = block.get("cells", [])
        budgets = {cell.get("activation_delta", {}).get("cache_budget_bytes") for cell in cells}
        ok = (block.get("passed") is True and seen.get(block.get("block")) == 1 and
              len(cells) == len(ARMS) and {c.get("arm") for c in cells} == set(ARMS) and
              len(budgets) == 1 and None not in budgets and
              all(c.get("passed") is True and c.get("shadow_verification") is False and
                  c.get("verified_requests") == REQUESTS_PER_CELL and
                  c.get("verified_output_tokens") == REQUESTS_PER_CELL * 64 and
                  len(c.get("requests", [])) == REQUESTS_PER_CELL and
                  c.get("prompt_order") == block.get("prompts") and
                  all(r.get("passed") is True for r in c.get("requests", [])) and
                  all(type(c.get("activation_delta", {}).get("dispatcher", {}).get(key)) is int
                      for key in DELTA_FIELDS + (
                          "prefetch_unused_resident", "prefetch_unused_resident_bytes")) and
                  math.isfinite(c.get("output_throughput_tokens_per_s", 0)) and
                  c.get("output_throughput_tokens_per_s", 0) > 0 for c in cells))
        (valid if ok else invalid).append(block)
    campaign_budgets = {
        cell["activation_delta"]["cache_budget_bytes"]
        for block in valid for cell in block["cells"]
    }
    budget_consistent = len(campaign_budgets) <= 1
    result = {
        "protocol": PROTOCOL, "required_blocks": BLOCKS,
        "valid_blocks": len(valid),
        "invalid_block_numbers": [b.get("block") for b in invalid],
        "complete": (len(valid) == BLOCKS and budget_consistent and
                     {b["block"] for b in valid} == set(range(1, BLOCKS + 1))),
        "valid_cells": len(valid) * len(ARMS),
        "verified_measured_requests": len(valid) * len(ARMS) * REQUESTS_PER_CELL,
        "verified_output_tokens": len(valid) * len(ARMS) * REQUESTS_PER_CELL * 64,
        "primary": "384 tokens / full six-request wall window including final prefetch drain",
        "cache_budget_consistent": budget_consistent,
        "cache_budget_bytes": next(iter(campaign_budgets)) if len(campaign_budgets) == 1 else None,
        "modes": {}, "paired": {},
    }
    if not valid:
        return result
    by_arm = {arm: [next(c for c in b["cells"] if c["arm"] == arm) for b in valid]
              for arm in ARMS}
    for arm, rows in by_arm.items():
        counters = {
            key: sum(row["activation_delta"]["dispatcher"][key] for row in rows)
            for key in DELTA_FIELDS + (
                "prefetch_unused_resident", "prefetch_unused_resident_bytes")
        }
        result["modes"][arm] = {
            key: statistics.median(row[key] for row in rows)
            for key in ("output_throughput_tokens_per_s", "first_text_ttft_median_ms",
                        "e2e_median_ms", "final_drain_s")
        }
        result["modes"][arm].update(
            counters_sum=counters,
            prefill_demand_hit_rate=(
                counters["demand_prefill_hits"] / counters["demand_prefill_accesses"]),
            decode_demand_hit_rate=(
                counters["demand_decode_hits"] / counters["demand_decode_accesses"]),
        )
    comparisons = (
        ("native-prefetch-on", "native-prefetch-off"),
        ("bpf-prefetch-on", "bpf-prefetch-off"),
        ("bpf-prefetch-off", "native-prefetch-off"),
        ("bpf-prefetch-on", "native-prefetch-on"),
    )
    rng = random.Random(SEED + 1)
    samples = [[rng.randrange(len(valid)) for _ in valid] for _ in range(10000)]
    for numerator, denominator in comparisons:
        logs = [math.log(a["output_throughput_tokens_per_s"] /
                         b["output_throughput_tokens_per_s"])
                for a, b in zip(by_arm[numerator], by_arm[denominator])]
        boot = sorted(math.exp(statistics.mean(logs[i] for i in sample)) for sample in samples)
        result["paired"][f"{numerator}/{denominator}"] = {
            "geometric_throughput_ratio": math.exp(statistics.mean(logs)),
            "paired_block_bootstrap_ci95": [boot[249], boot[9749]] if len(valid) >= 2 else None,
            "interpretation": "ratio > 1 favors numerator",
        }
    return result


def preflight(output, port, driver_stage, seed=SEED):
    lease = base.LeaseSet.acquire()
    try:
        output.mkdir(parents=True, exist_ok=False)
        inventory = runtime_inventory(paper.admit(port, driver_stage))
        arms = schedule(seed)[0]["arms"]
        result = {"protocol": f"{PROTOCOL}-preflight", "passed": False,
                  "performance_result": False, "arms": arms, "cells": []}
        base.atomic_write_json(output / "manifest.json", {
            **protocol_manifest(seed), "preflight": True,
            "preflight_requests_per_cell": 1,
            "driver_stage": str(driver_stage.resolve()),
            "runtime_inventory": inventory,
        })
        try:
            for arm in arms:
                paper.emit(f"prefetch ablation preflight: {arm}")
                result["cells"].append(run_cell(
                    arm, output / arm, port, [1], driver_stage, inventory))
            result["passed"] = True
        except BaseException as exc:
            result.update(error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            base.atomic_write_json(output / "result.json", result)
    finally:
        lease.close()


def validate_preflight(path, expected_runtime):
    path = path.resolve()
    result_path = path / "result.json"
    if not result_path.is_file():
        raise base.GateError("prefetch ablation preflight result is missing")
    result = json.loads(result_path.read_text())
    cells = result.get("cells", [])
    if (result.get("protocol") != f"{PROTOCOL}-preflight" or
            result.get("passed") is not True or result.get("performance_result") is not False or
            len(cells) != len(ARMS) or {cell.get("arm") for cell in cells} != set(ARMS)):
        raise base.GateError("preflight did not pass all four causal arms")
    goldens = json.loads((paper.OLD_CORRECTNESS / "result.json").read_text())
    prompts = json.loads(base.PROMPTS.read_text())
    import paper_result_audit as raw_audit
    from audit_prefetch_ablation import _launch as audit_launch
    for cell in cells:
        arm = cell["arm"]
        cell_dir = path / arm
        admission = json.loads((cell_dir / "admission.json").read_text())
        if admission.get("runtime_inventory") != expected_runtime:
            raise base.GateError(f"preflight runtime differs for {arm}")
        mode, _ = ARM_CONFIG[arm]
        audit_launch(cell_dir, arm)
        requests = cell.get("requests", [])
        if (cell.get("passed") is not True or cell.get("verified_requests") != 1 or
                cell.get("verified_output_tokens") != 64 or len(requests) != 1 or
                requests[0].get("passed") is not True or
                requests[0].get("text") != goldens["goldens"][0]):
            raise base.GateError(f"preflight exact correctness is incomplete for {arm}")
        raw_audit._request(cell_dir, 1, 1,
                           prompts["records"][1]["prompt_token_ids"],
                           goldens["goldens"][0], requests[0])
        observed_delta = activation_delta(
            arm, cell["activation_before"], cell["activation_after"],
            expected_requests=1)
        if observed_delta != cell.get("activation_delta"):
            raise base.GateError(f"preflight activation delta differs for {arm}")
        raw_audit._engagement(cell, expected_generated_tokens=64)
        raw_audit._log(cell_dir, mode, cell)
        if (base.validate_gpu_telemetry(
                cell_dir / "gpu-telemetry.csv", allow_fixed_power_cap=True) !=
                cell.get("gpu_telemetry")):
            raise base.GateError(f"preflight GPU telemetry differs for {arm}")
        if cell.get("cleanup_errors") != [] or cell.get("server_exit_code") != 0:
            raise base.GateError(f"preflight cleanup failed for {arm}")
        base.validate_pre_server_safety(admission["safety"])
        base.validate_post_server_safety(admission["safety"], cell["safety_after"])
        if cell["interrupt_warnings_before"] != cell["interrupt_warnings_after"]:
            raise base.GateError(f"preflight added an RM interrupt warning for {arm}")
    return {"path": str(path), "result": base.file_metadata(result_path)}


def run(output, port, driver_stage, preflight_path, seed=SEED, max_new_blocks=BLOCKS):
    lease = base.LeaseSet.acquire()
    try:
        output.mkdir(parents=True, exist_ok=True)
        manifest_path = output / "manifest.json"
        inventory = runtime_inventory(paper.admit(port, driver_stage))
        preflight_evidence = validate_preflight(preflight_path, inventory)
        manifest = {**protocol_manifest(seed), "driver_stage": str(driver_stage.resolve()),
                    "runtime_inventory": inventory, "warmup_prompt": 0,
                    "required_real_preflight": preflight_evidence}
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise base.GateError("resume protocol/runtime/schedule differs; use new output directory")
        else:
            if any(output.iterdir()):
                raise base.GateError("nonempty output has no matching manifest")
            base.atomic_write_json(manifest_path, manifest)
        completed = []
        new_blocks = 0
        for item in manifest["schedule"]:
            attempts = sorted(output.glob(f"block-{item['block']:02d}-attempt-*"))
            passed_attempts = []
            for attempt in attempts:
                path = attempt / "result.json"
                if path.exists():
                    previous = json.loads(path.read_text())
                    if previous.get("passed"):
                        passed_attempts.append(attempt)
            if len(passed_attempts) > 1:
                raise base.GateError("duplicate successful block; refusing selective resume")
            if passed_attempts:
                from audit_prefetch_ablation import audit_block
                completed.append(audit_block(passed_attempts[0], item, inventory))
                continue
            if new_blocks >= max_new_blocks:
                continue
            attempt = output / f"block-{item['block']:02d}-attempt-{len(attempts) + 1:02d}"
            attempt.mkdir(exist_ok=False)
            block = {**item, "passed": False, "cells": []}
            try:
                for arm in item["arms"]:
                    paper.emit(f"block {item['block']}/{BLOCKS}, {arm}")
                    block["cells"].append(run_cell(
                        arm, attempt / arm, port, item["prompts"], driver_stage, inventory))
                block["passed"] = True
                completed.append(block)
                new_blocks += 1
            except BaseException as exc:
                block.update(error_type=type(exc).__name__, error=str(exc))
                raise
            finally:
                base.atomic_write_json(attempt / "result.json", block)
                base.atomic_write_json(output / "analysis.json", analyze(completed))
            from audit_prefetch_ablation import audit_block
            completed[-1] = audit_block(attempt, item, inventory)
        summary = analyze(completed)
        base.atomic_write_json(output / "analysis.json", summary)
        paper.emit(f"valid paired blocks: {summary['valid_blocks']}/{BLOCKS}; complete={summary['complete']}")
    finally:
        lease.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    dry = subparsers.add_parser("dry-run", help="print and validate the CPU-only protocol matrix")
    dry.add_argument("--seed", type=int, default=SEED)
    execute = subparsers.add_parser("run", help="execute the real serialized GPU campaign")
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--port", type=int, default=18230)
    execute.add_argument("--driver-stage", type=Path, required=True)
    execute.add_argument("--preflight", type=Path, required=True)
    execute.add_argument("--seed", type=int, default=SEED)
    execute.add_argument("--max-new-blocks", type=int, default=BLOCKS)
    preflight_parser = subparsers.add_parser(
        "preflight", help="run one real exact-correctness request in each arm")
    preflight_parser.add_argument("--output", type=Path, required=True)
    preflight_parser.add_argument("--port", type=int, default=18230)
    preflight_parser.add_argument("--driver-stage", type=Path, required=True)
    preflight_parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.command == "dry-run":
        manifest = protocol_manifest(args.seed)
        if (manifest["planned_cells"] != 20 or
                manifest["planned_measured_requests"] != 120):
            raise base.GateError("protocol matrix is not exactly 20 cells / 120 requests")
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    if args.command == "preflight":
        preflight(args.output.resolve(), args.port, args.driver_stage, args.seed)
        return
    if not 1 <= args.max_new_blocks <= BLOCKS:
        parser.error("--max-new-blocks must be 1..5; the full protocol always requires five")
    signal.signal(signal.SIGINT, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt(signum)))
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt(signum)))
    run(args.output.resolve(), args.port, args.driver_stage, args.preflight.resolve(),
        args.seed, args.max_new_blocks)


if __name__ == "__main__":
    main()
