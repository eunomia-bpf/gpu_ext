"""Frozen five-arm adaptive-prefetch governor campaign on GPT-OSS-120B.

Protocol (adaptive-prefetch/plan.md, frozen before first preflight): each cell
is one held-out A->B->A group of six requests (A={cohort row 0,1}, B={2,3},
A'={0,1} repeated) under one policy arm; every SSE response must exactly equal
the frozen same-frontend golden for that cohort row.  Arms share the
paper-v3 predictor, executor, eviction selector, cache budget and exact SSE
frontend; they differ only in the speculative byte-admission governor:

- demand-only:      paper-native executor with MOE_REVISION_PREFETCH=0
                    (same predictor, eviction, and budget; zero speculation).
- unbounded-native: published MoE-Infinity v3 behavior (mode 1, unbounded).
- fixed-native:     live headroom rule with a frozen 512 MiB budget (mode 2).
- adaptive-native:  outcome/pressure adaptive budget, native C++ rule (mode 3).
- adaptive-bpf:     identical integer snapshot + rule through real host-uBPF
                    JIT (mode 3, no fallback).

Raw layout: raw/adaptive-prefetch-575/preflight-<stamp>/<arm> holds one
A->B->A cell per arm plus the golden-freeze run; raw/adaptive-prefetch-575/
full-<stamp>/block-NN-attempt-M/<arm> holds the five paired blocks.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import signal
import statistics
import subprocess
import time
from pathlib import Path

import run_moe_head_to_head as base
import run_paper_policy as paper
import run_paper_comparison as prior
import run_prefetch_ablation as ablation

HERE = Path(__file__).resolve().parent
COHORT = HERE / "adaptive-prefetch/held-out-cohort.json"
SCHEDULE = HERE / "adaptive-prefetch/schedule.json"
RAW_ROOT = HERE / "raw/adaptive-prefetch-575"
PROTOCOL = "moe-adaptive-prefetch-governor-1"
CONFIG = "moe_infinity_075"
BLOCKS = 5
ARMS = (
    "demand-only",
    "unbounded-native",
    "fixed-native",
    "adaptive-native",
    "adaptive-bpf",
)

# Frozen plan.md budgets: initial/fixed 512 MiB, floor 64 MiB, ceiling 2 GiB
# (initial x 4), quantum 128 MiB, enforced by the published rule.
INITIAL_BUDGET = 512 << 20
EXPECTED_GOVERNOR_MODE = {
    "demand-only": 0,
    "unbounded-native": 1,
    "fixed-native": 2,
    "adaptive-native": 3,
    "adaptive-bpf": 3,
}
# One A->B->A cell: positions 0,1 are group A; 2,3 disjoint group B; 4,5 repeat A.
CELL_POSITIONS = (0, 1, 2, 3, 0, 1)


def cohort_rows():
    rows = json.loads(COHORT.read_text())
    if len(rows) != 4:
        raise base.GateError(f"held-out cohort must hold exactly four rows: {len(rows)}")
    if len({row["index"] for row in rows}) != 4:
        raise base.GateError("held-out cohort rows are not disjoint")
    used = {record["source_index"]
            for record in json.loads(base.PROMPTS.read_text())["records"]}
    overlap = used.intersection(row["index"] for row in rows)
    if overlap:
        raise base.GateError(f"held-out cohort repeats prompts.json rows: {sorted(overlap)}")
    for position, row in enumerate(rows):
        if len(row["prompt_token_ids"]) != 512:
            raise base.GateError(f"cohort row {position} is not 512 tokens")
    return rows


def goldens():
    path = RAW_ROOT / "held-out-goldens.json"
    if not path.is_file():
        raise base.GateError(
            "held-out goldens are frozen by preflight and are missing; run preflight first")
    return json.loads(path.read_text())


def schedule(seed=20260909):
    """Per frozen schedule.json: fixed A1 A2 B1 B2 A1' A2' positions; only the
    arm order inside each paired block is randomized (seed 20260909)."""
    frozen = json.loads(SCHEDULE.read_text())
    if frozen["protocol"] != PROTOCOL or frozen["seed"] != 20260904:
        raise base.GateError(f"schedule.json does not match {PROTOCOL}")
    rng = random.Random(seed)
    result = []
    for number, block in enumerate(frozen["blocks"], 1):
        if block["block"] != number:
            raise base.GateError("schedule.json block numbering is inconsistent")
        arms = list(ARMS)
        rng.shuffle(arms)
        result.append({"block": number, "arms": arms,
                       "request_positions": list(block["request_positions"])})
    return result


def protocol_manifest(seed=20260909):
    planned = schedule(seed)
    requests_per_cell = len(CELL_POSITIONS)
    return {
        "protocol": PROTOCOL,
        "seed": seed,
        "cohort": [row["index"] for row in cohort_rows()],
        "cell_positions": list(CELL_POSITIONS),
        "schedule": planned,
        "required_blocks": BLOCKS,
        "arms_per_block": len(ARMS),
        "requests_per_cell": requests_per_cell,
        "planned_cells": BLOCKS * len(ARMS),
        "planned_measured_requests": BLOCKS * len(ARMS) * requests_per_cell,
        "planned_verified_output_tokens": BLOCKS * len(ARMS) * requests_per_cell * 64,
        "input_output_tokens_per_request": [512, 64],
        "memory_budget": 0.75,
        "kv_blocks": 128,
        "temporary_overload": False,
        "executor": "paper-activation-dispatcher",
        "governor_budgets_bytes": {
            "initial_or_fixed": INITIAL_BUDGET,
            "floor": 64 << 20,
            "ceiling": INITIAL_BUDGET * 4,
            "quantum": 128 << 20,
        },
        "correctness": "every measured SSE response exactly equals the frozen held-out golden",
    }


def runtime_inventory(admission):
    inventory = ablation.runtime_inventory(admission)
    inventory["files"].extend(base.file_metadata(path) for path in (
        Path(__file__), COHORT, SCHEDULE,
        base.EXTENSION / ".output/libmoe_spec_admission.so",
        base.EXTENSION / ".output/moe_spec_admission.bin",
    ))
    return inventory


def governor_arm_env(arm):
    if arm == "demand-only":
        return {}
    if arm == "unbounded-native":
        return {"MOE_REVISION_SPEC_ADMISSION": "unbounded"}
    if arm == "fixed-native":
        return {"MOE_REVISION_SPEC_ADMISSION": "fixed",
                "MOE_REVISION_SPEC_BUDGET_BYTES": str(INITIAL_BUDGET)}
    if arm == "adaptive-native":
        return {"MOE_REVISION_SPEC_ADMISSION": "adaptive-native",
                "MOE_REVISION_SPEC_BUDGET_BYTES": str(INITIAL_BUDGET)}
    return {"MOE_REVISION_SPEC_ADMISSION": "adaptive-bpf",
            "MOE_REVISION_SPEC_BUDGET_BYTES": str(INITIAL_BUDGET),
            "MOE_SPEC_ADMISSION_CODE": str(
                base.EXTENSION / ".output/moe_spec_admission.bin"),
            "MOE_SPEC_ADMISSION_LIBRARY": str(
                base.EXTENSION / ".output/libmoe_spec_admission.so")}


def launch_arm(arm, output, port):
    mode = "paper-bpf" if arm == "adaptive-bpf" else "paper-native"
    prefetch = arm != "demand-only"
    argv, cwd = base.server_command(CONFIG, port, output, paper.STORE)
    argv[4:6] = [str(HERE / "paper_server.py")]
    env = base.controlled_environment(CONFIG, cuda129_triton=True)
    artifacts = base.EXTENSION / ".output"
    env.update(MOE_REVISION_POLICY=mode, MOE_REVISION_VERIFY="0",
               MOE_EXPERT_POLICY_LIBRARY=str(artifacts / "libmoe_expert_policy.so"),
               MOE_EXPERT_RANK_CODE=str(artifacts / "moe_expert_policy_rank.bin"),
               MOE_EXPERT_SCORED_CODE=str(artifacts / "moe_expert_policy_scored.bin"),
               MOE_EXPERT_MATCH_CODE=str(artifacts / "moe_expert_policy_match.bin"),
               MOE_REVISION_PREFETCH="1" if prefetch else "0")
    env.update(governor_arm_env(arm))
    base.atomic_write_json(output / "launch.json", {"argv": argv, "cwd": str(cwd), "env": env})
    log = (output / "server.log").open("x")
    process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    return process, log


def governor_gate(arm, before, after, expected_requests):
    """Exact governor engagement/conservation plus the shared ablation delta."""
    delta = ablation.activation_delta(arm_for_ablation(arm), before, after,
                                      expected_requests=expected_requests)
    gov_before = before.get("governor") or {}
    governor_after = after.get("governor") or {}
    keys = ("governor_admission_calls", "governor_admission_submitted",
            "governor_admission_admitted", "governor_budget_updates",
            "governor_budget_unchanged", "governor_budget_decreases",
            "governor_budget_increases")
    for state in (gov_before := before.get("governor") or {}, governor_after):
        for key in keys:
            if key not in state or state[key] < 0:
                raise base.GateError(f"governor stats missing {key}")
    governor = base.counter_delta(gov_before, governor_after, keys)
    if (gov_before.get("governor_mode") != EXPECTED_GOVERNOR_MODE[arm] or
            governor_after.get("governor_mode") != EXPECTED_GOVERNOR_MODE[arm]):
        raise base.GateError(f"governor mode differs from arm {arm}: "
                             f"{gov_before.get('governor_mode')}->{governor_after.get('governor_mode')}")
    if arm in ("demand-only", "unbounded-native"):
        if any(governor.values()):
            raise base.GateError(f"{arm} unexpectedly ran governor admission/updates")
        return delta, None, governor_after
    if governor_after.get("governor_initial_budget_bytes") != INITIAL_BUDGET:
        raise base.GateError("governor initial budget differs from frozen plan")
    if arm == "fixed-native" and governor_after.get("governor_budget_bytes") != INITIAL_BUDGET:
        raise base.GateError("fixed arm budget drifted from the frozen 512 MiB")
    submitted = delta["dispatcher"]["prefetch_submitted"]
    if submitted > 0:
        if arm == "unbounded-native":
            # mode 1 admits without bookkeeping counters.
            if governor["governor_admission_calls"] != 0:
                raise base.GateError("unbounded arm unexpectedly counted admissions")
        if governor["governor_admission_admitted"] > governor["governor_admission_submitted"]:
            raise base.GateError("governor admitted more than submitted")
        # prefetch_submitted counts published identities; the governor's
        # admission_submitted counts offered candidates. Conservation holds
        # between published and ADMITTED candidates.
        if governor["governor_admission_admitted"] != submitted:
            raise base.GateError("governor admitted/prefetch_submitted disagree")
        if arm == "adaptive-native" or arm == "adaptive-bpf":
            if governor["governor_budget_updates"] <= 0:
                raise base.GateError("adaptive governor received no completed-request outcomes")
    else:
        if governor["governor_admission_calls"] or governor["governor_admission_submitted"]:
            raise base.GateError(f"governor admission ran without publication in {arm}")
    return delta, governor, governor_after


def arm_for_ablation(arm):
    if arm == "demand-only":
        return "native-prefetch-off"
    return "bpf-prefetch-on" if arm == "adaptive-bpf" else "native-prefetch-on"


def freeze_goldens(output, port, driver_stage):
    """One identity-checked cold server freezes exact held-out texts."""
    rows = cohort_rows()
    goldens_path = RAW_ROOT / "held-out-goldens.json"
    if goldens_path.is_file():
        existing = json.loads(goldens_path.read_text())
        if [entry["source_index"] for entry in existing["goldens"]] == [r["index"] for r in rows]:
            return existing
        raise base.GateError("frozen goldens do not match the held-out cohort")
    goldens_path.parent.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(base.PROMPTS.read_text())
    output.mkdir(parents=True, exist_ok=False)
    server, log = paper.launch("paper-native", output, port, verify=False)
    result = None
    try:
        paper.emit(f"golden freeze: cold model loading PID {server.pid}")
        base.wait_ready(server, port, output / "server.log", 1800)
        identity = base.check_server_identity(CONFIG, port, prompts, output / "server.log")
        old = json.loads((paper.OLD_CORRECTNESS / "result.json").read_text())
        warm = base.nonstream_completion(CONFIG, port, prompts["records"][0]["prompt_token_ids"],
                                         output / "warmup.json", timeout=600)
        if warm["text"] != old["warmup"]["text"]:
            raise base.GateError("identity warmup differs from retained same-frontend golden")
        entries = []
        for position, row in enumerate(rows):
            response = base.nonstream_completion(CONFIG, port, row["prompt_token_ids"],
                                                 output / f"freeze-{position}.json", timeout=600)
            entries.append({"cell_position": position, "source_index": row["index"],
                            "text": response["text"]})
        result = {"protocol": PROTOCOL, "identity": identity, "goldens": entries,
                  "frozen": "same-frontend nonstream, paper-native, identity-checked"}
    finally:
        if server is not None:
            base.stop_owned_process_group(server)
            result_exit = server.returncode
        log.close()
        if result is not None:
            if result_exit != 0:
                raise base.GateError("golden-freeze server exited nonzero")
            log_validate = output / "server.log"
            base.validate_log(log_validate)
            prior.sync_cell_artifacts(output)
            goldens_path.parent.mkdir(parents=True, exist_ok=True)
            import os
            temporary = goldens_path.with_suffix(".json.tmp")
            base.atomic_write_json(temporary, result)
            os.replace(temporary, goldens_path)
            # goldens_path.parent is the campaign dir holding attempt
            # subdirectories; fsync the file and its directory directly
            # instead of the flat-cell walk in sync_cell_artifacts.
            with goldens_path.open("rb") as stream:
                os.fsync(stream.fileno())
            goldens_dir_fd = os.open(goldens_path.parent,
                                     os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(goldens_dir_fd)
            finally:
                os.close(goldens_dir_fd)
    return result




def run_cell(arm, output, port, request_positions, driver_stage, expected_runtime=None):
    """One A->B->A cell under one governor arm with the exact-SSE machinery."""
    output.mkdir(parents=True, exist_ok=False)
    rows = cohort_rows()
    frozen = goldens()
    result = {"protocol": PROTOCOL, "arm": arm, "mode": "paper-bpf" if arm == "adaptive-bpf" else "paper-native",
              "prefetch_enabled": arm != "demand-only", "passed": False,
              "request_positions": list(request_positions),
              "cohort_rows": [rows[p]["index"] for p in request_positions],
              "shadow_verification": False,
              "execution_domain": "host-ubpf-jit" if arm == "adaptive-bpf" else "native",
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
        server, log = launch_arm(arm, output, port)
        paper.emit(f"{arm}: cold model loading PID {server.pid}")
        base.wait_ready(server, port, output / "server.log", 1800)
        prompts = json.loads(base.PROMPTS.read_text())
        result["identity"] = base.check_server_identity(CONFIG, port, prompts, output / "server.log")
        cold = base.http_json(port, "/revision/activation")
        if cold["controller"].get("eamc_entries", 0) or cold["controller"].get("completed_requests", 0):
            raise base.GateError("fresh server inherited EAMC state")
        result["activation_cold"] = cold
        old = json.loads((paper.OLD_CORRECTNESS / "result.json").read_text())
        result["warmup"] = base.nonstream_completion(
            CONFIG, port, prompts["records"][0]["prompt_token_ids"],
            output / "warmup.json", timeout=600)
        if result["warmup"]["text"] != old["warmup"]["text"]:
            raise base.GateError("excluded warmup differs from same-frontend golden")
        activation_before = base.http_json(port, "/revision/activation/drain", {}, timeout=600)
        before = base.engagement_snapshot(CONFIG, port, output, server.pid, current_deployment=True)
        if any(member["affinity"] != list(range(8)) for member in before["process_io"]["members"]):
            raise base.GateError("server process tree escaped CPU 0-7")
        prior.reject_build_contention()
        telemetry, telemetry_log, telemetry_path = base.start_gpu_telemetry(output)
        started = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        for seq, position in enumerate(request_positions, 1):
            prior.reject_build_contention()
            row = rows[position]
            expected = frozen["goldens"][position]["text"]
            paper.emit(f"{arm}: measured request {seq}/{len(request_positions)}, "
                       f"cohort row {row['index']} (position {position})")
            result["requests"].append(prior.stream_request(
                port, row["prompt_token_ids"], expected,
                output / f"request-{seq:02d}-row-{row['index']}.sse"))
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
        delta, governor, governor_after_state = governor_gate(
            arm, activation_before, activation_after,
            expected_requests=len(request_positions))
        result.update(
            activation_before=activation_before, activation_after=activation_after,
            activation_delta=delta, governor_delta=governor,
            governor_after=governor_after_state,
            engagement_before=before, engagement_after=after,
            engagement_delta=base.validate_measured_engagement(
                CONFIG, before, after, current_deployment=True,
                expected_generated_tokens=len(request_positions) * 64),
            gpu_telemetry=base.validate_gpu_telemetry(telemetry_path, allow_fixed_power_cap=True),
        )
        duration = (ended - started) / 1e9
        result.update(
            block_start_ns=started, block_end_ns=ended, duration_s=duration,
            final_drain_s=(ended - last_eof) / 1e9,
            verified_requests=len(request_positions),
            verified_output_tokens=len(request_positions) * 64,
            output_throughput_tokens_per_s=len(request_positions) * 64 / duration,
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
    """Whole-block geometric ratios with 10k-draw paired-block bootstrap CIs."""
    valid = []
    invalid = []
    seen = {}
    for block in blocks:
        seen[block.get("block")] = seen.get(block.get("block"), 0) + 1
    for block in blocks:
        cells = block.get("cells", [])
        ok = (block.get("passed") is True and seen.get(block.get("block")) == 1 and
              len(cells) == len(ARMS) and {c.get("arm") for c in cells} == set(ARMS) and
              all(c.get("passed") is True and c.get("shadow_verification") is False and
                  c.get("verified_requests") == len(CELL_POSITIONS) and
                  c.get("verified_output_tokens") == len(CELL_POSITIONS) * 64 and
                  len(c.get("requests", [])) == len(CELL_POSITIONS) and
                  c.get("request_positions") == block.get("request_positions") and
                  all(r.get("passed") is True for r in c.get("requests", [])) and
                  math.isfinite(c.get("output_throughput_tokens_per_s", 0)) and
                  c.get("output_throughput_tokens_per_s", 0) > 0 for c in cells))
        (valid if ok else invalid).append(block)
    result = {
        "protocol": PROTOCOL, "required_blocks": BLOCKS,
        "valid_blocks": len(valid),
        "invalid_block_numbers": [b.get("block") for b in invalid],
        "complete": len(valid) == BLOCKS and {b["block"] for b in valid} == set(range(1, BLOCKS + 1)),
        "valid_cells": len(valid) * len(ARMS),
        "verified_measured_requests": len(valid) * len(ARMS) * len(CELL_POSITIONS),
        "verified_output_tokens": len(valid) * len(ARMS) * len(CELL_POSITIONS) * 64,
        "primary": "384 tokens / full six-request A->B->A wall window including final drain",
        "modes": {}, "paired": {},
    }
    if not valid:
        return result
    by_arm = {arm: [next(c for c in b["cells"] if c["arm"] == arm) for b in valid]
              for arm in ARMS}
    byte_keys = ("prefetch_hit_bytes", "prefetch_wasted_bytes",
                 "prefetch_unused_resident_bytes",
                 "demand_prefetch_wait_ns", "demand_cache_wait_ns")
    for arm, rows in by_arm.items():
        counters = {key: sum(row["activation_delta"]["dispatcher"].get(key, 0) for row in rows)
                    for key in ablation.DELTA_FIELDS}
        counters.update({key: sum(row["activation_delta"]["dispatcher"].get(key, 0) for row in rows)
                         for key in byte_keys})
        result["modes"][arm] = {
            key: statistics.median(row[key] for row in rows)
            for key in ("output_throughput_tokens_per_s", "first_text_ttft_median_ms",
                        "e2e_median_ms", "final_drain_s")
        }
        result["modes"][arm].update(
            counters_sum=counters,
            governor_budget_final_bytes=int(
                statistics.median(int(row.get("governor_after", {}).get("governor_budget_bytes", 0))
                                  for row in rows)),
            governor_increases=sum((row.get("governor_delta") or {}).get("governor_budget_increases", 0)
                                   for row in rows),
            governor_decreases=sum((row.get("governor_delta") or {}).get("governor_budget_decreases", 0)
                                   for row in rows),
        )
    comparisons = (
        ("adaptive-bpf", "unbounded-native"),
        ("adaptive-native", "unbounded-native"),
        ("adaptive-bpf", "adaptive-native"),
        ("fixed-native", "adaptive-bpf"),
        ("unbounded-native", "demand-only"),
        ("adaptive-bpf", "demand-only"),
    )
    rng = random.Random(20260910)
    samples = [[rng.randrange(max(len(valid), 1)) for _ in valid] for _ in range(10000)]
    for numerator, denominator in comparisons:
        logs = [math.log(a["output_throughput_tokens_per_s"] /
                         b["output_throughput_tokens_per_s"])
                for a, b in zip(by_arm[numerator], by_arm[denominator])]
        boot = sorted(math.exp(statistics.mean(logs[i] for i in sample)) for sample in samples)
        result["paired"][f"{numerator}/{denominator}"] = {
            "geometric_throughput_ratio": math.exp(statistics.mean(logs)),
            "paired_block_bootstrap_ci95": [boot[249], boot[9749]] if len(valid) >= 2 else None,
            "interpretation": "ratio > 1 favors numerator; interval crossing 1 is inconclusive",
        }
    return result


def preflight(output, port, driver_stage, seed=20260909):
    lease = base.LeaseSet.acquire()
    try:
        output.mkdir(parents=True, exist_ok=False)
        inventory = runtime_inventory(paper.admit(port, driver_stage))
        arms = schedule(seed)[0]["arms"]
        result = {"protocol": f"{PROTOCOL}-preflight", "passed": False,
                  "performance_result": False, "arms": arms, "cells": []}
        base.atomic_write_json(output / "manifest.json", {
            **protocol_manifest(seed), "preflight": True,
            "preflight_requests_per_cell": len(CELL_POSITIONS),
            "driver_stage": str(driver_stage.resolve()),
            "golden_freeze": {"path": str((RAW_ROOT / "held-out-goldens.json").resolve()),
                              "frozen_from": freeze_goldens(
                                  output / "golden-freeze", port, driver_stage)["identity"]},
            "runtime_inventory": inventory,
        })
        try:
            for arm in arms:
                paper.emit(f"adaptive-prefetch preflight: {arm}")
                result["cells"].append(run_cell(
                    arm, output / arm, port, CELL_POSITIONS, driver_stage, inventory))
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
        raise base.GateError("adaptive-prefetch preflight result is missing")
    result = json.loads(result_path.read_text())
    cells = result.get("cells", [])
    if (result.get("protocol") != f"{PROTOCOL}-preflight" or
            result.get("passed") is not True or result.get("performance_result") is not False or
            len(cells) != len(ARMS) or {cell.get("arm") for cell in cells} != set(ARMS)):
        raise base.GateError("preflight did not pass all five adaptive arms")
    frozen = goldens()
    prompts = json.loads(base.PROMPTS.read_text())
    import paper_result_audit as raw_audit
    for cell in cells:
        arm = cell["arm"]
        cell_dir = path / arm
        admission = json.loads((cell_dir / "admission.json").read_text())
        if admission.get("runtime_inventory") != expected_runtime:
            raise base.GateError(f"preflight runtime differs for {arm}")
        requests = cell.get("requests", [])
        if (cell.get("passed") is not True or
                cell.get("verified_requests") != len(CELL_POSITIONS) or
                cell.get("verified_output_tokens") != len(CELL_POSITIONS) * 64 or
                len(requests) != len(CELL_POSITIONS) or
                any(not r.get("passed") for r in requests) or
                any(r.get("text") != frozen["goldens"][position]["text"]
                    for position, r in zip(cell.get("request_positions", []), requests))):
            raise base.GateError(f"preflight exact correctness is incomplete for {arm}")
        observed_delta, observed_governor, observed_after = governor_gate(
            arm, cell["activation_before"], cell["activation_after"],
            expected_requests=len(CELL_POSITIONS))
        if observed_delta != cell.get("activation_delta"):
            raise base.GateError(f"preflight activation delta differs for {arm}")
        if observed_governor != cell.get("governor_delta"):
            raise base.GateError(f"preflight governor delta differs for {arm}")
        raw_audit._engagement(cell, expected_generated_tokens=len(CELL_POSITIONS) * 64)
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


def run(output, port, driver_stage, preflight_path, seed=20260909, max_new_blocks=BLOCKS):
    lease = base.LeaseSet.acquire()
    try:
        output.mkdir(parents=True, exist_ok=True)
        manifest_path = output / "manifest.json"
        inventory = runtime_inventory(paper.admit(port, driver_stage))
        preflight_evidence = validate_preflight(preflight_path, inventory)
        manifest = {**protocol_manifest(seed), "driver_stage": str(driver_stage.resolve()),
                    "runtime_inventory": inventory,
                    "warmup_prompt_row": 0,
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
                completed.append(passed_attempts[0].name)
                continue
            if new_blocks >= max_new_blocks:
                continue
            attempt = output / f"block-{item['block']:02d}-attempt-{len(attempts) + 1:02d}"
            attempt.mkdir(exist_ok=False)
            block = {**item, "passed": False, "cells": []}
            try:
                for arm in item["arms"]:
                    paper.emit(f"adaptive-prefetch block {item['block']}/{BLOCKS}, {arm}")
                    block["cells"].append(run_cell(
                        arm, attempt / arm, port, item["request_positions"],
                        driver_stage, inventory))
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
        paper.emit(f"adaptive-prefetch valid paired blocks: {summary['valid_blocks']}/{BLOCKS}; "
                   f"complete={summary['complete']}")
    finally:
        lease.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    dry = subparsers.add_parser("dry-run", help="print and validate the CPU-only protocol matrix")
    dry.add_argument("--seed", type=int, default=20260909)
    execute = subparsers.add_parser("run", help="execute the real serialized GPU campaign")
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--port", type=int, default=18240)
    execute.add_argument("--driver-stage", type=Path,
                         default=Path("/opt/gpubpf/modules/575.57.08/gpreempt-e7d46fa5-6.15.11"))
    execute.add_argument("--preflight", type=Path, required=True)
    execute.add_argument("--seed", type=int, default=20260909)
    execute.add_argument("--max-new-blocks", type=int, default=BLOCKS)
    preflight_parser = subparsers.add_parser(
        "preflight", help="freeze goldens and run one A->B->A cell in each arm")
    preflight_parser.add_argument("--output", type=Path, required=True)
    preflight_parser.add_argument("--port", type=int, default=18240)
    preflight_parser.add_argument("--driver-stage", type=Path,
                                  default=Path("/opt/gpubpf/modules/575.57.08/gpreempt-e7d46fa5-6.15.11"))
    preflight_parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    if args.command == "dry-run":
        manifest = protocol_manifest(args.seed)
        if (manifest["planned_cells"] != 25 or
                manifest["planned_measured_requests"] != 150):
            raise base.GateError("protocol matrix is not exactly 25 cells / 150 requests")
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
