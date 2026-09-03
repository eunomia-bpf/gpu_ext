"""EB preflight or five complete FIFO/native/BPF blocks via FineMoE's runner.

Preflight retains exact logits and enables BPF shadow. Full requires that same
runtime's passed preflight, then uses neither logits capture nor native shadow.
"""
import argparse
import importlib.util
import itertools
import math
from pathlib import Path
import random
import signal
import statistics
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
FINE = HERE.parents[1] / "finemoe"
spec = importlib.util.spec_from_file_location("finemoe_common_controller", FINE / "compare.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)  # stdlib and the existing safety helpers only.
from build_adapter import stop_owned

require, read_json, metadata = common.require, common.read_json, common.metadata
ARMS = ("fifo", "native", "bpf")
SEED = 20260903
POOL_BYTES = 16834658304
EXPERT_BYTES = 17301504
ORIGINAL_ENVIRONMENT = common.child_environment


def expected_decoding(data):
    # The original golden has no decoding_configuration field. Verify against
    # its unchanged checkpoint and the exact overrides in FineMoE generate().
    checkpoint = read_json(Path(data["model"]["snapshot"]) / "generation_config.json")
    return {"checkpoint_fields": {key: checkpoint[key] for key in
            ("do_sample", "repetition_penalty", "eos_token_id", "pad_token_id", "temperature", "top_k", "top_p")},
            "explicit_overrides": {"do_sample": False, "min_new_tokens": 16, "max_new_tokens": 16,
                                   "pad_token_id": 151643}}


def audit_cell(result, data, arm, numerical, golden):
    require(type(numerical) is bool and result["correctness_only"] is numerical and
            result["check_logits"] is numerical and result["status"] == "passed" and
            result["engine_cleanup_returned"] is True and result["arm"] == arm,
            "not a complete EB cell in the requested mode")
    require(type(result["capacity"]) is int and 1 <= result["capacity"] <= 60,
            "invalid entry capacity")
    require(result["model"] == data["model"] and result["runtime_versions"] == golden["runtime_versions"] and
            result["golden_absolute_tolerance"] == golden["absolute_tolerance"] == 0.0 and
            result["decoding_configuration"] == expected_decoding(data),
            "model/runtime or original exact-logit tolerance differs")
    require([r["question_id"] for r in result["warmup"]] == [r["question_id"] for r in data["warmup"]] and
            [r["question_id"] for r in result["requests"]] == [r["question_id"] for r in data["evaluation"]] and
            result["evaluation_generated_tokens"] == 128 and result["correctness_generated_tokens"] == 144,
            "official one-warmup/eight-request cohort differs")
    previous_ready = 0
    for request, row in zip(result["warmup"] + result["requests"], data["warmup"] + data["evaluation"]):
        expected = golden["requests"][str(row["question_id"])]
        check = request["correctness"]
        require(request["input_ids"] == row["input_ids"] == expected["input_ids"] and
                request["generated_ids"] == expected["generated_ids"] and len(request["generated_ids"]) == 16,
                "original-model input or exact generated tokens differ")
        require(check["exact_token_match"] is True and check["checked_generated_tokens"] == 16 and
                check["logits_checked"] is numerical, "missing actual per-request token/mode check")
        if numerical:
            require(check["compared_logits"] == 16 * 151936 and check["max_abs_error"] == 0.0,
                    "missing full, exact numerical gate")
        else:
            require(not ({"compared_logits", "max_abs_error"} & check.keys()) and "logits_file" not in request,
                    "timed cell contains numerical capture or fabricated numerical evidence")
        times = request["token_ready_ns"]
        require(len(times) == 16 and previous_ready <= request["begin_ns"] < times[0] and
                all(a <= b for a, b in zip(times, times[1:])) and times[-1] <= request["verified_ready_ns"],
                "real generated-token events incomplete")
        previous_ready = request["verified_ready_ns"]
        require(request["ttft_ns"] == times[0] - request["begin_ns"] and
                math.isclose(request["tpot_ns"], (times[-1] - times[0]) / 15, rel_tol=1e-12),
                "TTFT/TPOT differ from real token events")
    require(result["clock"] == "perf_counter_ns" and
            result["warmup"][-1]["verified_ready_ns"] <= result["begin_ns"] <= result["requests"][0]["begin_ns"] and
            previous_ready <= result["end_ns"] <= result["drain_begin_ns"] <= result["drain_end_ns"],
            "application/warmup/drain timestamp boundaries differ")
    metrics = audit_engagement(result, arm, numerical)
    metrics.update(correctness_only=numerical, evaluation_requests=8, correctness_requests=9,
                   evaluation_generated_tokens=128, correctness_generated_tokens=144)
    if numerical:
        require("tokens_per_second" not in result, "untimed preflight published throughput")
        metrics.update(compared_logits=9 * 16 * 151936, maximum_absolute_error=0.0)
    else:
        begin, end, drained = (result[key] for key in
            ("application_native_begin_ns", "application_native_end_ns", "native_drained_ns"))
        expected = {"elapsed_seconds": (end - begin) / 1e9, "tokens_per_second": 128e9 / (end - begin),
                    "elapsed_seconds_including_drain": (drained - begin) / 1e9,
                    "drain_seconds": (result["drain_end_ns"] - result["drain_begin_ns"]) / 1e9}
        require(all(type(result[key]) in (int, float) and math.isfinite(result[key]) and
                    math.isclose(result[key], value, rel_tol=1e-12, abs_tol=1e-12)
                    for key, value in expected.items()), "timing is not raw-derived")
        require(all(type(result[key]) in (int, float) and math.isfinite(result[key]) and result[key] >= 0
                    for key in ("cpu_seconds", "cpu_seconds_including_drain")) and
                result["cpu_seconds_including_drain"] >= result["cpu_seconds"], "invalid measured CPU cost")
        metrics.update(expected, tokens_per_second_including_drain=128e9 / (drained - begin),
                       median_ttft_ms=statistics.median(r["ttft_ns"] for r in result["requests"]) / 1e6,
                       median_tpot_ms=statistics.median(r["tpot_ns"] for r in result["requests"]) / 1e6,
                       cpu_seconds=result["cpu_seconds"], cpu_seconds_including_drain=result["cpu_seconds_including_drain"])
    return metrics


def audit_engagement(result, arm, numerical):
    """Actual executor/selector facts shared by both modes, not numerical claims."""
    before, after = result["eb_before"], result["eb_after"]
    fields = ("decisions", "jit_calls", "admissions", "evictions")
    require(all(type(s[field]) is int and s[field] >= 0 for s in (before, after) for field in fields),
            "invalid EB counters")
    delta = {field: after[field] - before[field] for field in fields}
    require(all(v >= 0 for v in delta.values()) and 0 < delta["evictions"] <= delta["admissions"] and
            after["registered_layers"] == 24, "actual EB eviction/admission/layer engagement missing")
    if arm == "bpf":
        require(after["jit_calls"] == after["decisions"] > 0 and delta["jit_calls"] == delta["decisions"],
                "actual BPF decisions did not all execute JIT")
        if numerical:
            shadow = result["shadow"]
            require(shadow is not None and all(type(v) is int for v in shadow.values()) and
                    shadow["checks"] == shadow["jit_calls"] == after["jit_calls"] and shadow["mismatches"] == 0 and
                    result["selector_library"] == str(HERE / "build/libeb_shadow.so") and
                    result["real_selector_library"] == str(HERE / "build/libeb_policy.so"),
                    "actual-input JIT/native shadow coverage is incomplete")
    else:
        require(result["shadow"] is None and before["jit_calls"] == after["jit_calls"] == 0,
                "non-BPF arm used a JIT or shadow")
    if not numerical or arm != "bpf":
        require(result["shadow"] is None and result["real_selector_library"] is None and
                result["selector_library"] == str(HERE / "build/libeb_policy.so"),
                "timed/native/FIFO arm did not directly use the real selector")
    begin, end, drained = (result[key] for key in
        ("application_native_begin_ns", "application_native_end_ns", "native_drained_ns"))
    require(result["application_clock"] == "steady_clock" and begin < end <= drained,
            "copy ledger lacks a common-clock bounded window")
    snap = result["after"]
    require(snap["drained"] is True and snap["copy_fields"] == common.FIELDS and snap["clock"] == "steady_clock",
            "common copy ledger not drained")
    counts, copies = snap["counters"], snap["copies"]
    for ident, row in enumerate(copies, 1):
        require(len(row) == 8 and all(type(v) is int and v >= 0 for v in row) and row[0] == ident and
                row[2] == EXPERT_BYTES and row[3] == 0 and begin <= row[4] <= row[5] <= drained and
                row[6] == row[7] == 0, "invalid whole-expert demand copy record")
    require(counts["demand_copy_started"] == counts["demand_copy_completed"] == len(copies) == delta["admissions"] and
            counts["demand_copy_bytes"] == len(copies) * EXPERT_BYTES,
            "real whole-expert copy and EB admission counts differ")
    require(all(type(v) is int and v >= 0 for v in counts.values()), "invalid common counters")
    require(all(value == 0 for name, value in counts.items() if name.startswith("prefetch_")) and
            counts["compute_release_sync_errors"] == 0,
            "speculative work or copy/compute failure in current-batch-only EB")
    require(counts["expert_demand_uses"] == counts["expert_demand_cache_hits"] + counts["expert_demand_cache_misses"] ==
            delta["decisions"] > 0 and counts["expert_demand_cache_misses"] == delta["admissions"] and
            counts["compute_release_syncs"] >= counts["expert_demand_uses"],
            "acquire/release lifetime, miss or decision accounting differs")
    require(snap["pool_capacity_bytes"] == result["before"]["pool_capacity_bytes"] == POOL_BYTES and
            0 < snap["pool_resident_bytes"] <= counts["peak_pool_resident_bytes"] <= POOL_BYTES and
            after["resident_sparse_bytes"] == snap["resident_sparse_bytes"] <= snap["sparse_budget_bytes"] and
            0 < after["resident_sparse_bytes"] <= 24 * result["capacity"] * EXPERT_BYTES and
            snap["resident_sparse_bytes"] + snap["resident_dense_bytes"] == snap["pool_resident_bytes"],
            "original strict pool, whole-expert K or residency conservation failed")
    return {"eb_delta": delta, "pool_capacity_bytes": POOL_BYTES}


def audit_retained_logits(directory, golden_dir, golden, result):
    # Independently recompute every retained original/current array pair after
    # the worker has exited. This is untimed correctness, never a performance cell.
    import numpy as np
    ids = [request["question_id"] for request in result["warmup"] + result["requests"]]
    require({path.name for path in directory.glob("*.npy")} ==
            {f"question-{qid}-logits.npy" for qid in ids} and
            {path.name for path in directory.glob("question-*-result.json")} ==
            {f"question-{qid}-result.json" for qid in ids}, "missing/extra retained numerical artifacts")
    for request in result["warmup"] + result["requests"]:
        qid = request["question_id"]
        name = f"question-{qid}-logits.npy"
        require(request["logits_file"] == name, "unexpected retained output path")
        actual = np.load(directory / name, allow_pickle=False, mmap_mode="r")
        expected = np.load(golden_dir / golden["requests"][str(qid)]["logits_file"],
                           allow_pickle=False, mmap_mode="r")
        require(actual.shape == expected.shape == (16, 1, 151936) and
                actual.dtype == expected.dtype == np.dtype("float32") and
                np.isfinite(actual).all() and np.isfinite(expected).all() and np.array_equal(actual, expected),
                f"persisted complete logits differ on question {qid}")
        retained = read_json(directory / f"question-{qid}-result.json")
        require(retained["status"] == "passed" and retained["request"] == request and
                retained["golden_absolute_tolerance"] == 0.0,
                "per-request raw evidence differs from the completed result")
    return {"arrays_recomputed": 9, "compared_logits": 9 * 16 * 151936,
            "exact_equal": True, "absolute_tolerance": 0.0}


def inventory(source):
    paths = [HERE / name for name in ("correctness.py", "inference_eb.py", "analyze_results.py", "build_adapter.py",
             "shadow_bridge.cpp", "build/libeb_policy.so", "build/eb_policy.bin", "build/libeb_shadow.so")]
    paths += [FINE / name for name in ("inference.py", "compare.py", "policy_runtime.py",
              "finemoe_copy_ledger.h", "finemoe_runtime_safety.h", "source-inventory.json", "dataset-mtbench-v1.json")]
    paths.append(FINE.parent / "moe-infinity/run_moe_head_to_head.py")
    paths.append(FINE / "analyze_results.py")
    paths += sorted((source / "finemoe").rglob("*.py"))
    paths += sorted((source / "core").rglob("*.h")) + sorted((source / "core").rglob("*.cpp"))
    paths.append(source / "core/eb_section_vi/adapter_live.inc")
    binaries = list((source / "finemoe/ops/prefetch").glob("prefetch_op*.so"))
    require(len(binaries) == 1 and not binaries[0].is_symlink(), "exactly one private built offloader required")
    return {str(path.resolve()): metadata(path) for path in paths + binaries}


def environment(arm, capacity, inherited=None, *, numerical=True):
    env, changes, removed = ORIGINAL_ENVIRONMENT(inherited)
    for name in tuple(env):
        if name.startswith("EB_SECTION_VI_") or name == "FINEMOE_POLICY":
            del env[name]
            removed.append(name)
    changes.update(EB_SECTION_VI_ARM=arm,
                   EB_SECTION_VI_CAPACITY=str(capacity), EB_SECTION_VI_BYTECODE=str(HERE / "build/eb_policy.bin"),
                   EB_SECTION_VI_LIBRARY=str(HERE / "build/libeb_policy.so"))
    if numerical:
        changes["EB_SECTION_VI_CORRECTNESS_ONLY"] = "1"
    if numerical and arm == "bpf":
        changes.update(EB_SECTION_VI_UNTIMED_SHADOW="1",
                       EB_SECTION_VI_REAL_LIBRARY=str(HERE / "build/libeb_policy.so"),
                       EB_SECTION_VI_LIBRARY=str(HERE / "build/libeb_shadow.so"))
    env.update(changes)
    return env, changes, removed


def command(args, _stage, output, arm):
    result = ["taskset", "-c", "8-11", str(FINE / ".venv/bin/python"), "-B", str(HERE / "inference_eb.py"),
            "--source", str(args.source), "--data", str(args.data), "--golden", str(args.golden),
            "--offload", str(FINE / "deps/qwen-offload-cache"), "--output", str(output),
            "--arm", arm, "--capacity", str(args.capacity)]
    return result + (["--check-logits"] if args.mode == "preflight" else [])


def orders():
    rows = list(itertools.permutations(ARMS))
    random.Random(SEED).shuffle(rows)
    return [list(row) for row in rows[:5]]


def reference_inventory(directory):
    return {str(path): metadata(path) for path in directory.iterdir()
            if path.name == "golden.json" or path.suffix == ".npy"}


def audit_saved_cell(directory, manifest, golden, numerical):
    """Recheck saved launch, cleanup, real policy and numerical-mode evidence."""
    arm = directory.name
    result, launch = (read_json(directory / name) for name in ("result.json", "launch.json"))
    worker = read_json(directory / "worker-result.json")
    require(result["status"] == "passed" and result["returncode"] == 0 and
            not any(result.get(key) for key in ("error", "cleanup_error", "cleanup_errors")) and
            result["stage"] == "cell" and result["arm"] == arm and result["diagnostic"] is False and
            launch["status"] == "running", "saved cell or owned cleanup failed")
    for key in ("stage", "arm", "command", "diagnostic", "environment", "environment_removed_names",
                "safety_before", "runtime_before"):
        require(result[key] == launch[key], f"saved launch/result differs: {key}")
    require(result["runtime_before"] == result["runtime_after"] == manifest["runtime"] and
            worker["capacity"] == manifest["capacity"] and worker["private_source"] == manifest["source"],
            "saved runtime/source/K differs")
    args = SimpleNamespace(source=Path(manifest["source"]), data=FINE / "dataset-mtbench-v1.json",
                           golden=Path(manifest["golden"]), capacity=manifest["capacity"],
                           mode="preflight" if numerical else "full")
    require(result["command"] == command(args, "cell", directory, arm) and
            result["environment"] == environment(arm, args.capacity, {}, numerical=numerical)[1],
            "actual child command/environment differs")
    require((directory / "worker.log").is_file(), "raw worker log missing")
    common.base.validate_pre_server_safety(result["safety_before"])
    common.base.validate_post_server_safety(result["safety_before"], result["safety_after"])
    require(common.base.validate_gpu_telemetry(directory / "gpu-telemetry.csv", allow_fixed_power_cap=True) ==
            result["telemetry"], "raw telemetry differs")
    metrics = audit_cell(worker, manifest["data"], arm, numerical, golden)
    require(metrics == result["metrics"], "raw metrics differ from saved gate")
    if not numerical:
        require(not list(directory.glob("*.npy")) and not list(directory.glob("question-*-result.json")),
                "formal timing contains retained numerical arrays/request dumps")
    return worker, metrics


def validate_preflight(directory, expected, golden):
    saved = read_json(directory / "campaign.json")
    require(saved["complete"] is True and saved["mode"] == "correctness_only" and
            saved["valid_cells"] == 3 and saved["valid_blocks"] == 1 and saved["arms"] == list(ARMS) and
            saved["orders"] == [list(ARMS)] and saved["preflight"] is None and
            set(saved["numerical_audits"]) == set(ARMS) and
            {path.name for path in directory.iterdir() if path.is_dir()} == set(ARMS) and
            not list(directory.rglob("*failure*.json")), "three-arm preflight is incomplete")
    for key in ("source", "capacity", "runtime", "data", "model_files", "reference_files", "golden"):
        require(saved[key] == expected[key], f"preflight/formal input differs: {key}")
    workers = {}
    previous = 0
    for arm in ARMS:
        path = directory / arm
        worker, _metrics = audit_saved_cell(path, saved, golden, True)
        audit = audit_retained_logits(path, Path(saved["golden"]), golden, worker)
        require(audit == saved["numerical_audits"][arm], "preflight persisted numerical audit differs")
        require(previous <= worker["application_native_begin_ns"], "preflight cells overlap")
        previous = worker["native_drained_ns"]
        workers[arm] = worker
    matching_decisions(workers)
    return previous


def matching_decisions(workers):
    for key in ("decisions", "admissions", "evictions", "resident_sparse_bytes", "registered_layers"):
        require(workers["native"]["eb_after"][key] == workers["bpf"]["eb_after"][key],
                f"real native/BPF final {key} differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--mode", choices=("preflight", "full"), default="preflight")
    parser.add_argument("--preflight", type=Path, help="completed same-runtime three-arm numerical campaign")
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.is_symlink(), "preserve existing attempt; choose fresh output")
    args.output, args.source = args.output.resolve(), args.source.resolve()
    require(args.source.is_relative_to(HERE / "build") and args.output.is_relative_to(HERE / "raw") and
            1 <= args.capacity <= 60, "private source/raw paths or capacity invalid")
    args.data, args.golden = FINE / "dataset-mtbench-v1.json", FINE / "raw/golden-v4/stage"
    numerical = args.mode == "preflight"
    require(numerical or args.preflight is not None, "full requires the actual three-arm preflight")
    require(not numerical or args.preflight is None, "preflight cannot reuse an earlier preflight")
    if args.preflight:
        args.preflight = args.preflight.resolve()
        require(args.preflight.is_relative_to(HERE / "raw"), "preflight must be this experiment's raw campaign")
    common.validate_reference(args.golden, "golden")
    data, golden = read_json(args.data), read_json(args.golden / "golden.json")
    frozen = inventory(args.source)
    references = reference_inventory(args.golden)
    models = common.model_inventory()
    manifest = {"mode": "correctness_only" if numerical else "full", "complete": False, "arms": list(ARMS),
                "capacity": args.capacity, "data": data, "runtime": frozen, "model_files": models,
                "reference_files": references, "golden": str(args.golden), "source": str(args.source),
                "valid_cells": 0, "valid_blocks": 0, "seed": SEED,
                "orders": [list(ARMS)] if numerical else orders(),
                "preflight": str(args.preflight) if args.preflight else None, "numerical_audits": {}}
    previous_drained = 0 if numerical else validate_preflight(args.preflight, manifest, golden)
    # Process-local callbacks adapt the existing runner without editing its
    # frozen source or copying its telemetry/lease/failure-retention framework.
    common.command, common.audit_cell = command, audit_cell
    common.inventory = lambda: inventory(args.source)
    common.base.stop_owned_process_group = common.base.stop_exact_process = stop_owned
    def interrupted(signum, _frame):
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_IGN)
        raise InterruptedError(f"EB controller received signal {signum}")
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    lease = None
    try:
        lease = common.base.LeaseSet.acquire()
        args.output.mkdir(parents=True, exist_ok=False)
        common.base.atomic_write_json(args.output / "campaign.json", manifest)
        for block, order in enumerate(manifest["orders"]):
            results = {}
            for arm in order:
                require(common.model_inventory() == models and reference_inventory(args.golden) == references,
                        "original model/reference files changed before child")
                common.child_environment = lambda **_unused: environment(arm, args.capacity, numerical=numerical)
                directory = (args.output if numerical else args.output / f"block-{block:02d}") / arm
                common.run_stage(args, "cell", directory, frozen, data, arm)
                result = read_json(directory / "worker-result.json")
                require(result["capacity"] == args.capacity and result["private_source"] == str(args.source) and
                        previous_drained <= result["application_native_begin_ns"],
                        "worker source/K differs or real cell intervals overlap")
                previous_drained = result["native_drained_ns"]
                if numerical:
                    manifest["numerical_audits"][arm] = audit_retained_logits(directory, args.golden, golden, result)
                else:
                    require(not list(directory.glob("*.npy")) and not list(directory.glob("question-*-result.json")),
                            "formal cell wrote preparation arrays/request dumps")
                require(references == reference_inventory(args.golden) and common.model_inventory() == models,
                        "original model/reference files changed")
                results[arm] = result
                manifest["valid_cells"] += 1
                common.base.atomic_write_json(args.output / "campaign.json", manifest)
            matching_decisions(results)
            manifest["valid_blocks"] += 1
            common.base.atomic_write_json(args.output / "campaign.json", manifest)
        manifest["complete"] = True
        common.base.atomic_write_json(args.output / "campaign.json", manifest)
    except BaseException as error:
        if args.output.is_dir():
            common.base.atomic_write_json(args.output / "campaign-failure.json", {
                "complete": False, "error": f"{type(error).__name__}: {error}",
                "note": "retain all cells; preflight/partial blocks never count as full performance blocks"})
        raise
    finally:
        if lease is not None:
            lease.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
