"""Read-only, complete-five-block FineMoE analysis. JSON on stdout only.

Reuses the real cell correctness/engagement gate, but reconstructs display
timings and copy partitions from request events and copy-generation lifetimes.
No GPU access, inferred missing cells, epsilon ratios, or performance filtering.
Run with this workload's .venv/bin/python (NumPy is required for saved logits).
"""
import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import random
import statistics
from types import SimpleNamespace

import numpy as np

import compare

ARMS = ("demand-only", "all-positive", "finemoe-c", "finemoe-bpf")
SEED, DRAWS = 20260903, 10000
COMPARISONS = (
    ("all-positive", "demand-only"), ("finemoe-c", "demand-only"),
    ("finemoe-bpf", "demand-only"), ("finemoe-c", "all-positive"),
    ("finemoe-bpf", "all-positive"), ("finemoe-bpf", "finemoe-c"),
)
read = compare.read_json
require = compare.require


def reconstruct(worker):
    """Independent arithmetic; call audit_cell first to validate raw records."""
    begin, end, drained = (worker[key] for key in (
        "application_native_begin_ns", "application_native_end_ns", "native_drained_ns"))
    requests = [{"question_id": row["question_id"],
                 "generated_tokens": len(row["generated_ids"]),
                 "ttft_ms": (row["token_ready_ns"][0] - row["begin_ns"]) / 1e6,
                 "tpot_ms": (row["token_ready_ns"][-1] - row["token_ready_ns"][0]) /
                            (len(row["generated_ids"]) - 1) / 1e6}
                for row in worker["requests"]]
    tokens = sum(row["generated_tokens"] for row in requests)
    metrics = {
        "tokens_per_second": tokens * 1e9 / (end - begin),
        "tokens_per_second_including_drain": tokens * 1e9 / (drained - begin),
        "application_seconds": (end - begin) / 1e9,
        "including_drain_seconds": (drained - begin) / 1e9,
        "post_window_seconds": (drained - end) / 1e9,
        "drain_seconds": (worker["drain_end_ns"] - worker["drain_begin_ns"]) / 1e9,
        "median_ttft_ms": statistics.median(row["ttft_ms"] for row in requests),
        "median_tpot_ms": statistics.median(row["tpot_ms"] for row in requests),
        "cpu_seconds": worker["cpu_seconds"],
        "cpu_seconds_including_drain": worker["cpu_seconds_including_drain"],
        "peak_pool_resident_bytes": worker["after"]["counters"]["peak_pool_resident_bytes"],
    }
    require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                for value in metrics.values()), "nonfinite or negative display metric")
    require(metrics["cpu_seconds_including_drain"] >= metrics["cpu_seconds"],
            "CPU cost decreased while draining")
    fields = [f"{kind}_copy_{unit}" for kind in ("demand", "prefetch")
              for unit in ("completed", "bytes")]
    fields += [f"prefetch_{category}_{unit}" for category in
               ("first_use", "evicted_unused", "resident_unused") for unit in ("copies", "bytes")]
    partitions = {name: dict.fromkeys(fields, 0) for name in ("drained", "in_window", "tail")}
    for kind in ("demand", "prefetch"):
        for category in ("started_after_window", "inflight_at_deadline"):
            partitions["tail"][f"{kind}_{category}_bytes"] = 0
    for _, _, size, speculative, started, completed, used, evicted in worker["after"]["copies"]:
        kind = "prefetch" if speculative else "demand"
        window = "in_window" if completed <= end else "tail"
        for name in ("drained", window):
            partition = partitions[name]
            partition[f"{kind}_copy_completed"] += 1
            partition[f"{kind}_copy_bytes"] += size
            if speculative:
                boundary = end if name == "in_window" else drained
                category = ("first_use" if used and used <= boundary else
                            "evicted_unused" if evicted and evicted <= boundary else "resident_unused")
                partition[f"prefetch_{category}_copies"] += 1
                partition[f"prefetch_{category}_bytes"] += size
        if window == "tail":
            category = "started_after_window" if started > end else "inflight_at_deadline"
            partitions["tail"][f"{kind}_{category}_bytes"] += size
    for window, partition in partitions.items():
        require(sum(partition[f"prefetch_{category}_bytes"] for category in
                    ("first_use", "evicted_unused", "resident_unused")) == partition["prefetch_copy_bytes"],
                f"{window} speculative-byte partition does not conserve")
        metrics.update({f"{window}.{key}": value for key, value in partition.items()})
    return metrics, requests


def check_derived(metrics, audited):
    for key in ("tokens_per_second", "tokens_per_second_including_drain", "drain_seconds",
                "post_window_seconds", "median_ttft_ms", "median_tpot_ms", "cpu_seconds",
                "cpu_seconds_including_drain", "peak_pool_resident_bytes"):
        require(math.isclose(metrics[key], audited[key], rel_tol=1e-12, abs_tol=1e-12),
                f"independent raw arithmetic differs: {key}")
    for window, saved in (("drained", "copies_through_drain"),
                          ("in_window", "copies_completed_in_application_window"),
                          ("tail", "copies_completed_after_application_window")):
        for key, value in audited[saved].items():
            # Started==completed is already gated after drain; display counts completed.
            if not key.endswith("_copy_started"):
                require(metrics[f"{window}.{key}"] == value, f"independent copy partition differs: {key}")


def expected_orders():
    rows = list(itertools.permutations(ARMS))
    random.Random(SEED).shuffle(rows)
    return [list(row) for row in rows[:5]]


def validate_matrix(directory, manifest, mode="full"):
    blocks = 5 if mode == "full" else 1
    require(manifest.get("schema") == "finemoe_dynamic_set_v1" and manifest.get("mode") == mode and
            manifest.get("complete") is True and manifest.get("diagnostic") is False and
            manifest.get("valid_blocks") == blocks and manifest.get("seed") == SEED and
            manifest.get("orders") == expected_orders()[:blocks],
            f"not the complete frozen {blocks}-block {mode} campaign")
    require(not any(manifest.get(key) for key in ("rejected", "incomplete", "unexpected", "error")),
            "campaign retains rejected/incomplete/unexpected attempts")
    require(not list(directory.rglob("*failure*.json")), "campaign contains a retained failure")
    cells = [directory / f"block-{block:02d}" / arm
             for block, order in enumerate(manifest["orders"]) for arm in order]
    expected = set(cells)
    require({p for p in directory.iterdir() if p.is_dir()} == {p.parent for p in cells},
            "unexpected/missing block or attempt directory")
    require({p for block in {p.parent for p in cells} for p in block.iterdir() if p.is_dir()} == expected,
            "unexpected/missing arm attempt directory")
    for filename in ("launch.json", "result.json", "worker-result.json"):
        require({p.parent for p in directory.rglob(filename)} == expected,
                f"unexpected/missing attempts: {filename}")
    return cells


def validate_references(manifest):
    data, golden_dir, history_dir = manifest["data"], Path(manifest["golden"]), Path(manifest["history"])
    require(data == read(compare.HERE / "dataset-mtbench-v1.json"), "frozen 64/8/1 cohort differs")
    references = {}
    for directory, mode in ((golden_dir, "golden"), (history_dir, "history")):
        compare.validate_reference(directory, mode)
        prep = read(directory.parent / "campaign.json")
        require(prep["data"] == data, f"{mode} data differs")
        if mode == "history":
            require(prep["golden"] == manifest["golden"] and
                    prep["reference_files"] == {path: item for path, item in manifest["reference_files"].items()
                                                 if Path(path).parent == golden_dir},
                    "history was built from a different golden reference")
        for path in directory.iterdir():
            if path.name in ("golden.json", "history.json") or path.suffix == ".npy":
                references[str(path)] = compare.metadata(path)
    require(references == manifest["reference_files"], "golden/history file inventory changed")
    for name in ("store~embed~1000.npy", "store~traj~1000.npy"):
        require(str(history_dir / name) in references, "full frozen history array missing")
    golden, history = read(golden_dir / "golden.json"), read(history_dir / "history.json")
    for record in (golden, history):
        require(record["status"] == "passed" and record["model"] == data["model"] and
                read(Path(record["data"])) == data, "reference model/data/status differs")
    require(history["runtime_versions"] == golden["runtime_versions"] and
            history["store_capacity"] == history["store_data_size"] == 1000 and
            history["question_ids"] == [row["question_id"] for row in data["history"]],
            "history versions, capacity, or cohort differs")
    require([row["question_id"] for row in history["requests"]] == history["question_ids"],
            "history raw request coverage differs")
    for row, frozen in zip(history["requests"], data["history"]):
        expected = golden["requests"][str(row["question_id"])]
        require(row["input_ids"] == frozen["input_ids"] == expected["input_ids"] and
                len(row["generated_ids"]) == 16 and row["generated_ids"] == expected["generated_ids"] and
                row["correctness"]["exact_token_match"] is True and
                row["correctness"]["checked_generated_tokens"] == 16,
                "history raw tokens differ from original golden")
    decoding = history["decoding_configuration"]
    checkpoint = read(Path(data["model"]["snapshot"]) / "generation_config.json")
    require(decoding["checkpoint_fields"] == {key: checkpoint[key] for key in
            ("do_sample", "repetition_penalty", "eos_token_id", "pad_token_id", "temperature", "top_k", "top_p")} and
            decoding["explicit_overrides"] == {"do_sample": False, "min_new_tokens": 16,
                                               "max_new_tokens": 16, "pad_token_id": 151643},
            "history decoding configuration differs from checkpoint and frozen overrides")
    require(manifest["runtime"] and manifest["model_files"], "runtime/model inventory absent")
    for item in manifest["model_files"]:
        require(compare.metadata(Path(item["path"])) == item, "original model file inventory changed")
    return golden


def saved_cell(directory, manifest, golden, numerical=False):
    arm = directory.name
    result, launch = read(directory / "result.json"), read(directory / "launch.json")
    worker = read(directory / "worker-result.json")
    require(result["status"] == "passed" and result["returncode"] == 0 and
            not any(result.get(key) for key in ("error", "cleanup_error", "cleanup_errors")),
            f"failed/incomplete cell: {directory}")
    require(result["stage"] == "cell" and result["arm"] == arm and result.get("diagnostic") is False and
            launch["status"] == "running", "cell identity/launch differs")
    for key in ("stage", "arm", "command", "diagnostic", "environment", "environment_removed_names",
                "safety_before", "runtime_before"):
        require(launch[key] == result[key], f"saved launch/result differ: {key}")
    require(result["runtime_before"] == result["runtime_after"] == manifest["runtime"],
            "mixed runtime files across cell/campaign")
    args = SimpleNamespace(mode="preflight" if numerical else "full", data=Path(worker["data"]),
                           golden=Path(manifest["golden"]), history=Path(manifest["history"]))
    require(read(args.data) == manifest["data"] and worker["clock"] == "perf_counter_ns" and
            result["command"] == compare.command(args, "cell", directory, arm), "actual worker command/data differs")
    require(worker["decoding_configuration"] == read(args.history / "history.json")["decoding_configuration"],
            "cell/history decoding configuration differs")
    require(result["environment"] == compare.child_environment({})[1] and
            set(result["environment_removed_names"]) <= {"PYTORCH_CUDA_ALLOC_CONF", "PYTHONFAULTHANDLER"},
            "actual common environment differs")
    require((directory / "worker.log").is_file(), "raw worker log missing")
    compare.base.validate_pre_server_safety(result["safety_before"])
    compare.base.validate_post_server_safety(result["safety_before"], result["safety_after"])
    require(compare.base.validate_gpu_telemetry(directory / "gpu-telemetry.csv", allow_fixed_power_cap=True) ==
            result["telemetry"], "raw telemetry differs from saved summary")
    audited = compare.audit_cell(worker, manifest["data"], arm, numerical, golden)
    require(audited == result["metrics"], "raw correctness/engagement audit differs from saved metrics")
    metrics, requests = reconstruct(worker)
    check_derived(metrics, audited)
    return {"block": int(directory.parent.name.split("-")[1]), "arm": arm,
            "path": str(directory), "metrics": metrics, "requests": requests,
            "application_begin_ns": worker["application_native_begin_ns"],
            "application_end_ns": worker["application_native_end_ns"],
            "drained_ns": worker["native_drained_ns"],
            "pool_capacity_bytes": audited["pool_capacity_bytes"],
            "policy_delta": audited["policy_delta"],
            "executor_counters": worker["after"]["counters"],
            "observed_dynamic_set_reduction": audited["observed_dynamic_set_reduction"],
            "observed_actual_prefetch_copy": audited["observed_actual_prefetch_copy"]}


def logit_difference(actual_path, reference_path):
    """Compare saved float32 arrays in token-sized slices, without model imports."""
    actual = np.load(actual_path, mmap_mode="r", allow_pickle=False)
    reference = np.load(reference_path, mmap_mode="r", allow_pickle=False)
    require(actual.shape == reference.shape and actual.ndim == 3 and
            actual.shape[:2] == (16, 1) and actual.shape[2] > 0 and
            actual.dtype == reference.dtype == np.dtype("float32"), "saved logit shape/dtype differs")
    maximum = 0.0
    for output, original in zip(actual, reference):
        require(np.isfinite(output).all() and np.isfinite(original).all(), "saved logits contain NaN/Inf")
        delta = np.abs(output - original)
        require(np.isfinite(delta).all(), "saved logit difference is nonfinite")
        maximum = max(maximum, float(np.max(delta)))
    return {"shape": list(actual.shape), "compared_logits": int(actual.size), "max_abs_error": maximum}


def preflight_numerics(directory, worker, golden, data, golden_dir):
    tolerance = golden["absolute_tolerance"]
    require(math.isfinite(tolerance) and tolerance >= 0 and
            tolerance == golden["same_arm_repeat_max_abs_error"], "fixed golden tolerance differs")
    records = worker["warmup"] + worker["requests"]
    expected_ids = [row["question_id"] for row in data["warmup"] + data["evaluation"]]
    require([row["question_id"] for row in records] == expected_ids, "preflight 1+8 raw coverage differs")
    require({path.name for path in directory.glob("*.npy")} ==
            {f"question-{qid}-logits.npy" for qid in expected_ids} and
            {path.name for path in directory.glob("question-*-result.json")} ==
            {f"question-{qid}-result.json" for qid in expected_ids}, "missing/extra preflight numerical artifacts")
    checks = []
    for row in records:
        qid = row["question_id"]
        retained = read(directory / f"question-{qid}-result.json")
        expected = golden["requests"][str(qid)]
        require(retained["status"] == "passed" and not retained.get("error") and
                retained["golden_absolute_tolerance"] == tolerance and
                retained["expected_generated_ids"] == row["generated_ids"] == expected["generated_ids"] and
                row["input_ids"] == expected["input_ids"], "retained preflight record/token mismatch")
        saved = retained["request"]
        # Evaluation's final timestamp is taken after saving/checking the raw
        # preparation evidence. All other raw fields must be identical.
        require({key: value for key, value in saved.items() if key != "verified_ready_ns"} ==
                {key: value for key, value in row.items() if key != "verified_ready_ns"} and
                row["token_ready_ns"][-1] <= saved["verified_ready_ns"] <= row["verified_ready_ns"],
                "retained preflight request differs from worker record")
        filename = f"question-{qid}-logits.npy"
        require(row["logits_file"] == filename and expected["logits_file"] == filename,
                "preflight/golden logit file identity differs")
        measured = logit_difference(directory / filename, golden_dir / filename)
        require(measured["max_abs_error"] <= tolerance and
                measured["max_abs_error"] == row["correctness"]["max_abs_error"] and
                measured["compared_logits"] == row["correctness"]["compared_logits"],
                "independent saved-array numerical comparison failed")
        checks.append({"question_id": qid, **measured})
    return checks


def audit_preflight(manifest, golden):
    require(manifest.get("preflight"), "full manifest lacks its actual numerical preflight path")
    directory = Path(manifest["preflight"]).resolve()
    preflight = read(directory / "campaign.json")
    paths = validate_matrix(directory, preflight, "preflight")
    for key in ("runtime", "data", "reference_files", "model_files", "golden", "history"):
        require(preflight[key] == manifest[key], f"preflight/full reference differs: {key}")
    reports, cells, workers = {}, [], {}
    for path in paths:
        cells.append(saved_cell(path, preflight, golden, numerical=True))
        worker = read(path / "worker-result.json")
        workers[path.name] = worker
        reports[path.name] = preflight_numerics(path, worker, golden, manifest["data"], Path(manifest["golden"]))
    require(workers["finemoe-c"]["policy_after"]["events"] == workers["finemoe-bpf"]["policy_after"]["events"],
            "C/BPF actual-input selector/API event streams differ")
    require(len({row["pool_capacity_bytes"] for row in cells}) == 1 and
            all(a["drained_ns"] <= b["application_begin_ns"] for a, b in zip(cells, cells[1:])),
            "preflight common pool/timeline differs")
    return {"path": str(directory), "complete": True, "cells": 4, "raw_logit_arrays": 36,
            "fixed_absolute_tolerance": golden["absolute_tolerance"], "numerical_checks": reports,
            "last_drained_ns": cells[-1]["drained_ns"], "pool_capacity_bytes": cells[0]["pool_capacity_bytes"]}


def percentile(values, q):
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = int(index)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (index - low)


def paired(candidate, reference, draws):
    require(len(candidate) == len(reference) == 5 and
            all(math.isfinite(v) and v >= 0 for v in candidate + reference), "invalid paired values")
    differences = [a - b for a, b in zip(candidate, reference)]
    sampled = [sum(differences[i] for i in draw) / 5 for draw in draws]
    result = {"candidate": candidate, "reference": reference,
              "paired_absolute_differences": differences,
              "mean_paired_absolute_difference": statistics.mean(differences),
              "absolute_difference_ci95": [percentile(sampled, .025), percentile(sampled, .975)],
              "paired_ratios": [a / b if b > 0 else None for a, b in zip(candidate, reference)]}
    if all(a > 0 and b > 0 for a, b in zip(candidate, reference)):
        logs = [math.log(a) - math.log(b) for a, b in zip(candidate, reference)]
        samples = [math.exp(sum(logs[i] for i in draw) / 5) for draw in draws]
        result.update(ratio_defined=True, geometric_mean_ratio=math.exp(statistics.mean(logs)),
                      ratio_ci95=[percentile(samples, .025), percentile(samples, .975)])
    else:
        result.update(ratio_defined=False, geometric_mean_ratio=None, ratio_ci95=None,
                      ratio_unavailable_reason="zero value: no positive log-ratio; no epsilon or cell removal")
    return result


def aggregate(cells):
    rng = random.Random(SEED)
    draws = [tuple(rng.randrange(5) for _ in range(5)) for _ in range(DRAWS)]
    by_arm = {arm: sorted((row for row in cells if row["arm"] == arm), key=lambda r: r["block"])
              for arm in ARMS}
    require(all([row["block"] for row in rows] == list(range(5)) for rows in by_arm.values()),
            "cannot aggregate partial or duplicate blocks")
    keys = list(cells[0]["metrics"])
    require(all(set(row["metrics"]) == set(keys) for row in cells), "display metrics differ across cells")
    medians = {arm: {key: statistics.median(row["metrics"][key] for row in rows) for key in keys}
               for arm, rows in by_arm.items()}
    comparisons = {}
    for candidate, reference in COMPARISONS:
        comparisons[f"{candidate}_over_{reference}"] = {
            key: paired([r["metrics"][key] for r in by_arm[candidate]],
                        [r["metrics"][key] for r in by_arm[reference]], draws) for key in keys}
    return medians, comparisons


def analyze(directory):
    directory = directory.resolve()
    manifest = read(directory / "campaign.json")
    paths = validate_matrix(directory, manifest)
    golden = validate_references(manifest)
    preflight = audit_preflight(manifest, golden)
    cells = [saved_cell(path, manifest, golden) for path in paths]
    require(len({row["pool_capacity_bytes"] for row in cells}) == 1, "common pool capacity changed")
    require(cells[0]["pool_capacity_bytes"] == preflight["pool_capacity_bytes"] and
            preflight["last_drained_ns"] <= cells[0]["application_begin_ns"],
            "full pool/timeline differs from numerical preflight")
    require(all(a["drained_ns"] <= b["application_begin_ns"] for a, b in zip(cells, cells[1:])),
            "raw common-clock intervals overlap or disagree with randomized order")
    medians, comparisons = aggregate(cells)
    return {"schema": "finemoe_independent_results_v1", "complete": True,
            "campaign": str(directory), "valid_cells": 20, "valid_blocks": 5,
            "arms": list(ARMS), "orders": manifest["orders"], "cells": cells,
            "preflight": preflight, "history_raw_tokens_checked": 64 * 16,
            "arm_medians": medians, "comparisons": comparisons,
            "statistics": {"paired_blocks": 5, "bootstrap_draws": DRAWS, "seed": SEED,
                           "method": "whole-paired-block percentile bootstrap; geometric mean ratios",
                           "zero_values": "retain cells; undefined log-ratio; paired absolute mean difference/CI",
                           "median_ci": "not estimated; paired CIs are not median error bars"},
            "provenance": {key: manifest[key] for key in
                           ("source_revision", "runtime", "model_files", "reference_files", "golden", "history", "preflight")},
            "limits": ["Dynamic-set component port, not complete FineMoE reproduction or kernel-UVM execution.",
                       "Logical completed payload bytes, not measured PCIe traffic or saved transfer time.",
                       "in_window uses completed_ns <= application end and lifecycle state at that cutoff; "
                       "tail counts only copies completed later, classified at final drain.",
                       "drained also includes later eviction of an in-window copy; category changes are not tail bytes.",
                       "Resident-unused is right-censored, not evidence of wasted copies.",
                       "CPU and executor counters are recorded deltas, not reconstructed OS snapshots.",
                       "Formal/history token IDs and all 36 saved preflight logit arrays are independently rechecked; "
                       "formal timing does not save or compare full logits.",
                       "Recorded runtime inventories must match all 20 cells; available model/reference metadata is rechecked.",
                       "A ratio CI crossing one is inconclusive, not equivalence; no sign-based cell selection."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    try:
        output, status = analyze(args.campaign), 0
    except (OSError, ValueError, KeyError, TypeError, compare.base.GateError) as exc:
        output, status = {"complete": False, "error": f"{type(exc).__name__}: {exc}"}, 2
    print(json.dumps(output, indent=2, allow_nan=False))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
