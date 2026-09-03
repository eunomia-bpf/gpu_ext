"""Thin real FineMoE controller, reusing the project's GPU leases and safety gates.

Preparation is staged: golden -> history -> four-arm preflight -> five blocks.
No synthetic runner, native fallback, model reduction, or partial-block success.
"""
import argparse
from collections import Counter
import itertools
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "moe-infinity"))
import run_moe_head_to_head as base

ARMS = ("demand-only", "all-positive", "finemoe-c", "finemoe-bpf")
SEED = 20260903
FIELDS = "id,node,bytes,speculative,started_ns,completed_ns,first_use_ns,evicted_ns"


def require(condition, message):
    if not condition:
        raise base.GateError(message)


def read_json(path):
    return json.loads(path.read_text())


def metadata(path):
    stat = path.stat()
    # Keep official snapshot filenames, not internal content-addressed cache targets.
    return {"path": str(path.absolute()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def inventory():
    source = HERE / "deps/FineMoE-EuroSys26"
    paths = [HERE / name for name in (
        "inference.py", "compare.py", "policy_runtime.py", "finemoe_copy_ledger.h", "finemoe_runtime_safety.h",
        "build/libfinemoe_policy.so", "build/finemoe_policy.bin", "source-inventory.json")]
    paths += [HERE.parent / "moe-infinity/run_moe_head_to_head.py"]
    paths += sorted((source / "finemoe").rglob("*.py"))
    paths += sorted((source / "finemoe/ops/prefetch").glob("*.so"))
    require(len(list((source / "finemoe/ops/prefetch").glob("*.so"))) == 1,
            "exactly one private compiled offload extension required")
    for name in ("core/model/model_topology.cpp", "core/prefetch/task_scheduler.cpp",
                 "core/prefetch/archer_prefetch_handle.cpp", "core/python/py_archer_prefetch.cpp",
                 "core/memory/memory_pool.cpp"):
        paths.append(source / name)
    return {str(path.resolve()): metadata(path) for path in paths}


def model_inventory():
    model = read_json(HERE / "source-inventory.json")["model"]
    result = []
    for row in model["files"]:
        path = Path(model["snapshot"]) / row["name"]
        require(path.stat().st_size == row["bytes"], f"original model file size differs: {path.name}")
        result.append(metadata(path))
    return result


def policy_delta(result):
    before, after = result["policy_before"], result["policy_after"]["stats"]
    return {key: value - before.get(key, 0) for key, value in after.items()
            if type(value) is int}


def audit_policy_events(result, arm, numerical):
    after = result["policy_after"]
    events, stats = after["events"], after["stats"]
    if not numerical:
        require(not events and stats["shadow"] is False, "formal timing contains shadow/capture work")
        return
    require(stats["shadow"] is True, "numerical preflight lacks independent oracle")
    selectors = [event for event in events if event["event"] == "selector"]
    require(len(selectors) == stats["prediction_maps"] > 0, "raw prediction maps absent")
    selected = rows = 0
    for event in selectors:
        probabilities, masks, delta = event["probabilities"], event["masks"], event["delta"]
        require(len(probabilities) == len(masks) == event["layer_end"] - event["layer_start"] and
                math.isfinite(delta) and 0 <= delta <= 1, "invalid actual selector band/delta")
        for vector, mask in zip(probabilities, masks):
            require(len(vector) == 60 and all(math.isfinite(v) and 0 <= v <= 1 for v in vector),
                    "missing original full 60-expert probabilities")
            if arm == "demand-only":
                expected = 0
            elif arm == "all-positive":
                expected = sum(1 << i for i, value in enumerate(vector) if value > 0)
            else:
                expected, cumulative = 0, 0.0
                if any(vector):
                    for count, expert in enumerate(sorted(range(60), key=lambda i: (-vector[i], i)), 1):
                        cumulative += vector[expert]  # Independent sequential binary64 oracle.
                        expected |= 1 << expert
                        if count >= 4 and cumulative >= delta:
                            break
            require(mask == expected, "raw actual-input mask differs from independent paper oracle")
            selected += mask.bit_count()
            rows += 1
    require(rows == stats["selector_rows"] and selected == stats["selected_candidates"],
            "actual captured selection counts differ")
    candidates = [e for e in events if e["event"] == "engine_candidates"]
    enqueues = [e for e in events if e["event"] == "engine_enqueue"]
    require(sum(len(e["tensor_ids"]) for e in candidates) == selected == len(enqueues) and
            len(enqueues) == stats.get("engine_enqueue_calls", 0), "captured engine admission differs")


def audit_cell(result, data, arm, numerical, golden):
    require(result["status"] == "passed" and result["arm"] == arm, "wrong worker status/arm")
    require(result["model"] == data["model"] and result["check_logits"] is numerical,
            "wrong model or numerical stage")
    require(result["runtime_versions"] == golden["runtime_versions"], "golden/arm software versions differ")
    audit_policy_events(result, arm, numerical)
    requests = result["requests"]
    require([r["question_id"] for r in requests] == [r["question_id"] for r in data["evaluation"]],
            "held-out request order differs")
    previous = result["begin_ns"]
    require([r["question_id"] for r in result["warmup"]] == [r["question_id"] for r in data["warmup"]],
            "disjoint warmup missing")
    for request in result["warmup"] + requests:
        expected = golden["requests"][str(request["question_id"])]
        require(request["input_ids"] == expected["input_ids"] and request["generated_ids"] == expected["generated_ids"],
                "raw generated token IDs differ from original-model golden")
        check = request["correctness"]
        require(check["exact_token_match"] is True and check["checked_generated_tokens"] == 16 and
                check["logits_checked"] is numerical, "missing real warmup/request checks")
        if numerical:
            require(check["compared_logits"] > 0 and math.isfinite(check["max_abs_error"]) and
                    check["max_abs_error"] <= golden["absolute_tolerance"], "numerical gate failed")
    require(result["golden_absolute_tolerance"] == golden["absolute_tolerance"], "numerical tolerance changed")
    for request, frozen in zip(requests, data["evaluation"]):
        require(request["input_ids"] == frozen["input_ids"] and len(request["generated_ids"]) == 16,
                "wrong real request/token count")
        times = request["token_ready_ns"]
        require(len(times) == 16 and previous <= request["begin_ns"] < times[0] and
                all(a <= b for a, b in zip(times, times[1:])) and
                times[-1] <= request["verified_ready_ns"] <= result["end_ns"], "invalid raw token timestamps")
        previous = request["verified_ready_ns"]
        require(request["ttft_ns"] == times[0] - request["begin_ns"] and
                math.isclose(request["tpot_ns"], (times[-1] - times[0]) / 15, rel_tol=1e-12),
                "TTFT/TPOT differ from raw generated-token events")
        check = request["correctness"]
        require(check["exact_token_match"] is True and check["checked_generated_tokens"] == 16,
                "missing actual golden token check")
        require(check["logits_checked"] is numerical, "wrong numerical coverage")
        if numerical:
            require(check["compared_logits"] > 0 and math.isfinite(check["max_abs_error"]) and
                    check["max_abs_error"] <= result["golden_absolute_tolerance"], "logit gate failed")
    begin, cutoff, drained = (result[key] for key in
        ("application_native_begin_ns", "application_native_end_ns", "native_drained_ns"))
    require(result["application_clock"] == "steady_clock" and begin < cutoff <= drained,
            "application/copy common clock missing")
    duration = (cutoff - begin) / 1e9
    require(duration > 0 and result["generated_tokens"] == 128 and
            math.isclose(result["elapsed_seconds"], duration, rel_tol=1e-12) and
            math.isclose(result["tokens_per_second"], 128 / duration, rel_tol=1e-12), "throughput is not raw-derived")
    require(result["end_ns"] <= result["drain_begin_ns"] <= result["drain_end_ns"] and
            math.isclose(result["drain_seconds"], (result["drain_end_ns"] - result["drain_begin_ns"]) / 1e9,
                         rel_tol=1e-12, abs_tol=1e-12) and
            math.isclose(result["elapsed_seconds_including_drain"], (drained - begin) / 1e9, rel_tol=1e-12),
            "drain/tail cost missing or inconsistent")
    snap = result["after"]
    require(snap["drained"] is True and snap["copy_fields"] == FIELDS and snap["clock"] == "steady_clock",
            "copy ledger schema/drain absent")
    counts = snap["counters"]
    rebuilt, in_window, tail = Counter(), Counter(), Counter()
    for expected_id, row in enumerate(snap["copies"], 1):
        require(len(row) == 8 and all(type(v) is int and v >= 0 for v in row), "malformed raw copy row")
        ident, node, size, spec, started, finished, used, evicted = row
        require(ident == expected_id and size > 0 and spec in (0, 1) and begin <= started <= finished <= drained,
                "copy incomplete or IDs/time invalid")
        require((not used or finished <= used <= cutoff) and (not evicted or finished <= evicted <= drained) and
                (not used or not evicted or used <= evicted), "copy lifecycle order invalid")
        prefix = "prefetch" if spec else "demand"
        rebuilt[f"{prefix}_copy_started"] += 1
        rebuilt[f"{prefix}_copy_completed"] += 1
        rebuilt[f"{prefix}_copy_bytes"] += size
        if finished <= cutoff:
            in_window[f"{prefix}_copy_completed"] += 1
            in_window[f"{prefix}_copy_bytes"] += size
            if spec:
                boundary_category = ("first_use" if used and used <= cutoff else
                                     "evicted_unused" if evicted and evicted <= cutoff else "resident_unused")
                in_window[f"prefetch_{boundary_category}_copies"] += 1
                in_window[f"prefetch_{boundary_category}_bytes"] += size
        else:
            tail[f"{prefix}_copy_completed"] += 1
            tail[f"{prefix}_copy_bytes"] += size
            tail[f"{prefix}_{'started_after_window' if started > cutoff else 'inflight_at_deadline'}_bytes"] += size
        if spec:
            category = "first_use" if used else "evicted_unused" if evicted else "resident_unused"
            rebuilt[f"prefetch_{category}_copies"] += 1
            rebuilt[f"prefetch_{category}_bytes"] += size
    fields = [f"{p}_copy_{s}" for p in ("demand", "prefetch") for s in ("started", "completed", "bytes")]
    fields += [f"prefetch_{c}_{u}" for c in ("first_use", "evicted_unused", "resident_unused") for u in ("copies", "bytes")]
    for field in fields:
        require(counts[field] == rebuilt[field], f"raw copy conservation differs: {field}")
    require(counts["prefetch_queue_enqueued"] == counts["prefetch_queue_canceled"] + counts["prefetch_queue_dequeued"],
            "drained queue accounting does not conserve")
    require(counts["prefetch_copy_errors"] == counts["compute_release_sync_errors"] == 0,
            "copy or compute completion failed")
    require(counts["expert_demand_uses"] == counts["expert_demand_cache_hits"] + counts["expert_demand_cache_misses"] > 0,
            "real expert demand accounting absent")
    require(counts["compute_release_syncs"] >= counts["expert_demand_uses"], "expert lifetime synchronization missing")
    require(0 < snap["pool_resident_bytes"] <= counts["peak_pool_resident_bytes"] <= snap["pool_capacity_bytes"],
            "common memory pool exceeded its fixed capacity")
    require(snap["resident_sparse_bytes"] <= snap["sparse_budget_bytes"] and
            snap["resident_sparse_bytes"] + snap["resident_dense_bytes"] == snap["pool_resident_bytes"],
            "resident cache/budget accounting differs")
    delta = policy_delta(result)
    require(all(value >= 0 for value in delta.values()), "policy counter decreased")
    require(delta.get("selector_rows", 0) > 0, "real frozen history did not reach selector")
    require(sum(delta.get(f"cardinality_{i}", 0) for i in range(61)) == delta["selector_rows"] and
            sum(i * delta.get(f"cardinality_{i}", 0) for i in range(61)) == delta["selected_candidates"],
            "raw cardinality histogram does not conserve rows/candidates")
    require(delta.get("engine_enqueue_calls", 0) == counts["prefetch_queue_enqueued"] + counts["prefetch_enqueue_resident_skip"],
            "actual enqueue API and common native executor counts differ")
    require(delta.get("selected_candidates", 0) == delta.get("engine_admitted_candidates", 0) == delta.get("engine_enqueue_calls", 0),
            "selection mask did not reach actual candidate/enqueue APIs")
    if arm == "demand-only":
        require(delta.get("engine_enqueue_calls", 0) == counts["prefetch_copy_started"] == 0,
                "demand-only unexpectedly speculated")
    else:
        require(delta.get("engine_enqueue_calls", 0) > 0, "prefetch arm never delivered real candidates")
    if arm in ("finemoe-c", "finemoe-bpf"):
        require(delta["policy_calls"] == delta["selector_rows"] and
                delta.get("oracle_checks", 0) == (delta["policy_calls"] if numerical else 0),
                "actual-input oracle/policy coverage incomplete")
    require(delta.get("jit_calls", 0) == (delta["policy_calls"] if arm == "finemoe-bpf" else 0),
            "actual BPF JIT coverage differs / native fallback")
    return {"tokens_per_second": 128 / duration,
            "tokens_per_second_including_drain": 128 / result["elapsed_seconds_including_drain"],
            "drain_seconds": result["drain_seconds"],
            "post_window_seconds": (drained - cutoff) / 1e9,
            "median_ttft_ms": statistics.median(r["ttft_ns"] for r in requests) / 1e6,
            "median_tpot_ms": statistics.median(r["tpot_ns"] for r in requests) / 1e6,
            "cpu_seconds": result["cpu_seconds"], "cpu_seconds_including_drain": result["cpu_seconds_including_drain"],
            "copies_through_drain": dict(rebuilt), "copies_completed_in_application_window": dict(in_window),
            "copies_completed_after_application_window": dict(tail),
            "observed_dynamic_set_reduction": arm in ("finemoe-c", "finemoe-bpf") and
                any(delta.get(f"cardinality_{i}", 0) for i in range(4, 60)),
            "observed_actual_prefetch_copy": counts["prefetch_copy_completed"] > 0,
            "policy_delta": delta, "pool_capacity_bytes": snap["pool_capacity_bytes"],
            "peak_pool_resident_bytes": counts["peak_pool_resident_bytes"]}


def orders(mode):
    permutations = list(itertools.permutations(ARMS))
    random.Random(SEED).shuffle(permutations)
    return [list(row) for row in permutations[:5 if mode == "full" else 1]]


def command(args, stage, output, arm=None):
    result = ["taskset", "-c", "8-11", str(HERE / ".venv/bin/python"), "-B",
              str(HERE / "inference.py"), "--stage", stage, "--data", str(args.data),
              "--output", str(output), "--offload", str(HERE / "deps/qwen-offload-cache")]
    for name in ("golden", "history"):
        if getattr(args, name, None):
            result += [f"--{name}", str(getattr(args, name))]
    if arm:
        result += ["--arm", arm]
    if args.mode == "preflight":
        result += ["--check-logits"]
    if getattr(args, "native_backtrace", False):
        require(args.mode == "golden" and stage == "golden" and arm is None,
                "native backtrace is only permitted for a diagnostic golden")
        debugger = ["/usr/bin/gdb", "--batch", "--nx", "--return-child-result",
                    "-iex", "set auto-load off", "-iex", "set debuginfod enabled off",
                    "-ex", "set confirm off", "-ex", "set pagination off",
                    "-ex", "set startup-with-shell off", "-ex", "set print thread-events off",
                    "-ex", "run", "-ex", "bt 30", "-ex", "x/8i $pc", "-ex", "kill", "--args"]
        result = result[:3] + debugger + result[3:]
    return result


def child_environment(inherited=None, *, native_backtrace=False):
    env = dict(os.environ if inherited is None else inherited)
    removed = []
    # Never record the inherited value. PyTorch's legacy alias can override the
    # canonical setting, so suppress it identically in preparation and all arms.
    for name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTHONFAULTHANDLER"):
        if name in env:
            del env[name]
            removed.append(name)
    changes = {"CUDA_VISIBLE_DEVICES": "0", "FINEMOE_EXCLUSIVE_LEASE": "1",
               "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "4",
               "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "PYTORCH_ALLOC_CONF": "expandable_segments:True",
               "TORCH_DISABLE_NATIVE_JIT": "1"}
    if native_backtrace:
        changes["PYTHONFAULTHANDLER"] = "1"
    env.update(changes)
    return env, changes, removed


def validate_reference(directory, expected_mode):
    campaign = read_json(directory.parent / "campaign.json")
    require(campaign.get("diagnostic") is not True,
            "debugger output is diagnostic-only, never an experiment reference")
    require(campaign["mode"] == expected_mode and campaign["complete"] is True,
            "preparation reference did not complete normally")
    result = read_json(directory / "result.json")
    require(result["status"] == "passed" and result.get("diagnostic") is not True and
            not result.get("cleanup_error"), "preparation reference failed or was diagnostic")


def process_identity(pid):
    try:
        record = (Path("/proc") / str(pid) / "stat").read_text()
        fields = record[record.rfind(")") + 2:].split()
        return {"pid": pid, "state": fields[0], "ppid": int(fields[1]),
                "pgid": int(fields[2]), "session": int(fields[3]), "start_ticks": int(fields[19])}
    except (OSError, ValueError, IndexError):
        return None


def remember_debugger_children(process, owned):
    if process.poll() is not None:
        return
    for pid in base.descendants(process.pid):
        identity = process_identity(pid)
        if pid != process.pid and identity is not None:
            owned[pid] = identity
            require(identity["session"] == process.pid, "debugger child escaped the owned session")


def stop_debugger_children(owned):
    # GDB may give its inferior a separate PGID. Only a previously observed
    # descendant with the same PID/start time/session is eligible for cleanup.
    def living():
        found = []
        for pid, original in owned.items():
            current = process_identity(pid)
            if current and current["state"] != "Z" and all(current[key] == original[key]
                    for key in ("pid", "session", "start_ticks")):
                found.append(pid)
        return found
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in living():
            try:
                os.kill(pid, sig)
                if sig == signal.SIGTERM:
                    os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 3
        while living() and time.monotonic() < deadline:
            time.sleep(.1)
        if not living():
            return
    require(not living(), "owned debugger inferior survived bounded cleanup")


def durable_artifacts(directory):
    # Writers have exited; preserve raw logs/numerics before the final result commit.
    for path in directory.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())


def run_stage(args, stage, directory, frozen, data, arm=None):
    directory.mkdir(parents=True, exist_ok=False)
    before = base.safety_snapshot()
    base.validate_pre_server_safety(before)
    require(inventory() == frozen, "runtime files changed before child launch")
    diagnostic = bool(getattr(args, "native_backtrace", False))
    env, env_changes, removed_names = child_environment(native_backtrace=diagnostic)
    cmd = command(args, stage, directory, arm)
    result = {"status": "running", "stage": stage, "arm": arm, "command": cmd, "diagnostic": diagnostic,
              "environment": env_changes, "environment_removed_names": removed_names,
              "safety_before": before, "runtime_before": frozen}
    base.atomic_write_json(directory / "launch.json", result)
    process = telemetry = stream = None
    debugger_children = {}
    error = None
    try:
        telemetry, stream, telemetry_path = base.start_gpu_telemetry(directory)
        with (directory / "worker.log").open("x") as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True, cwd=HERE)
            print(json.dumps({"stage": stage, "arm": arm, "pid": process.pid, "directory": str(directory)}), flush=True)
            if diagnostic:
                deadline = time.monotonic() + args.timeout
                while True:
                    remember_debugger_children(process, debugger_children)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(cmd, args.timeout)
                    try:
                        result["returncode"] = process.wait(timeout=min(1., remaining))
                        break
                    except subprocess.TimeoutExpired:
                        continue
            else:
                result["returncode"] = process.wait(timeout=args.timeout)
        require(result["returncode"] == 0, f"real {stage}/{arm} child failed: {result['returncode']}")
        artifact = directory / {"golden": "golden.json", "history": "history.json", "cell": "worker-result.json"}[stage]
        record = read_json(artifact)
        require(record["status"] == "passed", "child did not complete real work")
        if stage == "cell":
            result["metrics"] = audit_cell(record, data, arm, args.mode == "preflight", read_json(args.golden / "golden.json"))
    except BaseException as exc:
        error = exc
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cleanup_errors = []
        def cleanup(label, operation):
            nonlocal error
            try:
                return operation()
            except BaseException as exc:
                cleanup_errors.append(f"{label}: {type(exc).__name__}: {exc}")
                error = error or exc
                return None
        if process is not None:
            if diagnostic:
                cleanup("remember descendants", lambda: remember_debugger_children(process, debugger_children))
            cleanup("owned process group", lambda: base.stop_owned_process_group(process))
            if diagnostic:
                result["owned_debugger_descendants"] = list(debugger_children.values())
                cleanup("owned debugger inferiors", lambda: stop_debugger_children(debugger_children))
        if telemetry is not None:
            cleanup("telemetry process", lambda: base.stop_exact_process(telemetry))
        if stream is not None:
            cleanup("telemetry stream", stream.close)
        result["safety_after"] = cleanup("post safety", lambda: base.wait_for_post_server_safety(before))
        result["telemetry"] = cleanup("telemetry audit", lambda: base.validate_gpu_telemetry(
            directory / "gpu-telemetry.csv", allow_fixed_power_cap=True))
        result["runtime_after"] = cleanup("runtime inventory", inventory)
        cleanup("runtime freeze", lambda: require(result["runtime_after"] == frozen, "runtime files changed during child"))
        if cleanup_errors:
            result["cleanup_errors"] = cleanup_errors
            result["cleanup_error"] = "; ".join(cleanup_errors)
        result["status"] = "failed" if error else "passed"
        durable_artifacts(directory)
        base.atomic_write_json(directory / "result.json", result)
    if error:
        raise error
    print(json.dumps({"stage": stage, "arm": arm, "status": "passed", "metrics": result.get("metrics")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("golden", "history", "preflight", "full"), required=True)
    parser.add_argument("--data", type=Path, default=HERE / "dataset-mtbench-v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--golden", type=Path)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--native-backtrace", action="store_true",
                        help="diagnostic golden only; never accepted as a golden/history/performance reference")
    args = parser.parse_args()
    require(not args.native_backtrace or args.mode == "golden", "--native-backtrace requires --mode golden")
    for name in ("data", "output", "golden", "history", "preflight"):
        if getattr(args, name):
            setattr(args, name, getattr(args, name).resolve())
    require(not args.output.exists(), "output already exists; preserve previous failure and use a new directory")
    require(args.mode == "golden" or args.golden, "real original-model golden is required")
    require(args.mode not in ("preflight", "full") or args.history, "real full history is required")
    if args.golden:
        validate_reference(args.golden, "golden")
    if args.history:
        validate_reference(args.history, "history")
    frozen, data = inventory(), read_json(args.data)
    references = {}
    for folder in (args.golden, args.history):
        if folder:
            for path in sorted(folder.iterdir()):
                if path.name in ("golden.json", "history.json") or path.suffix == ".npy":
                    references[str(path)] = metadata(path)
    if args.mode == "full":
        require(args.preflight is not None, "full run requires four-arm numerical preflight")
        preflight = read_json(args.preflight / "campaign.json")
        require(preflight["complete"] is True and preflight["mode"] == "preflight" and
                preflight.get("diagnostic") is not True and
                preflight["runtime"] == frozen and preflight["data"] == data and
                preflight["reference_files"] == references,
                "preflight incomplete or runtime/data changed")
        for arm in ARMS:
            cell = args.preflight / "block-00" / arm
            require(read_json(cell / "result.json")["status"] == "passed", "preflight cleanup failed")
            audit_cell(read_json(cell / "worker-result.json"), data, arm, True, read_json(args.golden / "golden.json"))
    lease = base.LeaseSet.acquire()
    try:
        args.output.mkdir(parents=True, exist_ok=False)
        manifest = {"schema": "finemoe_dynamic_set_v1", "mode": args.mode, "complete": False,
                    "diagnostic": args.native_backtrace,
                    "seed": SEED, "orders": orders(args.mode), "data": data, "runtime": frozen,
                    "model_files": model_inventory(), "golden": str(args.golden) if args.golden else None,
                    "reference_files": references,
                    "history": str(args.history) if args.history else None, "valid_blocks": 0,
                    "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()}
        base.atomic_write_json(args.output / "campaign.json", manifest)
        if args.mode in ("golden", "history"):
            run_stage(args, args.mode, args.output / "stage", frozen, data)
        else:
            cells = []
            for block, order in enumerate(manifest["orders"]):
                for arm in order:
                    cells.append(run_stage(args, "cell", args.output / f"block-{block:02d}" / arm,
                                           frozen, data, arm))
                budgets = [c["metrics"]["pool_capacity_bytes"] for c in cells]
                require(len(set(budgets)) == 1, "arms resolved different memory pool capacities")
                if args.mode == "preflight":
                    native = read_json(args.output / f"block-{block:02d}/finemoe-c/worker-result.json")
                    bpf = read_json(args.output / f"block-{block:02d}/finemoe-bpf/worker-result.json")
                    require(native["policy_after"]["events"] == bpf["policy_after"]["events"],
                            "real C/BPF probability/threshold/mask/engine-enqueue events differ")
                manifest["valid_blocks"] += 1
                base.atomic_write_json(args.output / "campaign.json", manifest)
        manifest["complete"] = True
        base.atomic_write_json(args.output / "campaign.json", manifest)
    except BaseException as exc:
        if args.output.is_dir():
            base.atomic_write_json(args.output / "campaign-failure.json", {
                "complete": False, "mode": args.mode, "error": f"{type(exc).__name__}: {exc}",
                "time_ns": time.time_ns(), "note": "retain all partial cells; do not count incomplete paired blocks"})
        raise
    finally:
        lease.close()


if __name__ == "__main__":
    main()
