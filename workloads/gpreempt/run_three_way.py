#!/usr/bin/env python3
"""Original config-A, paired native/original/BPF runs; no module or service changes."""
from __future__ import annotations

import argparse
import fcntl
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

import run_smoke

safety = run_smoke.safety
HERE = Path(__file__).resolve().parent
EXTENSION = HERE.parents[1] / "extension/.output"
ARMS = ("native", "original_gpreempt", "bpf_gpreempt")
TASKS = ("vgg_rt", "resnet152_be")
DEFAULT_GDRCOPY = HERE / "deps/gdrcopy-2.5.2"


def orders(blocks: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    result = []
    while len(result) < blocks:
        cycle = list(itertools.permutations(ARMS))
        rng.shuffle(cycle)
        result.extend(map(list, cycle))
    return result[:blocks]


def validate_config(config: dict) -> None:
    if config.get("time") != 60 or len(config.get("tasks", [])) != 2:
        raise ValueError("the formal comparison preserves original config A: two roles and 60 seconds")
    for task, name, model, role in zip(config["tasks"], TASKS, ("vgg", "resnet152"), (0, 1)):
        client, load = task["client"], task["load"]
        if (task["id"] != name or client["name"] != name or client["model_name"] != model
                or client["priority"] != role or client["preprocess_time"] != 200
                or client["use_cuda_graph"] is not True or client["batch_size"] != 1
                or load["type"] != "periodic" or load["frequency"] != 100 or load["priority"] != 0):
            raise ValueError("config A workload, roles, rates, graphs or preprocessing changed")


def parse_fields(log: str, prefix: str) -> dict:
    rows = [line for line in log.splitlines() if line.startswith(prefix)]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one {prefix} record")
    return dict(re.findall(r"(\w+)=([^\s]+)", rows[0]))


def parse_report(log: str) -> dict:
    decoder = json.JSONDecoder()
    candidates = []
    for match in re.finditer(r"(?m)^\{", log):
        try:
            report, _ = decoder.raw_decode(log[match.start():])
        except ValueError:
            continue
        if isinstance(report, dict) and "benchmarkTime(s)" in report:
            candidates.append(report)
    if len(candidates) != 1 or candidates[0]["benchmarkTime(s)"] != 60:
        raise ValueError("missing unique original 60-second DISB report")
    report = candidates[0]
    checks = [json.loads(line.split(" ", 1)[1]) for line in log.splitlines()
              if line.startswith("GPREEMPT_VALIDATION ")]
    if len(checks) != 2 or {row["task"] for row in checks} != set(TASKS):
        raise ValueError("missing both full-output numerical validation records")
    checks = {row["task"]: row for row in checks}
    if len(report.get("results", [])) != 2:
        raise ValueError("missing both workload results")
    metrics = {}
    for result in report["results"]:
        name = result["clientName"]
        if name not in TASKS or name in metrics:
            raise ValueError("unexpected or duplicate workload")
        analyzers = [row for row in result["analyzers"] if row.get("type") == "basic"]
        if len(analyzers) != 1:
            raise ValueError("missing unique original basic analyzer")
        analyzer = analyzers[0]
        samples = analyzer.get("requestLatencyNs", [])
        count = analyzer.get("completedRequests")
        if (type(count) is not int or count < 100 or len(samples) != count
                or any(type(value) is not int or value <= 0 for value in samples)
                or analyzer.get("latencyDefinition") != "sum_of_original_six_recorded_stages"):
            raise ValueError("p99 requires at least 100 real, positive request samples per role")
        check = checks[name]
        if (check["timed_checked"] != count or check["checked"] != count + 110
                or check["atol"] != 1e-6 or check["rtol"] != 1e-4
                or not math.isfinite(check["max_absolute_error"]) or check["max_absolute_error"] < 0):
            raise ValueError("numerical counts or tolerances do not match every original warmup/timed request")
        throughput = analyzer["avgThroughput(req/s)"]
        if not math.isfinite(throughput) or abs(throughput - count / 60) > 1e-6:
            raise ValueError("throughput is inconsistent with original duration and completed requests")
        metrics[name] = {"completed_requests": count, "throughput_rps": throughput,
                         "nominal_schedule_rps": 100, "observed_arrivals": None, "observed_drops": None,
                         "mean_latency_us": statistics.mean(samples) / 1000,
                         "p99_latency_us": sorted(samples)[math.ceil(count * 0.99) - 1] / 1000,
                         "numerics": check}
    return {"metrics": metrics, "report": report}


def check_engagement(arm: str, client_log: str, loader_log: str, flag_transport: str = "gdr") -> dict:
    if arm == "native":
        if any(marker in client_log for marker in ("gpreempt_context_registered:", "gpreempt_hint_ready:",
                                                    "gpreempt_bridge_stats:", "gpreempt_flag_transport:",
                                                    "gpreempt_flag_cleanup:")) or loader_log:
            raise ValueError("native arm unexpectedly engaged GPreempt policy")
        return {"backend": "native_single_context_stream_priorities"}
    transport = parse_fields(client_log, "gpreempt_flag_transport:")
    cleanup = parse_fields(client_log, "gpreempt_flag_cleanup:")
    if (flag_transport not in {"gdr", "host_mapped"} or transport.get("transport") != flag_transport
            or transport.get("portable") != ("1" if flag_transport == "host_mapped" else "0")
            or transport.get("original_gdr") != ("1" if flag_transport == "gdr" else "0")
            or cleanup.get("transport") != flag_transport or cleanup.get("status") != "passed"
            or cleanup.get("slots") != "1"):
        raise ValueError("flag transport mismatch, silent fallback, or incomplete flag cleanup")
    fields = parse_fields(client_log, "gpreempt_bridge_stats:")
    backend = "ubpf-jit" if arm == "bpf_gpreempt" else "original-c"
    if fields.get("backend") != backend or int(fields.get("errors", -1)) != 0:
        raise ValueError("wrong decision backend or nonzero bridge errors")
    for key in ("preprocess", "due", "infer", "reset", "hint", "block", "release"):
        if int(fields.get(key, 0)) <= 0:
            raise ValueError(f"full reserved-hint policy did not exercise {key}")
    result = {"backend": backend, "bridge": fields, "flag_transport": transport, "flag_cleanup": cleanup}
    if arm == "original_gpreempt":
        if loader_log or "gpreempt_context_registered:" in client_log:
            raise ValueError("original C arm unexpectedly used kernel BPF")
        return result
    if "gpreempt_hint_ready: backend=ubpf-jit" not in client_log:
        raise ValueError("actual host JIT readiness was not observed")
    registrations = [dict(re.findall(r"(\w+)=(\d+)", line)) for line in client_log.splitlines()
                     if line.startswith("gpreempt_context_registered:")]
    if len(registrations) != 2 or {int(row["role"]) for row in registrations} != {0, 1}:
        raise ValueError("missing both real BPF context registrations")
    for row in registrations:
        if (not 1 <= int(row["engine"]) <= 8 or int(row["timeslice_us"]) !=
                (1000000 if int(row["role"]) == 0 else 1)):
            raise ValueError("context engine or requested timeslice mismatch")
    for key in ("cuda_context", "tsg_id"):
        if len({int(row[key]) for row in registrations}) != 2:
            raise ValueError("the two roles did not use distinct contexts/TSGs")
    if len({(row["hclient"], row["htsg"]) for row in registrations}) != 2:
        raise ValueError("the two roles share an RM handle pair")
    kernel = parse_fields(loader_log, "gpreempt_policy_stats:")
    for key in ("scope_enter", "scope_leave", "gr_init", "timeslice_ok", "alloc_captured", "registered", "destroy"):
        if int(kernel.get(key, -1)) != 2:
            raise ValueError(f"kernel {key} did not match the two owned role contexts")
    for key in ("unknown_engine", "setter_error", "alloc_error", "register_error", "bind_shadow_mismatch",
                "map_error", "scope_error"):
        if int(kernel.get(key, -1)) != 0:
            raise ValueError(f"kernel policy error: {key}")
    control_lc = int(kernel.get("control_lc", 0))
    control_be = int(kernel.get("control_be", 0))
    if (control_lc <= 0 or control_be <= 0 or
            int(kernel.get("control_override", -1)) != control_lc + control_be):
        raise ValueError("both role contexts must exercise the persistent native-control BPF hook")
    for key in ("scopes", "registered", "ended"):
        if int(fields.get(key, -1)) != 2:
            raise ValueError(f"bridge {key} does not match two contexts")
    result.update(kernel=kernel, registrations=registrations,
                  runtime_control_request_engagement={"lc": control_lc, "be": control_be},
                  hardware_timeslice_proven_by_shadow_counters=False)
    return result


class Leases:
    """Same shared lock paths, without O_CREAT on existing sticky-/tmp files."""
    def __init__(self):
        self.files = []
        try:
            for name in ("gpu0", "struct-ops"):
                path = Path(f"/tmp/gpubpf-revision-{name}.lock")
                try:
                    stream = path.open("r+")
                except FileNotFoundError:
                    stream = path.open("x+")
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BaseException:
                    stream.close()
                    raise
                self.files.append(stream)
        except BaseException:
            self.close()
            raise
    def close(self):
        for stream in reversed(self.files):
            stream.close()
        self.files.clear()


def group_members(pgid: int) -> list[int]:
    members = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if fields[0] != "Z" and int(fields[2]) == pgid and int(fields[3]) == pgid:
                members.append(int(path.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    return members


def stop_owned(process) -> None:
    if process is None:
        return
    # Every child starts a new session whose known PGID equals its original PID.
    # Check surviving group members even after the leader has exited.
    for sig, seconds in ((signal.SIGINT, 8), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        process.poll()
        if not group_members(process.pid):
            process.wait(timeout=1)
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            continue
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            process.poll()
            if not group_members(process.pid):
                process.wait(timeout=1)
                return
            time.sleep(0.1)
    raise RuntimeError(f"owned process group {process.pid} survived bounded cleanup")


def environment(arm: str, pin: Path, gdrcopy: Path = DEFAULT_GDRCOPY) -> dict:
    env = {"PATH": "/usr/local/cuda-12.9/bin:/usr/bin:/bin", "LANG": "C.UTF-8",
           "CUDA_VISIBLE_DEVICES": "0", "GPREEMPT_POLICY": "original",
           "LD_LIBRARY_PATH": f"{EXTENSION}:{HERE / 'build/ninja'}:{gdrcopy / 'src'}:"
                              "/usr/local/cuda-12.9/lib64:/usr/local/lib"}
    if arm == "bpf_gpreempt":
        env.update(GPREEMPT_POLICY="bpf", GPREEMPT_BPF_MAPS=str(pin),
                   GPREEMPT_HINT_CODE=str(EXTENSION / "gpreempt_hint.bin"))
    return env


def model_assets() -> dict:
    assets = {}
    for name, layers in (("vgg", 19), ("resnet152", 152)):
        directory = HERE / "deps/upstream/model" / name
        spec = json.loads((directory / "model-spec.json").read_text())
        if (spec["model"] != name or spec["layers"] != layers or spec["architecture"] != "sm_120"
                or spec["dtype"] != "float32" or spec["input_shape"] != [1, 3, 224, 224]
                or spec["output_shape"] != [1, 1000] or spec["parameter_seed"] != 0
                or spec["input_formula"] != "((element_index % 257) - 128) / 128.0"):
            raise ValueError(f"model export settings disagree with the common workload: {name}")
        inventory = {}
        for filename in ("mod.cu", "mod.cubin", "mod.json", "host.json", "mod.params", "reference.f32"):
            path = directory / filename
            size = path.stat().st_size
            if size <= 0 or (filename == "reference.f32" and size != 4000):
                raise ValueError(f"missing/invalid model asset: {path}")
            inventory[filename] = {"path": str(path), "bytes": size}
        assets[name] = {"specification": spec, "inventory": inventory}
    return assets


def client_command(arm: str, config: Path, flag_transport: str) -> list[str]:
    if arm not in ARMS or flag_transport not in {"gdr", "host_mapped"}:
        raise ValueError("unknown policy arm or flag transport")
    executable = "baseclient" if arm == "native" else "gpreemptclient"
    command = [str(HERE / "build/ninja" / executable), str(config)]
    if arm != "native":
        command += ["--flag-transport", flag_transport]
    return command


def run_cell(directory: Path, arm: str, config: Path, timeout: int, gdrcopy: Path = DEFAULT_GDRCOPY,
             flag_transport: str = "gdr") -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    result = {"status": "failed", "arm": arm,
              "flag_transport": "not_used" if arm == "native" else flag_transport,
              "comparison_variant": "original_gdr" if flag_transport == "gdr" else "host_mapped_compatibility"}
    before = None
    client = loader = telemetry = None
    streams = []
    pin = Path(f"/sys/fs/bpf/gpreempt-{os.getpid()}-{time.monotonic_ns()}")
    try:
        command = client_command(arm, config, flag_transport)
        before = safety.safety_snapshot()
        safety.validate_pre_server_safety(before)
        if before["gpu"]["driver"] != "575.57.08":
            raise RuntimeError("the prepared 575 compatibility driver is required")
        if arm != "native" and flag_transport == "gdr" and not Path("/dev/gdrdrv").exists():
            raise RuntimeError("GDRCopy must be separately prepared; runner never changes drivers/devices")
        result["safety_before"] = before
        version = Path("/sys/module/gdrdrv/version")
        result["gdrdrv_module_version"] = version.read_text().strip() if version.exists() else None
        telemetry, telemetry_stream, telemetry_path = safety.start_gpu_telemetry(directory)
        streams.append(telemetry_stream)
        if arm == "bpf_gpreempt":
            loader_stream = (directory / "loader.log").open("x")
            streams.append(loader_stream)
            command = [str(EXTENSION / "gpreempt_policy"), "--library", str(EXTENSION / "libgpreempt_bridge.so"),
                       "--pin-dir", str(pin), "--duration", str(timeout + 30)]
            loader = subprocess.Popen(command, stdout=loader_stream, stderr=subprocess.STDOUT,
                                      start_new_session=True, env=environment(arm, pin, gdrcopy))
            deadline = time.monotonic() + 15
            while "gpreempt_policy_ready:" not in (directory / "loader.log").read_text():
                if loader.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("BPF loader did not become ready")
                time.sleep(0.1)
        command = client_command(arm, config, flag_transport)
        result.update(command=command, environment=environment(arm, pin, gdrcopy), timeout_seconds=timeout)
        client_stream = (directory / "client.log").open("x")
        streams.append(client_stream)
        start = time.monotonic()
        client = subprocess.Popen(command, stdout=client_stream, stderr=subprocess.STDOUT,
                                  start_new_session=True, env=environment(arm, pin, gdrcopy))
        while client.poll() is None:
            if time.monotonic() - start >= timeout:
                raise TimeoutError("owned GPreempt client exceeded its bound")
            if loader is not None and loader.poll() is not None:
                raise RuntimeError("BPF policy exited before the client")
            time.sleep(0.2)
        result.update(returncode=client.returncode, process_wall_seconds=time.monotonic() - start)
        stop_owned(client)
        stop_owned(loader)
        if client.returncode != 0 or (loader is not None and loader.returncode != 0):
            raise RuntimeError("client or attached policy exited unsuccessfully")
        client_log = (directory / "client.log").read_text(errors="replace")
        loader_log = (directory / "loader.log").read_text(errors="replace") if loader else ""
        parsed = parse_report(client_log)
        # Full samples remain separately available for recalculation, not just p99.
        safety.atomic_write_json(directory / "request-report.json", parsed.pop("report"))
        result.update(parsed)
        result["engagement"] = check_engagement(arm, client_log, loader_log, flag_transport)
        result["status"] = "passed"
    except BaseException as exc:
        result["error"] = str(exc)
        raise
    finally:
        errors = []
        for process in (client, loader, telemetry):
            try:
                stop_owned(process)
            except BaseException as exc:
                errors.append(str(exc))
        for stream in streams:
            stream.close()
        try:
            if before is not None:
                result["safety_after"] = safety.wait_for_post_server_safety(before)
            if loader is not None and pin.exists():
                raise RuntimeError(f"owned BPF pins survived loader cleanup: {pin}")
            if telemetry is not None:
                result["telemetry"] = safety.validate_gpu_telemetry(telemetry_path, allow_fixed_power_cap=True)
        except BaseException as exc:
            errors.append(str(exc))
        if errors:
            result.update(status="failed", cleanup_errors=errors)
        safety.atomic_write_json(directory / "result.json", result)
        if errors:
            raise RuntimeError("; ".join(errors))
    return result


def summarize(results: list[dict], blocks: int, flag_transport: str = "gdr") -> dict:
    ratios = []
    for block in range(blocks):
        cells = {row["arm"]: row for row in results if row["block"] == block}
        if set(cells) != set(ARMS) or any(row["status"] != "passed" for row in cells.values()):
            continue
        if any(cells[arm].get("flag_transport") != flag_transport for arm in ARMS if arm != "native"):
            continue
        native = cells["native"]["metrics"]
        original = cells["original_gpreempt"]["metrics"]
        bpf = cells["bpf_gpreempt"]["metrics"]
        ratios.append({"block": block,
            "original_over_native_lc_p99": original[TASKS[0]]["p99_latency_us"] / native[TASKS[0]]["p99_latency_us"],
            "bpf_over_native_lc_p99": bpf[TASKS[0]]["p99_latency_us"] / native[TASKS[0]]["p99_latency_us"],
            "bpf_over_original_lc_p99": bpf[TASKS[0]]["p99_latency_us"] / original[TASKS[0]]["p99_latency_us"],
            "bpf_over_original_be_throughput": bpf[TASKS[1]]["throughput_rps"] / original[TASKS[1]]["throughput_rps"]})
    return {"flag_transport": flag_transport,
            "comparison_variant": "original_gdr" if flag_transport == "gdr" else "host_mapped_compatibility",
            "original_gdr_transport": flag_transport == "gdr",
            "valid_paired_blocks": len(ratios), "formal_5_block_complete": len(ratios) >= 5,
            "requested_blocks_complete": len(ratios) == blocks,
            "paired_ratios": ratios, "medians": {key: statistics.median(row[key] for row in ratios)
            for key in ratios[0] if key != "block"} if ratios else {},
            "interpretation": "LC p99 ratio lower is better; BE throughput ratio higher is better; inspect sample counts"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--cell-timeout", type=int, default=240)
    parser.add_argument("--cooldown-seconds", type=int, default=10)
    parser.add_argument("--gdrcopy-dir", type=Path, default=DEFAULT_GDRCOPY)
    parser.add_argument("--flag-transport", choices=("gdr", "host_mapped"), default="gdr",
                        help="same explicit transport in both policy arms; host_mapped is not original GDRCopy")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.blocks <= 30 or not 90 <= args.cell_timeout <= 3500 or not 0 <= args.cooldown_seconds <= 60:
        parser.error("invalid block count, timeout, or cooldown")
    config = json.loads((HERE / "deps/upstream/config/A.json").read_text())
    validate_config(config)
    plan = {"seed": args.seed, "orders": orders(args.blocks, args.seed), "config": config,
            "timed_seconds_per_cell": 60, "cell_timeout_seconds": args.cell_timeout,
            "cooldown_seconds": args.cooldown_seconds, "formal_required_blocks": 5,
            "p99_definition": "nearest rank over original six-stage service latency; not arrival-to-completion",
            "load_semantics": "periodic newest-only: skip stale slots, standalone-latency phase offset; last admitted request may finish after cutoff",
            "arrival_and_drop_counts": "not instrumented; do not infer exact drops from 6000 minus completions",
            "gdrcopy_directory": str(args.gdrcopy_dir.resolve()),
            "flag_transport": args.flag_transport,
            "comparison_variant": "original_gdr" if args.flag_transport == "gdr" else "host_mapped_compatibility",
            "privilege": "all three clients and loader run as root; no permission changes"}
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    if args.output is None:
        parser.error("--output is required for actual experiments")
    if os.geteuid() != 0:
        parser.error("run all three arms consistently as root; pinned control maps remain root-only")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    safety.atomic_write_json(output / "plan.json", plan)
    safety.atomic_write_json(output / "config-A.json", config)
    results = []
    lease = None
    run_error = None
    def interrupted(signum, _frame):
        raise InterruptedError(f"signal {signum}; cleaning up only owned experiment processes")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        # Pure file checks before touching CUDA; never run a model with missing exports.
        safety.atomic_write_json(output / "model-assets.json", model_assets())
        lease = Leases()
        for block, order in enumerate(plan["orders"]):
            for arm in order:
                print(f"START block={block} arm={arm}", flush=True)
                result = run_cell(output / f"block-{block:02d}" / arm, arm, output / "config-A.json",
                                  args.cell_timeout, args.gdrcopy_dir.resolve(), args.flag_transport)
                result["block"] = block
                results.append(result)
                safety.atomic_write_json(output / "progress.json", {"completed_cells": len(results), "last_block": block, "last_arm": arm})
                safety.atomic_write_json(output / "summary.json", summarize(results, args.blocks, args.flag_transport))
                print(f"PASS block={block} arm={arm}", flush=True)
                if len(results) < args.blocks * 3:
                    time.sleep(args.cooldown_seconds)
    except BaseException as exc:
        run_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        summary = summarize(results, args.blocks, args.flag_transport)
        summary.update(status="failed" if run_error else "completed", error=run_error)
        safety.atomic_write_json(output / "summary.json", summary)
        if lease is not None:
            lease.close()


if __name__ == "__main__":
    main()
