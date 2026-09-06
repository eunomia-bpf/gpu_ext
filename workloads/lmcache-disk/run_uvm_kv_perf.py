#!/usr/bin/env python3
"""Minimal performance-only runner: recompute vs lmcache_disk vs lmcache_disk_uvm_kv.

A minimal copy of run_perf_only.py that measures only warm performance on the
existing Qwen3 workload over rotated complete blocks (default one block; every
arm exactly once per block in a rotating order, one attempt per cell) with
identical fixed prompts and server settings in every cell, started through the
existing server and request functions of lmcache_primitives (start_server,
wait_ready, streamed_completion, stop_owned_server, server_environment).

The third arm, lmcache_disk_uvm_kv, is lmcache_disk plus a UVM-backed
kv_cache allocation pool: server_environment extends the identical LMCache
local-disk environment with UVM_KV_PLUGIN=1 and the absolute UVM_KV_PLUGIN_SO
path to the prepared allocator (workloads/vllm/vllm/uvm_test/uvm_allocator.so)
so the installed vllm.general_plugins entry point (uvm_kv_plugin) routes the
kv_cache tag through that UVM pluggable allocator.

This runner deliberately calls none of the validation, admission, or retry
machinery: no validate-cell, compare-outputs, analyze, engagement checks
(validate_log), correctness/store checks (wait_for_cold_store,
sync_and_verify_disk), O_DIRECT trace validation, driver/version or GPU
safety admission, no inter-cell GPU-idle wait, no process lease, and no
file-identity or provenance metadata. Every attempted request, parsed
performance number, barrier outcome, cleanup error, and process return code
is recorded exactly once. A nonzero or failed outcome is preserved and the
campaign continues; no cell is retried and no record is removed. The only
extra wait is one non-gating causal cache-store barrier (parsing the server
log via request_log_values only) that keeps the warm phase warm without
judging anything and can never abort a cell; its observations are retained
as records.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import time
from typing import Any


HERE = Path(__file__).resolve().parent
PRIMITIVES_PATH = HERE / "lmcache_primitives.py"
SPEC = importlib.util.spec_from_file_location("perf_only_primitives", PRIMITIVES_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load LMCache primitives from {PRIMITIVES_PATH}")
ops = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ops)

KIND = "lmcache_uvm_kv_perf"
CONFIGS = ("recompute", "lmcache_disk", "lmcache_disk_uvm_kv")
DEFAULT_BLOCKS = 1
DEFAULT_PORT = 18080
DEFAULT_STORE_BARRIER_TIMEOUT_S = 120.0
DEFAULT_STORE_BARRIER_POLL_S = 1.0
REQUEST_LABELS = ("cold", "warm")


class DeferredStop:
    """Deferred stop between cells; never interrupts an owned cell."""

    def __init__(self):
        self.signum = None

    def request(self, signum, _frame):
        if self.signum is None:
            self.signum = signum

    def signum_name(self) -> str:
        return signal.Signals(self.signum).name if self.signum is not None else ""


def output_root(output: Path | None) -> Path:
    if output is None:
        timestamp = time.strftime("%Y%m%dT%H%M%S")
        output = HERE / "raw" / f"{KIND}-{timestamp}"
    return output.resolve()


def load_fixed_prompts() -> dict[str, Any]:
    """Read the pinned prompt artifact directly; no regeneration or checking."""
    return json.loads(ops.PROMPTS.read_text())


def rotation_orders(blocks: int) -> list[list[str]]:
    """Rotated complete block orders over the three arms.

    Block ``b`` rotates the fixed base order left by ``b`` positions (period
    three); every block still runs each arm exactly once.
    """
    if blocks < 1:
        raise ValueError(f"--blocks must be at least 1, got {blocks}")
    base = list(CONFIGS)
    return [base[index % len(base):] + base[:index % len(base)]
            for index in range(blocks)]


def request_entry(phase: str, index: int, response: dict[str, Any] | None = None,
                  error: str | None = None, attempted: bool = True,
                  reason: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"phase": phase, "prefix_index": index,
                             "request_id": f"lmc-p{index}-{phase}", "attempted": attempted}
    if reason is not None:
        entry["reason"] = reason
    if error is not None:
        entry["error"] = error
    if response is not None:
        entry.update(response)
    return entry


def store_barrier(config: str, log_path: Path, engine_request_id: str,
                  expected_store_tokens: int, prompt_tokens: int,
                  timeout_s: float, poll_s: float) -> dict[str, Any]:
    """Non-gating wait for the cold store line; a timeout is recorded, never fatal."""
    if config == "recompute":
        return {"applicable": False, "satisfied": None, "waited_s": 0.0}
    if timeout_s <= 0:
        return {"applicable": True, "satisfied": False, "waited_s": 0.0, "disabled": True}
    start = time.perf_counter_ns()
    deadline = time.monotonic() + timeout_s
    seen: dict[str, Any] = {"runtime_ids": [], "request_totals": [], "stores": []}
    while True:
        read_error = None
        try:
            log = log_path.read_text(errors="replace")
        except OSError as error:
            log = ""
            read_error = f"{type(error).__name__}: {error}"
        values = ops.request_log_values(log, engine_request_id)
        seen = {key: values.get(key) for key in ("runtime_ids", "request_totals", "stores")}
        if read_error is not None:
            seen["read_error"] = read_error
        if (len(values["runtime_ids"]) == 1
                and values["request_totals"] == [prompt_tokens]
                and values["stores"] == [[expected_store_tokens, expected_store_tokens]]):
            return {"applicable": True, "satisfied": True,
                    "waited_s": (time.perf_counter_ns() - start) / 1e9, "log_values": seen}
        if time.monotonic() >= deadline:
            break
        time.sleep(poll_s)
    return {"applicable": True, "satisfied": False,
            "waited_s": (time.perf_counter_ns() - start) / 1e9, "log_values": seen}


def worker_affinity(proc) -> list[int] | None:
    try:
        return sorted(os.sched_getaffinity(proc.pid))
    except (OSError, AttributeError):
        return None


def warm_aggregates(record: dict[str, Any], warm_start_ns: int,
                    warm_end_ns: int) -> dict[str, Any] | None:
    warm = [entry for entry in record["requests"] if entry["phase"] == "warm"]
    succeeded = [entry for entry in warm if entry["attempted"] and "ttft_ms" in entry]
    if not succeeded:
        return None
    ttft = [float(entry["ttft_ms"]) for entry in succeeded]
    elapsed_s = (warm_end_ns - warm_start_ns) / 1e9
    output_tokens = sum(int(entry["usage"]["completion_tokens"]) for entry in succeeded
                        if entry.get("usage", {}).get("completion_tokens") is not None)
    return {
        "sequential": True,
        "requests": len(succeeded),
        "attempts": len(warm),
        "failures": len(warm) - len(succeeded),
        "output_tokens": output_tokens,
        "elapsed_s": elapsed_s,
        "requests_per_s": len(succeeded) / elapsed_s if elapsed_s > 0 else None,
        "output_tokens_per_s": output_tokens / elapsed_s if elapsed_s > 0 else None,
        "warm_ttft_values_ms": ttft,
        "warm_ttft_median_ms": statistics.median(ttft),
        "warm_ttft_p95_ms": (statistics.quantiles(ttft, n=20)[-1] if len(ttft) >= 2
                             else statistics.median(ttft)),
        "warm_ttft_max_ms": max(ttft),
        "excludes": ["server startup", "cold population", "cold-store barriers", "shutdown"],
    }


def run_cell(config: str, block: int, position: int, run_dir: Path, port: int,
             model_path: Path, prefixes: list[dict[str, Any]], expected_driver: str,
             store_barrier_timeout_s: float) -> dict[str, Any]:
    """One arm, once: no gates, no retry; every number and exit code is kept."""
    record: dict[str, Any] = {
        "schema": 1, "kind": KIND, "config": config, "block": block, "position": position,
        "port": port, "expected_driver_parameter": expected_driver,
        "prompt_count": len(prefixes), "cached_tokens": ops.PREFIX_TOKENS,
        "output_tokens": ops.OUTPUT_TOKENS,
        "started_ns": time.time_ns(), "ready": False, "ready_error": None,
        "requests": [], "barriers": [], "cleanup_errors": [],
        "server_returncode": None, "error": None,
    }
    log_path = run_dir / "server.log"
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=False)
    record["environment"] = ops.server_environment(config, cache_dir, expected_driver)
    proc = None
    log_file = None
    stopped = False
    launched = False
    try:
        try:
            proc, log_file, argv, launch = ops.start_server(
                config, model_path, cache_dir, port, log_path, expected_driver=expected_driver)
            launched = True
        except FileExistsError as error:
            record["error"] = f"server log already exists; launch not attempted: {error}"
        if launched:
            record["command"] = argv
            record["launch_command"] = launch
            record["worker_cpu_affinity"] = worker_affinity(proc)
            try:
                ops.wait_ready(proc, port, log_path)
                record["ready"] = True
            except ops.GateError as error:
                record["ready_error"] = f"{type(error).__name__}: {error}"
            if record["ready"]:
                for item in prefixes:
                    index = item["index"]
                    if proc.poll() is not None:
                        record["requests"].append(request_entry(
                            "cold", index, attempted=False,
                            reason=f"server process exited with return code {proc.returncode}"))
                        continue
                    try:
                        cold = ops.streamed_completion(port, item["cold_token_ids"],
                                                        f"lmc-p{index}-cold")
                    except Exception as error:  # noqa: BLE001 - preserved, never fatal
                        record["requests"].append(request_entry(
                            "cold", index, error=f"{type(error).__name__}: {error}"))
                    else:
                        record["requests"].append(request_entry("cold", index, response=cold))
                        record["barriers"].append(store_barrier(
                            config, log_path, cold["engine_request_id"],
                            int(item["expected_store_tokens"]), len(item["cold_token_ids"]),
                            store_barrier_timeout_s, DEFAULT_STORE_BARRIER_POLL_S))
                warm_start = time.perf_counter_ns()
                for item in prefixes:
                    index = item["index"]
                    if proc.poll() is not None:
                        record["requests"].append(request_entry(
                            "warm", index, attempted=False,
                            reason=f"server process exited with return code {proc.returncode}"))
                        continue
                    try:
                        warm = ops.streamed_completion(port, item["warm_token_ids"],
                                                        f"lmc-p{index}-warm")
                    except Exception as error:  # noqa: BLE001
                        record["requests"].append(request_entry(
                            "warm", index, error=f"{type(error).__name__}: {error}"))
                    else:
                        record["requests"].append(request_entry("warm", index, response=warm))
                warm_end = time.perf_counter_ns()
                record["warm_phase"] = warm_aggregates(record, warm_start, warm_end)
            else:
                for item in prefixes:
                    for phase in REQUEST_LABELS:
                        record["requests"].append(request_entry(
                            phase, item["index"], attempted=False, reason="server never became ready"))
    except BaseException as error:  # noqa: BLE001 - keep numbers, never abort the campaign
        record["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        if proc is not None:
            try:
                ops.stop_owned_server(proc, log_file)
                stopped = True
            except BaseException as error:  # noqa: BLE001
                record["cleanup_errors"].append(f"stop_owned_server: {type(error).__name__}: {error}")
                try:
                    log_file.close()
                except OSError:
                    pass
            record["server_returncode"] = proc.returncode
            if stopped and proc.returncode is None:
                record["cleanup_errors"].append("server return code unknown after bounded stop")
        else:
            try:
                if log_file is not None:
                    log_file.close()
            except OSError:
                pass
        record["finished_ns"] = time.time_ns()
        ops.atomic_write_json(run_dir / "result.json", record)
    return record


def cell_metrics(record: dict[str, Any]) -> dict[str, Any]:
    warm = record.get("warm_phase") or {}
    return {
        "warm_ttft_median_ms": warm.get("warm_ttft_median_ms"),
        "warm_requests_per_s": warm.get("requests_per_s"),
        "warm_output_tokens_per_s": warm.get("output_tokens_per_s"),
        "warm_requests_ok": warm.get("requests", 0),
        "warm_requests_failed": warm.get("failures", 0),
        "ready": record.get("ready", False),
        "server_returncode": record.get("server_returncode"),
    }


def arm_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    by_config: dict[str, list[dict[str, Any]]] = {config: [] for config in CONFIGS}
    for cell in cells:
        by_config.setdefault(cell["config"], []).append(cell["warm"])
    summary: dict[str, Any] = {}
    for config in CONFIGS:
        rows = by_config.get(config, [])
        summary[config] = {
            "cells": rows,
            "warm_ttft_median_ms": [row.get("warm_ttft_median_ms") for row in rows],
            "warm_requests_per_s": [row.get("warm_requests_per_s") for row in rows],
            "warm_output_tokens_per_s": [row.get("warm_output_tokens_per_s") for row in rows],
        }
        summary[config]["cell_count"] = len(rows)
    return summary


def write_summary(root: Path, campaign: dict[str, Any]) -> None:
    lines = [
        "# LMCache/UVM-KV three-arm performance-only comparison",
        "",
        f"- Kind: `{campaign['kind']}`",
        f"- Timestamp: `{campaign['timestamp']}`",
        f"- Driver parameter: `{campaign['params']['expected_driver']}`",
        f"- Blocks: `{campaign['params']['blocks']}` (rotating base order "
        f"{campaign['params']['configs']})",
        f"- Prompts: `{campaign['params']['prompts']['path']}` "
        f"({campaign['params']['prompts']['prefix_count']} prefixes x "
        f"{campaign['params']['prompts']['prefix_tokens']} cached tokens, "
        f"{campaign['params']['prompts']['output_tokens']} output tokens)",
        "",
        "No correctness, engagement, admission, retry, or filtering gates ran; "
        "nonzero server return codes are preserved per cell.",
        "",
        "## Cell metrics",
        "",
        "| Block | Position | Arm | ready | warm TTFT median (ms) | warm requests/s | "
        "warm out tok/s | warm ok/failed | server returncode |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for cell in campaign["cells"]:
        metrics = cell["warm"]

        def fmt(value: Any, name: str = "") -> str:
            return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"

        lines.append(
            f"| {cell['block']} | {cell['position']} | {cell['config']} | "
            f"{str(metrics['ready']).lower()} | {fmt(metrics['warm_ttft_median_ms'])} | "
            f"{fmt(metrics['warm_requests_per_s'])} | "
            f"{fmt(metrics['warm_output_tokens_per_s'])} | "
            f"{metrics['warm_requests_ok']}/{metrics['warm_requests_failed']} | "
            f"{metrics['server_returncode']} |")
    lines.extend(["", "## Per-arm values across attempted cells", "",
                  "| Arm | cells | warm TTFT medians (ms) | warm requests/s | warm out tok/s |",
                  "|---|---:|---|---|---|"])
    for config in CONFIGS:
        values = campaign["arm_summary"].get(config, {})
        def fmt_list(series: list[Any]) -> str:
            return ", ".join(f"{x:.4f}" for x in series
                              if isinstance(x, (int, float))) or "n/a"
        lines.append(
            f"| {config} | {values.get('cell_count', 0)} | "
            f"{fmt_list(values.get('warm_ttft_median_ms', []))} | "
            f"{fmt_list(values.get('warm_requests_per_s', []))} | "
            f"{fmt_list(values.get('warm_output_tokens_per_s', []))} |")
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_campaign(args) -> int:
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    root = output_root(args.output)
    orders = rotation_orders(args.blocks)
    prompts = load_fixed_prompts()
    prefixes = prompts["prefixes"]
    campaign: dict[str, Any] = {
        "kind": KIND, "timestamp": timestamp,
        "params": {
            "blocks": args.blocks, "expected_driver": args.expected_driver,
            "port": args.port, "store_barrier_timeout_s": args.store_barrier_timeout_s,
            "configs": list(CONFIGS), "attempts_per_cell": 1,
            "retry": False, "result_filtering": False, "gpu_idle_wait": False,
            "model": ops.MODEL_ID, "model_revision": ops.MODEL_REVISION,
            "prompts": {
                "path": str(ops.PROMPTS), "prefix_count": len(prefixes),
                "prefix_tokens": ops.PREFIX_TOKENS, "output_tokens": ops.OUTPUT_TOKENS,
                "model": prompts.get("model"), "model_revision": prompts.get("model_revision"),
            },
        },
        "block_orders": orders, "cells": [], "arm_summary": {}, "complete_cells": 0,
    }
    root.mkdir(parents=True, exist_ok=False)
    campaign["params"]["model_path"] = str(ops.resolve_model(local_only=True))
    stop = DeferredStop()
    previous = {sig: signal.signal(sig, stop.request)
                for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for block, order in enumerate(orders):
            for position, config in enumerate(order):
                run_dir = root / f"block-{block:02d}" / f"position-{position}-{config}"
                run_dir.parent.mkdir(parents=True, exist_ok=True)
                print(f"block={block} position={position} config={config}", flush=True)
                try:
                    record = run_cell(config, block, position, run_dir, args.port,
                                      Path(campaign["params"]["model_path"]), prefixes,
                                      args.expected_driver, args.store_barrier_timeout_s)
                except Exception as error:  # noqa: BLE001 - preserved, campaign continues
                    record = {"schema": 1, "kind": KIND, "config": config,
                              "block": block, "position": position, "port": args.port,
                              "ready": False,
                              "error": f"{type(error).__name__}: {error}",
                              "requests": [], "barriers": [], "cleanup_errors": [],
                              "server_returncode": None}
                campaign["cells"].append({"block": block, "position": position,
                                          "config": config, "run_dir": str(run_dir),
                                          "warm": cell_metrics(record)})
                campaign["arm_summary"] = arm_summary(campaign["cells"])
                campaign["complete_cells"] = sum(
                    1 for cell in campaign["cells"]
                    if cell["warm"]["warm_requests_per_s"] is not None)
                ops.atomic_write_json(root / "campaign.json", campaign)
                if stop.signum is not None:
                    raise InterruptedError(
                        f"deferred {stop.signum_name()} request; stopping between cells")
        complete = campaign["complete_cells"] == len(orders) * len(CONFIGS)
        write_summary(root, campaign)
        print((root / "summary.md").read_text(encoding="utf-8"), flush=True)
        return 0 if complete else 2
    except InterruptedError as error:
        campaign["stopped_early"] = str(error)
        ops.atomic_write_json(root / "campaign.json", campaign)
        write_summary(root, campaign)
        print(f"early stop: {error}", file=sys.stderr, flush=True)
        return 3
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def dry_run_plan(args) -> dict[str, Any]:
    prompts = load_fixed_prompts()
    return {
        "dry_run": True, "kind": KIND,
        "metric": "warm TTFT, warm request rate, and warm output-token rate per arm",
        "configs": list(CONFIGS),
        "blocks": args.blocks,
        "block_orders": rotation_orders(args.blocks),
        "expected_driver_parameter": args.expected_driver,
        "port": args.port,
        "store_barrier": {
            "timeout_s": args.store_barrier_timeout_s,
            "note": "non-gating log-parse wait; a timeout is recorded and the cell continues",
        },
        "prompts": {
            "path": str(ops.PROMPTS), "prefix_count": len(prompts["prefixes"]),
            "prefix_tokens": ops.PREFIX_TOKENS, "output_tokens": ops.OUTPUT_TOKENS,
            "model": prompts.get("model"), "model_revision": prompts.get("model_revision"),
        },
        "reuse": ["start_server", "wait_ready", "streamed_completion", "stop_owned_server",
                  "server_environment", "request_log_values (barrier only)",
                  "resolve_model", "atomic_write_json"],
        "removed_mechanisms": [
            "inter-cell GPU-idle wait",
            "process lease from the shared module",
            "file identity / provenance / integrity metadata",
        ],
        "bypassed_gates": [
            "validate-cell / compare-outputs / analyze",
            "engagement checks (validate_log)",
            "correctness/store checks (wait_for_cold_store, sync_and_verify_disk)",
            "O_DIRECT trace validation",
            "driver/version and GPU admission",
            "attempts-per-cell > 1, retry, result filtering",
        ],
        "preserved": [
            "per-cell server.log (server stdout/stderr)",
            "per-cell server return code, including nonzero",
            "per-request cold/warm ttft_ms, e2e_ms, status, usage counts, and token ids",
            "warm-phase request rate and output-token rate",
            "cleanup errors and barrier outcomes as records, never as rejections",
        ],
    }


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-driver", choices=ops.EXPERIMENT_DRIVERS,
                        default=ops.EXPECTED_DRIVER)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument("--store-barrier-timeout-s", type=float,
                        default=DEFAULT_STORE_BARRIER_TIMEOUT_S)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the fixed plan without touching GPU or output state")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("HF_HOME", "/home/yunwei37/.cache/huggingface")
    if args.dry_run:
        print(json.dumps(dry_run_plan(args), ensure_ascii=False, indent=2), flush=True)
        return 0
    try:
        return run_campaign(args)
    except (ops.GateError, ValueError, OSError) as error:
        print(f"NOT STARTED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
