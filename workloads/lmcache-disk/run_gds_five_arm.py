#!/usr/bin/env python3
"""Minimal LMCache performance runner: recompute, CPU, and three GDS policies.

The request lifecycle and per-cell raw result come from ``run_perf_only.py``.
This wrapper only supplies a five-arm rotation, the GDS backend environment,
append-only JSONL capture, and per-arm medians.  It intentionally adds no
correctness, engagement, admission, retry, GPU-idle, or clock-stability gates.
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
PERF_PATH = HERE / "run_perf_only.py"
SPEC = importlib.util.spec_from_file_location("gds_five_arm_perf", PERF_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load performance runner from {PERF_PATH}")
perf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(perf)
ops = perf.ops

KIND = "lmcache_gds_five_arm"
CONFIGS = ("recompute", "lmcache_cpu", "gds_fifo", "gds_native", "gds_bpf")
GDS_MODES = {"gds_fifo": "fifo", "gds_native": "native", "gds_bpf": "bpf"}
DEFAULT_BLOCKS = 5
DEFAULT_GDS_BUFFER_SIZE_MIB = 512
GDS_CONTROL = HERE / "gds-control"
BOOTSTRAP = GDS_CONTROL / "bootstrap"
RAW_NAME = "raw.jsonl"
SUMMARY_NAME = "summary.json"
_BASE_SERVER_ENVIRONMENT = ops.server_environment


def rotation_orders(blocks: int) -> list[list[str]]:
    """Return complete cyclic rotations; every arm occupies every position in five blocks."""
    if blocks < 1:
        raise ValueError(f"--blocks must be at least 1, got {blocks}")
    return [list(CONFIGS[offset:] + CONFIGS[:offset])
            for offset in (block % len(CONFIGS) for block in range(blocks))]


def gds_server_environment(config: str, cache_dir: Path, expected_driver: str,
                           gds_buffer_size_mib: int) -> dict[str, str]:
    """Build the ordinary environment, changing only the three GDS arms."""
    physical = "lmcache_disk" if config in GDS_MODES else config
    env = _BASE_SERVER_ENVIRONMENT(physical, cache_dir, expected_driver)
    if config not in GDS_MODES:
        return env

    env.pop("LMCACHE_LOCAL_DISK", None)
    env.pop("LMCACHE_MAX_LOCAL_DISK_SIZE", None)
    python_path = [str(BOOTSTRAP), str(GDS_CONTROL)]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env.update({
        "PYTHONPATH": os.pathsep.join(python_path),
        "LMCACHE_GDS_PATH": str(cache_dir),
        "LMCACHE_GDS_BUFFER_SIZE": str(gds_buffer_size_mib),
        "LMCACHE_USE_GDS": "True",
        "LMCACHE_GDS_BACKEND": "cufile",
        "LMCACHE_GDS_POLICY_MODE": GDS_MODES[config],
    })
    return env


def run_cell(config: str, block: int, position: int, run_dir: Path, port: int,
             model_path: Path, prefixes: list[dict[str, Any]], expected_driver: str,
             store_barrier_timeout_s: float,
             gds_buffer_size_mib: int) -> dict[str, Any]:
    """Delegate one cell while temporarily supplying its launch environment."""
    original = ops.server_environment

    def cell_environment(_config: str, cache_dir: Path,
                         driver: str = ops.EXPECTED_DRIVER) -> dict[str, str]:
        return gds_server_environment(config, cache_dir, driver, gds_buffer_size_mib)

    ops.server_environment = cell_environment
    try:
        record = perf.run_cell(config, block, position, run_dir, port, model_path,
                               prefixes, expected_driver, store_barrier_timeout_s)
        record["kind"] = KIND
        ops.atomic_write_json(run_dir / "result.json", record)
        return record
    finally:
        ops.server_environment = original


def cell_metrics(record: dict[str, Any]) -> dict[str, Any]:
    return perf.cell_metrics(record)


def median_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "warm_ttft_median_ms",
        "warm_requests_per_s",
        "warm_output_tokens_per_s",
    )
    per_arm: dict[str, Any] = {}
    for config in CONFIGS:
        rows = [cell["metrics"] for cell in cells if cell["config"] == config]
        medians = {}
        for name in metric_names:
            values = [float(row[name]) for row in rows
                      if isinstance(row.get(name), (int, float))]
            medians[name] = statistics.median(values) if values else None
        per_arm[config] = {
            "cells_attempted": len(rows),
            "cells_measured": sum(
                row.get("warm_requests_per_s") is not None for row in rows
            ),
            "medians": medians,
        }
    return {"kind": KIND, "cells_attempted": len(cells), "per_arm": per_arm}


def write_jsonl_record(raw_file, record: dict[str, Any]) -> None:
    raw_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    raw_file.flush()
    os.fsync(raw_file.fileno())


def run_campaign(args: argparse.Namespace) -> int:
    root = perf.output_root(args.output)
    orders = rotation_orders(args.blocks)
    prompts = perf.load_fixed_prompts()
    prefixes = prompts["prefixes"]
    root.mkdir(parents=True, exist_ok=False)
    model_path = ops.resolve_model(local_only=True)
    campaign: dict[str, Any] = {
        "kind": KIND,
        "timestamp": time.strftime("%Y%m%dT%H%M%S"),
        "params": {
            "blocks": args.blocks,
            "configs": list(CONFIGS),
            "port": args.port,
            "expected_driver": args.expected_driver,
            "store_barrier_timeout_s": args.store_barrier_timeout_s,
            "gds_buffer_size_mib": args.gds_buffer_size_mib,
            "attempts_per_cell": 1,
            "retry": False,
            "model_path": str(model_path),
        },
        "block_orders": orders,
        "cells": [],
    }
    stop = perf.DeferredStop()
    previous = {sig: signal.signal(sig, stop.request)
                for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with (root / RAW_NAME).open("x", encoding="utf-8") as raw_file:
            for block, order in enumerate(orders):
                for position, config in enumerate(order):
                    run_dir = root / f"block-{block:02d}" / f"position-{position}-{config}"
                    run_dir.parent.mkdir(parents=True, exist_ok=True)
                    print(f"block={block} position={position} config={config}", flush=True)
                    try:
                        record = run_cell(
                            config, block, position, run_dir, args.port, model_path,
                            prefixes, args.expected_driver, args.store_barrier_timeout_s,
                            args.gds_buffer_size_mib,
                        )
                    except Exception as error:  # preserve the attempt and continue
                        record = {
                            "schema": 1, "kind": KIND, "config": config,
                            "block": block, "position": position, "port": args.port,
                            "ready": False, "requests": [], "barriers": [],
                            "cleanup_errors": [], "server_returncode": None,
                            "error": f"{type(error).__name__}: {error}",
                        }
                        run_dir.mkdir(parents=True, exist_ok=True)
                        ops.atomic_write_json(run_dir / "result.json", record)
                    write_jsonl_record(raw_file, record)
                    campaign["cells"].append({
                        "block": block,
                        "position": position,
                        "config": config,
                        "run_dir": str(run_dir),
                        "metrics": cell_metrics(record),
                    })
                    campaign["summary"] = median_summary(campaign["cells"])
                    ops.atomic_write_json(root / "campaign.json", campaign)
                    ops.atomic_write_json(root / SUMMARY_NAME, campaign["summary"])
                    if stop.signum is not None:
                        campaign["stopped_early"] = (
                            f"deferred {stop.signum_name()} request; stopping between cells"
                        )
                        ops.atomic_write_json(root / "campaign.json", campaign)
                        return 3
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)

    expected_cells = args.blocks * len(CONFIGS)
    complete = all(
        cell["metrics"]["warm_requests_per_s"] is not None
        for cell in campaign["cells"]
    ) and len(campaign["cells"]) == expected_cells
    print(json.dumps(campaign["summary"], ensure_ascii=False, indent=2), flush=True)
    return 0 if complete else 2


def dry_run_plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dry_run": True,
        "kind": KIND,
        "configs": list(CONFIGS),
        "blocks": args.blocks,
        "block_orders": rotation_orders(args.blocks),
        "gds": {
            "buffer_size_mib": args.gds_buffer_size_mib,
            "backend": "cufile",
            "adapter_module": "lmcache_gds_backend_adapter",
            "policy_modes": GDS_MODES,
        },
        "outputs": [RAW_NAME, SUMMARY_NAME, "campaign.json", "per-cell result.json"],
        "reuse": ["run_perf_only.run_cell", "lmcache_primitives"],
        "gates": [],
        "retries": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-driver", choices=ops.EXPERIMENT_DRIVERS,
                        default=ops.EXPECTED_DRIVER)
    parser.add_argument("--port", type=int, default=perf.DEFAULT_PORT)
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument("--store-barrier-timeout-s", type=float,
                        default=perf.DEFAULT_STORE_BARRIER_TIMEOUT_S)
    parser.add_argument("--gds-buffer-size-mib", type=int,
                        default=DEFAULT_GDS_BUFFER_SIZE_MIB)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.gds_buffer_size_mib < 1:
        parser.error("--gds-buffer-size-mib must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
