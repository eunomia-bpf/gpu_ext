#!/usr/bin/env python3
"""Mixed-traffic LMCache GdsBackend runner: FIFO vs native vs BPF write deferral.

The existing sequential cold-then-warm workload cannot show policy benefit:
writes never overlap demand reads.  This runner drives the installed LMCache
0.5.4 ``GdsBackend`` directly (no vLLM server) with one mixed phase per cell:

- urgent demand reads of already-stored real 24 MiB KV objects.  Each read is
  a blocking ``get_blocking`` issued from its own worker thread so the cuFile
  reads of distinct objects overlap in time;
- asynchronous background writes of different real objects, submitted through
  the backend's ``submit_put_task`` and completed on the backend event loop.

All three modes receive the SAME explicit controlled policy inputs (HBM
pressure 801 permille, slack 10 ms, speculative_recomputable false).  These
are values chosen by the runner and labelled as controlled inputs; they are
not measured live HBM pressure.  Under them, demand reads always submit
immediately in every mode, while native and BPF defer each background write
by 10 ms and FIFO submits every write immediately.

Objects live in the backend's bounded GPU staging pool
(``gds_buffer_size`` MiB, default 256, sized for an RTX 5090 that already
holds a live experiment).  Storage is real cuFile I/O with ``use_direct_io``
on the cell's fresh cache directory.  The same executor code and the same
deterministic traffic plan run for every mode; only the decider differs.

Per request the record captures offer, submitted, and completion times plus
bytes.  Metrics are read end-to-end p50/p99 (offer through completion),
write completion throughput, total storage bandwidth, and the adapter's
decision counts.  No correctness, clock, preflight, admission, or retry
gates are added; failures are recorded in the cell and the campaign
continues.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
import signal
import statistics
import sys
import threading
import time
from typing import Any, Optional


HERE = Path(__file__).resolve().parent
GDS_CONTROL = HERE / "gds-control"
KIND = "lmcache_gds_mixed_backend"
CONFIGS = ("gds_fifo", "gds_native", "gds_bpf")
POLICY_MODES = {"gds_fifo": "fifo", "gds_native": "native", "gds_bpf": "bpf"}
DEFAULT_BLOCKS = 5
DEFAULT_GDS_BUFFER_SIZE_MIB = 256
DEFAULT_OBJECT_BYTES = 24 * 1024 * 1024
DEFAULT_READS = 4
DEFAULT_WRITES = 6
DEFAULT_READ_STAGGER_S = 0.002
DEFAULT_WRITE_STAGGER_S = 0.004
DEFAULT_DST_DEVICE = "cuda:0"
WAIT_TIMEOUT_S = 120.0
EXPECTED_LMCACHE_VERSION = "0.5.4"
RAW_NAME = "raw.jsonl"
SUMMARY_NAME = "summary.json"

CONTROLLED_POLICY_INPUTS: dict[str, Any] = {
    "hbm_pressure_permille": 801,
    "slack_ns": 10_000_000,
    "speculative_recomputable": False,
    "label": "explicit controlled inputs, not measured live HBM pressure",
}


@dataclass(frozen=True)
class TrafficPlan:
    """Deterministic per-cell traffic, identical for every policy mode."""

    reads: int
    writes: int
    object_bytes: int
    read_stagger_s: float
    write_stagger_s: float
    pool_bytes: int

    @property
    def objects(self) -> int:
        return self.reads + self.writes

    def to_dict(self) -> dict[str, Any]:
        return {
            "reads": self.reads,
            "writes": self.writes,
            "object_bytes": self.object_bytes,
            "read_stagger_s": self.read_stagger_s,
            "write_stagger_s": self.write_stagger_s,
            "pool_bytes": self.pool_bytes,
            "total_offered_bytes": self.objects * self.object_bytes,
        }


def make_plan(args: argparse.Namespace) -> TrafficPlan:
    object_bytes = args.object_mib * 1024 * 1024
    pool_bytes = args.gds_buffer_size_mib * 1024 * 1024
    plan = TrafficPlan(
        reads=args.reads,
        writes=args.writes,
        object_bytes=object_bytes,
        read_stagger_s=args.read_stagger_ms / 1000.0,
        write_stagger_s=args.write_stagger_ms / 1000.0,
        pool_bytes=pool_bytes,
    )
    validate_plan(plan)
    return plan


def validate_plan(plan: TrafficPlan) -> None:
    if plan.reads < 1 or plan.writes < 1:
        raise ValueError("--reads and --writes must each be at least 1")
    if plan.object_bytes < 4096 or plan.object_bytes % 4096 != 0:
        raise ValueError(
            f"object bytes {plan.object_bytes} must be a 4096-byte multiple "
            "for O_DIRECT staging"
        )
    if plan.pool_bytes < plan.objects * plan.object_bytes:
        raise ValueError(
            f"staging pool {plan.pool_bytes} bytes cannot hold all "
            f"{plan.objects} in-flight objects of {plan.object_bytes} bytes; "
            "raise --gds-buffer-size-mib or lower --reads/--writes"
        )


def rotation_orders(blocks: int) -> list[list[str]]:
    """Return complete cyclic rotations; every mode occupies every position."""
    if blocks < 1:
        raise ValueError(f"--blocks must be at least 1, got {blocks}")
    return [list(CONFIGS[offset:] + CONFIGS[:offset])
            for offset in (block % len(CONFIGS) for block in range(blocks))]


def _percentile_nearest_rank(values: list[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), -(-pct * len(ordered) // 100)))
    return ordered[rank - 1]


def compute_metrics(requests: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive the cell metrics from per-request offer/submitted/completed records."""
    completed = [r for r in requests if r.get("status") == "completed"]
    reads = [r for r in completed if r["role"] == "read_demand"]
    writes = [r for r in completed if r["role"] == "write_background"]
    read_latency_ms = [
        (r["completed_s"] - r["offer_s"]) * 1000.0 for r in reads
    ]
    metrics: dict[str, Any] = {
        "completed_reads": len(reads),
        "completed_writes": len(writes),
        "read_end_to_end_p50_ms": statistics.median(read_latency_ms) if read_latency_ms else None,
        "read_end_to_end_p99_ms": _percentile_nearest_rank(read_latency_ms, 99),
    }
    if writes:
        window = max(r["completed_s"] for r in writes) - min(r["offer_s"] for r in writes)
        write_bytes = sum(r["bytes"] for r in writes)
        metrics["write_completion_throughput_mib_s"] = (
            write_bytes / window / 1024**2 if window > 0 else None
        )
        metrics["write_completion_ops_s"] = len(writes) / window if window > 0 else None
    if completed:
        overall_window = max(r["completed_s"] for r in completed) - min(
            r["offer_s"] for r in completed
        )
        overall_bytes = sum(r["bytes"] for r in completed)
        metrics["total_storage_bandwidth_mib_s"] = (
            overall_bytes / overall_window / 1024**2 if overall_window > 0 else None
        )
    return metrics


def cell_metrics(record: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(record.get("metrics", {}))
    decision_counts = record.get("decision_counts", {})
    metrics["defer_decisions"] = decision_counts.get("defer")
    return metrics


def median_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "read_end_to_end_p50_ms",
        "read_end_to_end_p99_ms",
        "write_completion_throughput_mib_s",
        "total_storage_bandwidth_mib_s",
    )
    per_mode: dict[str, Any] = {}
    for config in CONFIGS:
        rows = [cell["metrics"] for cell in cells if cell["config"] == config]
        medians = {}
        for name in metric_names:
            values = [float(row[name]) for row in rows
                      if isinstance(row.get(name), (int, float))]
            medians[name] = statistics.median(values) if values else None
        per_mode[config] = {
            "cells_attempted": len(rows),
            "cells_measured": sum(
                row.get("read_end_to_end_p50_ms") is not None for row in rows
            ),
            "medians": medians,
        }
    return {"kind": KIND, "cells_attempted": len(cells), "per_mode": per_mode}


def write_jsonl_record(raw_file, record: dict[str, Any]) -> None:
    raw_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    raw_file.flush()
    os.fsync(raw_file.fileno())


def atomic_write_json(path: Path, value: Any) -> None:
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def output_root(output: Path | None) -> Path:
    if output is None:
        output = HERE / "raw" / f"{KIND}-{time.strftime('%Y%m%dT%H%M%S')}"
    return output.resolve()


@dataclass
class BackendParts:
    """Lazy-imported LMCache pieces; kept out of module import time."""

    torch: Any
    GdsBackend: Any
    LMCacheEngineConfig: Any
    LMCacheMetadata: Any
    MemoryFormat: Any
    CacheEngineKey: Any
    adapter: Any


def load_backend_parts() -> BackendParts:
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed_version = version("lmcache")
    except PackageNotFoundError as error:
        raise RuntimeError("LMCache is not installed") from error
    if installed_version != EXPECTED_LMCACHE_VERSION:
        raise RuntimeError(
            f"this runner targets LMCache {EXPECTED_LMCACHE_VERSION}, "
            f"found {installed_version}"
        )
    import torch

    from lmcache.utils import CacheEngineKey
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.memory_management import MemoryFormat
    from lmcache.v1.metadata import LMCacheMetadata
    from lmcache.v1.storage_backend.gds_backend import GdsBackend

    if str(GDS_CONTROL) not in sys.path:
        sys.path.insert(0, str(GDS_CONTROL))
    import lmcache_gds_backend_adapter as adapter

    return BackendParts(
        torch=torch,
        GdsBackend=GdsBackend,
        LMCacheEngineConfig=LMCacheEngineConfig,
        LMCacheMetadata=LMCacheMetadata,
        MemoryFormat=MemoryFormat,
        CacheEngineKey=CacheEngineKey,
        adapter=adapter,
    )


def make_engine_config(parts: BackendParts, cache_dir: Path,
                       gds_buffer_size_mib: int):
    """Construct the config the same way the installed backend consumes it."""
    config = parts.LMCacheEngineConfig(
        gds_path=str(cache_dir),
        gds_buffer_size=gds_buffer_size_mib,
        use_gds=True,
        gds_backend="cufile",
        extra_config={"use_direct_io": True},
    )
    config._user_set_keys = {
        "gds_path", "gds_buffer_size", "use_gds", "gds_backend", "extra_config",
    }
    return config


def make_metadata(parts: BackendParts, object_bytes: int):
    torch = parts.torch
    return parts.LMCacheMetadata(
        model_name="gds-mixed-runner",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.uint8,
        kv_shape=(1, 1, 1, 1, object_bytes),
    )


def start_event_loop() -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, name="gds-mixed-loop",
                              daemon=True)
    thread.start()
    return loop, thread


def stop_event_loop(loop: asyncio.AbstractEventLoop,
                    thread: threading.Thread) -> None:
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=10.0)


def make_keys(parts: BackendParts, plan: TrafficPlan) -> list[Any]:
    keys = []
    for index in range(plan.objects):
        keys.append(
            parts.CacheEngineKey(
                model_name="gds-mixed",
                world_size=1,
                worker_id=0,
                chunk_hash=0x7AC5E0 + index,
                dtype=parts.torch.uint8,
                request_configs=None,
            )
        )
    return keys


def wrap_submit_timing(backend: Any, log: dict[Any, dict[str, Any]],
                       log_lock: threading.Lock, t0: float) -> None:
    """Stamp ``submitted_s`` when LMCache actually starts the save coroutine.

    Installed before the policy adapter, so both the immediate FIFO path and
    the adapter's deferred path reach the real save through this stamp.
    """
    real_save = backend._async_save_bytes_to_disk

    async def timed_save(key, memory_obj, on_complete_callback=None):
        with log_lock:
            log[key]["submitted_s"] = time.perf_counter() - t0
        return await real_save(key, memory_obj, on_complete_callback)

    backend._async_save_bytes_to_disk = timed_save


def install_policy(backend: Any, mode: str, parts: BackendParts) -> Any:
    provider = parts.adapter.EnvironmentRequestProvider(
        parts.adapter.Telemetry(**{
            key: CONTROLLED_POLICY_INPUTS[key]
            for key in ("hbm_pressure_permille", "slack_ns",
                        "speculative_recomputable")
        })
    )
    return parts.adapter.install_backend(backend, mode=mode,
                                         request_provider=provider)


def allocate_object(backend: Any, parts: BackendParts, object_bytes: int,
                    value: int) -> Any:
    torch = parts.torch
    memory_obj = backend.allocate(
        torch.Size([object_bytes]),
        torch.uint8,
        fmt=parts.MemoryFormat.BINARY,
        eviction=False,
        busy_loop=True,
    )
    if memory_obj is None:
        return None
    memory_obj.tensor.view(torch.uint8).fill_(value)
    return memory_obj


def prestore_read_objects(backend: Any, parts: BackendParts, keys: list[Any],
                          plan: TrafficPlan) -> dict[str, Any]:
    """Store the urgent-read objects through the unadorned backend."""
    start = time.perf_counter()
    completed = 0
    errors: list[str] = []
    for index in range(plan.reads):
        memory_obj = allocate_object(backend, parts, plan.object_bytes,
                                    0x40 + index)
        if memory_obj is None:
            errors.append(f"read-{index}: staging pool allocation failed")
            continue
        try:
            future = backend.submit_put_task(keys[index], memory_obj)
            future.result(timeout=WAIT_TIMEOUT_S)
        except Exception as error:
            errors.append(f"read-{index}: {type(error).__name__}: {error}")
        else:
            completed += 1
        finally:
            memory_obj.ref_count_down()
    return {
        "objects": plan.reads,
        "object_bytes": plan.object_bytes,
        "completed": completed,
        "errors": errors,
        "duration_s": time.perf_counter() - start,
    }


def run_mixed_traffic(backend: Any, parts: BackendParts, keys: list[Any],
                      plan: TrafficPlan,
                      log: dict[Any, dict[str, Any]],
                      log_lock: threading.Lock,
                      t0: float) -> None:
    """Run the overlapping demand reads and background writes for one cell."""
    write_memory_objs: list[Any] = []
    for index in range(plan.writes):
        memory_obj = allocate_object(backend, parts, plan.object_bytes,
                                    0xA0 + index)
        if memory_obj is None:
            key = keys[plan.reads + index]
            with log_lock:
                log[key]["status"] = "allocation_failed"
                log[key]["error"] = "staging pool allocation failed"
            write_memory_objs.append(None)
        else:
            write_memory_objs.append(memory_obj)

    phase_start = time.perf_counter()

    def read_worker(index: int) -> None:
        key = keys[index]
        target = phase_start + index * plan.read_stagger_s
        delay = target - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        with log_lock:
            record = log[key]
            record["offer_s"] = time.perf_counter() - t0
            record["submitted_s"] = record["offer_s"]
        try:
            memory_obj = backend.get_blocking(key)
            with log_lock:
                record["completed_s"] = time.perf_counter() - t0
                if memory_obj is None:
                    record["status"] = "get_blocking_none"
                else:
                    record["bytes"] = memory_obj.get_size()
                    record["status"] = "completed"
            if memory_obj is not None:
                memory_obj.ref_count_down()
        except Exception as error:
            with log_lock:
                record["completed_s"] = time.perf_counter() - t0
                record["status"] = "error"
                record["error"] = f"{type(error).__name__}: {error}"

    readers = [
        threading.Thread(target=read_worker, args=(index,),
                         name=f"gds-mixed-read-{index}", daemon=True)
        for index in range(plan.reads)
    ]
    for reader in readers:
        reader.start()

    write_futures: list[tuple[Any, Any, Any]] = []
    for index in range(plan.writes):
        key = keys[plan.reads + index]
        memory_obj = write_memory_objs[index]
        target = phase_start + index * plan.write_stagger_s
        delay = target - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        if memory_obj is None:
            continue

        def on_complete(_key, record=log[key]):
            with log_lock:
                record["completed_s"] = time.perf_counter() - t0
                record["status"] = "completed"

        with log_lock:
            record = log[key]
            record["offer_s"] = time.perf_counter() - t0
            record["bytes"] = memory_obj.get_size()
        try:
            future = backend.submit_put_task(key, memory_obj,
                                             on_complete_callback=on_complete)
        except Exception as error:
            with log_lock:
                record["completed_s"] = time.perf_counter() - t0
                record["status"] = "error"
                record["error"] = f"{type(error).__name__}: {error}"
            memory_obj.ref_count_down()
            continue
        write_futures.append((key, memory_obj, future))

    for reader in readers:
        reader.join(timeout=WAIT_TIMEOUT_S)

    for index in range(plan.reads):
        with log_lock:
            record = log[keys[index]]
            if record["status"] == "pending":
                record["status"] = "not_completed"
                if record["completed_s"] is None:
                    record["completed_s"] = time.perf_counter() - t0

    for key, memory_obj, future in write_futures:
        record = log[key]
        try:
            future.result(timeout=WAIT_TIMEOUT_S)
        except Exception as error:
            with log_lock:
                record["completed_s"] = time.perf_counter() - t0
                if record["status"] == "pending":
                    record["status"] = "error"
                    record["error"] = f"{type(error).__name__}: {error}"
        with log_lock:
            if record["completed_s"] is None:
                record["completed_s"] = time.perf_counter() - t0
            if record["status"] == "pending":
                record["status"] = "save_not_confirmed"
        memory_obj.ref_count_down()


def gds_effective_backend_facts(backend: Any) -> dict[str, Any]:
    return {
        "use_gds": bool(backend.use_gds),
        "gds_backend": backend.gds_backend,
        "use_direct_io": bool(backend.use_direct_io),
        "fstype": backend.fstype,
        "gds_path": str(backend.gds_path),
        "buffer_size_mib": int(backend.config.gds_buffer_size),
    }


def run_cell(config: str, block: int, position: int, run_dir: Path,
             plan: TrafficPlan, dst_device: str) -> dict[str, Any]:
    """One rotated cell: fresh cache, prestore, one policy mode, mixed traffic."""
    parts = load_backend_parts()
    cache_dir = run_dir / "cache"
    record: dict[str, Any] = {
        "schema": 1,
        "kind": KIND,
        "config": config,
        "block": block,
        "position": position,
        "policy_mode": POLICY_MODES[config],
        "controlled_policy_inputs": dict(CONTROLLED_POLICY_INPUTS),
        "gds": {},
        "traffic": dict(plan.to_dict(), prestore=None),
        "requests": [],
        "decision_counts": {},
        "metrics": {},
        "wall_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cleanup_errors": [],
    }
    run_dir.mkdir(parents=True, exist_ok=False)
    cache_dir.mkdir(parents=True, exist_ok=False)

    loop, loop_thread = start_event_loop()
    backend: Any = None
    adapter: Any = None
    try:
        backend = parts.GdsBackend(
            make_engine_config(parts, cache_dir, plan.pool_bytes // 1024**2),
            make_metadata(parts, plan.object_bytes),
            loop,
            dst_device,
        )
        record["gds"] = gds_effective_backend_facts(backend)
        keys = make_keys(parts, plan)
        prestore = prestore_read_objects(backend, parts, keys, plan)
        record["traffic"]["prestore"] = prestore

        log_lock = threading.Lock()
        log: dict[Any, dict[str, Any]] = {}
        for index, key in enumerate(keys):
            role = "read_demand" if index < plan.reads else "write_background"
            prefix = "r" if index < plan.reads else "w"
            number = index if index < plan.reads else index - plan.reads
            log[key] = {
                "id": f"{prefix}{number}",
                "role": role,
                "object": f"{'read' if index < plan.reads else 'write'}-{number}",
                "bytes": plan.object_bytes,
                "offer_s": None,
                "submitted_s": None,
                "completed_s": None,
                "status": "pending",
            }
        record["requests"] = [log[key] for key in keys]

        t0 = time.perf_counter()
        record["t0_wall"] = time.strftime("%Y-%m-%dT%H:%M:%S%z",
                                          time.localtime())
        wrap_submit_timing(backend, log, log_lock, t0)
        adapter = install_policy(backend, POLICY_MODES[config], parts)
        run_mixed_traffic(backend, parts, keys, plan, log, log_lock, t0)
        record["decision_counts"] = dict(adapter.stats)
        record["metrics"] = compute_metrics(record["requests"])
        return record
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        return record
    finally:
        cleanup_errors: list[str] = record["cleanup_errors"]
        if adapter is not None:
            try:
                adapter.close()
            except Exception as error:
                cleanup_errors.append(f"adapter.close: {error}")
        if backend is not None:
            try:
                backend.close()
            except Exception as error:
                cleanup_errors.append(f"backend.close: {error}")
        try:
            stop_event_loop(loop, loop_thread)
        except Exception as error:
            cleanup_errors.append(f"loop stop: {error}")
        record["cleanup_errors"] = cleanup_errors


class DeferredStop:
    """Deferred stop between cells; never interrupts an owned cell."""

    def __init__(self):
        self.signum = None

    def request(self, signum, _frame):
        if self.signum is None:
            self.signum = signum

    def signum_name(self) -> str:
        return signal.Signals(self.signum).name if self.signum is not None else ""


def run_campaign(args: argparse.Namespace) -> int:
    plan = make_plan(args)
    root = output_root(args.output)
    orders = rotation_orders(args.blocks)
    root.mkdir(parents=True, exist_ok=False)
    campaign: dict[str, Any] = {
        "kind": KIND,
        "timestamp": time.strftime("%Y%m%dT%H%M%S"),
        "params": {
            "blocks": args.blocks,
            "configs": list(CONFIGS),
            "dst_device": args.dst_device,
            "gds_buffer_size_mib": args.gds_buffer_size_mib,
            "object_mib": args.object_mib,
            "reads": args.reads,
            "writes": args.writes,
            "read_stagger_ms": args.read_stagger_ms,
            "write_stagger_ms": args.write_stagger_ms,
            "attempts_per_cell": 1,
            "retry": False,
            "controlled_policy_inputs": dict(CONTROLLED_POLICY_INPUTS),
        },
        "block_orders": orders,
        "cells": [],
    }
    stop = DeferredStop()
    previous = {sig: signal.signal(sig, stop.request)
                for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with (root / RAW_NAME).open("x", encoding="utf-8") as raw_file:
            for block, order in enumerate(orders):
                for position, config in enumerate(order):
                    run_dir = root / f"block-{block:02d}" / f"position-{position}-{config}"
                    print(f"block={block} position={position} config={config}", flush=True)
                    record = run_cell(config, block, position, run_dir, plan,
                                      args.dst_device)
                    atomic_write_json(run_dir / "result.json", record)
                    write_jsonl_record(raw_file, record)
                    campaign["cells"].append({
                        "block": block,
                        "position": position,
                        "config": config,
                        "run_dir": str(run_dir),
                        "metrics": cell_metrics(record),
                    })
                    campaign["summary"] = median_summary(campaign["cells"])
                    atomic_write_json(root / "campaign.json", campaign)
                    atomic_write_json(root / SUMMARY_NAME, campaign["summary"])
                    if stop.signum is not None:
                        campaign["stopped_early"] = (
                            f"deferred {stop.signum_name()} request; stopping between cells"
                        )
                        atomic_write_json(root / "campaign.json", campaign)
                        return 3
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)

    expected_cells = args.blocks * len(CONFIGS)
    complete = (
        len(campaign["cells"]) == expected_cells
        and all(
            cell["metrics"].get("read_end_to_end_p50_ms") is not None
            for cell in campaign["cells"]
        )
    )
    print(json.dumps(campaign["summary"], ensure_ascii=False, indent=2), flush=True)
    return 0 if complete else 2


def dry_run_plan(args: argparse.Namespace) -> dict[str, Any]:
    plan = make_plan(args)
    return {
        "dry_run": True,
        "kind": KIND,
        "configs": list(CONFIGS),
        "blocks": args.blocks,
        "block_orders": rotation_orders(args.blocks),
        "dst_device": args.dst_device,
        "traffic": plan.to_dict(),
        "gds": {
            "buffer_size_mib": args.gds_buffer_size_mib,
            "backend": "cufile",
            "use_direct_io": True,
            "adapter_module": "lmcache_gds_backend_adapter",
            "policy_modes": POLICY_MODES,
            "expected_lmcache_version": EXPECTED_LMCACHE_VERSION,
        },
        "controlled_policy_inputs": dict(CONTROLLED_POLICY_INPUTS),
        "outputs": [RAW_NAME, SUMMARY_NAME, "campaign.json", "per-cell result.json"],
        "reused": ["installed LMCache 0.5.4 GdsBackend",
                   "committed lmcache_gds_backend_adapter",
                   "committed lmcache_gds_policy_adapter deciders"],
        "gates": [],
        "retries": False,
        "attempts_per_cell": 1,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS,
                        help="rotated blocks; 1 for the first measurement")
    parser.add_argument("--reads", type=int, default=DEFAULT_READS,
                        help="overlapping urgent demand reads of stored objects")
    parser.add_argument("--writes", type=int, default=DEFAULT_WRITES,
                        help="asynchronous background writes of distinct objects")
    parser.add_argument("--object-mib", type=int, default=DEFAULT_OBJECT_BYTES // 1024**2,
                        help="object size in MiB (default 24, the real KV object size)")
    parser.add_argument("--read-stagger-ms", type=float,
                        default=DEFAULT_READ_STAGGER_S * 1000)
    parser.add_argument("--write-stagger-ms", type=float,
                        default=DEFAULT_WRITE_STAGGER_S * 1000)
    parser.add_argument("--gds-buffer-size-mib", type=int,
                        default=DEFAULT_GDS_BUFFER_SIZE_MIB,
                        help="bounded GPU staging pool in MiB")
    parser.add_argument("--dst-device", default=DEFAULT_DST_DEVICE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.gds_buffer_size_mib < 1:
        parser.error("--gds-buffer-size-mib must be at least 1")
    if args.object_mib < 1:
        parser.error("--object-mib must be at least 1")
    if args.dst_device and not args.dst_device.startswith("cuda"):
        parser.error("--dst-device must start with 'cuda'")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps(dry_run_plan(args), ensure_ascii=False, indent=2), flush=True)
        return 0
    try:
        return run_campaign(args)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"NOT STARTED: {type(error).__name__}: {error}", file=sys.stderr,
              flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
