#!/usr/bin/env python3
"""Four-arm LMCache GDS asynchronous-prefetch serving runner.

Arms (same model, driver config, fixed prompts, 768 MiB KV cache, and
256 MiB GDS staging in every arm):

- ``gds_demand_fifo``: demand-driven disk retrieval, existing FIFO policy,
  async prefetch disabled (reference serving path).
- ``gds_eager_async``: eager async prefetch on the same real lookup
  opportunity, FIFO policy.
- ``gds_async_native`` / ``gds_async_bpf``: adaptive prefetch with
  identical information through the native decider or the BPF decider.

Async arms opt into the agreed backend interface with
``LMCACHE_GDS_ASYNC_PREFETCH=1`` plus ordinary LMCache
``LMCACHE_ENABLE_ASYNC_LOADING=True``; the policy mode stays
``LMCACHE_GDS_POLICY_MODE`` fifo/native/bpf.  Every new cell launches the
server with ``--max-num-seqs 2`` identically: with the running limit at 1
the vLLM scheduler breaks out of the waiting loop before the connector
lookup, so a queued request could never start its lookup while one request
is already running.

The cell lifecycle reuses ``run_perf_only.py`` and
``lmcache_primitives.py``: server start/stop, readiness, the existing
request payload semantics, the non-gating cold-store barrier, and per-cell
raw capture.  Eight sequential cold population requests populate the disk,
then eight bounded-concurrent warm HTTP completions overlap so disk
retrieval can precede GPU consumption.  Every warm request carries the
top-level ``kv_transfer_params`` entry ``lmcache.prefetch_deadline_ns``:
host monotonic nanoseconds at the actual HTTP send plus
``--prefetch-lead-ms`` (default 0).  It is an explicit application
deadline, not a predicted scheduler use time, and every arm receives the
same field with the same lead.

Five rotated blocks, fresh raw directory, no retries, overwriting,
filtering, or gates; failures and server stderr are preserved per cell.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import signal
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Any


HERE = Path(__file__).resolve().parent
PERF_PATH = HERE / "run_perf_only.py"
SPEC = importlib.util.spec_from_file_location("gds_async_prefetch_perf", PERF_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load performance runner from {PERF_PATH}")
perf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(perf)
ops = perf.ops

KIND = "gds-async-prefetch-575"
CONFIGS = ("gds_demand_fifo", "gds_eager_async", "gds_async_native", "gds_async_bpf")
POLICY_MODES = {
    "gds_demand_fifo": "fifo",
    "gds_eager_async": "fifo",
    "gds_async_native": "native",
    "gds_async_bpf": "bpf",
}
DEMAND_ARMS = frozenset({"gds_demand_fifo"})
ASYNC_ENV = {
    "LMCACHE_GDS_ASYNC_PREFETCH": "1",
    "LMCACHE_ENABLE_ASYNC_LOADING": "True",
}
MAX_NUM_SEQS = 2
DEFAULT_BLOCKS = 5
DEFAULT_GDS_BUFFER_SIZE_MIB = 256
DEFAULT_KV_CACHE_MEMORY_BYTES = 768 * 1024 * 1024
DEFAULT_EXPECTED_DRIVER = "575.57.08"
DEFAULT_WARM_STAGGER_MS = 250.0
DEFAULT_WARM_CONCURRENCY = 4
DEFAULT_PREFETCH_LEAD_MS = 0.0
GDS_CONTROL = HERE / "gds-control"
BOOTSTRAP = GDS_CONTROL / "bootstrap"
RAW_NAME = "raw.jsonl"
SUMMARY_NAME = "summary.json"
_BASE_SERVER_ENVIRONMENT = ops.server_environment
_BASE_SERVER_ARGV = ops.server_argv


def output_root(output: Path | None) -> Path:
    if output is None:
        output = HERE / "raw" / f"{KIND}-{time.strftime('%Y%m%dT%H%M%S')}"
    return output.resolve()


def rotation_orders(blocks: int) -> list[list[str]]:
    """Return complete cyclic rotations; every arm occupies every position in five blocks."""
    if blocks < 1:
        raise ValueError(f"--blocks must be at least 1, got {blocks}")
    return [list(CONFIGS[offset:] + CONFIGS[:offset])
            for offset in (block % len(CONFIGS) for block in range(blocks))]


def gds_server_environment(config: str, cache_dir: Path, expected_driver: str,
                           gds_buffer_size_mib: int) -> dict[str, str]:
    """Build the GDS arm environment; async arms add the two agreed opt-ins."""
    env = _BASE_SERVER_ENVIRONMENT("lmcache_disk", cache_dir, expected_driver)
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
        "LMCACHE_GDS_POLICY_MODE": POLICY_MODES[config],
        "LMCACHE_EXTRA_CONFIG": ops.canonical({"use_direct_io": True}),
    })
    if config not in DEMAND_ARMS:
        env.update(ASYNC_ENV)
    return env


def cell_server_argv(config: str, model_path: Path, port: int | str,
                     kv_cache_memory_bytes: int) -> list[str]:
    """Identical vLLM flags in every arm: max-num-seqs 2 plus the shared KV bytes."""
    return _BASE_SERVER_ARGV(config, model_path, port,
                             max_num_seqs=MAX_NUM_SEQS,
                             kv_cache_memory_bytes=kv_cache_memory_bytes)


def measured_streamed_completion(port: int, token_ids: list[int], request_id: str,
                                 entry: dict[str, Any],
                                 deadline_ns: int | None = None) -> dict[str, Any]:
    """One measured completion with the existing request payload semantics.

    With ``deadline_ns`` the payload adds the top-level
    ``kv_transfer_params`` entry ``lmcache.prefetch_deadline_ns``: host
    monotonic nanoseconds (``time.monotonic_ns``) at the actual HTTP send
    plus the lead, i.e. an explicit application deadline, not a predicted
    scheduler use time.  No correctness assertions are made on the
    response: observed status, usage, token IDs, text, and engine request
    IDs are recorded; a missing first token yields ``ttft_ms: None`` while
    the actual E2E is still preserved.  HTTP errors are raised for the
    caller to record.
    """
    payload = {"model": ops.MODEL_ID, "prompt": token_ids,
               "max_tokens": ops.OUTPUT_TOKENS, "temperature": 0, "seed": 0,
               "ignore_eos": True, "stream": True, "logprobs": 1,
               "return_token_ids": True,
               "stream_options": {"include_usage": True}}
    if deadline_ns is not None:
        payload["kv_transfer_params"] = {"lmcache.prefetch_deadline_ns": deadline_ns}
    sent_ns = time.perf_counter_ns()
    entry["sent_ns"] = sent_ns
    if deadline_ns is not None:
        entry["prefetch_deadline_ns"] = deadline_ns
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json", "X-Request-Id": request_id},
        method="POST")
    start = time.perf_counter_ns()
    first = None
    text_parts: list[str] = []
    generated_ids: list[int] = []
    engine_request_ids: set[str] = set()
    usage: dict[str, Any] = {}
    status = None
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            status = response.status
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                obj = json.loads(line[6:])
                engine_request_id = obj.get("id")
                if isinstance(engine_request_id, str) and engine_request_id:
                    engine_request_ids.add(engine_request_id)
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices", []):
                    piece = choice.get("text") or ""
                    piece_ids = choice.get("token_ids") or []
                    generated_tokens = (choice.get("logprobs") or {}).get("tokens") or []
                    if (generated_tokens or piece_ids) and first is None:
                        first = time.perf_counter_ns()
                    text_parts.append(piece)
                    generated_ids.extend(int(value) for value in piece_ids)
    except urllib.error.HTTPError as exc:
        entry["status"] = exc.code
        raise ops.GateError(
            f"HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
    end = time.perf_counter_ns()
    return {
        "request_header": request_id,
        "input_tokens": len(token_ids),
        "status": status,
        "ttft_ms": (first - start) / 1e6 if first is not None else None,
        "e2e_ms": (end - start) / 1e6,
        "usage": usage,
        "text": "".join(text_parts),
        "generated_token_ids": generated_ids,
        "engine_request_ids": sorted(engine_request_ids),
    }


def warm_task(port: int, item: dict[str, Any], scheduled_send_ns: int,
              lead_ns: int) -> dict[str, Any]:
    """Arrive at the scheduled time, then issue one bounded-concurrent warm request."""
    index = item["index"]
    request_id = f"lmc-p{index}-warm"
    entry: dict[str, Any] = {"phase": "warm", "prefix_index": index,
                             "request_id": request_id,
                             "scheduled_send_ns": scheduled_send_ns, "attempted": True}
    delay = scheduled_send_ns - time.perf_counter_ns()
    if delay > 0:
        time.sleep(delay / 1e9)
    try:
        entry.update(measured_streamed_completion(
            port, item["warm_token_ids"], request_id, entry,
            deadline_ns=time.monotonic_ns() + lead_ns))
    except Exception as error:  # noqa: BLE001 - preserved, never fatal
        entry["error"] = f"{type(error).__name__}: {error}"
    return entry


def warm_aggregates(record: dict[str, Any], warm_start_ns: int, warm_end_ns: int,
                    warm_concurrency: int, warm_stagger_ms: float,
                    prefetch_lead_ms: float) -> dict[str, Any] | None:
    warm = [entry for entry in record["requests"] if entry["phase"] == "warm"]
    completed = [entry for entry in warm
                 if entry.get("attempted") and "error" not in entry]
    elapsed_s = (warm_end_ns - warm_start_ns) / 1e9
    output_tokens = sum(int(entry["usage"]["completion_tokens"]) for entry in completed
                        if entry.get("usage", {}).get("completion_tokens") is not None)
    ttft = [float(entry["ttft_ms"]) for entry in completed
            if isinstance(entry.get("ttft_ms"), (int, float))]
    arrivals = []
    for entry in warm:
        scheduled = entry.get("scheduled_send_ns")
        sent = entry.get("sent_ns")
        arrivals.append({
            "prefix_index": entry["prefix_index"],
            "scheduled_send_ns": scheduled,
            "sent_ns": sent,
            "send_lag_ns": (sent - scheduled)
                           if (sent is not None and scheduled is not None) else None,
        })
    return {
        "sequential": False,
        "concurrency": warm_concurrency,
        "stagger_ms": warm_stagger_ms,
        "prefetch_lead_ms": prefetch_lead_ms,
        "requests": len(completed),
        "attempts": len(warm),
        "failures": len(warm) - len(completed),
        "output_tokens": output_tokens,
        "elapsed_s": elapsed_s,
        "requests_per_s": len(completed) / elapsed_s if elapsed_s > 0 else None,
        "output_tokens_per_s": output_tokens / elapsed_s if elapsed_s > 0 else None,
        "warm_ttft_samples": len(ttft),
        "warm_ttft_values_ms": ttft,
        "warm_ttft_median_ms": statistics.median(ttft) if ttft else None,
        "warm_ttft_p95_ms": (statistics.quantiles(ttft, n=20)[-1] if len(ttft) >= 2
                             else None),
        "warm_ttft_max_ms": max(ttft) if ttft else None,
        "arrivals": arrivals,
        "excludes": ["server startup", "cold population", "cold-store barriers", "shutdown"],
    }


def run_cell(config: str, block: int, position: int, run_dir: Path, port: int,
             model_path: Path, prefixes: list[dict[str, Any]], expected_driver: str,
             store_barrier_timeout_s: float, gds_buffer_size_mib: int,
             kv_cache_memory_bytes: int, warm_stagger_ms: float,
             warm_concurrency: int, prefetch_lead_ms: float) -> dict[str, Any]:
    """One arm, once: sequential cold population then overlapping warm requests."""
    record: dict[str, Any] = {
        "schema": 1, "kind": KIND, "config": config, "block": block, "position": position,
        "port": port, "expected_driver_parameter": expected_driver,
        "policy_mode": POLICY_MODES[config],
        "async_prefetch_enabled": config not in DEMAND_ARMS,
        "max_num_seqs": MAX_NUM_SEQS,
        "kv_cache_memory_bytes": kv_cache_memory_bytes,
        "gds_buffer_size_mib": gds_buffer_size_mib,
        "warm_stagger_ms": warm_stagger_ms,
        "warm_concurrency": warm_concurrency,
        "prefetch_lead_ms": prefetch_lead_ms,
        "prefetch_deadline_semantics": (
            "top-level kv_transfer_params.lmcache.prefetch_deadline_ns is host "
            "monotonic nanoseconds (time.monotonic_ns) at the actual HTTP send "
            "plus the prefetch lead; an explicit application deadline, not a "
            "predicted scheduler use time; the same field and lead are "
            "supplied in every arm"),
        "prompt_count": len(prefixes), "cached_tokens": ops.PREFIX_TOKENS,
        "output_tokens": ops.OUTPUT_TOKENS,
        "started_ns": time.time_ns(), "ready": False, "ready_error": None,
        "requests": [], "barriers": [], "warm_phase": None,
        "cleanup_errors": [], "server_returncode": None, "error": None,
    }
    log_path = run_dir / "server.log"
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=False)
    record["environment"] = gds_server_environment(config, cache_dir, expected_driver,
                                                   gds_buffer_size_mib)
    proc = None
    log_file = None
    stopped = False
    original_environment = ops.server_environment

    def cell_environment(_config: str, cache_dir: Path,
                         driver: str = ops.EXPECTED_DRIVER, **_options: Any) -> dict[str, str]:
        return gds_server_environment(config, cache_dir, driver, gds_buffer_size_mib)

    ops.server_environment = cell_environment
    try:
        try:
            proc, log_file, argv, launch = ops.start_server(
                config, model_path, cache_dir, port, log_path,
                expected_driver=expected_driver,
                max_num_seqs=MAX_NUM_SEQS,
                kv_cache_memory_bytes=kv_cache_memory_bytes)
        except FileExistsError:
            raise ops.GateError(
                "server log already exists; this cell would not be a first attempt")
        record["command"] = argv
        record["launch_command"] = launch
        record["worker_cpu_affinity"] = perf.worker_affinity(proc)
        try:
            ops.wait_ready(proc, port, log_path)
            record["ready"] = True
        except ops.GateError as error:
            record["ready_error"] = f"{type(error).__name__}: {error}"
        if record["ready"]:
            for item in prefixes:
                index = item["index"]
                request_id = f"lmc-p{index}-cold"
                if proc.poll() is not None:
                    record["requests"].append({
                        "phase": "cold", "prefix_index": index, "request_id": request_id,
                        "attempted": False,
                        "reason": (f"server process exited with return code "
                                   f"{proc.returncode}")})
                    continue
                entry: dict[str, Any] = {"phase": "cold", "prefix_index": index,
                                         "request_id": request_id, "attempted": True}
                try:
                    entry.update(measured_streamed_completion(
                        port, item["cold_token_ids"], request_id, entry))
                except Exception as error:  # noqa: BLE001 - preserved, never fatal
                    entry["error"] = f"{type(error).__name__}: {error}"
                record["requests"].append(entry)
                engine_ids = entry.get("engine_request_ids") or []
                if len(engine_ids) == 1:
                    record["barriers"].append(perf.store_barrier(
                        config, log_path, engine_ids[0],
                        int(item["expected_store_tokens"]), len(item["cold_token_ids"]),
                        store_barrier_timeout_s, perf.DEFAULT_STORE_BARRIER_POLL_S))
                else:
                    record["barriers"].append({
                        "applicable": True, "satisfied": False, "waited_s": 0.0,
                        "reason": f"no single stable engine request ID observed: {engine_ids}"})
            lead_ns = int(round(prefetch_lead_ms * 1_000_000))
            stagger_ns = int(round(warm_stagger_ms * 1_000_000))
            warm_start = time.perf_counter_ns()
            with ThreadPoolExecutor(max_workers=warm_concurrency,
                                    thread_name_prefix="warm") as pool:
                futures = [
                    pool.submit(warm_task, port, item,
                                warm_start + item["index"] * stagger_ns, lead_ns)
                    for item in prefixes
                ]
                for future in futures:
                    record["requests"].append(future.result())
            warm_end = time.perf_counter_ns()
            record["warm_phase"] = warm_aggregates(
                record, warm_start, warm_end, warm_concurrency,
                warm_stagger_ms, prefetch_lead_ms)
        else:
            for item in prefixes:
                for phase in perf.REQUEST_LABELS:
                    record["requests"].append({
                        "phase": phase, "prefix_index": item["index"],
                        "request_id": f"lmc-p{item['index']}-{phase}",
                        "attempted": False, "reason": "server never became ready"})
    except BaseException as error:  # noqa: BLE001 - keep numbers, never abort the campaign
        record["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        ops.server_environment = original_environment
        if proc is not None:
            try:
                ops.stop_owned_server(proc, log_file)
                stopped = True
            except BaseException as error:  # noqa: BLE001
                record["cleanup_errors"].append(
                    f"stop_owned_server: {type(error).__name__}: {error}")
                try:
                    log_file.close()
                except OSError:
                    pass
            record["server_returncode"] = proc.returncode
            if stopped and proc.returncode is None:
                record["cleanup_errors"].append("server return code unknown after bounded stop")
            try:
                ops.wait_gpu_idle()
            except BaseException as error:  # noqa: BLE001
                record["cleanup_errors"].append(
                    f"wait_gpu_idle: {type(error).__name__}: {error}")
        else:
            try:
                if log_file is not None:
                    log_file.close()
            except OSError:
                pass
        try:
            record["server_log_identity"] = ops.file_identity(log_path)
        except ops.GateError:
            record["server_log_identity"] = {"path": str(log_path.resolve()), "bytes": 0}
        record["finished_ns"] = time.time_ns()
        ops.atomic_write_json(run_dir / "result.json", record)
    return record
