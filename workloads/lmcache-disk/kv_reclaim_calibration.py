"""Recompute calibration helper for the KV reclaim campaign.

One real ``recompute`` server (``max_num_seqs=2``, the supplied
``kv_cache_memory_bytes``, the existing model/port/driver) serves the
supplied token arrays sequentially through the existing streamed completion
helper (16 generated tokens by default; actual usage and lengths recorded).
``recompute_ns_per_token`` is the observed end-to-end TTFT per prompt token
on that server: a proxy price, not pure GPU compute. No cache or cold-store
phase, no retries, no preflight. Readiness/server errors, every request
outcome, and the teardown return code are preserved in
``calibration.json`` and returned.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import run_perf_only as perf  # noqa: E402

ops = perf.ops

KIND = "kv-reclaim-recompute-calibration"
MAX_NUM_SEQS = 2


def run_calibration(*, run_dir: Path, model_path: Path, port: int,
                    token_arrays: list[list[int]], expected_driver: str,
                    kv_cache_memory_bytes: int) -> dict[str, Any]:
    """Serve the supplied arrays on one recompute server; retain all outcomes."""
    run_dir = Path(run_dir)
    record: dict[str, Any] = {
        "schema": 1, "kind": KIND, "port": port, "model_path": str(model_path),
        "expected_driver_parameter": expected_driver,
        "kv_cache_memory_bytes": kv_cache_memory_bytes,
        "max_num_seqs": MAX_NUM_SEQS, "output_tokens": ops.OUTPUT_TOKENS,
        "note": ("recompute_ns_per_token is the observed end-to-end TTFT per "
                 "prompt token on the recompute server; a proxy, not pure GPU compute"),
        "started_ns": time.time_ns(), "ready": False, "ready_error": None,
        "requests": [], "recompute_ns_per_token": None, "elapsed_s": None,
        "total_output_tokens": None, "output_tokens_per_s": None,
        "cleanup_errors": [], "server_returncode": None, "error": None,
    }
    created = proc = log_file = None
    stopped = False
    seq_start_ns = seq_end_ns = None
    try:
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise RuntimeError(f"run_dir already exists, never overwritten: {run_dir}") from error
        created = True
        cache_dir = run_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=False)
        log_path = run_dir / "server.log"
        try:
            proc, log_file, argv, launch = ops.start_server(
                "recompute", model_path, cache_dir, port, log_path,
                expected_driver=expected_driver, max_num_seqs=MAX_NUM_SEQS,
                kv_cache_memory_bytes=kv_cache_memory_bytes)
            record["command"] = argv
            record["launch_command"] = launch
            try:
                ops.wait_ready(proc, port, log_path)
                record["ready"] = True
            except Exception as error:  # noqa: BLE001 - retained, never fatal
                record["ready_error"] = f"{type(error).__name__}: {error}"
            if record["ready"]:
                seq_start_ns = time.perf_counter_ns()
                for index, token_ids in enumerate(token_arrays):
                    entry: dict[str, Any] = {
                        "index": index, "request_id": f"cal-p{index}-recompute",
                        "prompt_tokens": len(token_ids), "attempted": True}
                    if proc.poll() is not None:
                        entry["attempted"] = False
                        entry["error"] = f"server process exited with return code {proc.returncode}"
                    else:
                        try:
                            entry["response"] = ops.streamed_completion(
                                port, token_ids, entry["request_id"])
                        except Exception as error:  # noqa: BLE001 - retained, never fatal
                            entry["error"] = f"{type(error).__name__}: {error}"
                    record["requests"].append(entry)
                seq_end_ns = time.perf_counter_ns()
            else:
                for index, token_ids in enumerate(token_arrays):
                    record["requests"].append({
                        "index": index, "request_id": f"cal-p{index}-recompute",
                        "prompt_tokens": len(token_ids), "attempted": False,
                        "error": "server never became ready"})
        finally:
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
            elif log_file is not None:
                try:
                    log_file.close()
                except OSError:
                    pass
        if seq_start_ns is not None and seq_end_ns is not None:
            completed = [entry for entry in record["requests"] if entry.get("response")]
            ns_values = [
                float(entry["response"]["ttft_ms"]) * 1e6 / entry["prompt_tokens"]
                for entry in completed
                if isinstance(entry["response"].get("ttft_ms"), (int, float))
                and float(entry["response"]["ttft_ms"]) > 0]
            record["recompute_ns_per_token"] = (round(statistics.median(ns_values))
                                                if ns_values else None)
            total = sum(int(entry["response"]["usage"]["completion_tokens"])
                        for entry in completed
                        if entry["response"]["usage"].get("completion_tokens") is not None)
            elapsed_s = (seq_end_ns - seq_start_ns) / 1e9
            record["elapsed_s"] = elapsed_s
            record["total_output_tokens"] = total
            record["output_tokens_per_s"] = total / elapsed_s if elapsed_s > 0 else None
    except BaseException as error:  # noqa: BLE001 - recorded before return
        record["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
    record["finished_ns"] = time.time_ns()
    if created:
        try:
            ops.atomic_write_json(run_dir / "calibration.json", record)
        except BaseException as error:  # noqa: BLE001
            record["cleanup_errors"].append(
                f"calibration.json write: {type(error).__name__}: {error}")
    return record
