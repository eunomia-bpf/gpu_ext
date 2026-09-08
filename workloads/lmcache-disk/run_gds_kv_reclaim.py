#!/usr/bin/env python3
"""Three-arm LMCache disk-aware KV reclaim serving runner (thin real runner).

Arms (same model Qwen/Qwen3-30B-A3B-FP8, same GDS/cuFile demand transport,
same FIFO per-IO policy, identical fixed prompts, capacity, arrivals, and
generation in every arm):

- ``stock``: the disk-aware KV reclaim opt-in is left off; stock vLLM victim.
- ``native``: ``LMCACHE_KV_RECLAIM=1`` with the native scalar decider.
- ``bpf``: identical reclaim policy through the kernel BPF decider.

All three arms run the existing demand-driven LMCache GDS backend
(``LMCACHE_USE_GDS`` + cuFile, ``LMCACHE_GDS_POLICY_MODE=fifo``): the
reclaim opt-in is independent of per-IO admission, and no async-prefetch
campaign repeats here (no async opt-ins, no prefetch deadlines).

Exactly one real recompute calibration phase runs before the 15 cells
(5 rotated blocks x 3 arms, fresh server process per cell): the existing
serving helper measures the same fixed prompts with short generation on a
``recompute`` server with the same capacity limits; the positive median
TTFT per prompt token is the end-to-end recompute proxy price (explicitly
not pure GPU compute) and is supplied identically to every reclaim arm via
``LMCACHE_KV_RECLAIM_RECOMPUTE_NS_PER_TOKEN``.  A failed or non-positive
calibration aborts with real errors retained; no invented constant exists.

Warm workload: eight bounded-concurrent streamed completions built from the
existing real token arrays; warm prompt = warm continuation tokens truncated
to 1536 tokens (even prefix index) or 1024 (odd prefix index); up to
``--warm-output-tokens`` generated tokens with identical stop behavior in
every arm, actual generated token counts recorded from usage, never assumed.
With a 384 MiB KV pool and two running sequences, 1536-prompt requests
overlapping generation push the pool past its actual capacity (1536+1024
prompt tokens plus generated tails).  Arrival order rotates by block while
remaining identical across the three arms within a block.  A warm request
whose stream raises keeps on the same request record the data actually
observed before the error (send time, HTTP status when a response started,
first-token time when one arrived, observed text / token IDs / usage) plus
the end time, stays labeled a failure, and infers no missing token counts or
completed-output aggregates.  Every finished warm future is additionally
checkpointed to the cell's ``warm-progress.json`` through the existing
atomic writer as it completes, so one slow request cannot hide later
completed requests; the final ``result.json`` remains the single final
per-cell record in planned request order.

With ``--resume`` on an existing ``--output`` root, complete compatible
per-cell ``result.json`` records (failed cells included) are reused
verbatim instead of rerunning those cells, nonempty directories without
``result.json`` are left untouched and reported unfinished (no age or
timeout assumption), and only missing cells run in the same planned
rotated order; existing ``raw.jsonl`` lines and per-cell files are never
rewritten.

The runner imports its helpers (``run_gds_async_prefetch`` ->
``run_perf_only`` -> ``lmcache_primitives``) and adds no lifecycle code, no
correctness gates, no retries, no admission, no wall-clock timeout.  Every
cold request, store barrier, warm request, teardown step, server exit code,
raw response, error, and adapter diagnostics snapshot is recorded exactly
once and preserved; a failure never aborts a cell.  Adapter diagnostics
arrive through the ``LMCACHE_KV_RECLAIM_DIAG_OUT`` exit dump
(``diagnostics_payload`` JSON: ``policies[].counters/decisions/recovery``,
``backing.read_stats`` including registry-observed disk read costs, and
``backing_activated``); a missing dump is recorded, never hidden and never
used to discard performance.  The runner takes no file locks: root
serializes runs under /tmp/gpubpf-revision-gpu0.lock and
/tmp/gpubpf-revision-struct-ops.lock; nested lock acquisition would
deadlock.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import run_gds_async_prefetch as campaign_base  # noqa: E402

ops = campaign_base.ops  # lmcache_primitives: server lifecycle, HTTP, teardown
perf = campaign_base.perf  # run_perf_only: barriers, deferred stop, prompts

KIND = "gds-kv-reclaim-575"
ARMS = ("stock", "native", "bpf")
RECLAIM_MODES = {"stock": None, "native": "native", "bpf": "bpf"}
RECLAIM_ENV = "LMCACHE_KV_RECLAIM"
MODE_ENV = "LMCACHE_KV_RECLAIM_MODE"
RECOMPUTE_NS_PER_TOKEN_ENV = "LMCACHE_KV_RECLAIM_RECOMPUTE_NS_PER_TOKEN"
UVM_DEVICE_ENV = "LMCACHE_KV_RECLAIM_UVM_DEVICE"
DIAG_OUT_ENV = "LMCACHE_KV_RECLAIM_DIAG_OUT"
UVM_DEVICE = "/dev/nvidia-uvm"
GDS_POLICY_MODE = "fifo"
MAX_NUM_SEQS = 2
DEFAULT_BLOCKS = 5
DEFAULT_PORT = perf.DEFAULT_PORT
DEFAULT_EXPECTED_DRIVER = "575.57.08"
DEFAULT_GDS_BUFFER_SIZE_MIB = 256
DEFAULT_KV_CACHE_MEMORY_BYTES = 384 * 1024 * 1024
DEFAULT_WARM_OUTPUT_TOKENS = 1024
DEFAULT_WARM_STAGGER_MS = 250.0
DEFAULT_WARM_CONCURRENCY = 4
DEFAULT_STORE_BARRIER_TIMEOUT_S = perf.DEFAULT_STORE_BARRIER_TIMEOUT_S
DEFAULT_STORE_BARRIER_POLL_S = perf.DEFAULT_STORE_BARRIER_POLL_S
WARM_PREFIX_LENGTHS = (1536, 1024)
ADAPTER_DIAG_WAIT_S = 10.0
ADAPTER_DIAG_POLL_S = 0.25
RAW_NAME = "raw.jsonl"
SUMMARY_NAME = "summary.json"
DIAG_NAME = "kv-reclaim-diagnostics.json"
WARM_PROGRESS_NAME = "warm-progress.json"
BOOTSTRAP = HERE / "gds-control" / "bootstrap"
GDS_CONTROL = HERE / "gds-control"
LOG_KV_PATTERNS: dict[str, tuple[re.Pattern[str], type]] = {
    "gpu_kv_cache_size_tokens": (
        re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens"), int),
    "gpu_blocks": (
        re.compile(r"#\s*GPU blocks:\s*([\d,]+)"), int),
    "max_concurrency_for_model_len": (
        re.compile(
            r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*"
            r"([\d.]+)x"), float),
    "available_kv_cache_memory_gib": (
        re.compile(r"KV cache memory:\s*([\d.]+)\s*GiB"), float),
}


def output_root(output: Path | None) -> Path:
    if output is None:
        output = HERE / "raw" / f"{KIND}-{time.strftime('%Y%m%dT%H%M%S')}"
    return output.resolve()


def rotation_orders(blocks: int) -> list[list[str]]:
    """Complete cyclic arm rotations; every arm occupies every position."""
    if blocks < 1:
        raise ValueError(f"--blocks must be at least 1, got {blocks}")
    return [list(ARMS[offset:] + ARMS[:offset])
            for offset in (block % len(ARMS) for block in range(blocks))]


def warm_arrival_order(block: int, prefix_count: int) -> list[int]:
    """Rotated warm arrival order, identical for all arms within a block."""
    if prefix_count < 1:
        raise ValueError(f"prefix_count must be at least 1, got {prefix_count}")
    return [(block + offset) % prefix_count for offset in range(prefix_count)]


def warm_specs(prefixes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Warm prompt plans built only from the existing real token arrays.

    Even prefix indexes send the full 1536-token warm prefix; odd prefix
    indexes send a 1024-token truncation of the same real warm token array
    (``warm_token_ids[:1536]`` is exactly the stored 1536-token prefix, so
    the 1024-token prompt is a pure truncation, never a re-encode). With
    1024 generated tokens each, a 1536/1024 prompt pair needs 4608 tokens;
    a 1536/1536 pair needs 5120, exceeding the observed 4096-token pool.
    """
    specs = []
    for item in prefixes:
        index = int(item["index"])
        length = WARM_PREFIX_LENGTHS[index % len(WARM_PREFIX_LENGTHS)]
        token_ids = list(item["warm_token_ids"][:length])
        if len(token_ids) != length or token_ids != item["warm_token_ids"][:length]:
            raise ValueError(
                f"prefix {index} has no {length}-token warm truncation")
        specs.append({"index": index, "warm_prefix_tokens": length,
                      "prompt_tokens": length, "warm_token_ids": token_ids})
    return specs


def cell_server_environment(arm: str, cache_dir: Path, expected_driver: str,
                            gds_buffer_size_mib: int,
                            recompute_ns_per_token: int | None,
                            diag_out: str | None) -> dict[str, str]:
    """Identical GDS demand environment per arm; reclaim arms add opt-ins.

    Every arm gets the existing GDS/cuFile demand backend in the same FIFO
    per-IO policy mode.  ``native``/``bpf`` add the agreed reclaim opt-in
    environment (including the measured recompute proxy price and the
    per-cell diagnostics exit path); ``stock`` leaves the opt-in off.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm!r}")
    env = campaign_base._BASE_SERVER_ENVIRONMENT("lmcache_disk", cache_dir, expected_driver)
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
        "LMCACHE_GDS_POLICY_MODE": GDS_POLICY_MODE,
        "LMCACHE_EXTRA_CONFIG": ops.canonical({"use_direct_io": True}),
    })
    for key in (RECLAIM_ENV, MODE_ENV, RECOMPUTE_NS_PER_TOKEN_ENV,
                UVM_DEVICE_ENV, DIAG_OUT_ENV):
        env.pop(key, None)
    if arm != "stock":
        if recompute_ns_per_token is None or int(recompute_ns_per_token) <= 0:
            raise ValueError(
                "reclaim arms require a positive measured recompute price")
        env[RECLAIM_ENV] = "1"
        env[MODE_ENV] = RECLAIM_MODES[arm]
        env[RECOMPUTE_NS_PER_TOKEN_ENV] = str(int(recompute_ns_per_token))
        if diag_out:
            env[DIAG_OUT_ENV] = str(diag_out)
        if arm == "bpf":
            env[UVM_DEVICE_ENV] = UVM_DEVICE
    return env


def warm_burst_completion(port: int, token_ids: list[int], request_id: str,
                          warm_output_tokens: int,
                          record: dict[str, Any] | None = None
                          ) -> dict[str, Any]:
    """One bounded-concurrent warm completion; raw response, no assertions.

    Same request payload semantics as the existing helpers (temperature 0,
    seed 0, streaming, logprobs 1, usage), with ``max_tokens`` set to the
    campaign-wide generation bound and identical stop behavior
    (``ignore_eos``) in every arm.  Observed status, usage, actual generated
    token count, text, TTFT, E2E, and engine request IDs are recorded; a
    missing first token yields ``ttft_ms: None`` while E2E is preserved.
    HTTP errors are raised for the caller to record verbatim.  When a stream
    error is raised and ``record`` was supplied, the observations actually
    made before the error (send time, HTTP status when a response started,
    first-token time when one arrived, observed text / token IDs / usage,
    end time) are preserved on that record; nothing missing is inferred and
    the raised error is unchanged.
    """
    payload = {"model": ops.MODEL_ID, "prompt": token_ids,
               "max_tokens": warm_output_tokens, "temperature": 0, "seed": 0,
               "ignore_eos": True, "stream": True, "logprobs": 1,
               "return_token_ids": True,
               "stream_options": {"include_usage": True}}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json",
                 "X-Request-Id": request_id},
        method="POST")
    start = time.perf_counter_ns()
    first = None
    text_parts: list[str] = []
    generated_ids: list[int] = []
    engine_request_ids: set[str] = set()
    usage: dict[str, Any] = {}
    status = None

    def preserve_partial(response_status: int | None) -> None:
        """Keep the actually observed stream tail on the request record."""
        if record is None:
            return
        end = time.perf_counter_ns()
        record["sent_ns"] = start
        record["input_tokens"] = len(token_ids)
        record["status"] = (response_status
                            if response_status is not None else status)
        record["ttft_ms"] = ((first - start) / 1e6
                             if first is not None else None)
        record["end_ns"] = end
        record["e2e_ms"] = (end - start) / 1e6
        record["usage"] = usage
        record["text"] = "".join(text_parts)
        record["generated_token_ids"] = list(generated_ids)
        record["engine_request_ids"] = sorted(engine_request_ids)

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
                    if (piece or piece_ids) and first is None:
                        first = time.perf_counter_ns()
                    text_parts.append(piece)
                    generated_ids.extend(int(value) for value in piece_ids)
    except urllib.error.HTTPError as exc:
        preserve_partial(exc.code)
        raise ops.GateError(
            f"HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
    except Exception:
        preserve_partial(None)
        raise
    end = time.perf_counter_ns()
    decode_s = (end - first) / 1e9 if first is not None else None
    completion_tokens = usage.get("completion_tokens")
    output_tokens_per_s = None
    if decode_s is not None and decode_s > 0 and completion_tokens is not None:
        output_tokens_per_s = int(completion_tokens) / decode_s
    return {
        "sent_ns": start,
        "input_tokens": len(token_ids),
        "status": status,
        "ttft_ms": (first - start) / 1e6 if first is not None else None,
        "e2e_ms": (end - start) / 1e6,
        "decode_s": decode_s,
        "usage": usage,
        "completion_tokens_observed": completion_tokens,
        "generated_tokens_observed": len(generated_ids),
        "generated_token_ids": generated_ids,
        "text": "".join(text_parts),
        "output_tokens_per_s": output_tokens_per_s,
        "engine_request_ids": sorted(engine_request_ids),
        "stop_semantics": {"ignore_eos": True,
                           "max_tokens": warm_output_tokens},
    }


def warm_task(port: int, spec: dict[str, Any], scheduled_send_ns: int,
              warm_output_tokens: int) -> dict[str, Any]:
    """Arrive at the scheduled time, then issue one warm request.

    Failures are preserved on the entry together with the partial stream
    data actually observed before the error, and never abort the burst.
    """
    index = int(spec["index"])
    entry: dict[str, Any] = {
        "phase": "warm", "prefix_index": index,
        "warm_prefix_tokens": spec["warm_prefix_tokens"],
        "request_id": f"lmc-p{index}-warm",
        "scheduled_send_ns": scheduled_send_ns, "attempted": True,
    }
    delay = scheduled_send_ns - time.perf_counter_ns()
    if delay > 0:
        time.sleep(delay / 1e9)
    try:
        entry.update(warm_burst_completion(
            port, spec["warm_token_ids"], entry["request_id"],
            warm_output_tokens, record=entry))
    except Exception as error:  # noqa: BLE001 - preserved, never fatal
        entry["error"] = f"{type(error).__name__}: {error}"
    return entry


def warm_burst_aggregates(record: dict[str, Any], warm_start_ns: int,
                          warm_end_ns: int, warm_concurrency: int,
                          warm_stagger_ms: float,
                          warm_output_tokens: int) -> dict[str, Any]:
    """Aggregate the preserved warm phase; no filtering, no gating."""
    warm = [entry for entry in record["requests"] if entry["phase"] == "warm"]
    completed = [entry for entry in warm
                 if entry.get("attempted") and "error" not in entry]
    elapsed_s = (warm_end_ns - warm_start_ns) / 1e9
    output_tokens = sum(int(entry["usage"]["completion_tokens"])
                        for entry in completed
                        if entry.get("usage", {}).get("completion_tokens")
                        is not None)
    ttft = [float(entry["ttft_ms"]) for entry in completed
            if isinstance(entry.get("ttft_ms"), (int, float))]
    generate_rates = [float(entry["output_tokens_per_s"]) for entry in completed
                      if isinstance(entry.get("output_tokens_per_s"),
                                    (int, float))]
    generated_counts = [int(entry["generated_tokens_observed"])
                        for entry in completed
                        if entry.get("generated_tokens_observed") is not None]
    arrivals = []
    for entry in warm:
        scheduled = entry.get("scheduled_send_ns")
        sent = entry.get("sent_ns")
        arrivals.append({
            "prefix_index": entry["prefix_index"],
            "scheduled_send_ns": scheduled,
            "sent_ns": sent,
            "send_lag_ns": (sent - scheduled)
                           if (sent is not None and scheduled is not None)
                           else None,
        })
    return {
        "concurrency": warm_concurrency,
        "stagger_ms": warm_stagger_ms,
        "output_tokens_bound": warm_output_tokens,
        "requests": len(completed),
        "attempts": len(warm),
        "failures": len(warm) - len(completed),
        "output_tokens": output_tokens,
        "generated_tokens_observed_sum": sum(generated_counts),
        "elapsed_s": elapsed_s,
        "requests_per_s": len(completed) / elapsed_s if elapsed_s > 0 else None,
        "output_tokens_per_s": output_tokens / elapsed_s
                              if elapsed_s > 0 else None,
        "warm_ttft_samples": len(ttft),
        "warm_ttft_values_ms": ttft,
        "warm_ttft_median_ms": statistics.median(ttft) if ttft else None,
        "warm_ttft_p95_ms": (statistics.quantiles(ttft, n=20)[-1]
                             if len(ttft) >= 2 else None),
        "warm_ttft_max_ms": max(ttft) if ttft else None,
        "output_tokens_per_s_values": generate_rates,
        "generated_tokens_observed_values": generated_counts,
        "arrivals": arrivals,
        "excludes": ["server startup", "cold population",
                     "cold-store barriers", "shutdown"],
    }


def scan_kv_pool_log(log: str) -> dict[str, Any]:
    """Derive the actual KV pool from server log lines; None when absent."""
    scan: dict[str, Any] = {}
    for name, (pattern, caster) in LOG_KV_PATTERNS.items():
        matches = [match.group(1) for match in pattern.finditer(log)]
        value = None
        if matches:
            try:
                value = caster(matches[0].replace(",", ""))
            except ValueError:
                value = None
        scan[name] = {"found": bool(matches), "value": value,
                      "values": matches[:8]}
    return scan


def scan_preemption_log(log: str, sample_cap: int = 32) -> dict[str, Any]:
    """Informational preemption-line scan; never a gate, never discarded."""
    samples: list[dict[str, Any]] = []
    count = 0
    for line_no, line in enumerate(log.splitlines(), 1):
        if "preempt" in line.lower():
            count += 1
            if len(samples) < sample_cap:
                samples.append({"line": line_no, "text": line[:400]})
    return {"count": count, "sample_cap": sample_cap, "samples": samples}


def collect_adapter_diagnostics(record: dict[str, Any]) -> None:
    """Collect the adapter's exit diagnostics; report absence, never hide it.

    For reclaim arms the configured ``LMCACHE_KV_RECLAIM_DIAG_OUT`` exit
    dump (``diagnostics_payload`` JSON) is read after teardown with a short
    bounded poll so a clean EngineCore exit can finish writing it; the wait
    is recorded.  A missing or unreadable dump is reported in
    ``adapter_diagnostics`` and never discards the cell's performance.
    """
    path_value = record.get("adapter_diagnostics_path")
    entry: dict[str, Any] = {
        "expected": bool(record.get("reclaim_enabled")),
        "path": path_value,
        "arrived": False,
        "waited_s": 0.0,
        "error": None,
        "payload": None,
    }
    if path_value:
        path = Path(path_value)
        started = time.monotonic_ns()
        while True:
            if path.is_file():
                try:
                    entry["payload"] = json.loads(path.read_text())
                    entry["arrived"] = True
                except Exception as error:  # noqa: BLE001 - reported, not hidden
                    entry["error"] = (
                        f"diagnostics parse failed: {type(error).__name__}: "
                        f"{error}")
                break
            if (time.monotonic_ns() - started) / 1e9 >= ADAPTER_DIAG_WAIT_S:
                break
            time.sleep(ADAPTER_DIAG_POLL_S)
        entry["waited_s"] = (time.monotonic_ns() - started) / 1e9
    record["adapter_diagnostics"] = entry


def adapter_policy_summary(record: dict[str, Any]) -> dict[str, Any]:
    """Small per-cell numeric summary inside cell metrics; raw payload stays
    whole in the record.  A content summary, not an integrity digest: no
    hash of anything is recorded anywhere in this runner."""
    entry = record.get("adapter_diagnostics") or {}
    payload = entry.get("payload") or {}
    policies = payload.get("policies") or []
    counters: dict[str, int] = {}
    for policy in policies:
        for key, value in (policy.get("counters") or {}).items():
            try:
                counters[key] = counters.get(key, 0) + int(value)
            except (TypeError, ValueError):
                counters[key] = counters.get(key, 0)
    return {
        "expected": entry.get("expected"),
        "arrived": entry.get("arrived"),
        "diagnostics_error": entry.get("error"),
        "enabled": payload.get("enabled"),
        "mode": payload.get("mode"),
        "recompute_ns_per_token": payload.get("recompute_ns_per_token"),
        "backing_activated": payload.get("backing_activated"),
        "policies_installed": len(policies),
        "counters_total": counters or None,
        "decision_records": sum(len(policy.get("decisions") or [])
                                for policy in policies),
        "recovery_records": sum(len(policy.get("recovery") or [])
                                for policy in policies),
        "backing_read_stats": (payload.get("backing") or {}).get("read_stats"),
    }


def run_cell(arm: str, block: int, position: int, run_dir: Path, port: int,
             model_path: Path, prefixes: list[dict[str, Any]],
             warm_order: list[int], expected_driver: str,
             store_barrier_timeout_s: float, gds_buffer_size_mib: int,
             kv_cache_memory_bytes: int, warm_output_tokens: int,
             warm_stagger_ms: float, warm_concurrency: int,
             recompute_ns_per_token: int) -> dict[str, Any]:
    """One arm, once: sequential cold population, then overlapping warm burst."""
    specs = warm_specs(prefixes)
    specs_by_index = {spec["index"]: spec for spec in specs}
    record: dict[str, Any] = {
        "schema": 1, "kind": KIND, "arm": arm, "block": block,
        "position": position, "port": port,
        "expected_driver_parameter": expected_driver,
        "reclaim_mode": RECLAIM_MODES[arm],
        "reclaim_enabled": arm != "stock",
        "gds_policy_mode": GDS_POLICY_MODE,
        "per_io_admission_identical_arms": True,
        "max_num_seqs": MAX_NUM_SEQS,
        "kv_cache_memory_bytes": kv_cache_memory_bytes,
        "gds_buffer_size_mib": gds_buffer_size_mib,
        "warm_output_tokens_bound": warm_output_tokens,
        "warm_stagger_ms": warm_stagger_ms,
        "warm_concurrency": warm_concurrency,
        "recompute_ns_per_token": recompute_ns_per_token,
        "recompute_price_semantics": (
            "measured end-to-end recompute proxy (positive median TTFT per "
            "prompt token from the one calibration phase), not pure GPU "
            "compute; identical for every arm"),
        "warm_prefix_tokens": {spec["index"]: spec["warm_prefix_tokens"]
                                for spec in specs},
        "warm_order": list(warm_order),
        "adapter_diagnostics_path": None,
        "cold_output_tokens": ops.OUTPUT_TOKENS,
        "prompt_count": len(prefixes),
        "started_ns": time.time_ns(), "ready": False, "ready_error": None,
        "requests": [], "barriers": [], "warm_phase": None,
        "cache_footprint": None, "kv_pool_log": None,
        "preemption_log": None, "adapter_diagnostics": None,
        "cleanup_errors": [], "server_returncode": None, "error": None,
    }
    log_path = run_dir / "server.log"
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=False)
    diag_path = run_dir / DIAG_NAME
    if arm != "stock":
        record["adapter_diagnostics_path"] = str(diag_path)
    record["environment"] = cell_server_environment(
        arm, cache_dir, expected_driver, gds_buffer_size_mib,
        recompute_ns_per_token, record["adapter_diagnostics_path"])
    proc = None
    log_file = None
    stopped = False
    original_environment = ops.server_environment
    original_output_tokens = ops.OUTPUT_TOKENS

    def cell_environment(_config: str, cache_dir_arg: Path, _driver: str,
                         **_options: Any) -> dict[str, str]:
        return cell_server_environment(
            arm, cache_dir_arg, expected_driver, gds_buffer_size_mib,
            recompute_ns_per_token, record["adapter_diagnostics_path"])

    ops.server_environment = cell_environment
    try:
        try:
            proc, log_file, argv, launch = ops.start_server(
                "lmcache_disk", model_path, cache_dir, port, log_path,
                expected_driver=expected_driver,
                max_num_seqs=MAX_NUM_SEQS,
                kv_cache_memory_bytes=kv_cache_memory_bytes)
        except FileExistsError:
            raise ops.GateError(
                "server log already exists; this cell would not be a first "
                "attempt")
        record["command"] = argv
        record["launch_command"] = launch
        record["worker_cpu_affinity"] = perf.worker_affinity(proc)
        try:
            ops.wait_ready(proc, port, log_path)
            record["ready"] = True
        except ops.GateError as error:
            record["ready_error"] = f"{type(error).__name__}: {error}"
        if record["ready"]:
            # Sequential cold population: short generation on the existing
            # helper; the shared output-token setting is set once for the
            # whole sequential phase and restored around it.
            ops.OUTPUT_TOKENS = 16
            for item in prefixes:
                index = item["index"]
                request_id = f"lmc-p{index}-cold"
                if proc.poll() is not None:
                    record["requests"].append({
                        "phase": "cold", "prefix_index": index,
                        "request_id": request_id, "attempted": False,
                        "reason": ("server process exited with return code "
                                   f"{proc.returncode}")})
                    continue
                entry: dict[str, Any] = {
                    "phase": "cold", "prefix_index": index,
                    "request_id": request_id, "attempted": True}
                try:
                    entry.update(campaign_base.measured_streamed_completion(
                        port, item["cold_token_ids"], request_id, entry))
                except Exception as error:  # noqa: BLE001 - preserved
                    entry["error"] = f"{type(error).__name__}: {error}"
                record["requests"].append(entry)
                engine_ids = entry.get("engine_request_ids") or []
                if len(engine_ids) == 1:
                    record["barriers"].append(perf.store_barrier(
                        "lmcache_disk", log_path, engine_ids[0],
                        int(item["expected_store_tokens"]),
                        len(item["cold_token_ids"]),
                        store_barrier_timeout_s, DEFAULT_STORE_BARRIER_POLL_S))
                else:
                    record["barriers"].append({
                        "applicable": True, "satisfied": False, "waited_s": 0.0,
                        "reason": ("no single stable engine request ID "
                                   f"observed: {engine_ids}")})
            warm_start = time.perf_counter_ns()
            stagger_ns = int(round(warm_stagger_ms * 1_000_000))
            with ThreadPoolExecutor(
                max_workers=warm_concurrency, thread_name_prefix="warm"
            ) as pool:
                futures = [
                    pool.submit(warm_task, port, specs_by_index[index],
                                warm_start + offset * stagger_ns,
                                warm_output_tokens)
                    for offset, index in enumerate(warm_order)
                ]
                future_offset = {future: offset
                                 for offset, future in enumerate(futures)}
                warm_results: dict[int, dict[str, Any]] = {}
                progress = {
                    "schema": 1, "kind": KIND, "arm": arm, "block": block,
                    "position": position,
                    "warm_scheduled": len(futures),
                    "warm_completed": 0, "warm_failures": 0,
                    "requests": [], "updated_ns": None,
                }
                pending = set(futures)
                while pending:
                    done, pending = wait(
                        pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        warm_results[future_offset[future]] = future.result()
                        progress["warm_completed"] = len(warm_results)
                        progress["warm_failures"] = sum(
                            1 for result in warm_results.values()
                            if "error" in result)
                        progress["requests"] = [
                            warm_results[offset]
                            for offset in sorted(warm_results)]
                        progress["updated_ns"] = time.time_ns()
                        ops.atomic_write_json(
                            run_dir / WARM_PROGRESS_NAME, progress)
                record["requests"].extend(
                    warm_results[offset] for offset in range(len(futures)))
            warm_end = time.perf_counter_ns()
            record["warm_phase"] = warm_burst_aggregates(
                record, warm_start, warm_end, warm_concurrency,
                warm_stagger_ms, warm_output_tokens)
        else:
            for item in prefixes:
                for phase in perf.REQUEST_LABELS:
                    record["requests"].append({
                        "phase": phase, "prefix_index": item["index"],
                        "request_id": f"lmc-p{item['index']}-{phase}",
                        "attempted": False,
                        "reason": "server never became ready"})
    except BaseException as error:  # noqa: BLE001 - keep numbers, never abort
        record["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        ops.server_environment = original_environment
        ops.OUTPUT_TOKENS = original_output_tokens
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
                record["cleanup_errors"].append(
                    "server return code unknown after bounded stop")
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
            log = log_path.read_text(errors="replace") if log_path.exists() else ""
        except OSError:
            log = ""
        record["kv_pool_log"] = scan_kv_pool_log(log)
        record["preemption_log"] = scan_preemption_log(log)
        collect_adapter_diagnostics(record)
        try:
            files = ops.disk_files(cache_dir)
            record["cache_footprint"] = {
                "files": len(files),
                "bytes": sum(path.stat().st_size for path in files),
            }
        except OSError as error:
            record["cache_footprint"] = {
                "error": f"{type(error).__name__}: {error}"}
        try:
            record["server_log_identity"] = ops.file_identity(log_path)
        except ops.GateError:
            record["server_log_identity"] = {
                "path": str(log_path.resolve()), "bytes": 0}
        record["finished_ns"] = time.time_ns()
        ops.atomic_write_json(run_dir / "result.json", record)
    return record


def cell_metrics(record: dict[str, Any]) -> dict[str, Any]:
    """Per-cell metrics from the preserved warm phase plus diagnostics."""
    warm = record.get("warm_phase") or {}
    kv_pool = record.get("kv_pool_log") or {}
    return {
        "arm": record.get("arm"),
        "reclaim_enabled": record.get("reclaim_enabled"),
        "reclaim_mode": record.get("reclaim_mode"),
        "warm_requests": warm.get("requests"),
        "warm_attempts": warm.get("attempts"),
        "warm_failures": warm.get("failures"),
        "warm_output_tokens": warm.get("output_tokens"),
        "warm_elapsed_s": warm.get("elapsed_s"),
        "warm_requests_per_s": warm.get("requests_per_s"),
        "warm_output_tokens_per_s": warm.get("output_tokens_per_s"),
        "warm_ttft_median_ms": warm.get("warm_ttft_median_ms"),
        "warm_ttft_p95_ms": warm.get("warm_ttft_p95_ms"),
        "warm_ttft_max_ms": warm.get("warm_ttft_max_ms"),
        "barriers_satisfied": sum(
            1 for barrier in (record.get("barriers") or [])
            if barrier.get("satisfied")),
        "barriers_recorded": len(record.get("barriers") or []),
        "kv_pool_tokens": (kv_pool.get("gpu_kv_cache_size_tokens") or {})
                          .get("value"),
        "preemption_log_count": (record.get("preemption_log") or {})
                                .get("count"),
        "adapter": adapter_policy_summary(record),
        "ready": record.get("ready", False),
        "server_returncode": record.get("server_returncode"),
        "error": record.get("error"),
    }


ADAPTER_COUNTER_KEYS = (
    "seam_invocations", "override_picks", "stock_kept",
    "forced_zero_lookups", "route_consumed_full_recompute",
    "route_consumed_disk_prefix", "read_pricing_missing_decisions",
    "coverage_unknown_candidates", "disk_route_recompute_fallback",
)


def median_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-arm aggregation over every attempted cell; nothing is dropped.

    Failed cells simply contribute ``None`` entries to the per-cell lists so
    the medians reflect only real measurements while all attempts stay
    visible; no desirable-ratio tuning and no bad-cell filtering exists.
    """
    metric_names = ("warm_requests", "warm_failures", "warm_output_tokens",
                    "warm_elapsed_s", "warm_requests_per_s",
                    "warm_output_tokens_per_s", "warm_ttft_median_ms",
                    "warm_ttft_p95_ms", "warm_ttft_max_ms",
                    "kv_pool_tokens", "preemption_log_count")
    median_names = ("warm_requests_per_s", "warm_output_tokens_per_s",
                    "warm_ttft_median_ms")
    per_arm: dict[str, Any] = {}
    for arm in ARMS:
        rows = [cell["metrics"] for cell in cells if cell.get("arm") == arm]
        medians = {}
        for name in median_names:
            values = [float(row[name]) for row in rows
                      if isinstance(row.get(name), (int, float))]
            medians[name] = statistics.median(values) if values else None
        adapter_rows = [row.get("adapter") or {} for row in rows]
        counters: dict[str, list[Any]] = {}
        for key in ADAPTER_COUNTER_KEYS:
            counters[key] = [
                (row.get("counters_total") or {}).get(key) for row in adapter_rows]
        per_arm[arm] = {
            "reclaim_enabled": arm != "stock",
            "reclaim_mode": RECLAIM_MODES[arm],
            "cells_attempted": len(rows),
            "cells_measured": sum(
                row.get("warm_requests_per_s") is not None for row in rows),
            "diagnostics_expected": [row.get("expected") for row in adapter_rows],
            "diagnostics_arrived": [row.get("arrived") for row in adapter_rows],
            "backing_activated": [row.get("backing_activated")
                                  for row in adapter_rows],
            "backing_read_stats": [row.get("backing_read_stats")
                                   for row in adapter_rows],
            "adapter_counters": counters,
            "decision_records": [row.get("decision_records")
                                 for row in adapter_rows],
            "recovery_records": [row.get("recovery_records")
                                 for row in adapter_rows],
            "medians": medians,
            "per_cell_values": {
                name: [row.get(name) for row in rows] for name in metric_names},
            "server_returncodes": [row.get("server_returncode")
                                   for row in rows],
            "errors": [row.get("error") for row in rows],
        }
    return {"kind": KIND, "cells_attempted": len(cells), "per_arm": per_arm}


def resume_scan(
    root: Path, args: argparse.Namespace, prefixes: list[dict[str, Any]],
    price: int, orders: list[list[str]],
) -> tuple[dict[tuple[int, int, str], tuple[Path, dict[str, Any]]],
           dict[tuple[int, int, str], tuple[Path, dict[str, Any]]],
           list[dict[str, Any]]]:
    """Classify every planned cell directory under an existing --resume root.

    A nonempty directory whose ``result.json`` parses to a record matching
    ``kind``/``schema``, the planned ``arm``/``block``/``position``, and the
    ordinary settings fields (expected driver, capacity, warm burst knobs,
    prompt plan, measured recompute price; JSON ``warm_prefix_tokens`` keys
    are strings) via plain value equality, never hashes, is adopted
    verbatim; failed full records are adopted the same way and never
    rerun, and no adopted field is rewritten.  Only a recognizable record
    carrying the matching campaign identity and an actual ``error`` marker
    whose full knob fields are merely absent (top-level exception results)
    is a completed failed attempt: preserved verbatim, never rerun, and
    never described as a compatible measured cell.  Any other sparse or
    wrong-valued record is incompatible, left untouched, and reported
    unfinished, as is a nonempty directory without a usable
    ``result.json``; no age or timeout assumption is made.  Absent or
    empty directories remain candidates for a fresh run.
    """
    prefix_plan = {str(spec["index"]): spec["warm_prefix_tokens"]
                   for spec in warm_specs(prefixes)}
    adopted: dict[tuple[int, int, str], tuple[Path, dict[str, Any]]] = {}
    completed: dict[tuple[int, int, str], tuple[Path, dict[str, Any]]] = {}
    unfinished: list[dict[str, Any]] = []
    for block, order in enumerate(orders):
        for position, arm in enumerate(order):
            planned = {"arm": arm, "block": block, "position": position}
            run_dir = root / f"block-{block:02d}" / f"position-{position}-{arm}"
            if not run_dir.is_dir() or not any(run_dir.iterdir()):
                continue
            result_path = run_dir / "result.json"
            if not result_path.is_file():
                unfinished.append({
                    **planned, "run_dir": str(run_dir),
                    "reason": ("nonempty cell directory without result.json; "
                               "left untouched and not rerun"),
                    "contents": sorted(child.name
                                       for child in run_dir.iterdir())[:16],
                })
                continue
            try:
                loaded = json.loads(result_path.read_text())
                if not isinstance(loaded, dict):
                    raise ValueError("result.json is not a JSON object")
            except Exception as error:  # noqa: BLE001 - real failure, kept
                unfinished.append({
                    **planned, "run_dir": str(run_dir),
                    "result_json": str(result_path),
                    "reason": (f"result.json unreadable: "
                               f"{type(error).__name__}: {error}"),
                })
                continue
            expected: dict[str, Any] = {
                **planned, "kind": KIND, "schema": 1,
                "expected_driver_parameter": args.expected_driver,
                "max_num_seqs": MAX_NUM_SEQS,
                "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
                "gds_buffer_size_mib": args.gds_buffer_size_mib,
                "warm_output_tokens_bound": args.warm_output_tokens,
                "warm_stagger_ms": args.warm_stagger_ms,
                "warm_concurrency": args.warm_concurrency,
                "gds_policy_mode": GDS_POLICY_MODE,
                "prompt_count": len(prefixes),
                "warm_prefix_tokens": prefix_plan,
                "warm_order": list(warm_arrival_order(block, len(prefixes))),
                "recompute_ns_per_token": int(price),
            }
            observed = dict(loaded)
            observed.setdefault("schema", 1)
            mismatches = [name for name, value in expected.items()
                          if observed.get(name) != value]
            if mismatches:
                present_wrong = [name for name in mismatches
                                 if name in loaded]
                identity_ok = all(
                    loaded.get(name) == value
                    for name, value in (("kind", KIND),
                                        ("arm", planned["arm"]),
                                        ("block", planned["block"]),
                                        ("position", planned["position"])))
                sparse_failure = (identity_ok and not present_wrong
                                  and bool(loaded.get("error")))
                if sparse_failure:
                    completed[(block, position, arm)] = (run_dir, loaded)
                    continue
                missing = [name for name in ("kind", "arm", "block",
                                             "position")
                           if name not in loaded]
                if not loaded.get("error"):
                    missing.append("error")
                unfinished.append({
                    **planned, "run_dir": str(run_dir),
                    "result_json": str(result_path),
                    "reason": "incompatible record not adopted",
                    "incompatible_fields": present_wrong or missing,
                })
                continue
            adopted[(block, position, arm)] = (run_dir, loaded)
    return adopted, completed, unfinished


def resume_import_raw(
    root: Path,
    reused: dict[tuple[int, int, str], tuple[Path, dict[str, Any]]],
    failed: dict[tuple[int, int, str], tuple[Path, dict[str, Any]]],
) -> int:
    """Append reused and preserved records to raw.jsonl once each.

    Existing lines stay verbatim; one line per record, in planned
    (block, position, arm) order; a (block, position, arm) already present
    in raw.jsonl is not appended again.  Same JSONL writer shape as the
    per-cell append in run_campaign.
    """
    records = {**reused, **failed}
    raw_path = root / RAW_NAME
    seen: set[tuple[Any, Any, Any]] = set()
    if raw_path.is_file():
        for line in raw_path.read_text(errors="replace").splitlines():
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                seen.add((obj.get("block"), obj.get("position"),
                          obj.get("arm")))
    imported = 0
    with raw_path.open("a", encoding="utf-8") as raw_file:
        for key, (_run_dir, record) in sorted(records.items()):
            if key in seen:
                continue
            raw_file.write(json.dumps(record, ensure_ascii=False,
                                      separators=(",", ":")) + "\n")
            imported += 1
        raw_file.flush()
        os.fsync(raw_file.fileno())
    return imported


def resume_calibration_source(
    root: Path, previous_campaign: dict[str, Any] | None,
) -> Path | dict[str, Any] | None:
    """Pick an already-measured calibration for --resume without rerunning.

    Preference: ``<root>/calibration/calibration.json``; then the source
    path recorded as ``reused_from`` by a previous campaign; then the
    embedded calibration record of a previous campaign when it carries a
    positive measured price.  Returns a source path, a reusable record, or
    None when nothing has been measured yet.  An explicit
    ``--calibration-result`` always wins and bypasses this helper.
    """
    candidates = [root / "calibration" / "calibration.json"]
    previous = (previous_campaign or {}).get("calibration")
    if isinstance(previous, dict) and isinstance(previous.get("reused_from"),
                                                str):
        candidates.append(Path(previous["reused_from"]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if isinstance(previous, dict) and previous.get("recompute_ns_per_token") \
            is not None:
        try:
            price = int(previous["recompute_ns_per_token"])
        except (TypeError, ValueError):
            return None
        if price > 0:
            return dict(previous)
    return None


def calibration_phase(campaign: dict[str, Any], root: Path, args: Any,
                      model_path: Path, prefixes: list[dict[str, Any]],
                      reused_record: dict[str, Any] | None = None
                      ) -> int | None:
    """The one real recompute calibration, owned by kv_reclaim_calibration.

    This runner does not implement any calibration lifecycle: it imports the
    agreed helper, supplies the actual warm token arrays (1536/1024) it
    prepared, and receives ``{recompute_ns_per_token, error, ready,
    requests, server_returncode}`` with calibration.json/server.log written
    under ``root/calibration``.  The returned record is embedded verbatim in
    the campaign.  A missing helper, a failed run, or a non-positive price
    aborts the campaign before any cell with the real failure retained; no
    invented or dummy price is ever served to a cell, and nothing changes
    the environment of an already-launched cell server.  ``reused_record``
    embeds an already-measured calibration record verbatim (``rerun``
    marked false) without any new run.
    """
    source = getattr(args, "calibration_result", None)
    result: dict[str, Any] | None = None
    if reused_record is not None:
        result = dict(reused_record)
        result["rerun"] = False
        campaign["calibration"] = result
    elif source:
        # The one real calibration already ran; reuse its exact raw result
        # (this file) instead of calling the helper again.  The source path
        # travels with the campaign record.
        try:
            loaded = json.loads(Path(source).read_text())
            if not isinstance(loaded, dict):
                raise ValueError("calibration result is not a JSON object")
            result = dict(loaded)
            result["reused_from"] = str(Path(source).resolve())
            result["rerun"] = False
            campaign["calibration"] = result
        except Exception as error:  # noqa: BLE001 - real failure, retained
            campaign["calibration"] = {
                "phase": "recompute_calibration",
                "recompute_ns_per_token": None, "ready": None, "requests": [],
                "server_returncode": None,
                "error": (f"calibration result reuse from {source} failed: "
                          f"{type(error).__name__}: {error}"),
            }
    else:
        token_arrays = [spec["warm_token_ids"] for spec in warm_specs(prefixes)]
        try:
            from kv_reclaim_calibration import run_calibration
        except Exception as error:  # noqa: BLE001 - real failure, retained
            campaign["calibration"] = {
                "phase": "recompute_calibration",
                "run_dir": str(root / "calibration"),
                "recompute_ns_per_token": None, "ready": None, "requests": [],
                "server_returncode": None,
                "error": (f"calibration helper import failed: "
                          f"{type(error).__name__}: {error}"),
            }
        else:
            try:
                result = run_calibration(
                    run_dir=root / "calibration", model_path=model_path,
                    port=args.port, token_arrays=token_arrays,
                    expected_driver=args.expected_driver,
                    kv_cache_memory_bytes=args.kv_cache_memory_bytes)
            except Exception as error:  # noqa: BLE001 - real failure
                campaign["calibration"] = {
                    "phase": "recompute_calibration",
                    "run_dir": str(root / "calibration"),
                    "recompute_ns_per_token": None, "ready": None,
                    "requests": [], "server_returncode": None,
                    "error": (f"calibration helper failed: "
                              f"{type(error).__name__}: {error}"),
                }
            else:
                campaign["calibration"] = result
    if not isinstance(result, dict):
        return None
    price = result.get("recompute_ns_per_token")
    if price is None:
        return None
    try:
        price = int(price)
    except (TypeError, ValueError):
        return None
    return price if price > 0 else None


def run_campaign(args: argparse.Namespace) -> int:
    root = output_root(args.output)
    orders = rotation_orders(args.blocks)
    prefixes = perf.load_fixed_prompts()["prefixes"]
    specs = warm_specs(prefixes)
    campaign: dict[str, Any] = {
        "kind": KIND,
        "timestamp": time.strftime("%Y%m%dT%H%M%S"),
        "params": {
            "blocks": args.blocks,
            "arms": list(ARMS),
            "reclaim_modes": RECLAIM_MODES,
            "port": args.port,
            "expected_driver": args.expected_driver,
            "store_barrier_timeout_s": args.store_barrier_timeout_s,
            "gds_buffer_size_mib": args.gds_buffer_size_mib,
            "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
            "gds_policy_mode": GDS_POLICY_MODE,
            "max_num_seqs": MAX_NUM_SEQS,
            "warm_output_tokens_bound": args.warm_output_tokens,
            "warm_concurrency": args.warm_concurrency,
            "warm_stagger_ms": args.warm_stagger_ms,
            "cold_output_tokens": ops.OUTPUT_TOKENS,
            "warm_prefix_tokens": {spec["index"]: spec["warm_prefix_tokens"]
                                   for spec in specs},
            "attempts_per_cell": 1,
            "retry": False,
            "campaign_wall_clock_timeout": None,
            "model_path": None,
        },
        "block_orders": orders,
        "warm_orders": {},
        "calibration": None,
        "cells": [], "summary": None,
    }
    stop = perf.DeferredStop()
    previous = {sig: signal.signal(sig, stop.request)
                for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        if args.resume:
            if not root.is_dir():
                raise ops.GateError(
                    f"--resume requires an existing output directory: {root}")
        else:
            root.mkdir(parents=True, exist_ok=False)
        model_path = Path(ops.resolve_model(local_only=True))
        campaign["params"]["model_path"] = str(model_path)
        campaign["warm_orders"] = {
            str(block): warm_arrival_order(block, len(prefixes))
            for block in range(args.blocks)}
        previous_campaign: dict[str, Any] | None = None
        if args.resume:
            campaign_path = root / "campaign.json"
            if campaign_path.is_file():
                try:
                    loaded_campaign = json.loads(campaign_path.read_text())
                except Exception as error:  # real failure, file untouched
                    raise ops.GateError(
                        "existing campaign.json is unreadable; resume "
                        f"leaves it untouched: {type(error).__name__}: "
                        f"{error}") from error
                if (not isinstance(loaded_campaign, dict)
                        or loaded_campaign.get("kind") != KIND):
                    raise ops.GateError(
                        "existing campaign.json is not a matching campaign "
                        "record; resume leaves it untouched")
                previous_campaign = loaded_campaign
                campaign["timestamp"] = (previous_campaign.get("timestamp")
                                         or campaign["timestamp"])
        if not args.resume:
            ops.atomic_write_json(root / "campaign.json", campaign)
        reused = None
        if args.resume and not args.calibration_result:
            auto = resume_calibration_source(root, previous_campaign)
            if isinstance(auto, Path):
                args.calibration_result = auto
            else:
                reused = auto
        price = calibration_phase(campaign, root, args, model_path, prefixes,
                                  reused_record=reused)
        if not args.resume:
            ops.atomic_write_json(root / "campaign.json", campaign)
        if price is None:
            print("NOT STARTED: recompute calibration did not yield a "
                  "positive measured price; no cells ran, raw retained at "
                  f"{root / 'calibration'}", file=sys.stderr, flush=True)
            return 2
        if stop.signum is not None:
            campaign["stopped_early"] = (
                f"deferred {stop.signum_name()} request; stopping after "
                "calibration")
            if not args.resume:
                ops.atomic_write_json(root / "campaign.json", campaign)
            return 3
        resumed: dict[tuple[int, int, str], tuple[Path, dict[str, Any]]] = {}
        completed: dict[tuple[int, int, str], tuple[Path, dict[str, Any]]] = {}
        unfinished: list[dict[str, Any]] = []
        unfinished_keys: set[tuple[int, int, str]] = set()
        imports = 0
        if args.resume:
            resumed, completed, unfinished = resume_scan(
                root, args, prefixes, price, orders)
            unfinished_keys = {(entry["block"], entry["position"],
                                entry["arm"]) for entry in unfinished}
            imports = resume_import_raw(root, resumed, completed)
            campaign["resume"] = {
                "adopted": [{"block": block, "position": position,
                             "arm": arm, "run_dir": str(run_dir),
                             "result_json": str(run_dir / "result.json")}
                            for (block, position, arm), (run_dir, _record)
                            in sorted(resumed.items())],
                "completed_failures": [
                    {"block": block, "position": position, "arm": arm,
                     "run_dir": str(run_dir),
                     "result_json": str(run_dir / "result.json"),
                     "error": record.get("error")}
                    for (block, position, arm), (run_dir, record)
                    in sorted(completed.items())],
                "unfinished": unfinished,
                "raw_imported_records": imports,
            }
            if previous_campaign is not None:
                campaign["resume"]["previous_params"] = (
                    previous_campaign.get("params"))
            ops.atomic_write_json(root / "campaign.json", campaign)
        for block, order in enumerate(orders):
            warm_order = warm_arrival_order(block, len(prefixes))
            for position, arm in enumerate(order):
                run_dir = root / f"block-{block:02d}" / f"position-{position}-{arm}"
                if (block, position, arm) in unfinished_keys:
                    print(f"block={block} position={position} arm={arm} "
                          "resume=unfinished; cell left untouched",
                          flush=True)
                    continue
                run_dir.parent.mkdir(parents=True, exist_ok=True)
                adoption = resumed.get((block, position, arm))
                failed_attempt = completed.get((block, position, arm))
                if adoption is not None:
                    print(f"block={block} position={position} arm={arm} "
                          "resume=reused complete result.json", flush=True)
                    record = adoption[1]
                elif failed_attempt is not None:
                    print(f"block={block} position={position} arm={arm} "
                          "resume=preserved completed failed attempt; "
                          "not rerun", flush=True)
                    record = failed_attempt[1]
                else:
                    print(f"block={block} position={position} arm={arm}",
                          flush=True)
                    try:
                        record = run_cell(
                            arm=arm, block=block, position=position,
                            run_dir=run_dir, port=args.port,
                            model_path=model_path, prefixes=prefixes,
                            warm_order=warm_order,
                            expected_driver=args.expected_driver,
                            store_barrier_timeout_s=args.store_barrier_timeout_s,
                            gds_buffer_size_mib=args.gds_buffer_size_mib,
                            kv_cache_memory_bytes=args.kv_cache_memory_bytes,
                            warm_output_tokens=args.warm_output_tokens,
                            warm_stagger_ms=args.warm_stagger_ms,
                            warm_concurrency=args.warm_concurrency,
                            recompute_ns_per_token=price)
                    except Exception as error:  # noqa: BLE001 - preserved
                        record = {
                            "schema": 1, "kind": KIND, "arm": arm,
                            "block": block, "position": position,
                            "port": args.port,
                            "ready": False, "requests": [], "barriers": [],
                            "warm_phase": None, "cleanup_errors": [],
                            "server_returncode": None,
                            "error": f"{type(error).__name__}: {error}",
                        }
                        run_dir.mkdir(parents=True, exist_ok=True)
                        ops.atomic_write_json(run_dir / "result.json", record)
                    with (root / RAW_NAME).open("a",
                                                encoding="utf-8") as raw_file:
                        raw_file.write(json.dumps(record, ensure_ascii=False,
                                                  separators=(",", ":"))
                                      + "\n")
                        raw_file.flush()
                        os.fsync(raw_file.fileno())
                campaign["cells"].append({
                    "block": block, "position": position, "arm": arm,
                    "run_dir": str(run_dir),
                    "metrics": cell_metrics(record)})
                campaign["summary"] = median_summary(campaign["cells"])
                ops.atomic_write_json(root / "campaign.json", campaign)
                ops.atomic_write_json(root / SUMMARY_NAME, campaign["summary"])
                if stop.signum is not None:
                    campaign["stopped_early"] = (
                        f"deferred {stop.signum_name()} request; stopping "
                        "between cells")
                    ops.atomic_write_json(root / "campaign.json", campaign)
                    return 3
        expected_cells = args.blocks * len(ARMS)
        complete = (len(campaign["cells"]) == expected_cells
                    and all(cell["metrics"]["warm_requests_per_s"] is not None
                            for cell in campaign["cells"]))
        print(json.dumps(campaign["summary"], ensure_ascii=False, indent=2),
              flush=True)
        return 0 if complete else 2
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def dry_run_plan(args: argparse.Namespace) -> dict[str, Any]:
    prefixes = perf.load_fixed_prompts()["prefixes"]
    specs = warm_specs(prefixes)
    return {
        "dry_run": True, "kind": KIND,
        "metric": ("warm burst TTFT/E2E/output rate and actual generated "
                   "token counts per arm, plus registry-observed disk read "
                   "stats and reclaim decisions from adapter exit "
                   "diagnostics"),
        "arms": list(ARMS),
        "reclaim_modes": RECLAIM_MODES,
        "blocks": args.blocks,
        "block_orders": rotation_orders(args.blocks),
        "warm_orders": {str(block): warm_arrival_order(block, len(prefixes))
                        for block in range(args.blocks)},
        "warm_prefix_tokens": {spec["index"]: spec["warm_prefix_tokens"]
                               for spec in specs},
        "warm_output_tokens_bound": args.warm_output_tokens,
        "warm_concurrency": args.warm_concurrency,
        "warm_stagger_ms": args.warm_stagger_ms,
        "cold_output_tokens": ops.OUTPUT_TOKENS,
        "expected_driver_parameter": args.expected_driver,
        "port": args.port,
        "capacity": {"kv_cache_memory_bytes": args.kv_cache_memory_bytes,
                     "max_num_seqs": MAX_NUM_SEQS,
                     "gds_buffer_size_mib": args.gds_buffer_size_mib,
                     "gds_policy_mode": GDS_POLICY_MODE,
                     "kv_pool_tokens_source": "server log scan, not assumed"},
        "calibration": {
            "owner": "kv_reclaim_calibration.run_calibration",
            "reuse_result_path": (str(args.calibration_result)
                                  if getattr(args, "calibration_result", None)
                                  else None),
            "abort_when": "no positive measured price; no dummy fallback",
        },
        "adapter_env": {
            "stock": {},
            "native": {RECLAIM_ENV: "1", MODE_ENV: "native",
                       RECOMPUTE_NS_PER_TOKEN_ENV: "<measured>",
                       DIAG_OUT_ENV: "<run_dir>/kv-reclaim-diagnostics.json"},
            "bpf": {RECLAIM_ENV: "1", MODE_ENV: "bpf",
                    RECOMPUTE_NS_PER_TOKEN_ENV: "<measured>",
                    UVM_DEVICE_ENV: UVM_DEVICE,
                    DIAG_OUT_ENV: "<run_dir>/kv-reclaim-diagnostics.json"},
            "shared_by_all_arms": {
                "LMCACHE_USE_GDS": "True",
                "LMCACHE_GDS_BACKEND": "cufile",
                "LMCACHE_GDS_POLICY_MODE": GDS_POLICY_MODE,
                "LMCACHE_GDS_BUFFER_SIZE": str(args.gds_buffer_size_mib),
            },
        },
        "diagnostics_contract": ("adapter exit dump collected per reclaim "
                                 "cell; missing dumps are recorded and never "
                                 "discard performance"),
        "reuse": ["run_gds_async_prefetch.measured_streamed_completion (cold)",
                  "run_perf_only: DeferredStop, store_barrier, "
                  "worker_affinity, load_fixed_prompts, defaults",
                  "lmcache_primitives: start_server, wait_ready, "
                  "stop_owned_server, wait_gpu_idle, server_environment, "
                  "resolve_model, file_identity, atomic_write_json, "
                  "disk_files, canonical, GateError"],
        "not_claimed": ["automatic offload complete",
                        "physical HBM release from fixed pool block return",
                        "hardware NVMe-GPU P2P (compatibility transport only)"],
        "preserved": ["every cold/warm request attempt with raw response or "
                      "error", "barrier outcomes (non-gating)",
                      "server return codes and cleanup errors",
                      "missing adapter diagnostics as records",
                      "calibration raw and errors as baseline evidence"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Three-arm LMCache disk-aware KV reclaim campaign")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true",
                        help=("resume an existing --output directory: reuse "
                              "complete compatible per-cell result.json "
                              "records (failed cells included, never rerun), "
                              "leave nonempty cells without result.json "
                              "untouched as unfinished, and run only missing "
                              "cells in the planned rotated order"))
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--expected-driver", default=DEFAULT_EXPECTED_DRIVER)
    parser.add_argument("--store-barrier-timeout-s", type=float,
                        default=DEFAULT_STORE_BARRIER_TIMEOUT_S)
    parser.add_argument("--gds-buffer-size-mib", type=int,
                        default=DEFAULT_GDS_BUFFER_SIZE_MIB)
    parser.add_argument("--kv-cache-memory-bytes", type=int,
                        default=DEFAULT_KV_CACHE_MEMORY_BYTES)
    parser.add_argument("--warm-output-tokens", type=int,
                        default=DEFAULT_WARM_OUTPUT_TOKENS)
    parser.add_argument("--warm-concurrency", type=int,
                        default=DEFAULT_WARM_CONCURRENCY)
    parser.add_argument("--warm-stagger-ms", type=float,
                        default=DEFAULT_WARM_STAGGER_MS)
    parser.add_argument("--calibration-result", type=Path, default=None,
                        help=("reuse an existing calibration.json (an "
                              "explicit path always wins); omitted means an "
                              "existing measured calibration is reused when "
                              "available (resume), otherwise "
                              "kv_reclaim_calibration.run_calibration runs "
                              "into <root>/calibration"))
    parser.add_argument("--dry-run", action="store_true",
                        help="print the fixed plan without touching GPU or output state")
    args = parser.parse_args(argv)
    if args.blocks < 1:
        parser.error("--blocks must be at least 1")
    if args.gds_buffer_size_mib < 1:
        parser.error("--gds-buffer-size-mib must be at least 1")
    if args.kv_cache_memory_bytes < 1:
        parser.error("--kv-cache-memory-bytes must be at least 1")
    if args.warm_output_tokens < 1:
        parser.error("--warm-output-tokens must be at least 1")
    if args.warm_concurrency < 1:
        parser.error("--warm-concurrency must be at least 1")
    if args.warm_stagger_ms < 0:
        parser.error("--warm-stagger-ms must be non-negative")
    if args.resume and args.output is None:
        parser.error("--resume requires an existing --output directory")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps(dry_run_plan(args), ensure_ascii=False, indent=2),
              flush=True)
        return 0
    try:
        return run_campaign(args)
    except (ops.GateError, ValueError, OSError) as error:
        print(f"NOT STARTED: {type(error).__name__}: {error}",
              file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
