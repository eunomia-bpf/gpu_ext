#!/usr/bin/env python3
"""Official POD/FlashAttention operator cells; no GPU work at import time.

The coordinator owns the GPU lease and, for pod_bpf, starts the private loader
before this process with the existing bpftime agent preloaded. Each invocation
runs one arm over the fixed shapes (or the one-shape real preflight).
"""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import statistics
import struct
import sys
import time

ROOT = Path(__file__).resolve().parent
ARMS = ("official_serial", "official_streams", "pod_inline", "pod_cuda", "pod_bpf")
MODEL_HEADS = {"llama-3-8b": 8, "yi-6b": 4}
META_NAMES = ("nsmid", "grid_ctas", "prefill_blocks", "decode_blocks", "factor_p",
              "factor_d", "smem_bytes", "threads", "fused_op", "mode", "trace")
CTX_NAMES = ("counters", "abi_version", "nsmid", "smid", "prefill_slots", "decode_slots",
             "proportional", "grid_ctas", "out_op", "out_cta", "status", "engine", "ticket",
             "first_op", "first_claim", "fallback_claim", "reserved")
UNSET = 0xffffffff


def audit_decisions(meta, counters, contexts, engine):
    """Recompute the original rule/claims, not a producer success flag."""
    p = (meta["prefill_blocks"] + meta["factor_p"] - 1) // meta["factor_p"]
    d = (meta["decode_blocks"] + meta["factor_d"] - 1) // meta["factor_d"]
    n = meta["nsmid"]
    if len(counters) != n + 2 or len(contexts) != p + d or meta["grid_ctas"] != p + d:
        raise ValueError("wrong counter/context/work extent")
    tickets, claims, work = defaultdict(list), ([], []), ([], [])
    address = contexts[0]["counters"]
    if not address:
        raise ValueError("null counter address")
    for c in contexts:
        if (c["abi_version"] != 1 or c["engine"] != engine or c["status"] != 1 or
            c["nsmid"] != n or not 0 <= c["smid"] < n or c["counters"] != address or
            c["prefill_slots"] != p or c["decode_slots"] != d or
            c["grid_ctas"] != p + d or c["proportional"] != (meta["fused_op"] & 1)):
            raise ValueError("invalid actual device context or wrong execution engine")
        ticket = c["ticket"]
        if c["proportional"]:
            expected_op = (int(ticket % (d // p + 1) > 0) if p <= d else
                           int(ticket % (p // d + 1) >= p // d))
        else:
            expected_op = ticket % 2
        if c["first_op"] != expected_op:
            raise ValueError("device SM ticket does not implement original POD rule")
        tickets[c["smid"]].append(ticket)
        claims[expected_op].append(c["first_claim"])
        exhausted = c["first_claim"] >= (p, d)[expected_op]
        out_op = 1 - expected_op if exhausted else expected_op
        out_cta = c["fallback_claim"] if exhausted else c["first_claim"]
        if not exhausted and c["fallback_claim"] != UNSET:
            raise ValueError("spurious fallback")
        if exhausted:
            claims[out_op].append(c["fallback_claim"])
        if c["out_op"] != out_op or c["out_cta"] != out_cta or not 0 <= out_cta < (p, d)[out_op]:
            raise ValueError("actual attention decision is not the claimed valid task")
        work[out_op].append(out_cta)
    for sm in range(n):
        if sorted(tickets[sm]) != list(range(counters[sm])):
            raise ValueError("missing/duplicate actual SM ticket")
    for op, slots in enumerate((p, d)):
        if sorted(claims[op]) != list(range(counters[n + op])):
            raise ValueError("missing/duplicate atomic global claim")
        if sorted(work[op]) != list(range(slots)):
            raise ValueError("not exactly-once attention work")
    return {"physical_ctas": p + d, "prefill_slots": p, "decode_slots": d,
            "prefill_logical_blocks": meta["prefill_blocks"],
            "decode_logical_blocks": meta["decode_blocks"],
            "fallbacks": sum(c["fallback_claim"] != UNSET for c in contexts),
            "observed_sm_ids": sorted(sm for sm, ts in tickets.items() if ts),
            "engine": engine}


def shape_order(block, preflight):
    shapes = [(name, bs) for name in MODEL_HEADS for bs in (32, 64, 96, 128, 192)]
    if preflight:
        return [("llama-3-8b", 32)]
    random.Random(20260903 + block).shuffle(shapes)
    return shapes


def audit_bridge(before, after, expected_launches, smem_bytes, mode):
    if after["launches"] - before["launches"] != expected_launches:
        raise ValueError("not every actual POD launch entered the shared-memory bridge")
    if mode == "cuda" and after["runtime_redirects"] - before["runtime_redirects"] != expected_launches:
        raise ValueError("CUDA adapter control did not use the common driver bridge")
    if (after["prepared_functions"] < 1 or after["requested_dynamic_bytes"] != smem_bytes or
        after["verified_dynamic_bytes"] < smem_bytes or
        after["static_shared_bytes"] + after["verified_dynamic_bytes"] > after["device_optin_bytes"]):
        raise ValueError("actual launch shared-memory opt-in was not verified")
    return {"before": before, "after": after, "expected_launches": expected_launches}


def run_arm(args, result, output_file):
    # Nothing below is a CPU test: the coordinator must own an exclusive GPU slot.
    sys.path[:0] = [str(ROOT / "build/python"), str(ROOT / "deps/vattention/pod_attn")]
    import torch
    import fused_attn
    from pod_attn.flash_attn_interface import flash_attn_with_kvcache
    from pod_attn.fused_attn_interface import true_fused_attn_with_kvcache

    bridge_stats = None
    if args.arm in ("pod_cuda", "pod_bpf"):
        mode = args.arm.removeprefix("pod_")
        if (os.environ.get("POD_LAUNCH_BRIDGE") != mode or
            "libpod_launch_bridge.so" not in os.environ.get("LD_PRELOAD", "")):
            raise RuntimeError("both adapter arms require the scoped checked launch bridge")
        import ctypes
        class BridgeStats(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                ("launches", "prepared_functions", "runtime_redirects", "requested_dynamic_bytes",
                 "verified_dynamic_bytes", "static_shared_bytes", "device_optin_bytes")]
        accessor = ctypes.CDLL(None).pod_bridge_get_stats
        accessor.argtypes = [ctypes.POINTER(BridgeStats), ctypes.c_uint64]
        accessor.restype = ctypes.c_int
        def bridge_stats():
            stats = BridgeStats()
            if accessor(ctypes.byref(stats), ctypes.sizeof(stats)):
                raise RuntimeError("launch bridge statistics ABI mismatch")
            return {name: getattr(stats, name) for name, _ in stats._fields_}
    elif os.environ.get("POD_LAUNCH_BRIDGE", "off") != "off":
        raise RuntimeError("official/inline arms must retain their original CUDA launch path")

    if args.arm == "pod_bpf":
        if "libbpftime-agent.so" not in os.environ.get("LD_PRELOAD", ""):
            raise RuntimeError("device-BPF arm requires the existing bpftime CUDA agent")
        if not os.environ.get("BPFTIME_GLOBAL_SHM_NAME", "").startswith("pod_attention_"):
            raise RuntimeError("device-BPF arm requires the owned private loader")
    if tuple(torch.cuda.get_device_capability()) != (12, 0):
        raise RuntimeError("this frozen hardware experiment requires sm_120")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.cuda.get_device_properties(0)
    result.update({"arm": args.arm, "block": args.block, "preflight": args.preflight,
              "torch": torch.__version__, "torch_cuda": torch.version.cuda,
              "device": device.name, "sm_count": device.multi_processor_count,
              "memory_bytes": device.total_memory, "cells": []})
    shapes = shape_order(args.block, args.preflight)
    result["shape_order"] = shapes
    result["estimator"] = "arithmetic mean of all timed complete operator CUDA-event latencies"
    for name, bs in shapes:
        seed = 20260904 + (0 if name == "llama-3-8b" else 1000) + bs
        g = torch.Generator(device="cuda").manual_seed(seed)
        hkv = MODEL_HEADS[name]
        def rand(shape):
            return torch.randn(shape, dtype=torch.float16, device="cuda", generator=g)
        q_p, k_p, v_p = rand((1, 8192, 32, 128)), rand((1, 8192, hkv, 128)), rand((1, 8192, hkv, 128))
        q_d, k_d, v_d = rand((bs, 1, 32, 128)), rand((bs, 8192, hkv, 128)), rand((bs, 8192, hkv, 128))
        lengths_p = torch.full((1,), 8192, dtype=torch.int32, device="cuda")
        lengths_d = torch.full((bs,), 8191, dtype=torch.int32, device="cuda")
        def prefill():
            return flash_attn_with_kvcache(q_p, k_p, v_p, cache_seqlens=lengths_p, causal=True)
        def decode():
            return flash_attn_with_kvcache(q_d, k_d, v_d, cache_seqlens=lengths_d, causal=False)
        gold = (prefill(), decode())
        torch.cuda.synchronize()

        def check_pair(outputs, refs):
            maximum = 0.0
            for output, reference in zip(outputs, refs):
                if output is None or output.shape != reference.shape or not torch.isfinite(output).all().item():
                    raise RuntimeError("missing, invalid-shaped or nonfinite attention output")
                diff = (output.float() - reference.float()).abs()
                maximum = max(maximum, diff.max().item())
                if not torch.allclose(output.float(), reference.float(), atol=1e-3, rtol=1e-5):
                    raise RuntimeError(f"attention mismatch: max_abs={maximum}, fixed atol=1e-3 rtol=1e-5")
            return maximum

        def fp32_reference_check(q, k, v, output, length, causal):
            maximum = 0.0
            for b in range(0, q.shape[0], 4):
                kk = k[b:b+4, :length].float().repeat_interleave(32 // hkv, dim=2).transpose(1, 2)
                vv = v[b:b+4, :length].float().repeat_interleave(32 // hkv, dim=2).transpose(1, 2)
                for row in range(0, q.shape[1], 128):
                    qq = q[b:b+4, row:row+128].float().transpose(1, 2)
                    scores = torch.matmul(qq, kk.transpose(-1, -2)) * (128 ** -0.5)
                    if causal:
                        positions = torch.arange(row, row + qq.shape[2], device="cuda")
                        keys = torch.arange(length, device="cuda")
                        scores.masked_fill_(keys[None, :] > positions[:, None], -math.inf)
                    reference = torch.matmul(torch.softmax(scores, dim=-1), vv).transpose(1, 2)
                    maximum = max(maximum, check_pair((output[b:b+4, row:row+128],), (reference,)))
            return maximum

        reference_error = max(fp32_reference_check(q_p, k_p, v_p, gold[0], 8192, True),
                              fp32_reference_check(q_d, k_d, v_d, gold[1], 8191, False))
        streams = (torch.cuda.Stream(), torch.cuda.Stream())
        joins = (torch.cuda.Event(), torch.cuda.Event())
        current = torch.cuda.current_stream()
        pod_mode = args.arm.removeprefix("pod_") if args.arm.startswith("pod_") else None
        if pod_mode:
            fused_attn.pod_configure(pod_mode, True)

        def operator():
            if args.arm == "official_serial":
                return prefill(), decode()
            if args.arm == "official_streams":
                outputs = []
                # Both streams acquire the same current-stream prefix before
                # either completion is joined; otherwise decode becomes serial.
                for stream in streams:
                    stream.wait_stream(current)
                for stream, join, fn in zip(streams, joins, (prefill, decode)):
                    with torch.cuda.stream(stream):
                        outputs.append(fn())
                        join.record(stream)
                for join in joins:
                    current.wait_event(join)
                return tuple(outputs)
            return true_fused_attn_with_kvcache(q_p, k_p, v_p, q_d, k_d, v_d,
                       cache_seqlens_p=lengths_p, cache_seqlens_d=lengths_d,
                       causal=True, fused_params=15)

        bridge_before = bridge_stats() if bridge_stats else None
        outputs = operator()
        torch.cuda.synchronize()
        error = check_pair(outputs, gold)
        diagnostic = None
        if pod_mode:
            metadata, counters, contexts, errors = fused_attn.pod_last_launch()
            if errors.cpu().item() != 0:
                raise RuntimeError("POD executor rejected actual device selector output")
            meta = dict(zip(META_NAMES, metadata.tolist()))
            raw_context = bytes(contexts.cpu().flatten().tolist())
            decisions = [dict(zip(CTX_NAMES, row)) for row in struct.iter_unpack("<Q16I", raw_context)]
            counts = counters.cpu().tolist()
            diagnostic = {"metadata": meta, "counters": counts, "contexts": decisions,
                          "audit": audit_decisions(meta, counts, decisions, 2 if pod_mode == "bpf" else 1)}
            fused_attn.pod_configure(pod_mode, False)
        for _ in range(10):
            outputs = operator()
        torch.cuda.synchronize()
        error = max(error, check_pair(outputs, gold))
        samples = []
        for _ in range(3 if args.preflight else 100):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            wall_start = time.perf_counter_ns()
            start.record(current)
            outputs = operator()
            end.record(current)
            end.synchronize()
            wall_ms = (time.perf_counter_ns() - wall_start) / 1e6
            elapsed_ms = start.elapsed_time(end)
            if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
                raise RuntimeError("invalid complete operator event timing")
            error = max(error, check_pair(outputs, gold))
            if pod_mode:
                metadata, counters, _, errors = fused_attn.pod_last_launch()
                meta = dict(zip(META_NAMES, metadata.tolist()))
                counts = counters.cpu().tolist()
                if errors.cpu().item() or sum(counts[:meta["nsmid"]]) != meta["grid_ctas"]:
                    raise RuntimeError("timed operator did not execute all device decisions")
            samples.append({"cuda_ms": elapsed_ms, "host_wall_ms": wall_ms})
        bridge = (audit_bridge(bridge_before, bridge_stats(), 1 + 10 + len(samples),
                              meta["smem_bytes"], pod_mode) if bridge_stats else None)
        cell = {"model": name, "kv_heads": hkv, "query_heads": 32, "head_dim": 128,
                "prefill_batch": 1, "prefill_length": 8192, "decode_batch": bs,
                "decode_query_length": 1, "decode_cache_extent": 8192, "decode_valid_kv": 8191,
                "dtype": "float16", "seed": seed, "fused_params": 15 if pod_mode else None, "warmups": 10,
                "samples": samples, "mean_cuda_ms": statistics.fmean(x["cuda_ms"] for x in samples),
                "mean_host_wall_ms": statistics.fmean(x["host_wall_ms"] for x in samples),
                "max_abs_vs_official": error, "official_max_abs_vs_fp32": reference_error,
                "atol": 1e-3, "rtol": 1e-5, "diagnostic": diagnostic, "launch_bridge": bridge}
        result["cells"].append(cell)
        output_file.seek(0)
        json.dump(result, output_file, indent=2)
        output_file.write("\n")
        output_file.truncate()
        output_file.flush()
        print(f"POD_CELL arm={args.arm} model={name} bs={bs} mean_cuda_ms={cell['mean_cuda_ms']:.6f}", flush=True)
    result["complete"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--block", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Refuse replacement before importing CUDA. Persist failure, never a silent None/-1.
    with args.output.open("x") as out:
        result = {"complete": False, "arm": args.arm, "block": args.block,
                  "preflight": args.preflight, "cells": []}
        try:
            run_arm(args, result, out)
        except BaseException as exc:
            result.update({"complete": False, "error": f"{type(exc).__name__}: {exc}"})
            out.seek(0)
            json.dump(result, out, indent=2)
            out.write("\n")
            out.truncate()
            raise
        out.seek(0)
        json.dump(result, out, indent=2)
        out.write("\n")
        out.truncate()


if __name__ == "__main__":
    main()
