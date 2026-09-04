#!/usr/bin/env python3
"""Official POD/FlashAttention operator cells; no GPU work at import time.

The coordinator owns the GPU lease and, for pod_bpf, starts the private loader
before this process with the existing bpftime agent preloaded. Each invocation
runs one arm over the fixed shapes (or the one-shape real preflight).
"""
import time

# This is intentionally the first executable module statement.  The parent
# records the time immediately before Popen, so the two monotonic timestamps
# bound interpreter/preload startup without pretending that this is exec time.
PROCESS_MAIN_NS = time.monotonic_ns()

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

ROOT = Path(__file__).resolve().parent
ARMS = ("official_serial", "official_streams", "pod_inline", "pod_cuda", "pod_bpf")
MODEL_HEADS = {"llama-3-8b": 8, "yi-6b": 4}
META_NAMES = ("nsmid", "grid_ctas", "prefill_blocks", "decode_blocks", "factor_p",
              "factor_d", "smem_bytes", "threads", "fused_op", "mode", "trace")
CTX_NAMES = ("counters", "abi_version", "nsmid", "smid", "prefill_slots", "decode_slots",
             "proportional", "grid_ctas", "out_op", "out_cta", "status", "engine", "ticket",
             "first_op", "first_claim", "fallback_claim", "reserved")
UNSET = 0xffffffff
NUMERIC_PROTOCOL = 'pod-fp16-upstream-match-v2'
STDLIB_IMPORTS_DONE_NS = time.monotonic_ns()


def fp32_diagnostic_name(model, batch, phase):
    if model not in MODEL_HEADS or batch not in (32, 64, 96, 128, 192) or phase not in ('prefill', 'decode'):
        raise ValueError('unknown fixed diagnostic shape/phase')
    return f'fp32-characterization-{model}-bs{batch}-{phase}'


def half_precision_evidence(actual, reference):
    """Adjacent representable values, not an adjusted correctness threshold."""
    if not math.isfinite(actual) or not math.isfinite(reference):
        raise ValueError('precision diagnosis requires finite actual/reference values')
    bits = struct.unpack('<H', struct.pack('<e', actual))[0]
    half = lambda value: struct.unpack('<e', struct.pack('<H', value))[0]
    if half(bits) != actual:
        raise ValueError('actual diagnostic value is not exactly FP16')
    if actual == 0:
        lower, upper = half(0x8001), half(1)
    elif actual > 0:
        lower, upper = half(bits - 1), half(bits + 1)
    else:
        lower, upper = half(bits + 1), half(bits - 1)
    nearest = struct.unpack('<e', struct.pack('<e', reference))[0]
    threshold = 1e-3 + 1e-5 * abs(reference)
    return dict(actual_fp16=actual, reference_fp32=reference, adjacent_fp16_lower=lower,
                adjacent_fp16_upper=upper, nearest_fp16_to_reference=nearest,
                actual_is_nearest_fp16=actual == nearest,
                minimum_fp16_absolute_error=abs(nearest - reference),
                actual_absolute_error=abs(actual - reference), fixed_allowed_error=threshold,
                nearest_fp16_satisfies_fixed_tolerance=abs(nearest - reference) <= threshold)


def save_fp32_failure(directory, metadata, arrays):
    """Save only one real query/head and its effective keys; never overwrite."""
    if set(arrays) != {'q', 'k', 'v', 'actual', 'fp32_reference'}:
        raise ValueError('incomplete real FP32 diagnosis arrays')
    save_numeric_arrays(directory, metadata, arrays)


def save_numeric_arrays(directory, metadata, arrays):
    import numpy as np
    directory = Path(directory)
    directory.mkdir()
    files = {}
    for name, value in arrays.items():
        if name not in ('q', 'k', 'v', 'actual', 'fp32_reference', 'official'):
            raise ValueError('unexpected numeric array name')
        filename = name + '.npy'
        with (directory / filename).open('xb') as file:
            np.save(file, value, allow_pickle=False)
        files[name] = dict(filename=filename, dtype=str(value.dtype), shape=list(value.shape),
                           bytes=(directory / filename).stat().st_size)
    with (directory / 'diagnostic.json').open('x') as file:
        json.dump(dict(complete=True, **metadata, arrays=files), file, indent=2)
        file.write('\n')


def recompute_saved_fp64(directory):
    """Offline CPU-only recheck; importing this function does not import torch."""
    import numpy as np
    directory = Path(directory)
    meta = json.loads((directory / 'diagnostic.json').read_text())
    if meta.get('complete') is not True or meta.get('atol') != 1e-3 or meta.get('rtol') != 1e-5:
        raise ValueError('incomplete diagnostic or changed fixed threshold')
    arrays = {}
    for name in ('q', 'k', 'v', 'actual', 'fp32_reference'):
        spec = meta['arrays'][name]
        if spec['filename'] != name + '.npy':
            raise ValueError('unexpected diagnostic array path')
        value = np.load(directory / spec['filename'], allow_pickle=False)
        if list(value.shape) != spec['shape'] or str(value.dtype) != spec['dtype'] or not np.isfinite(value).all():
            raise ValueError('diagnostic array differs from its declared shape/type')
        if value.dtype != (np.float32 if name == 'fp32_reference' else np.float16):
            raise ValueError('not the real FP16 inputs/output and FP32 oracle row')
        arrays[name] = value.astype(np.float64)
    q, k, v, actual, fp32 = (arrays[name] for name in ('q', 'k', 'v', 'actual', 'fp32_reference'))
    width, keys = meta['head_dim'], meta['effective_keys']
    if (width != 128 or q.shape != (width,) or actual.shape != (width,) or fp32.shape != (width,)
            or k.shape != (keys, width) or v.shape != k.shape or keys < 1
            or keys != (meta['query_index'] + 1 if meta['causal'] else meta['valid_kv'])
            or not 0 <= meta['query_index'] < meta['query_length']
            or not 0 < keys <= meta['valid_kv'] or meta['scale'] != 128 ** -0.5):
        raise ValueError('saved query/key extent or scale does not match the real mask')
    scores = (k @ q) * meta['scale']
    weights = np.exp(scores - scores.max())
    fp64 = (weights / weights.sum()) @ v
    nearest = fp64.astype(np.float16).astype(np.float64)
    errors, floor = np.abs(actual - fp64), np.abs(nearest - fp64)
    threshold = 1e-3 + 1e-5 * np.abs(fp64)
    index = int(errors.argmax())
    result = dict(numeric_protocol=meta.get('numeric_protocol', 'pod-fp32-hard-gate-v1'),
        scope='saved real query/head only, not a full-shape pass',
        max_abs_actual_vs_fp64=float(errors.max()), max_abs_fp32_vs_fp64=float(np.abs(fp32 - fp64).max()),
        max_abs_nearest_fp16_vs_fp64=float(floor.max()), actual_exceeding_fixed_tolerance=int((errors > threshold).sum()),
        nearest_fp16_exceeding_fixed_tolerance=int((floor > threshold).sum()),
        max_excess_above_best_final_fp16_rounding=float((errors - floor).max()),
        worst_dimension=index, actual=float(actual[index]), fp32_reference=float(fp32[index]),
        fp64_reference=float(fp64[index]), nearest_fp16=float(nearest[index]),
        limitation='Excess over final FP16 rounding does not isolate softmax, GEMM, or online rescaling.')
    if keys == 2 and meta['causal'] and meta['query_index'] == 1:
        # A real two-key row fits one upstream tile: model the unnormalized
        # exponential's conversion to Element=half before PV, then normalize
        # using the unquantized sum. This is not a hardware rounding simulator.
        half_p = weights.astype(np.float16).astype(np.float64)
        modeled64 = ((half_p @ v) / weights.sum()).astype(np.float16).astype(np.float64)
        raw32 = k.astype(np.float32) @ q.astype(np.float32)
        scale32 = np.float32(meta['scale'] * np.log2(np.e))
        max_scaled = np.float32(raw32.max() * scale32)
        exponent32 = (raw32.astype(np.float64) * float(scale32) - float(max_scaled)).astype(np.float32)
        exp32 = np.exp2(exponent32)
        half_p32 = exp32.astype(np.float16).astype(np.float32)
        pv32 = half_p32[0] * v[0].astype(np.float32) + half_p32[1] * v[1].astype(np.float32)
        inv_sum = np.float32(1) / exp32.sum(dtype=np.float32)
        modeled32 = (pv32 * inv_sum).astype(np.float16).astype(np.float64)
        result['two_key_source_model'] = dict(
            scope='saved two-key row only; source consistency, not exact isolation of GPU rounding',
            unnormalized_exp_fp64=weights.tolist(), half_p=half_p.tolist(),
            unnormalized_exp_fp32_exp2=exp32.tolist(), half_p_fp32=half_p32.tolist(),
            fp32_inverse_sum=float(inv_sum),
            final_half_only_matches=int((nearest == actual).sum()),
            half_p_fp64_model_matches=int((modeled64 == actual).sum()),
            half_p_fp32_model_matches=int((modeled32 == actual).sum()),
            half_p_fp64_model_max_abs_vs_actual=float(np.abs(modeled64 - actual).max()),
            half_p_fp32_model_max_abs_vs_actual=float(np.abs(modeled32 - actual).max()),
            worst_dimension_half_p_fp64=float(modeled64[index]),
            worst_dimension_half_p_fp32=float(modeled32[index]))
    return result


def save_cpu_fp64_report(directory):
    directory = Path(directory)
    result = recompute_saved_fp64(directory)
    with (directory / 'cpu-fp64-report.json').open('x') as file:
        json.dump(result, file, indent=2)
        file.write('\n')
    return result


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
    if args.phase_study:
        result['phase_timestamps'] = {
            'process_main_ns': PROCESS_MAIN_NS,
            'stdlib_imports_done_ns': STDLIB_IMPORTS_DONE_NS,
            'runtime_imports_start_ns': time.monotonic_ns(),
        }
    sys.path[:0] = [str(ROOT / "build/python"), str(ROOT / "deps/vattention/pod_attn")]
    import torch
    import fused_attn
    from pod_attn.flash_attn_interface import flash_attn_with_kvcache
    from pod_attn.fused_attn_interface import true_fused_attn_with_kvcache
    if args.phase_study:
        result['phase_timestamps']['runtime_imports_done_ns'] = time.monotonic_ns()

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
                 "verified_dynamic_bytes", "static_shared_bytes", "device_optin_bytes",
                 "first_launches")]
        class BridgeFirstLaunch(ctypes.Structure):
            _fields_ = [("monotonic_ns", ctypes.c_uint64), ("kernel", ctypes.c_char * 512)]
        accessor = ctypes.CDLL(None).pod_bridge_get_stats
        accessor.argtypes = [ctypes.POINTER(BridgeStats), ctypes.c_uint64]
        accessor.restype = ctypes.c_int
        first_accessor = ctypes.CDLL(None).pod_bridge_get_first_launch
        first_accessor.argtypes = [ctypes.POINTER(BridgeFirstLaunch), ctypes.c_uint64,
                                   ctypes.c_uint64]
        first_accessor.restype = ctypes.c_int
        def bridge_stats():
            stats = BridgeStats()
            if accessor(ctypes.byref(stats), ctypes.sizeof(stats)):
                raise RuntimeError("launch bridge statistics ABI mismatch")
            return {name: getattr(stats, name) for name, _ in stats._fields_}
        def bridge_first_launches():
            count = bridge_stats()['first_launches']
            records = []
            for index in range(count):
                record = BridgeFirstLaunch()
                if first_accessor(ctypes.byref(record), ctypes.sizeof(record), index):
                    raise RuntimeError("launch bridge first-launch record disappeared")
                kernel = bytes(record.kernel).split(b'\0', 1)[0].decode('ascii')
                if not kernel or record.monotonic_ns <= 0:
                    raise RuntimeError("invalid launch bridge first-launch record")
                records.append({'kernel': kernel, 'monotonic_ns': record.monotonic_ns})
            extra = BridgeFirstLaunch()
            if first_accessor(ctypes.byref(extra), ctypes.sizeof(extra), count) != 1:
                raise RuntimeError("launch bridge first-launch count is not closed")
            if len({record['kernel'] for record in records}) != len(records):
                raise RuntimeError("duplicate launch bridge first-launch kernel")
            return records
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
              "phase_study": args.phase_study,
              "torch": torch.__version__, "torch_cuda": torch.version.cuda,
              "device": device.name, "sm_count": device.multi_processor_count,
              "memory_bytes": device.total_memory, "cells": []})
    shapes = shape_order(args.block, args.preflight or args.phase_study)
    if args.phase_study and len(shapes) != 1:
        raise RuntimeError("phase study must retain exactly one frozen operator shape")
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

        def check_pair(outputs, refs, stage):
            maximum = 0.0
            for phase, output, reference in zip(('prefill', 'decode'), outputs, refs):
                if output is None or output.shape != reference.shape:
                    result['pair_failure_context'] = dict(model=name, decode_batch=bs, phase=phase, stage=stage,
                        actual_shape=list(output.shape) if output is not None else None,
                        official_shape=list(reference.shape), reason='missing or invalid-shaped output')
                    raise RuntimeError("missing, invalid-shaped or nonfinite attention output")
                diff = (output.float() - reference.float()).abs()
                finite = torch.isfinite(output).all().item() and torch.isfinite(reference).all().item()
                current_max = diff.max().item() if finite else math.inf
                maximum = max(maximum, current_max)
                if not finite or not torch.allclose(output.float(), reference.float(), atol=1e-3, rtol=1e-5):
                    flat, index = torch.nan_to_num(diff, nan=math.inf).argmax().item(), []
                    for extent in reversed(diff.shape):
                        index.append(flat % extent)
                        flat //= extent
                    batch, query, head, component = reversed(index)
                    q, k, v = (q_p, k_p, v_p) if phase == 'prefill' else (q_d, k_d, v_d)
                    effective_keys = query + 1 if phase == 'prefill' else 8191
                    kv_head = head // (32 // hkv)
                    directory = args.output.parent / f'pair-failure-{name}-bs{bs}-{phase}-{stage}'
                    metadata = dict(numeric_protocol=NUMERIC_PROTOCOL, arm=args.arm, model=name, decode_batch=bs,
                        phase=phase, stage=stage, seed=seed, comparison='tested operator vs official FA, both FP16',
                        scope='entire ' + phase + ' output', output_shape=list(output.shape), finite=finite,
                        checked_elements=diff.numel(), exceeding_elements=(~torch.isclose(output.float(), reference.float(), atol=1e-3, rtol=1e-5)).sum().item(),
                        max_abs_error=current_max if finite else None,
                        max_error_coordinate=[batch, query, head, component], kv_head=kv_head,
                        query_index=query, query_length=q.shape[1], valid_kv=8192 if phase == 'prefill' else 8191,
                        effective_keys=effective_keys, causal=phase == 'prefill', head_dim=128, scale=128 ** -0.5,
                        atol=1e-3, rtol=1e-5)
                    result['pair_failure_context'] = dict(**metadata, diagnostic_directory=directory.name)
                    try:
                        save_numeric_arrays(directory, metadata,
                            dict(q=q[batch, query, head].cpu().numpy(),
                                 k=k[batch, :effective_keys, kv_head].cpu().numpy(),
                                 v=v[batch, :effective_keys, kv_head].cpu().numpy(),
                                 actual=output[batch, query, head].cpu().numpy(),
                                 official=reference[batch, query, head].cpu().numpy()))
                    except Exception as error:
                        raise RuntimeError(f'attention mismatch: max_abs={maximum}, fixed atol=1e-3 rtol=1e-5; '
                                           f'pair diagnostic save failed: {error}') from error
                    raise RuntimeError(f"attention mismatch: max_abs={maximum}, fixed atol=1e-3 rtol=1e-5")
            return maximum

        def fp32_characterization(q, k, v, output, length, causal, phase):
            if (output is None or tuple(output.shape) != tuple(q.shape) or tuple(k.shape) != tuple(v.shape)
                    or q.shape[2:] != (32, 128) or k.shape[2:] != (hkv, 128)
                    or q.shape[0] != k.shape[0] or not 0 < length <= k.shape[1]
                    or (causal and q.shape[1] != length) or (not causal and q.shape[1] != 1)):
                raise RuntimeError('official output/input shape or reference mask extent differs')
            maximum, error_sum, error_square_sum = -1.0, 0.0, 0.0
            checked, exceeding, worst = 0, 0, None
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
                    actual = output[b:b+4, row:row+128]
                    if not torch.isfinite(actual).all().item() or not torch.isfinite(reference).all().item():
                        raise RuntimeError('nonfinite official output or FP32 reference')
                    diff = (actual.float() - reference).abs()
                    maximum_here = diff.max().item()
                    exceeding += (~torch.isclose(actual.float(), reference, atol=1e-3, rtol=1e-5)).sum().item()
                    checked += diff.numel()
                    error_sum += diff.sum(dtype=torch.float64).item()
                    error_square_sum += diff.double().square().sum().item()
                    if maximum_here > maximum:
                        maximum = maximum_here
                        flat, index = diff.argmax().item(), []
                        for extent in reversed(diff.shape):
                            index.append(flat % extent)
                            flat //= extent
                        batch_offset, query_offset, head, component = reversed(index)
                        batch, query = b + batch_offset, row + query_offset
                        # Keep only this output row alive while subsequent chunks
                        # are checked. Original Q/K/V tensors remain untouched.
                        worst = (batch, query, head, component,
                                 reference[batch_offset, query_offset, head].clone())
            summary = dict(numeric_protocol=NUMERIC_PROTOCOL, phase=phase,
                role='characterization_not_cross_precision_pass_gate', finite=True, shape_checked=True,
                mask='causal_prefix' if causal else 'valid_kv', output_shape=list(output.shape),
                checked_elements=checked, exceeding_elements=exceeding, max_abs_error=maximum,
                mean_abs_error=error_sum / checked, rms_error=math.sqrt(error_square_sum / checked),
                atol=1e-3, rtol=1e-5, diagnostic_directory=None)
            if exceeding:
                batch, query, head, component, reference_row = worst
                kv_head = head // (32 // hkv)
                effective_keys = query + 1 if causal else length
                actual_row = output[batch, query, head]
                diagnostic_dir = args.output.parent / fp32_diagnostic_name(name, bs, phase)
                metadata = dict(numeric_protocol=NUMERIC_PROTOCOL, arm=args.arm, model=name, decode_batch=bs, seed=seed, phase=phase,
                    comparison='official FlashAttention vs full-FP32 reference, before this shape tested operator',
                    scope='entire ' + phase + ' output', output_shape=list(output.shape),
                    checked_elements=checked, exceeding_elements=exceeding, max_abs_error=maximum,
                    mean_abs_error=error_sum / checked, rms_error=math.sqrt(error_square_sum / checked),
                    max_error_coordinate=[batch, query, head, component], kv_head=kv_head,
                    query_index=query, query_length=q.shape[1], valid_kv=length,
                    effective_keys=effective_keys, causal=causal, head_dim=128, scale=128 ** -0.5,
                    atol=1e-3, rtol=1e-5,
                    precision=half_precision_evidence(actual_row[component].item(), reference_row[component].item()))
                try:
                    save_fp32_failure(diagnostic_dir, metadata,
                        dict(q=q[batch, query, head].cpu().numpy(),
                             k=k[batch, :effective_keys, kv_head].cpu().numpy(),
                             v=v[batch, :effective_keys, kv_head].cpu().numpy(),
                             actual=actual_row.cpu().numpy(), fp32_reference=reference_row.cpu().numpy()))
                    summary['diagnostic_directory'] = diagnostic_dir.name
                except Exception as error:
                    raise RuntimeError(f'FP32 characterization max_abs={maximum}, atol=1e-3 rtol=1e-5; '
                                       f'diagnostic save failed: {error}') from error
            return summary

        reference_stats = dict(prefill=fp32_characterization(q_p, k_p, v_p, gold[0], 8192, True, 'prefill'),
                               decode=fp32_characterization(q_d, k_d, v_d, gold[1], 8191, False, 'decode'))
        reference_error = max(value['max_abs_error'] for value in reference_stats.values())
        result.setdefault('fp32_characterizations', {})[f'{name}:bs{bs}'] = reference_stats
        # Keep completed reference scans even if the following operator/JIT or
        # hard same-precision comparison fails. This is outside event timing.
        output_file.seek(0)
        json.dump(result, output_file, indent=2)
        output_file.write('\n')
        output_file.truncate()
        output_file.flush()
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
        if args.phase_study:
            result['phase_timestamps']['pre_first_diagnostic_ns'] = time.monotonic_ns()
        outputs = operator()
        torch.cuda.synchronize()
        if args.phase_study:
            result['phase_timestamps']['post_first_sync_ns'] = time.monotonic_ns()
        error = check_pair(outputs, gold, 'diagnostic')
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
        error = max(error, check_pair(outputs, gold, 'warmup-10'))
        if args.phase_study:
            result['phase_timestamps']['warmup_done_ns'] = time.monotonic_ns()
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
            error = max(error, check_pair(outputs, gold, f'timed-{len(samples) + 1:03d}'))
            if pod_mode:
                metadata, counters, _, errors = fused_attn.pod_last_launch()
                meta = dict(zip(META_NAMES, metadata.tolist()))
                counts = counters.cpu().tolist()
                if errors.cpu().item() or sum(counts[:meta["nsmid"]]) != meta["grid_ctas"]:
                    raise RuntimeError("timed operator did not execute all device decisions")
            samples.append({"cuda_ms": elapsed_ms, "host_wall_ms": wall_ms})
        if args.phase_study:
            result['phase_timestamps']['steady_complete_ns'] = time.monotonic_ns()
        bridge = (audit_bridge(bridge_before, bridge_stats(), 1 + 10 + len(samples),
                              meta["smem_bytes"], pod_mode) if bridge_stats else None)
        if bridge is not None and args.phase_study:
            bridge['first_launches'] = bridge_first_launches()
        cell = {"numeric_protocol": NUMERIC_PROTOCOL, "fp32_characterization": reference_stats,
                "model": name, "kv_heads": hkv, "query_heads": 32, "head_dim": 128,
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
    parser.add_argument("--phase-study", action="store_true",
                        help="one-shape fresh-process setup/first-launch/steady decomposition")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Refuse replacement before importing CUDA. Persist failure, never a silent None/-1.
    with args.output.open("x") as out:
        result = {"complete": False, "numeric_protocol": NUMERIC_PROTOCOL, "arm": args.arm, "block": args.block,
                  "preflight": args.preflight, "phase_study": args.phase_study, "cells": []}
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
