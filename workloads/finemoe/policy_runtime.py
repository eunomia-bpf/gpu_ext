"""Bounded FineMoE selector glue on the author's real Qwen offload executor.

FP32 routing probabilities/delta are unchanged. Prefix accumulation is explicitly
sequential binary64 (unlike the original demo's FP32 torch.cumsum), with an
independent Python arithmetic oracle. Real JIT failure never falls back to C.
"""
from collections import Counter
import ctypes as ct
import math
from pathlib import Path
import struct
import types

import torch


HERE = Path(__file__).resolve().parent
ARMS = ("demand-only", "all-positive", "finemoe-c", "finemoe-bpf")


class PolicyInput(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in ("count", "top_k", "threshold_bits", "reserved")]
    _fields_ += [("probability_bits", ct.c_uint32 * 60)]


class PolicyOutput(ct.Structure):
    _fields_ = [("mask", ct.c_uint64), ("cumulative_bits", ct.c_uint64),
                ("selected", ct.c_uint32), ("status", ct.c_uint32)]


class PolicyContext(ct.Structure):
    _fields_ = [("input", PolicyInput), ("output", PolicyOutput)]


def f32bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def python_oracle(probabilities, delta, top_k):
    if not 0 < len(probabilities) <= 60 or not 0 < top_k <= len(probabilities):
        raise ValueError("invalid FineMoE dimensions")
    if not math.isfinite(delta) or not 0 <= delta <= 1:
        raise ValueError("invalid FineMoE delta")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("invalid router probabilities")
    if not any(probabilities):
        return 0, 0, 0
    ordered = sorted(range(len(probabilities)), key=lambda expert: (-probabilities[expert], expert))
    cumulative = 0.0
    mask = 0
    for count, expert in enumerate(ordered, 1):
        cumulative += float(probabilities[expert])
        mask |= 1 << expert
        if count >= top_k and cumulative >= delta:
            break
    return mask, count, struct.unpack("<Q", struct.pack("<d", cumulative))[0]


class EngineObserver:
    """Records calls delivered to real native engine APIs; no policy decision."""
    def __init__(self, target, policy):
        self.target, self.policy = target, policy

    def __getattr__(self, name):
        return getattr(self.target, name)

    def replace_cache_candidates(self, tids):
        result = self.target.replace_cache_candidates(tids)
        self.policy.stats["engine_candidate_replacements"] += 1
        self.policy.stats["engine_admitted_candidates"] += len(tids)
        if self.policy.capture:
            self.policy.events.append({"event": "engine_candidates", "tensor_ids": list(tids)})
        return result

    def enqueue_prefetch(self, tid, device, probability):
        result = self.target.enqueue_prefetch(tid, device, probability)
        self.policy.stats["engine_enqueue_calls"] += 1
        if self.policy.capture:
            self.policy.events.append({"event": "engine_enqueue", "tensor_id": tid,
                                       "device": device, "probability": probability})
        return result


class FineMoePolicy:
    def __init__(self, arm, shadow=False, capture=False):
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm, self.shadow, self.capture = arm, shadow, capture
        self.stats, self.events = Counter(), []
        self.lib, self.handle = None, None
        if ct.sizeof(PolicyContext) != 280:
            raise RuntimeError("unexpected FineMoE policy ABI")
        if arm in ("finemoe-c", "finemoe-bpf"):
            self.lib = ct.CDLL(str(HERE / "build/libfinemoe_policy.so"))
            self.lib.finemoe_select_native.argtypes = [ct.POINTER(PolicyContext)]
            self.lib.finemoe_select_native.restype = ct.c_int
            self.lib.finemoe_jit_open.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_size_t]
            self.lib.finemoe_jit_open.restype = ct.c_void_p
            self.lib.finemoe_select_bpf.argtypes = [ct.c_void_p, ct.POINTER(PolicyContext)]
            self.lib.finemoe_select_bpf.restype = ct.c_int
            self.lib.finemoe_jit_calls.argtypes = [ct.c_void_p]
            self.lib.finemoe_jit_calls.restype = ct.c_uint64
            self.lib.finemoe_jit_close.argtypes = [ct.c_void_p]
        if arm == "finemoe-bpf":
            error = ct.create_string_buffer(512)
            self.handle = self.lib.finemoe_jit_open(
                str(HERE / "build/finemoe_policy.bin").encode(), error, len(error))
            if not self.handle:
                raise RuntimeError(f"FineMoE JIT failed: {error.value.decode()}")

    def close(self):
        if self.handle:
            self.lib.finemoe_jit_close(self.handle)
            self.handle = None

    def select(self, probabilities, delta, top_k):
        # Shared validation/packing is not selection. The C/BPF arms both rank
        # and perform the prefix test themselves on the same unsorted input.
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
            raise ValueError("invalid router probabilities")
        if not math.isfinite(delta) or not 0 <= delta <= 1:
            raise ValueError("invalid confidence threshold")
        if not 0 < len(probabilities) <= 60 or not 0 < top_k <= len(probabilities):
            raise ValueError("invalid policy dimensions")
        if self.arm == "demand-only":
            return 0
        if self.arm == "all-positive":
            return sum(1 << expert for expert, p in enumerate(probabilities) if p > 0)
        ctx = PolicyContext()
        ctx.input.count, ctx.input.top_k = len(probabilities), top_k
        ctx.input.threshold_bits = f32bits(delta)
        for expert, p in enumerate(probabilities):
            ctx.input.probability_bits[expert] = f32bits(p)
        if self.arm == "finemoe-bpf":
            status = self.lib.finemoe_select_bpf(self.handle, ct.byref(ctx))
        else:
            status = self.lib.finemoe_select_native(ct.byref(ctx))
        if status or ctx.output.status:
            raise RuntimeError(f"FineMoE policy rejected input: status={status}/{ctx.output.status}")
        self.stats["policy_calls"] += 1
        if self.shadow:
            expected = python_oracle(probabilities, delta, top_k)
            actual = (ctx.output.mask, ctx.output.selected, ctx.output.cumulative_bits)
            if actual != expected:
                raise RuntimeError(f"FineMoE independent oracle mismatch: {actual} != {expected}")
            self.stats["oracle_checks"] += 1
        return ctx.output.mask

    @torch.inference_mode()
    def process(self, matcher, layer_start, layer_end, score, expert_map):
        if expert_map.shape != (matcher.num_layers, matcher.num_experts):
            raise ValueError("unexpected expert-map shape")
        if not 0 <= layer_start < layer_end <= matcher.num_layers:
            raise ValueError("invalid predicted layer band")
        # This data movement and the author's search/store are shared by all arms.
        probabilities = expert_map.detach().to("cpu", dtype=torch.float32).clone()
        delta = float(torch.clamp(1 - score.to(dtype=torch.float32), 0, 1).cpu().item())
        probabilities[:layer_start].zero_()
        probabilities[layer_end:].zero_()
        priorities = torch.zeros_like(probabilities)
        rows = probabilities.tolist()
        masks = []
        for layer in range(layer_start, layer_end):
            mask = self.select(rows[layer], delta, matcher.top_k)
            selected = [bool(mask & (1 << expert)) for expert in range(matcher.num_experts)]
            selected_tensor = torch.tensor(selected, dtype=torch.bool)
            # Keep the demo's distance priority fixed; the experiment changes
            # only candidate admission. Selected zero probabilities retain K.
            decay = 1 - (layer - layer_start) / (layer_end + 1)
            priorities[layer] = torch.where(selected_tensor,
                                           probabilities[layer] * decay + 1e-6, 0.)
            self.stats["selector_rows"] += 1
            self.stats["selected_candidates"] += mask.bit_count()
            self.stats[f"cardinality_{mask.bit_count()}"] += 1
            masks.append(mask)
        self.stats["prediction_maps"] += 1
        if self.capture:
            self.events.append({"event": "selector", "layer_start": layer_start,
                                "layer_end": layer_end, "delta": delta, "masks": masks,
                                "probabilities": rows[layer_start:layer_end]})
        return priorities, probabilities

    def install(self, engine):
        matcher = engine.expert_map_matcher
        matcher.process_expert_map = types.MethodType(
            lambda bound, first, last, score, values: self.process(bound, first, last, score, values), matcher)
        prefetcher = engine.expert_prefetcher
        prefetcher.archer_engine = EngineObserver(prefetcher.archer_engine, self)

    def snapshot(self):
        result = dict(self.stats)
        result["jit_calls"] = int(self.lib.finemoe_jit_calls(self.handle)) if self.handle else 0
        result["arm"] = self.arm
        result["arithmetic"] = "exact binary32 input, sequential binary64 prefix, stable expert-ID ties"
        result["shadow"] = self.shadow
        return {"stats": result, "events": self.events if self.capture else []}
