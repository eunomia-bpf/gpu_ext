"""CPU diagnostic only; compare old packing and batch packing with real JIT."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import time

import numpy as np

from paper_policy import JitRanker, native_rank
from paper_policy_buffers import RankEntry, packed_select
import run_moe_head_to_head as base


def legacy_select(run, identities, scores):
    scores = np.ascontiguousarray(scores, dtype=np.float64)
    entries = (RankEntry * len(scores))()
    for i, bits in enumerate(scores.view(np.uint64)):
        entries[i] = RankEntry(identities[i], int(bits), i, 0)
    indices = (ctypes.c_uint32 * len(scores))()
    count = ctypes.c_uint32()
    rc = run(entries, len(scores), indices, len(scores), ctypes.byref(count))
    if rc != 0 or count.value > len(scores):
        raise RuntimeError("legacy JIT failed")
    result = list(indices[:count.value])
    if len(set(result)) != len(result) or any(i >= len(scores) for i in result):
        raise RuntimeError("legacy invalid indices")
    return result


def benchmark(output):
    library = base.EXTENSION / ".output/libmoe_expert_policy.so"
    rank = JitRanker(str(library), str(base.EXTENSION / ".output/moe_expert_policy_rank.bin"))
    # The old module may still define its own layout-identical ctypes class.
    rank.run.argtypes = [ctypes.POINTER(RankEntry), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
    rng = np.random.default_rng(17)
    cases = 0
    for size in (0, 1, 2, 7, 128, 1024, 4096):
        for _ in range(20):
            scores = rng.integers(-3, 4, size * 2).astype(np.float64)[::2]
            if size >= 7:
                scores[:7] = [np.nan, np.inf, -np.inf, -0., 0., 1., 1.]
            identities = list(range(size))
            expected = native_rank(scores)
            if legacy_select(rank.run, identities, scores) != expected or packed_select(rank.run, identities, scores) != expected:
                raise RuntimeError("packing changed the actual JIT result")
            cases += 1
    scores = rng.random(4096)
    identities = list(range(len(scores)))
    result = {"cpu_diagnostic_only": True, "gpu_performance_claimed": False,
              "affinity": sorted(os.sched_getaffinity(0)), "exact_equivalence_cases": cases,
              "candidate_count": 4096, "calls_per_variant": 300,
              "variant_order": ["legacy", "packed", "native"], "wall_ms_per_call": {}}
    for name, call in (("legacy", lambda: legacy_select(rank.run, identities, scores)),
                       ("packed", lambda: packed_select(rank.run, identities, scores)),
                       ("native", lambda: native_rank(scores))):
        started = time.perf_counter_ns()
        for _ in range(300):
            call()
        result["wall_ms_per_call"][name] = (time.perf_counter_ns() - started) / 300e6
    if output.exists():
        raise FileExistsError(output)
    base.atomic_write_json(output, result)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    benchmark(parser.parse_args().output)
