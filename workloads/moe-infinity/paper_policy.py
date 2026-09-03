"""Explicit paper-v3 MoE-Infinity policy port; no torch/CUDA at import time.

Source: arXiv:2401.14361v3, sections 4.3--4.7, Algorithm 1 and B.1.
This is not the current upstream GPT-OSS default (which leaves prediction off).
See activation-aware-port.md for conventions the paper does not specify.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
import threading

import numpy as np

from paper_policy_buffers import RankEntry, packed_select


EPSILON = 1e-8


def native_rank(scores: np.ndarray) -> list[int]:
    """Positive candidates, stable descending order (including positive inf)."""
    return sorted((i for i, value in enumerate(scores) if value > 0),
                  key=lambda i: float(scores[i]), reverse=True)


class JitRanker:
    def __init__(self, library: str, code: str, *, operation: str = "rank"):
        if not Path(library).is_absolute() or not Path(code).is_absolute():
            raise ValueError("BPF library and bytecode paths must be absolute")
        self.lib = ctypes.CDLL(library)
        if operation not in {"rank", "match"}:
            raise ValueError("unknown JIT selection operation")
        initialize = getattr(self.lib, f"moe_expert_{operation}_init_v1")
        initialize.argtypes = [ctypes.c_char_p]
        initialize.restype = ctypes.c_int
        self.run = getattr(self.lib, f"moe_expert_{operation}_v1")
        self.run.argtypes = [
            ctypes.POINTER(RankEntry), ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32)]
        self.run.restype = ctypes.c_int
        if initialize(code.encode()) != 0:
            raise RuntimeError("BPF rank JIT initialization failed; no fallback")

    def __call__(self, identities: list[int], scores: np.ndarray) -> list[int]:
        return packed_select(self.run, identities, scores)


class JitMatcher(JitRanker):
    def __init__(self, library: str, code: str):
        super().__init__(library, code, operation="match")

    def __call__(self, similarities: np.ndarray) -> list[int]:
        return super().__call__(list(range(len(similarities))), similarities)


class EAMCollection:
    """Bounded completed request-level traces, replacing the closest trace."""
    def __init__(self, layers: int, experts: int, capacity: int = 1000,
                 matcher=None, verify: bool = False):
        if min(layers, experts, capacity) <= 0:
            raise ValueError("positive EAMC dimensions/capacity required")
        self.layers, self.experts, self.capacity = layers, experts, capacity
        self.entries: list[np.ndarray] = []
        self.phases: list[str] = []
        self._rows = np.empty((capacity, layers * experts), dtype=np.float64)
        self._row_norms = np.empty(capacity, dtype=np.float64)
        self.replacements = 0
        self.matcher, self.verify = matcher, verify
        self.match_calls, self.match_mismatches = 0, 0

    def select_matches(self, similarities: np.ndarray) -> list[int]:
        def native():
            return np.flatnonzero(similarities == np.max(similarities)).tolist()
        if self.matcher is None:
            return native()
        result = self.matcher(similarities)
        self.match_calls += 1
        if not result:
            raise RuntimeError("BPF returned no match for finite nonempty similarities")
        if self.verify and result != native():
            self.match_mismatches += 1
            raise RuntimeError("BPF/native EAMC match mismatch")
        return result

    def _validate(self, matrix):
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (self.layers, self.experts):
            raise ValueError("EAM dimensions disagree with the model")
        if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
            raise ValueError("EAM must contain finite nonnegative counts")
        return matrix

    def similarities(self, matrix: np.ndarray) -> np.ndarray:
        query = self._validate(matrix).reshape(-1)
        if not self.entries:
            return np.empty(0, dtype=np.float64)
        rows = self._rows[:len(self.entries)]
        denominator = self._row_norms[:len(self.entries)] * np.linalg.norm(query)
        return np.divide(rows @ query, denominator, out=np.zeros(len(rows)),
                         where=denominator > 0)

    def insert(self, matrix: np.ndarray, phase: str) -> int | None:
        matrix = self._validate(matrix)
        if not np.any(matrix):
            return None
        if phase not in {"prefill", "decode"}:
            raise ValueError("phase must be prefill or decode")
        if len(self.entries) < self.capacity:
            self.entries.append(matrix.copy())
            self.phases.append(phase)
            index = len(self.entries) - 1
        else:
            # Appendix B.1: closest existing trace, NOT least-frequently used.
            index = self.select_matches(self.similarities(matrix))[0]
            self.entries[index] = matrix.copy()
            self.phases[index] = phase
            self.replacements += 1
        self._rows[index] = matrix.reshape(-1)
        # Keep precisely the original per-row norm operation, computed only
        # when an entry changes, not by restacking the EAMC at every layer.
        self._row_norms[index] = np.linalg.norm(self._rows[index:index + 1], axis=1)[0]
        return index

    def predict(self, current_iteration: np.ndarray) -> tuple[np.ndarray, list[int]]:
        current_iteration = self._validate(current_iteration)
        if not self.entries:
            # Explicit cold-start convention: neutral reuse, no prefetch below.
            return np.full_like(current_iteration, 1.0 / self.experts), []
        similarities = self.similarities(current_iteration)
        matched = self.select_matches(similarities)
        aggregate = np.sum([self.entries[i] for i in matched], axis=0)
        row_sum = aggregate.sum(axis=1, keepdims=True)
        probabilities = np.divide(aggregate, row_sum, out=np.zeros_like(aggregate),
                                  where=row_sum > 0)
        return probabilities, matched


@dataclass
class Prediction:
    reuse_scores: np.ndarray
    prefetch_identities: list[int]
    matched_entries: list[int]


class ActivationPolicy:
    """One active sequence, one iEAM per model forward, rEAM per phase."""
    def __init__(self, layers: int, experts: int, *, capacity: int = 1000,
                 ranker=None, matcher=None, verify_rank: bool = False):
        self.collection = EAMCollection(layers, experts, capacity, matcher, verify_rank)
        self.layers, self.experts = layers, experts
        self.ranker, self.verify_rank = ranker, verify_rank
        self.requests: dict[int, dict[str, np.ndarray]] = {}
        self._abort_lock = threading.Lock()
        self._aborted: set[int] = set()
        self.current_seq: int | None = None
        self.current_phase: str | None = None
        self.current_iteration: np.ndarray | None = None
        self.next_layer = 0
        self.stats = dict(iterations=0, routed_layers=0, predictions=0,
                          matched_predictions=0, rank_calls=0, rank_mismatches=0,
                          completed_requests=0, aborted_requests=0,
                          prefetch_candidates_selected=0)

    def begin_iteration(self, seq_id: int, is_prefill: bool) -> None:
        if self.current_iteration is not None:
            raise RuntimeError("overlapping model iterations are unsupported")
        self.current_seq = seq_id
        self.current_phase = "prefill" if is_prefill else "decode"
        self.requests.setdefault(seq_id, {
            phase: np.zeros((self.layers, self.experts), dtype=np.float64)
            for phase in ("prefill", "decode")})
        self.current_iteration = np.zeros((self.layers, self.experts), dtype=np.float64)
        self.next_layer = 0

    def mark_aborted(self, seq_id: int) -> None:
        # Called by the API thread. Never mutates an execution-owned EAM.
        with self._abort_lock:
            self._aborted.add(seq_id)

    def drain_aborted(self) -> None:
        # Called only by the model execution owner, outside an active forward.
        with self._abort_lock:
            aborted = self._aborted - ({self.current_seq} if self.current_seq is not None else set())
            self._aborted.difference_update(aborted)
        for seq_id in aborted:
            self.finish_request(seq_id, aborted=True)

    def observe(self, layer: int, counts) -> Prediction:
        if self.current_iteration is None or layer != self.next_layer:
            raise RuntimeError(f"expected routed layer {self.next_layer}, got {layer}")
        counts = np.asarray(counts, dtype=np.float64)
        if counts.shape != (self.experts,) or np.any(counts < 0) or not np.all(np.isfinite(counts)):
            raise ValueError("invalid per-expert activation counts")
        self.current_iteration[layer] = counts
        self.next_layer += 1
        self.stats["routed_layers"] += 1
        probabilities, matched = self.collection.predict(self.current_iteration)
        self.stats["predictions"] += 1
        self.stats["matched_predictions"] += bool(matched)

        # Algorithm 1: all-layer reuse, global layer proximity. No future mask.
        totals = probabilities.sum(axis=1, keepdims=True)
        normalized = np.divide(probabilities + EPSILON, totals,
                               out=np.zeros_like(probabilities), where=totals > 0)
        reuse = normalized * (1.0 - np.arange(self.layers)[:, None] / self.layers)

        # Section 4.5: future-only activation, proximity to current routed layer.
        future = probabilities[layer + 1:].copy()
        future *= (1.0 - (np.arange(layer + 1, self.layers) - layer)[:, None] / self.layers)
        if not matched:
            future.fill(0)  # no claim of a learned prediction with empty EAMC
        identities = [(i << 32) | j for i in range(layer + 1, self.layers)
                      for j in range(self.experts)]
        values = future.reshape(-1)
        if self.ranker is None:
            order = native_rank(values)
        else:
            order = self.ranker(identities, values)
            self.stats["rank_calls"] += 1
            if self.verify_rank and order != native_rank(values):
                self.stats["rank_mismatches"] += 1
                raise RuntimeError("BPF/native prefetch ordering mismatch")
        selected = [identities[i] for i in order]
        self.stats["prefetch_candidates_selected"] += len(selected)
        return Prediction(reuse.reshape(-1), selected, matched)

    def end_iteration(self, *, success: bool = True) -> None:
        if self.current_iteration is None:
            raise RuntimeError("no active iteration")
        try:
            if success:
                if self.next_layer != self.layers:
                    raise RuntimeError("incomplete routed iteration")
                self.requests[self.current_seq][self.current_phase] += self.current_iteration
                self.stats["iterations"] += 1
        finally:
            self.current_iteration = None
            self.current_seq = None
            self.current_phase = None
            self.drain_aborted()

    def finish_request(self, seq_id: int, *, aborted: bool = False) -> None:
        with self._abort_lock:
            aborted = aborted or seq_id in self._aborted
            self._aborted.discard(seq_id)
        traces = self.requests.pop(seq_id, None)
        if traces is None:
            return
        if aborted:
            self.stats["aborted_requests"] += 1
            return
        for phase in ("prefill", "decode"):
            self.collection.insert(traces[phase], phase)
        self.stats["completed_requests"] += 1

    def snapshot_stats(self) -> dict:
        return {**self.stats, "eamc_entries": len(self.collection.entries),
                "eamc_replacements": self.collection.replacements,
                "bpf_match_calls": self.collection.match_calls,
                "match_mismatches": self.collection.match_mismatches,
                "active_request_traces": len(self.requests)}
