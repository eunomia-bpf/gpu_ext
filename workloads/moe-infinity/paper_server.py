"""Same GPT-OSS serving frontend, with the explicit paper-v3 policy port."""

from __future__ import annotations

import os
import threading

from paper_policy import ActivationPolicy, JitMatcher, JitRanker
from moe_infinity.entrypoints.openai import revision_server
from moe_infinity.entrypoints.openai import api_server_v2 as server
from moe_infinity.serving.engine import ContinuousBatchingEngine


class PaperEngine(ContinuousBatchingEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._revision_abort_lock = threading.Lock()
        self._revision_aborts = set()
        mode = os.environ.get("MOE_REVISION_POLICY", "native-off")
        if mode not in {"native-off", "paper-native", "paper-bpf"}:
            raise ValueError("unknown MOE_REVISION_POLICY")
        self.revision_activation = None
        if mode == "native-off":
            return
        if self.speculative_draft is not None:
            raise ValueError("paper policy port requires non-speculative serving")
        if self.scheduler.max_batch_size != 1:
            raise ValueError("paper policy port currently requires max_batch_size=1")
        executor = self.engine.expert_executor
        if executor.prefetcher is not None:
            raise ValueError("cannot combine paper policy with the old prefetch engine")
        layers = self.engine.expert_predictor.num_layers
        experts = self.engine.expert_predictor.num_experts
        verify = os.environ.get("MOE_REVISION_VERIFY", "0") == "1"
        library = os.environ.get("MOE_EXPERT_POLICY_LIBRARY", "")
        ranker = JitRanker(library, os.environ["MOE_EXPERT_RANK_CODE"]) if mode == "paper-bpf" else None
        matcher = JitMatcher(library, os.environ["MOE_EXPERT_MATCH_CODE"]) if mode == "paper-bpf" else None
        self.revision_activation = ActivationPolicy(layers, experts, ranker=ranker, matcher=matcher,
                                                    verify_rank=verify)
        self.engine.expert_dispatcher.configure_activation_policy(
            2 if mode == "paper-bpf" else 1, library,
            os.environ.get("MOE_EXPERT_SCORED_CODE", ""), verify)
        executor.revision_activation = self.revision_activation

    def _execute_batch(self, batch):
        activation = self.revision_activation
        if activation is None:
            return super()._execute_batch(batch)
        if len(batch.seq_ids) != 1 or len(batch.is_prefill) != 1:
            raise ValueError("paper activation traces require exactly one sequence")
        activation.begin_iteration(batch.seq_ids[0], batch.is_prefill[0])
        try:
            result = super()._execute_batch(batch)
        except BaseException:
            activation.end_iteration(success=False)
            raise
        activation.end_iteration()
        return result

    def step(self):
        with self._revision_abort_lock:
            aborts = self._revision_aborts
            self._revision_aborts = set()
        for request_id in aborts:
            super().abort_request(request_id)
        if self.revision_activation:
            self.revision_activation.drain_aborted()
        outputs = super().step()
        if self.revision_activation:
            for output in outputs:
                if output.finished:
                    self.revision_activation.finish_request(output.seq_id)
            if any(output.finished for output in outputs):
                self.engine.expert_dispatcher.drain_activation_prefetch()
        return outputs

    def abort_request(self, request_id):
        seq_ids = list(self._request_to_seq_ids.get(request_id, []))
        if self.revision_activation:
            for seq_id in seq_ids:
                self.revision_activation.mark_aborted(seq_id)
        # The API thread only marks; the execution owner removes native sequence
        # state on its next step, never while a forward has released the GIL.
        with self._revision_abort_lock:
            self._revision_aborts.add(request_id)


server.ContinuousBatchingEngine = PaperEngine


@server.app.get("/revision/activation")
async def activation_stats():
    runtime = server.engine
    if runtime is None:
        return revision_server._unavailable("Service is starting")
    activation = runtime.revision_activation
    return {"algorithm": "arxiv-2401.14361v3-reimplementation",
            "mode": os.environ.get("MOE_REVISION_POLICY", "native-off"),
            "features": "shared-float64-EAMC-cosine-and-probability",
            "controller": activation.snapshot_stats() if activation else {},
            "dispatcher": runtime.engine.expert_dispatcher.get_activation_stats()}


@server.app.post("/revision/activation/drain")
async def drain_activation():
    runtime = server.engine
    if runtime is None:
        return revision_server._unavailable("Service is starting")
    runtime.engine.expert_dispatcher.drain_activation_prefetch()
    return await activation_stats()


if __name__ == "__main__":
    revision_server.main()
