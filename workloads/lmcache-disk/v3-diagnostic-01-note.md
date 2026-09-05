# V3 warm-divergence diagnostic-01 result note

One-prefix diagnostic (`raw/storage-575-v3-diagnostic-01`, prefix 0, 1550
warm input tokens, one generated token, top-20 logprobs retained). Arms:
`lmcache_cpu` (LMCache v0.5.4 with the V3 GPU connector, local CPU storage)
and `native_prefix` (native vLLM prefix caching, LMCache not engaged).
Frozen protocol: `max_tokens=1` for this diagnostic only, `temperature=0`,
`seed=0`, `return_token_ids=true`; streamed `choice.token_ids` and
`choice.text` were parsed.

Result:

- Engagement was exact in both arms: `cached_tokens=0` on the cold request
  and `cached_tokens=1536` on the warm request
  (`usage.prompt_tokens_details`).
- Both arms generated the same token (ID 2303, text `"  \n"`) for the cold
  and the warm request.
- Cross-arm top-20 logprob deltas on common tokens were 0.0 for both the
  cold and the warm request; within-arm cold-versus-warm deltas were at
  most 0.189 on common tokens.
- The `lmcache_cpu` arm logged six double-unpin warnings on the warm
  request; see `double-unpin-analysis.md`. Outputs still matched.

Conclusion: the one-token diagnostic did not reproduce the 16-token warm
divergence observed in `raw/storage-575-v3-correctness-02`, where only the
`recompute` arm diverged from the two cached arms on five of eight
prefixes. The eight-prefix protocol in
`run_native_prefix_correctness.py` is the formal re-check: the frozen
prompts.json pairs, `max_tokens=16`, `temperature=0`, `seed=0`,
`return_token_ids=true`, exact streamed `choice.token_ids` plus text for
every cold and warm request across `native_prefix`, `lmcache_cpu`, and
`lmcache_disk`; native engagement is gated on `cached_tokens=0/1536` per
prefix, and results whose responses lack token IDs are rejected. The
`recompute` arm remains a separate performance arm and is not part of this
correctness comparison.

## Future GPU command

On the 575.57.08 host, when the GPU is idle and this workload owns it:

    cd workloads/lmcache-disk
    PY=current-venv/bin/python
    $PY run_lmcache_disk.py run-cell --config lmcache_cpu \
        --output raw/storage-575-v3-correctness-03/lmcache_cpu \
        --expected-driver 575.57.08
    $PY run_lmcache_disk.py run-cell --config lmcache_disk \
        --output raw/storage-575-v3-correctness-03/lmcache_disk \
        --expected-driver 575.57.08
    $PY run_native_prefix_correctness.py run \
        --output raw/storage-575-v3-correctness-03/native_prefix \
        --expected-driver 575.57.08
    $PY run_native_prefix_correctness.py validate \
        raw/storage-575-v3-correctness-03/native_prefix
    $PY run_native_prefix_correctness.py compare \
        raw/storage-575-v3-correctness-03/native_prefix \
        raw/storage-575-v3-correctness-03/lmcache_cpu \
        raw/storage-575-v3-correctness-03/lmcache_disk
    $PY test_runner.py
    $PY test_native_prefix_correctness.py
    $PY test_v3_warm_diagnostic.py
    $PY test_batch_575.py

These newly run cells record `generated_token_ids` for every request;
legacy cells such as `storage-575-v3-correctness-02` do not and are
rejected by `compare` with a missing-token-ID error. No commit/push is
part of this step.
