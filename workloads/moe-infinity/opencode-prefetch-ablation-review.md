# OpenCode review: predictive-prefetch ablation

- Reviewer: OpenCode `opencode/ling-3.0-flash-fin-free`
- Session: `ses_f9523d139ffe8U72aUFJJd8W8c`
- Mode: read-only; writing, shell execution, delegation, and network access disabled
- Verdict: **PASS**; no blocker

The review covered the four-arm runner, independent raw audit, dispatcher
patch, source wiring, and tests. It accepted the matched native/BPF eviction by
prefetch-off/on design and its correctness and mechanism-engagement checks.
The experiment keeps host wait counters scoped as host observations rather than
GPU-stall attribution and does not reuse the earlier confounded `native-off`
row.

The installed `_store` extension still exposes the old four-argument
configuration ABI. The runner now detects that ordinary dynamic-symbol state
and fails before a real campaign; the patched five-argument extension must be
rebuilt before preflight. Local verification passed 92 focused host tests, 12
fresh-process analyzer tests, Python compilation, the fixed 20-cell dry run,
and the reverse patch-application check. No GPU cell was run during review.
