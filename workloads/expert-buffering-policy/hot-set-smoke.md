# Hot-set compiler smoke

Date: 2026-08-31

This is an implementation smoke, not the frozen eight-prompt calibration or a
performance sample.

The compiler consumed the exact-model marker trace described in
`exact-model-smoke.md`. It joined routes to source tensor bases, required equal
selected-expert sets across gate/down/up weights, deduplicated those three
observations per graph, and covered all 36 layers.

The one-graph trace had only four selected experts in layer 35, so the requested
top-ten compilation correctly failed closed. With `top-k=4` solely to exercise
the encoding path, compilation succeeded with 36 layers and 2,916 route events.
The resulting hot set then compiled against all 216 source layouts into 29,620
spanned two-MiB blocks: 27,737 cold, 1,259 hot, 189 shared, and 435 unclassified
gap blocks. The hot-overlap footprint was 2,678,063,104 bytes.

Disposition: **PASS** for route/layout joining, per-graph deduplication,
cross-weight consistency, deterministic ranking, source-layout filtering, and
block-class encoding. The eight frozen calibration prompts are still required
before a top-ten hot set can be admitted for the real policy experiment.

No file/content hashes, checksums, digests, or fingerprints were generated or
used.
