# Expert route calibration

Date: 2026-08-31

Status: **PASS** for freezing the top-ten expert set and its page-level
protection budget. This is setup evidence, not a performance result.

Eight ShareGPT prompts were selected with seed 1796 before execution. They are
disjoint from the nine prompts in the MoE evaluation workload, round-trip to
the same 512 token IDs in the pinned HF and GGUF tokenizers, and are recorded in
`calibration-prompts.json`.

One marker-enabled GPT-OSS-120B MXFP4 server ran all eight requests sequentially
with `--n-cpu-moe 36`. Seven requests reached the 64-token limit. One emitted a
model EOG after one completion token; it is retained exactly as observed. The
request artifact therefore records 4,096 prompt tokens and 449 completion
tokens rather than claiming 512 completions.

The trace recorded 216 source layouts, 457 graph ordinals, 305,547 route events,
and zero dropped events. Of these graphs, 449 exposed all 36 layers. The first
prompt-evaluation graph of each request exposed layers 0--34, while its
following graph exposed layer 35; no missing route is fabricated. Each
observed `(graph, layer)` had equal selected-expert sets for gate, down, and up
weights. Per-layer ranking counts each expert at most once per graph and breaks
ties by ascending expert ID. Every layer produced ten admitted experts.

The frozen hot set compiles against all 216 source tensors into a span of 29,620
two-MiB blocks. Of those, 25,725 are cold, 3,271 overlap at least one hot expert,
189 are shared/boundary blocks, and 435 are unclassified gaps. The hot overlap
is 6,901,727,232 bytes, below the fixed 8 GiB admission limit.

The address-bearing class table is intentionally regenerated while each policy
process is alive because model virtual addresses change between runs. It is not
reused from this completed calibration process.

No file/content hashes, checksums, digests, or fingerprints were generated or
used.
