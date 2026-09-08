# Automatic warp: remove the redundant unconditional ballot

Status: complete, 10 rotated blocks / 30 applications / 20 attached loaders.
All applications and loaders exit zero. Date: 2026-09-08, about 10:00 UTC.
This is a new implementation variant, not a rerun or replacement of the
[completed original campaign](../results-automatic-warp-shared-10blocks-20260908.ORjHAp/README.md).

## Change and result

Runtime `bpftime` commit **0c3b6d1** replaces the unconditional
`vote.ballot.sync(true, active_mask)` with the active mask itself. It removes
the redundant true predicate and ballot. Predicated hook sites are unchanged;
no lane-zero assumption, manual probe guard, event suppression or new map
contract is introduced. Root made this single bounded peephole under the
user's exception for simple edits; OpenCode retains the nontrivial batching
implementation. Both runtime library targets built successfully before timing.

The same original 14-instruction `cuda__shared` object is used in both
instrumented arms. Each cell has one 128-thread block per launch, eight
warmups and 128 measured launches. The metric is the CUDA-event elapsed
milliseconds over all 128 launches, excluding initial compilation/attachment
and final map readback.

| Configuration | Samples | Median elapsed ms | Minimum–maximum ms |
| --- | ---: | ---: | ---: |
| Native application | 10 | 0.2183040005 | 0.196447998–0.259615988 |
| Same BPF probe, automatic warp off | 10 | 0.3285920020 | 0.327039987–0.331488013 |
| Same BPF probe, automatic warp on | 10 | 0.5057600140 | 0.481375992–0.514400005 |

On/off within-block percent change has mean **+53.1303%**, median +54.0858%,
range +45.2167% to +55.8712%, and 95% bootstrap interval for the mean
**[+51.0982%, +54.5791%]**. Every pair remains adverse. The low on-arm sample
in block 8 is retained. Off/native paired mean change is +46.9751%
(95% interval +36.6210% to +57.1195%); on/native is +124.8700%
(+109.9809% to +139.8294%). These are not ratios of arm medians.

The earlier implementation measured +57.2860% paired on/off cost. This new
campaign has a lower point estimate, but the implementations were not
interleaved within one campaign, so the cross-campaign difference is not an
isolated causal measurement of the removed ballot. More importantly, this
peephole does **not** remove the substantial slowdown of automatic leader
execution on this probe. This is not a Table 1 throughput result, a generic
speedup, or completion of event batching.

## Execution and retained evidence

Protocol, original executable commands, hardware, environment, map and
rotation match the original campaign linked above. GPU: RTX 5090; driver:
575.57.08; CUDA: 12.9; LLVM: 15. Runtime libraries come from
`bpftime-auto-warp/build-auto-warp-575`, with verifier enabled and the same
build-only `<cstdint>` compatibility option. Native/off/on orders rotate
across ten blocks. Run IDs start at 91003 and are individually recorded in
`cells.csv`; fresh controls are collected for this new variant. No prior
cell or historical number is overwritten or pooled into these blocks.

All ten on arms report automatic eligibility and successful patched-module
loading. Every off/on loader reports key 0 = 6291613622346973184. Dynamic
scalar execution counts are not measured here. No custom correctness,
preflight, clock-precision or acceptance campaign was added.

Root held the GPU lock through the campaign and released it afterward.
Loader shutdown/readback occurred normally after each application completed;
no timeout expired. All 20 task-owned private 256 MiB transport segments
were deleted after their loaders completed. The data here consist of small
logs, the readbacks, all 30 rows in `cells.csv`, and `analysis.json`; no large
buffer or runtime binary is committed. Runtime-internal cache-key text is
omitted from logs without changing any timing, error or output value.

Analysis uses the same within-block ratios and 100000 whole-block bootstrap
resamples as the original campaign, seed 1797, with the PRNG and percentile
method recorded in JSON. All observations are included. The pending work
remains record-preserving output batching, Table 1's same-object comparison,
and the original device block/work/execution-count sweeps.
