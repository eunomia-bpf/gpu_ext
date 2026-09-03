# MoE expert decisions in userspace eBPF

This component executes actual bpftime/ubpf JIT bytecode on CPU. It neither
replaces the CUDA executor nor implements a GPU-page LFU policy. Model inference,
prediction tensor arithmetic, expert transfers, synchronization, and cache
ownership remain in the common MoE frontend. No GPU or driver attachment is
needed for the component's CPU tests.

## Current-source eviction component

The source reference is MoE-Infinity commit `b766f8f`,
`core/parallel/expert_dispatcher.cpp:379`, `FindExpertEvict`, with the
`uint64_t incache_visit_count` field at `core/model/model_topology.h:66`.
This selector is **not the complete activation-aware algorithm from paper v3**.
It is retained as an explicitly named current-source component, not relabeled
as the paper's probability-scored eviction.

Preserved semantics:

- Traverse the actual `cached_experts_[gpu]` unordered-set iteration order.
- Skip null nodes, non-CUDA nodes, pending dispatches, and non-IDLE nodes.
- Start the unsigned minimum at `INT_MAX`, not `UINT64_MAX`.
- Use strict `<`: retain the first tie; counts at or above `INT_MAX` never win.
- Do not change where the frontend increments `incache_visit_count`.

The C ABI in `moe_expert_policy.h` consumes 24-byte entries containing the
original identity, count, and four separately captured eligibility facts.
`moe_expert_policy_select_v1` returns an input index or `MOE_EXPERT_NONE` on
success. Any negative function return is fatal; there is no native fallback.
`moe_expert_policy_init_v1` accepts an absolute bytecode path; a null path reads
`MOE_EXPERT_POLICY_CODE`. The process cannot replace the initialized program.
The bridge emits an explicit JIT-ready event and exposes/emits call, candidate,
selection, no-victim, and error counters. Counters are engagement evidence,
not proof that the engine actually evicted or transferred an expert.

Both native and BPF comparison arms must consume the **same immutable snapshot**.
This preserves the selector's decisions on captured inputs; it does not prove
equivalence to unconstrained concurrent reads at different instants. The original
engine remains responsible for validating/lifetiming the selected expert and
performing its original transfer and cache update. The BPF snapshot is bounded
at 65,536 candidates and uses per-thread storage.

Build and test, reusing existing bpftime VM archives without rebuilding them:

```sh
taskset -c 8-15 make -C extension -f moe_expert_policy.mk -j2 all test
```

Artifacts: `extension/.output/libmoe_expert_policy.so` and
`extension/.output/moe_expert_policy.bin`. The standalone Makefile selects the
real `/usr/bin/g++-13`, not the local `c++` wrapper.

CPU validation completed: **90,100 original-transcription/JIT comparisons,
zero mismatches**, including 4 concurrent threads, 65,536 candidates, all
eligibility combinations, unsigned count boundaries, null nodes, and ties.
Nine invalid initialization/input cases were rejected. The independent oracle
reads raw mock nodes and atomics rather than reusing the BPF eligibility predicate.
This is component validation, not an end-to-end MoE performance result or a full
paper reproduction.

## Paper-v3 scored eviction and prefetch ordering

The activation-aware frontend port follows the separate paper-v3 implementation
in `workloads/moe-infinity/paper_policy.py`; see its accompanying
`activation-aware-port.md` for mathematical conventions and deviations from
the dormant upstream predictor. The policy source is
[MoE-Infinity, arXiv v3](https://arxiv.org/abs/2401.14361v3), Algorithm 1 and the
prefetch discussion. The current-source count selector above is not substituted
for the paper's probability-scored eviction.

The common frontend computes cosine similarities, selected-trace aggregation,
probabilities, and layer-decayed scores. It must hand the native and BPF selectors
the **same float64 bit patterns**, without converting them to float32. The C++
bridge does not calculate scores, filter candidates, or sort them. It copies
unsorted entries into bounded memory and calls the selected JIT program.

- `moe_expert_scored_select_v1`: choose the first eligible strictly smaller
  score, with initial minimum positive infinity. Four eligibility facts and
  input order are the same as the count selector. NaNs and positive infinity
  cannot win; negative infinity can; positive and negative zero compare equal.
- `moe_expert_rank_v1`: filter `score > 0`, retaining positive infinity and
  excluding all NaNs and nonpositive values; perform stable descending merge
  sort **inside BPF**. Equal scores retain the original unfiltered, layer-major
  input order. Each entry's ordinal must equal its input position. Return input
  indices, leaving identity mapping and prefetch enqueueing to the frontend.

Both BPF programs implement IEEE-754 float64 comparisons using integer bit
ordering, avoiding unsupported eBPF floating-point operations. Rank uses bounded
caller-owned scratch and O(n log n) work, not a C pre-sort or quadratic insertion
sort. Its bridge requires output capacity at least the input count, exposes
`moe_expert_rank_stats_v1`, and separately counts ranked indices and empty calls.
The scored selector has its own `moe_expert_scored_stats_v1` counters. Ready/error
and process-exit log records include `kind=paper_scored` or `kind=paper_rank`.

The existing SO exports all component interfaces. Initialize scored and
rank programs with absolute paths to these additional build outputs:

```text
extension/.output/moe_expert_policy_scored.bin
extension/.output/moe_expert_policy_rank.bin
```

The corresponding null-path environment variables are `MOE_EXPERT_SCORED_CODE`
and `MOE_EXPERT_RANK_CODE`. Missing programs and malformed snapshots fail closed;
there is no C decision fallback. The header gives the full 24-byte entry layouts
and separate output types.

CPU validation completed for each new component: **40,871 scored selections and
40,871 rankings, zero mismatches**, producing **2,376,217 ranked indices**.
The independent oracle uses native double comparisons and `std::stable_sort`,
not the BPF integer encoding or merge algorithm. Tests cover positive/negative
infinities, quiet and signaling NaNs, both zeros, subnormals, adjacent doubles
that would collapse under float32, all eligibility combinations, random IEEE
encodings, 65,536-candidate all-tie and reordered cases, and four concurrent
threads. Fourteen invalid input/initialization cases are rejected. Output-array
sentinels verify that the bridge does not overwrite the caller's boundaries.

This validates the discrete score-selection components against a native
specification. It does **not** by itself validate prediction math, EAMC lifecycle,
model correctness, actual prefetch/eviction actuation, or performance.

## EAMC nearest-trace and replacement selection

`moe_expert_match_v1` completes the discrete nearest-neighbor decision using
the common frontend's **unsorted float64 cosine similarities**, preserving
original EAMC entry order. Its BPF program finds the numeric maximum and returns
all equal indices in a single pass. The prediction frontend can aggregate all
returned ties; a full EAMC can replace the first returned index. Computing cosine
similarities and modifying the trace collection remain common frontend work.

The API reuses the rank entry layout, input ordinals, and output capacity rules.
Initialize with `extension/.output/moe_expert_policy_match.bin`, or set absolute
`MOE_EXPERT_MATCH_CODE` for a null init path. `moe_expert_match_stats_v1` has
independent `calls/candidates/matched/empty/errors` counters; log records use
`kind=paper_match`. It starts at negative infinity, ignores NaNs, accepts positive
infinity, treats both zeros as equal, returns all entries for an all-negative-
infinity input, and returns empty for empty/all-NaN input. Actual finite,
nonnegative cosine features satisfy a stricter frontend validation contract;
the broader comparison cases are explicit component tests, not expected traces.

The combined CPU suite now passes **40,871 match snapshots with 225,449 matched
indices**, in addition to the scored/rank counts above, with zero mismatches.
The three scored/rank/match interfaces reject 22 invalid initialization/input
cases in total. All component tests remain CPU-only and exercise the real JIT.

A concurrent frontend CPU test observed `dlopen: file too short` while the first
match-enabled SO was being linked in place; it failed closed, without fallback.
The standalone build now links to a temporary output and atomically renames the
successfully linked SO, keeping the previously published SO intact during future
builds. This build-time failure is not a model correctness or performance result.
