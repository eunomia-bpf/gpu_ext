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
