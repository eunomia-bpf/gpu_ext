# DALI-inspired WARC selector port (CPU-only)

This workload is a port of the Workload-Aware cache Replacement (WARC)
cache-replacement policy from DALI (arXiv:2602.03495v1, Section 4.3 and
Algorithm 2), exercised through this repository's gpubpf selector stack,
plus replay of the frozen 35,360-step real routing trace
(`workloads/expert-buffering-policy/raw/timing/block-01/llama_ncmoe32/
trace.jsonl`).

## Scope

- **What is ported:** the per-MoE-layer expert-cache replacement decision
  only — a fixed-size resident expert set per layer, a sliding
  workload-score window of `w_size` recent steps, and a symmetric swap of
  `u_size` experts at each window boundary (highest-scoring non-resident
  experts loaded, lowest-scoring resident experts evicted, deterministic
  lower-ID tie-breaks). Both selector arms and the trace replay run the
  35,360-step frozen trace policy.
- **What is not ported:** this is not the whole DALI system. There is no
  greedy CPU/GPU token-to-device assignment, no residual-based
  prefetching, and no multi-GPU scheduling.
- **Not a GPU-throughput reproduction:** everything is CPU-only, with no
  GPU and no driver. Reported numbers are policy-level metrics (decision
  hit rates, transferred bytes, per-step decision overhead), not
  end-to-end GPU throughput, latency, or energy.

## Selector arms

One shared library (`libdali_policy.so`) exports two arms over the shared
decision model (`dali_policy_model.h`):

- `native`: selection executed in C (the reference oracle).
- `bpf`: the same policy with every top-k/bottom-k selection executed by
  the existing gpubpf rank selector (`extension/.output/
  libmoe_expert_policy.so` + `moe_expert_policy_rank.bin`, host uBPF
  JIT). No new BPF program and no native fallback: any selector failure
  is fatal to the step (fail-closed), matching the component's
  contract.

The two arms must reach identical decisions on identical inputs.

## Validation

```
make -C workloads/dali-policy-port clean all test
```

CPU-only; reuses the prebuilt `extension/.output` selector artifacts and
never rebuilds them. `test_dali_policy` runs:

1. native-oracle unit checks (deterministic tie-breaks, contract
   rejections);
2. state invariants over 500 random steps (resident-set size,
   hit/miss partition, request counter);
3. fail-closed checks (bad library/bytecode rejected, null state
   rejected, native arm exposes no BPF counters);
4. differential test: all 48 parameter cells (cache size
   {8,16,32,64} x window {1,2,4,8} x swap size {1,2,4}) x 1,200
   steps x 4 layers of random routing, native vs bpf in lockstep,
   zero resident-set mismatches and identical cumulative metrics per
   cell;
5. trace comparison: the 35,360-step frozen trace (17 requests,
   213,986 routed events) replayed in lockstep on both arms, zero
   resident-set mismatches and identical cumulative metrics.

Gates are limited to decision agreement (zero native-vs-bpf
mismatches), contract/fail-closed behavior, and state invariants.
Selector engagement counts (`bpf_rank_calls`, `bpf_errors`,
`selector_errors`) and performance metrics (hit rates, transfer bytes,
decision overhead) are retained and printed as metadata only; they are
never a gate, retry, filter, or result rejection. Note that
`bpf_errors` is the gpubpf library's process-global counter, so it also
reflects the intentional fail-closed negative test in step 3.

## Layout

- `dali_policy_abi.h` — public C ABI of the two arms and stats.
- `dali_policy_model.h` — shared WARC decision model + native oracle.
- `dali_policy.c` — the two arms (native; gpubpf rank selector).
- `test_dali_policy.c` — CPU-only validation and metadata reporting.
- `Makefile` — build plus `make test`.
- `hooks/`, `raw/` — reserved, currently empty.
