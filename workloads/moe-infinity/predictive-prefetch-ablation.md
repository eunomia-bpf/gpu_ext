# Predictive-prefetch causal ablation harness

Status: **implemented and CPU-tested; no GPU result is claimed**.  The currently
installed `_store` extension still exposes the old four-argument activation
configuration ABI, so real preflight is intentionally fail-closed until the
patched extension is rebuilt.  Admission inspects the ordinary demangled
dynamic-symbol inventory and requires the new five-argument ABI; it does not
use a content hash.  The harness
exists to answer the unresolved attribution in
`results-paper-v3-protected-575.md`: the earlier `native-off` baseline used a
different executor and could take a temporary prefill overload slot, so its
throughput advantage must not be presented as evidence against predictive
prefetch.

## Hypothesis and factorial comparison

The tested hypothesis is that, under the same activation-aware executor, cache
budget, strict no-overload rule, demand-eviction eligibility, requests, and
correctness oracle, issuing the paper-v3 predictive prefetches changes serving
throughput and cache behavior.  The four arms are:

| Arm | Demand eviction selector | Predict/match/rank | Speculative issue |
| --- | --- | --- | --- |
| `native-prefetch-off` | native scored selector | enabled | disabled |
| `native-prefetch-on` | native scored selector | enabled | enabled |
| `bpf-prefetch-off` | real host-uBPF JIT selector | real host-uBPF match/rank | disabled |
| `bpf-prefetch-on` | real host-uBPF JIT selector | real host-uBPF match/rank | enabled |

The off arms intentionally retain prediction, cosine matching, ranking, and
score installation.  The C++ dispatcher suppresses only speculative queue
publication/copy.  This keeps control-path work common within each on/off pair
and makes the treatment the prefetch issue/execution path.  All arms configure
the paper dispatcher, whose full-cache demand path evicts a scored safe resident
instead of entering the upstream prefill overload branch.  Therefore the old
`native-off` measurements are not cells in this experiment.

Five fixed-seed randomized paired blocks each contain all four arms.  Every cell
uses one excluded warmup followed by six measured 512-input/64-output SSE
requests.  The full matrix is exactly 20 cells, 120 measured requests, and
7,680 exact-checked output tokens.  Primary throughput is 384 output tokens
divided by the complete six-request wall window including the final speculative
drain.  Paired block-bootstrap intervals report both prefetch effects and the
BPF/native ratios at prefetch off and on.

## Runtime evidence and fail-closed gates

The rebuilt dispatcher reports the following measured-window observations:

- prefill and decode demand accesses, hits, and misses;
- demand-copy count/bytes and successful demand evictions;
- speculative copy count/bytes, successful speculative evictions, first-use
  hit count/bytes, unused-before-use eviction count/bytes, and unused resident
  count/bytes after drain;
- demand waiting for an in-flight prefetch, demand waiting for an eligible
  cache victim, and the host time spent synchronizing completed prefetch copy
  events;
- temporary-slot uses/bytes/waits, split native/BPF demand-versus-prefetch
  eviction calls, cache budget, and the active prefetch toggle.

Every cell fails unless both phase access identities conserve, demand copies and
demand evictions engage, the cache budget remains positive and unchanged, the
temporary slot is disabled and unused, and outputs exactly match retained
same-frontend goldens.  Both BPF arms additionally require positive
measured-window JIT match calls, rank calls, and **demand-eviction** calls; warmup
engagement cannot satisfy the delta.  Prefetch-off rejects any submitted,
started, completed, resident, protected, or eviction-side speculative work.
Prefetch-on requires issued/completed copies and count/byte conservation:

```text
completed = first-use hits + unused-before-use evictions + unused residents
prefetch bytes = hit bytes + unused-eviction bytes + unused-resident bytes
```

Wait durations are host `steady_clock` observations around the actual blocking
loops.  `prefetch_copy_wait_ns` covers `cudaEventSynchronize` in the fetch
worker.  `demand_prefetch_wait_ns` covers demand waiting to claim a node whose
prefetch is in flight, and `demand_cache_wait_ns` covers retry time when no safe
victim can immediately be committed.  They are not PCIe analyzer measurements
or GPU-kernel stall attribution.  Logical byte counters use each expert node's
payload size and may count repeated transfers.

For this frozen single-sequence workload, dispatcher phase attribution uses the
actual expert input row count: 512-token prefill has more than one row and every
decode step has one.  The server already rejects batching and speculative
decoding.  This counter definition is not claimed to generalize to one-token
prefill or batched mixed-phase serving.

## Required real preflight and commands

The full run cannot start without a separately completed four-arm real
preflight from the identical runtime inventory.  Preflight starts a fresh real
server per arm, performs the same excluded warmup, runs one full 512+64 request,
checks its exact output, drains speculation, and applies all mechanism gates
above with an expected measured-request delta of one.  `run` rereads the raw
launch, admission, SSE, activation snapshots, golden text, and runtime inventory;
a producer `passed` flag alone is insufficient.

CPU-only matrix inspection:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B run_prefetch_ablation.py dry-run
```

Real preflight and full run, only after the coordinator grants the GPU slot:

```bash
.venv/bin/python -B run_prefetch_ablation.py preflight \
  --output raw/prefetch-ablation-575/preflight-01 \
  --driver-stage /opt/gpubpf/modules/575.57.08/gpreempt-<stage>-6.15.11

.venv/bin/python -B run_prefetch_ablation.py run \
  --preflight raw/prefetch-ablation-575/preflight-01 \
  --output raw/prefetch-ablation-575/timing-01 \
  --driver-stage /opt/gpubpf/modules/575.57.08/gpreempt-<stage>-6.15.11
```

This implementation has not built or loaded the modified CUDA extension and has
not executed either real command.  CPU source-wiring and synthetic accounting
tests are dependency evidence only.  A paper result requires the real preflight,
all five complete paired blocks, raw review, exact correctness, and the measured
BPF/copy/eviction engagement described above.
