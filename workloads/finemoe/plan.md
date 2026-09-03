# FineMoE dynamic prefetch sets on the official Qwen execution path

Status: **all 20 formal cells / five paired blocks completed and independently
audited**. See [the performance report](results-performance.md): dynamic sets
reduce evicted-unused payload versus all-positive, but demand-only is faster;
the BPF/C throughput difference is unresolved.
This is a FineMoE **dynamic-set component experiment**, not a reproduction of
the entire EuroSys evaluation or a renamed EAMC activation-count heuristic.

The following preserves preparation history; completed stages are not rerun.
GPU work was coordinated by the root lease.
The original Qwen checkpoint is downloaded (exact source revision and 16-file
size inventory in `source-inventory.json`), the official extension compiles, and
the private Transformers 4.49 overlay imports the compiled module. Final CPU
build05 passed: `build/offload-build-05.log` is 69,335 bytes; the private offload
extension is 58,219,416 bytes and advertises runtime revision
`dynamic-set-safety-20260903-v2`. Both standalone C++ safety/accounting tests and
all 35 Python protocol, author-method, native and actual-JIT tests passed.
Import/API checks confirmed CUDA was not initialized. Replayable
`dynamic-set.patch`, `common-runtime.patch`, `copy-accounting.patch`, and
`build-compat.patch` preserve the author source and common safety/instrumentation
changes. The first original-model golden, `raw/golden-v1`, failed during shard
loading with CUDA OOM before any request; cleanup passed. See
`results-preparation.md`. All stages/arms now use the same
`PYTORCH_ALLOC_CONF=expandable_segments:True` setting and remove an inherited
legacy `PYTORCH_CUDA_ALLOC_CONF` without logging its value. The experimental
allocator option is a proposed fragmentation remedy, not a demonstrated fix.
The v2 retry loaded all eight shards (27,925 MiB sampled peak) but ended with
SIGILL before any completed request. A golden-only native-stack diagnostic then
stopped at a SIGSEGV in libcuda's Triton module-load path; this is an observation,
not proof of the un-debugged SIGILL's root cause. RoPE's bmm is a source-backed
candidate, not a confirmed faulting operator. Before fresh normal golden-v3,
all stages/arms now set `TORCH_DISABLE_NATIVE_JIT=1` and the worker checks the
official `check_native_jit_disabled()` predicate, recording it in every raw
result's runtime state. This only disables PyTorch's automatic Triton/DSL operator
overrides, not the independently linked uBPF selector JIT or its engagement gate.
No Torch source, model mathematics, precision, cohort, budget or tolerance changed.
**The real history store and four-arm numerical preflight passed before formal
timing; they are not counted as performance blocks.**
Normal golden-v3 has now passed 73 requests and nine repeated-output checks;
its fixed reported absolute tolerance is 0.0 and teardown passed. It is retained
unchanged as preparation, but its repeat logits were not separately saved.
Final golden-v4 has also passed 73 originals + nine repeats, retaining both sets
of arrays. Independent CPU recomputation found all nine actual maximum errors
equal to 0.0 across 43,757,568 finite float32 values (18 arrays); the frozen
absolute tolerance remains 0.0. Raw token/timing checks, unchanged runtime
inventory and cleanup passed; sampled peaks were 28,041 MiB and 50 C. V4 is the
final reference, not an offload-policy performance result. The full 64/8/1 cohort
and all calculations stay fixed. See `results-preparation.md` for audit scope.
The first actual offload history attempt, `raw/history-v1`, then failed the exact
token gate on its first question (141), before any accepted history request or
store export; cleanup passed. Its cause is not established. A storage-only patch
now retains history actual/expected token records and preflight actual logits
before the existing gates, without changing tolerance or adding formal timing
writes. The storage-only patch passed all 46 Python regression tests.
The retention-only history-v2 retry then exposed a single token difference at
generated position 14. CPU checks confirmed the author loader omitted the
checkpoint generation configuration (repetition penalty 1.0 instead of HF's
1.05, plus different EOS defaults). `create_finemoe` now loads that complete local
configuration for all offload arms, preserving explicit greedy/16-token settings,
golden-v4 and tolerance 0.0. All 48 Python tests passed. History-v3 then passed all
64 requests (1,024 exact HF tokens), including question 141, with independent
request/cohort/configuration/cleanup checks. Its two actual float32 store arrays,
`(1000, 2048)` embeddings and `(1000, 24, 60)` full probabilities, contain 3,488,000
finite values and populate the full 1,000-entry capacity. History inputs remain
disjoint from held-out/warmup inputs. The observed token failure is resolved;
full-logit equivalence at unchanged tolerance 0.0 remains a four-arm preflight
gate, not a conclusion from history.
The first numerical preflight stopped on BPF warmup question 137: exact tokens,
but 155 prefill-only logit values differed (max 0.03125); all 15 later decode
steps were exactly equal. The zero-tolerance failure is retained. The author's
head computed all prefill positions while HF 4.49 requested only the last.
`common-runtime.patch` now aligns the `logits_to_keep` signature, propagation and
pre-head slice, retaining default-zero full-logit semantics. All 51 CPU tests
passed; no policy, tolerance or compiled extension changed. Fresh history-v4
then passed all 64 exact-token requests, full-store and independent cleanup
checks (18,399 MiB sampled peak). Four-arm preflight-v2 then passed: all 36 saved
actual arrays (87,515,136 finite FP32 values) were exactly equal to golden-v4 at
tolerance 0.0, and all 576 generated tokens matched. The observed prefill
discrepancy is eliminated on this cohort. C/BPF's complete 122,695-event traces
matched; actual JIT, downstream admission, copy accounting, common budget and
cleanup all passed. These numerical canaries are not performance measurements.

## Research Question

RQ1, verbatim from `docs/paper/tex/eval.tex`: “How much performance gain can
\sys's programmable memory and scheduling policies provide on oversubscribed
single-tenant workloads?” The uncertainty is whether FineMoE's probability- and
confidence-dependent set can reduce real unused expert transfers while a
matched host-BPF implementation preserves its decisions, correctness, and
application performance. The existing GPT-OSS EAMC waste motivates this question
but is **not** a comparable baseline for Qwen.

## Paper-Value Admission

Supporting expressibility and performance evidence: a different published
policy, with full routing-probability and embedding inputs absent from the old
count-based EAMC bridge. The load-bearing objection is that moving a small
selector into BPF may either not control actual transfers or cost more than it
saves. Actual copies and end-to-end generation distinguish those explanations.
A win supports a bounded application-aware policy port, not kernel-UVM or full
FineMoE equivalence; a loss or mixed result remains evidence and limits the claim.
An EAMC top-K tweak is cheaper but does not test this published algorithm.

## Expected And Alternative Outcomes

Expected: dynamic sets admit fewer candidates and unused bytes than the matched
all-positive control; C and BPF agree. Competing explanations: demand misses
offset saved bytes, confidence is poorly calibrated, or shared Python search /
host-copy overhead dominates. Fewer selector outputs without fewer completed
copies is not success. No useful-byte or latency improvement is guaranteed.

## Published Precedent And Real Assets

- [FineMoE, EuroSys 2026, author PDF](https://intellisys.haow.us/assets/pdf/Hanfei_FineMoE_EuroSys26.pdf),
  local `docs/reference/2026-yu-finemoe.pdf` (1,335,925 bytes).
- [Official demo](https://github.com/IntelliSys-Lab/FineMoE-EuroSys26), ordinary
  upstream revision `5c584686da077676e3854a363832e9e7b973a054`, cloned in ignored
  `deps/FineMoE-EuroSys26/`. Reuse its Qwen forward, full router probabilities,
  original token embeddings, ExpertMapStore/search, and real expert loader.
- Original model `Qwen/Qwen1.5-MoE-A2.7B-Chat`: BF16, 24 layers, 60 experts/layer,
  K=4; official weight index declares 28,631,568,384 bytes across eight shards.
  The model is not replaced by a smaller one. The demo's >=48 GB recommendation
  is not proof that a smaller explicit expert cache cannot run on this 32 GB GPU.
- Paper PDF pp. 7–9 (§4.1–4.3, Eq. 6–8): full per-token L×J softmax probabilities;
  cosine semantic / observed-prefix trajectory match; delta=clip(1−score,0,1);
  descending-probability **smallest prefix whose sum >= delta and size >= K**.
  Empty history means no prediction; invalid inputs fail closed, demand remains live.
- Existing `../moe-infinity/paper_policy.py` consumes routed expert counts, not
  these inputs. Reuse its host-JIT linkage and accounting *patterns*, not its
  history, model, prediction scores, or past numerical goldens.

### Source audit and required scope

The inspected official source has two direct selector-to-executor defects:

1. `finemoe/runtime/model_offload.py:219` counts cumulative sums <= delta without
   adding the crossing expert; e.g. [.4,.3,.2,.1], delta=.8, K=1 selects .7 mass.
2. The same file, line 251, adds 1e-6 to **all** experts in target layers;
   `finemoe/memory/expert_prefetcher.py:74–98` then uses `priority > 0` and enqueues
   all of them. A selector-only unit test would miss this second defect.

Replayable `dynamic-set.patch` fixes these two, with direct corrected-Python
oracle tests through the real prefetcher's candidate/queue calls. Preserve the
unmodified upstream revision; do not present buggy execution as a strong baseline.

Other differences are disclosed, held common, and not silently called repaired:
the demo uses d=6 instead of the paper's profiled d=3, linear rather than reciprocal
distance priority (`model_offload.py:227–235`), synchronous Python search calls
from model forward, and integer-truncated probability×frequency eviction keys
(`core/prefetch/task_scheduler.cpp:291–303`). The first experiment freezes d=6
and the shared demo executor, so it isolates Eq. 6–8 rather than all §4.5 behavior.
The common runtime patch repairs current-sequence trajectory lookup: model lines
873–887 iterate **all retained traces**, while `finish_entry()` does not remove
them. It now looks up the current sequence IDs in batch order. Zero-routed expert
modules are skipped, so their offload hooks cannot load unused weights and count
them as demand. GPU numerical/IO preflight remains required.

The common executor also retains acquired-node lock ownership until release and
waits for the real Torch CUDA compute stream before unlocking a used node. Queued
tasks no longer unlock mutexes they do not own. The sole GPU copy worker checks
space immediately before H2D: speculative eviction protects the prediction set;
demand may evict a predicted but unlocked/non-executing node using the original
eviction ordering. If no space exists, demand fails explicitly, never silently
exceeds the 0.5 pool budget. The pool allocator enforces that bound as a final
guard. Record pool capacity/resident/peak bytes and total GPU HBM separately.
These are shared safety and budget repairs, not BPF algorithm improvements.
Dense incoming bytes are reserved before applying a configured sparse cap;
sparse incoming bytes are reserved inside that cap. A common mutex now protects
demand transfer completion and readiness publication, preventing condition-variable
lost wakeups even though the readiness flag itself is atomic.

## Comparison

Four matched arms, two main baselines and one explicitly labeled ablation:

| Arm | Role | Behavior |
|---|---|---|
| demand-only | baseline / sanity | Same model and executor; no speculative enqueue. |
| all-positive | prefetch-set ablation | Same predicted maps, target layers, priorities, cache; all positive probabilities admitted. |
| finemoe-c | algorithm baseline | Correct Eq. 6–8 selector in native C, checked against corrected Python oracle. |
| finemoe-bpf | proposed mechanism | Same bounded input and outputs, actual host uBPF/bpftime JIT; no native fallback. |

The all-positive control answers unused-transfer reduction; demand-only shows
whether prefetching pays at all; C isolates the BPF mechanism cost. Search/store,
GPU→host input materialization, deterministic ties, executor, eviction, copy
streams, request order, cache bytes, and diagnostics remain common. BPF selects
the candidate set, not embeddings, GPU search, CUDA memcpy, or all cache runtime.
Use explicit expert IDs and separate selection masks; a zero/epsilon score must
not turn an unselected expert into a queued candidate. The implemented ABI keeps
the exact original FP32 probability and delta bit patterns, stably sorts expert
IDs, and accumulates the prefix in **sequential binary64**, not the original
FP32 `torch.cumsum`. C uses hardware double; BPF executes software positive-double
addition and all sort/prefix decisions. An independent Python-float oracle checks
both. This numerical convention is disclosed, not presented as unchanged demo
arithmetic or quantization. `policy_runtime.py` preserves selected zero-valued
experts needed for minimum K using the explicit bit mask. CPU tests also observe
the actual author's prefetcher calls reaching the engine API; those calls do not
by themselves prove completed copies.

## Workloads And Metrics

Use the original BF16 model, batch 1, at most 16 input and exactly 16 generated
tokens (demo truncation protocol). The official included LMSYS demo file contains
only 64 prompts, and CPU tokenization finds only 56 unique 16-token inputs.
Duplicate raw IDs are 3, 4, 13, 14, 27, 31, 37, and 52 (zero-based).
The full official LMSYS-Chat-1M is gated (unauthorized file access returned 401;
its page requires login/contact-sharing terms). No terms were accepted, no user
tokens read, and no cohort was shrunk. With explicit root approval, use the
[official public MT-Bench first-turn questions](https://github.com/lm-sys/FastChat/blob/b494d0c6b4e7935f1764f8439e75da3e66beccc7/fastchat/llm_judge/data/mt_bench/question.jsonl)
at source revision `b494d0c6b4e7935f1764f8439e75da3e66beccc7` (48,929 bytes).
All 80 first turns are unique after the original Qwen tokenizer's 16-token
truncation. `dataset-mtbench-v1.json` freezes seed 20260903, exact question/token
IDs, **64 history / 8 evaluation / 1 disjoint warmup** and seven unused rows.
This is an explicit dataset deviation: neither LMSYS's natural request
distribution nor MT-Bench answer quality is reproduced. Store capacity is 1000,
filled by real history-token trajectories, and history is frozen offline.
No online A→B→A
claim follows from this offline run. Reset model/cache per arm, import identical
store, warm up on a disjoint prompt, then measure the same eight requests.

Initial cache configuration: `device_memory_ratio=0.5`, not the demo's 0.9;
record the resolved sparse-cache byte budget and dense/KV allocation. All arms
must use the same resolved budget, actually offload, and stay below available
HBM. If this does not fit, report the concrete allocation limit before changing
any frozen parameter. No model/precision changes to make a failing cell pass.

- Primary application metric: generated tokens / wall time from first measured
  request submission through last verified completion, including policy/search
  and input-transfer overhead. TTFT and per-output-token latency are secondary.
- Mechanism metrics: candidate mask cardinality and cumulative mass; downstream
  admitted/enqueued/started/completed/canceled copies; completed payload bytes
  classified as **first demand use**, **evicted unused**, or **still resident unused**.
  These classes must conserve completed speculative bytes; canceled-before-copy
  bytes are separate, and end-of-window residency is not called waste. Drain
  in-flight copies before closing accounting. Both application begin/end and
  copy lifecycle use the same native steady clock: report byte partitions at
  application end separately from the post-window completed bytes and the final
  drained snapshot. Report drain duration, CPU cost and throughput including
  drain as well as primary application throughput; never label tail copies as
  in-window. Use copy-generation IDs and actual tensor byte sizes. All-60 dynamic
  sets or zero completed speculative copies are valid observed negative outcomes,
  not reasons to discard a cell with real selector/JIT/API engagement.
- Also record demand-copy bytes/wait, cache hit/miss, peak HBM, and per-arm CPU
  time. Logical payload bytes are not measured PCIe traffic or copy-time savings.
- Correctness: same BF16 checkpoint; deterministic greedy output token IDs and
  per-step numerical/logit comparison to the original Transformers full-model
  no-offload/no-prefetch golden. Absolute tolerance is frozen from that original
  HF model's same-arm repeated logits, before any offload-policy arm runs; the
  common offload demand-only arm must satisfy it too. It is not relaxed after
  BPF or another arm fails. All routed
  experts must execute exactly once as required, with no dropped requests/tokens.
  CPU oracle→C→JIT parity includes threshold crossings, ties, delta 0/1, zero
  histories, out-of-range/NaN inputs, and masks actually passed to enqueue.
- Five randomized complete paired blocks, seed 20260903; show every cell, median,
  paired geometric-mean ratios and 10,000 paired bootstrap 95% percentile CIs.
  A CI crossing 1 is inconclusive, not a formal equivalence claim.

## Planned Runs

| Run group | Role | Workload | Arms | Repetitions | Decision |
|---|---|---|---|---:|---|
| CPU | dependency | Paper boundaries + real-method downstream call capture | Python/C/JIT | deterministic cases | Exact algorithm/engagement parity. |
| preflight | dependency | Full 64/8/1 cohort, original real Qwen | four above | 1 each | Memory, numerical, copy-accounting and actual selector/JIT/API engagement; no result-direction gate. |
| full | supporting | frozen store, 8 held-out prompts | four above | 5 paired blocks | Useful/unused-byte and latency/throughput tradeoff. |

## Execution

Preparation used official source and normal model cache only; no
system-wide `setup.sh`, token logging, or shared-environment upgrades. A private
venv/build is required because the demo constrains Transformers <4.50 while the
current GPT-OSS environment is newer. Initial cost estimate: 26.67 GiB model download plus
roughly another model-sized offload store; 5–30 min download depending on network,
5–15 min extension build (unverified estimate), then time one real cell before
quoting the full GPU campaign cost. GPU fit was subsequently verified, and the
full campaign completed in the recorded 08:31:45–08:59:10 UTC safety window.
All GPU work used the root-owned exclusive lease.

Planned implementation files: `dynamic-set.patch`, `test_dynamic_set.py`,
`finemoe_policy.h`, `finemoe_policy.c`, `finemoe_policy.bpf.c`,
`finemoe_policy_bridge.cpp`, `prepare.py`, and a thin `compare.py` that calls the
official `MoE.generate`/forward path, not a replacement inference engine. The
existing project lease/telemetry, ordinary Git revisions/file inventories,
and simple completed-cell files are reused; no new experiment-control framework.

Exact immediate CPU command:
`CUDA_VISIBLE_DEVICES='' taskset -c 8 .venv/bin/python -B -m unittest -v test_dynamic_set.py test_policy_runtime.py test_finemoe_policy.py`
from this directory. Patch replay is `git -C deps/FineMoE-EuroSys26 apply --check
../../dynamic-set.patch` on the clean pinned source, then apply once.
The completed CPU data freeze was `.venv/bin/python -B prepare.py --output
dataset-mtbench-v1.json`. `inference.py` calls the original Transformers full-model
golden and the actual author `MoE.generate` path; `compare.py` is the thin
lease/safety/telemetry controller. Its staged preparation creates a real original
model golden and same-arm repeat tolerance, then the full 64-request history,
then all four numerical/oracle/copy-accounting canaries. Formal runs disable
full-logit transfer and shadow comparisons, retain exact golden token checks,
and use fresh processes with the same store/warmup. The failed v1 and v2 GPU
attempts and diagnostic are retained. Normal golden-v4 completed the full
73-request / 9-repeat protocol and independent two-array numerical audit;
history-v3 completed the full 64-request real history and store audit. Following
the retained preflight-v1 discrepancy and Python head-shape compatibility change,
fresh full history-v4 and four-arm preflight-v2 also passed independent raw
and actual-array audits. The five-block formal comparison, `raw/full-v1`, then
completed all 20 cells; its [report](results-performance.md) records the mixed
result and the independent analysis path. Next is revision synthesis, not another
run of these completed output directories.

Exact staged GPU commands, **only while the root grants the exclusive window**:

The completed diagnostic used `.venv/bin/python -B compare.py --mode golden
--native-backtrace --output raw/golden-sigill-gdb-01 --timeout 1200`. It cannot be
reused as a reference. Normal golden-v4, history-v4, preflight-v2 and full-v1 have
completed. The commands below record those executions, not instructions to
overwrite or restart their existing outputs.

```sh
.venv/bin/python -B compare.py --mode golden --output raw/golden-v4
.venv/bin/python -B compare.py --mode history --golden raw/golden-v4/stage --output raw/history-v4
.venv/bin/python -B compare.py --mode preflight --golden raw/golden-v4/stage --history raw/history-v4/stage --output raw/preflight-v2
.venv/bin/python -B compare.py --mode full --golden raw/golden-v4/stage --history raw/history-v4/stage --preflight raw/preflight-v2 --output raw/full-v1
```

The parent controller must retain its ordinary affinity (it runs telemetry on
CPU 16); only its worker is pinned to CPUs 8–11. Each output directory is unique.
Any failed stage remains on disk and prevents later success; no numerical
tolerance is widened for an arm. The completed final CPU rebuild was
`taskset -c 8-11 .venv/bin/python -B build_offload.py --log build/offload-build-05.log`
(explicit g++-13, MAX_JOBS=2); future rebuilds require a new log and must not
overlap another GPU timing window.
The small standalone tests use g++-13 on `test_copy_ledger.cpp` and
`test_runtime_safety.cpp`; `test_compare.py` independently rechecks raw lifecycle
and actual-input selector records without loading any CUDA model.

Raw requests, token IDs, timestamp events, candidate/copy accounting, runtime
inventory, configs, source revisions and all failures belong in the exact new
raw directory. Interrupted/incomplete paired blocks do not count or mix with
completed ones; keep failures and restart the whole affected block.

## Interpretation And Reproducibility Notes

Positive: fewer unused **completed** bytes, preserved correct output, and no
unaccounted application regression, with C/BPF parity. Mixed: byte reduction but
lower throughput, or no reduction in demand stall; report both. If demand-only
wins, report that speculation does not pay in this workload. If C beats BPF,
report mechanism overhead. Planned figure: unused/useful/resident bytes alongside
application throughput, with all five raw points and no cross-model speedup claim.
This remains a FineMoE dynamic-set component port on the official Qwen demo,
not full FineMoE/EuroSys reproduction, not a kernel-UVM execution result, and
not evidence that selected experts necessarily equal the actually routed K.
