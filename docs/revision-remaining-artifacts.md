# Remaining revision artifacts — 2026-09-03

This is an execution checklist, not evidence that the missing experiments or
release have completed. It supplements the
[verbatim commitments](revision-shepherd-comment.md) and
[current experiment status](revision-experiment-status.md). FineMoE,
Hummingbird and POD-Attention have their own plans; do not rerun them merely to
fill this checklist. GPU work is exclusive and coordinated by the main thread.

## 1. LMCache local disk: paused by user direction

Current update: the repaired 575 traced disk preflight passed its storage
engagement checks, but the subsequent full eight-prefix correctness comparison
failed: cold output matches across all three arms; CPU and disk warm output
match each other but disagree with recompute for every prefix. No formal
performance cells started. A shared KV layout/stride defect is a concrete
source-level candidate, not a validated root cause. Preserve those raw records
without changing their identity-sensitive permissions. The sequence below is
retained for history and possible resumption, **not an active execution queue**.
The user explicitly paused this work on 2026-09-03; keep the storage discussion
and disclose the deferred measurement in the revision response.

The highest-value remaining LMCache result is the already specified matched
**recompute / native LMCache CPU / native LMCache disk** comparison. It answers
whether disk retrieval saves first-token latency and what request-rate cost it
pays on this model and SSD. This is a runnable research-baseline measurement,
not yet a BPF policy port or evidence that the mechanism accelerates storage.
See the existing [protocol](../workloads/lmcache-disk/plan-v2.md).

### What actually exists

- Pinned LMCache 0.5.4 and vLLM 0.27.1+cu129 environments, both retained wheels,
  all nine recorded import files, and the official LMCache source are present.
  This check inspected files, not a new import/build/GPU qualification.
- Qwen3-30B-A3B-FP8 has all seven local shards, 32,449,487,768 bytes total,
  plus config/index/tokenizer files. The frozen ShareGPT source is present.
  No download or smaller-model substitution is needed based on this inventory.
- The [successful one-prefix smoke](../workloads/lmcache-disk/raw/code-smoke-prefix1-20260901-05/result.json)
  served one cold and one warm request on driver 610.43.02. It stored/retrieved
  exactly 1,536 tokens in six 24 MiB files. It had no syscall trace, and is not a
  throughput result, an eight-prefix test, or a CPU/recompute/BPF comparison.
- The current host is Linux 6.15.11 with driver 575.57.08. The old helper still
  defaults to an exact 610.43.02 requirement; using its old launch command
  unchanged will fail admission. All old 610 records must retain that identity.

### Harness repairs and remaining launch preparation

1. **Implemented, CPU-checked:** explicit `--expected-driver 575.57.08`, preserving
   the legacy 610 default and offline validation. New cells record expected and
   observed driver versions; mismatch and mixed-driver formal comparisons are
   rejected. Historical records without an explicit selection still require 610.
   This is a harness repair, not evidence that the runtime works on 575.
2. **Implemented, CPU-checked:** `analyze` requires **eight prefixes per formal
   cell**. `validate-cell` still accepts 1–8 prefixes for legitimate smokes, but
   smoke-sized cells cannot enter the formal comparison.
3. Reconcile the historical closed-protocol wording with the current main-thread
   execution queue and record the new 575 run as a disclosed stack change. The
   old three failures remain failures; changing directories is not evidence of
   repair. Do not reopen review loops merely to obtain a different verdict.
4. **Implemented, CPU-checked:** the runner now takes the two existing leases,
   pins workers to CPUs 8–15, retains CPU 16 for continuous GPU/kernel monitoring,
   checks pre/post safety, and cleans only owned process groups. New records
   preserve the boot and CPU placement; formal analysis rejects mixed boots.
   Do not add an outer duplicate lease or restrict the coordinator to 8–15.
   Source, GPU/SSD and actual runtime compatibility still need real admission.

Validation: 23 lightweight CPU tests passed independently on CPU 17; CLI parsing and the actual
historical 610 smoke's recorded environment also passed. The separate frozen-
prompt regeneration test was deferred because it reads the 672 MB source and
imports the tokenizer during another campaign's timing. No new GPU, model-load,
runtime-import or 575 compatibility test was run for these repairs. The four
scoped harness/protocol files were committed and pushed as `bbc4d3f`.

The runtime remains the existing official `vllm serve` path: eager execution,
native prefix caching off, max model length 4,096, one sequence, common memory
budget 0.98, DeepGEMM off, and 16 generated tokens. CPU retention is 8 GiB;
disk uses no CPU retention, a 2 GiB staging allocator and a 16 GiB disk tier.

### Required real sequence

1. Recheck source/environment, the same model, the NVMe filesystem and exclusive
   idle GPU on 575. File presence is not runtime compatibility.
2. Run **one full eight-prefix disk preflight with tracing**: 48 durable 24 MiB
   files, exact zero-hit cold stores and 1,536-token warm retrievals. Reparse
   actual successful `O_DIRECT` opens for every write/read path, not only the
   LMCache configuration log. No buffered fallback, missing chunk or native
   prefix-cache hit may count as disk retrieval.
3. Run the three complete correctness paths and compare exact response text for
   all cold/warm requests. Do not compare a warm request's different suffix
   against its cold request; compare identical requests across configurations.
4. Run the existing **ten complete paired blocks / 30 cells**, using all eight
   prefixes and `schedule.json` orders. Startup, cold population, persistence
   barriers and shutdown remain outside the eight-contiguous-warm-request
   measurement window. Preserve incomplete attempts and their reasons; do not
   select blocks by performance.
5. Recompute warm TTFT, request/s and output-token/s, paired uncertainty and
   disk-versus-both-control effects from the actual raw files. Publish valid
   negative or mixed results. The existing analyzer writes `analysis.json`, so
   invoking it is not a read-only status check.

Existing command entry points, from `workloads/lmcache-disk` (explicit 575
selection is implemented; the fresh real qualification below is still required):

```sh
./current-venv/bin/python -B run_lmcache_disk.py inspect \
  --expected-driver 575.57.08 --storage-root raw
./current-venv/bin/python -B run_lmcache_disk.py run-cell \
  --expected-driver 575.57.08 --config lmcache_disk \
  --output raw/storage-575-preflight-01/disk --trace
./current-venv/bin/python -B run_lmcache_disk.py validate-cell \
  raw/storage-575-preflight-01/disk --require-trace
./current-venv/bin/python -B run_lmcache_disk.py compare-outputs \
  raw/storage-575-correctness-01/recompute \
  raw/storage-575-correctness-01/lmcache_cpu \
  raw/storage-575-correctness-01/lmcache_disk
./current-venv/bin/python -B run_lmcache_disk.py analyze raw/storage-575-full-01
```

For correctness, invoke `run-cell` once for each of the three listed directories
with the matching `--config` and explicit driver; do not use `--prefix-limit 1`.
For full execution, use the same one-cell entry point at
`raw/storage-575-full-01/attempt-NN/position-P-CONFIG`, in the checked-in order.
Attempt 00 is CPU → recompute → disk; 01 is recompute → disk → CPU; 02 is disk →
CPU → recompute. Stop after the tenth complete block, not after ten arbitrary
attempts. Trace qualification is separate from untraced formal timing.

**Time estimate:** reserve 10–20 minutes for qualification and 30–60 minutes for
30 formal cells, then replace this provisional range with the first complete
eight-prefix cell's wall time. The earlier single-prefix server log spans about
20 seconds, but does not establish full-prefix or 575 timing. Model startup,
CPU-side admission and persistence dominate the reservation; this is not a
promise of completion within one hour or measured GPU utilization time.

### What is still missing for a BPF comparison

There is **no BPF configuration in this adapter**. Recompute is the no-cache
control; CPU and disk are original user-space LMCache backends, not native/BPF
implementations of one new policy. A successful three-arm campaign closes the
bounded disk-baseline measurement, not the user's general same-policy BPF test.

The [older vLLM UVM workload](../workloads/vllm/README.md) uses a different fork,
allocator, concurrency and request protocol. Its BPF page-prefetch result cannot
be appended to the above table as a matched cache-retention arm. A genuine next
BPF comparison needs a defined cache/tier decision, the same observed inputs and
executor in original-native and BPF modes, real BPF engagement, identical cache
budgets and correctness, then repeated matched cells. No such executable entry
point or credible timing estimate exists yet; do not invent a `--config bpf`
command or call UVM page migration a disk-backend port. Finish the promised
native storage-tier measurement before deciding whether that additional port
adds enough new evidence.

## 2. RTX 5090 / NVBit Table 1

The predeclared non-cross-clock subset is complete. The repaired
[preflight](../workloads/llama.cpp/observability_overhead/revision-rq4/raw/preflight-575-noncross-clock-04/README.md)
passes all five configurations, and the
[paper-value run](../workloads/llama.cpp/observability_overhead/revision-rq4/result-review.md)
passes all five correctness cells and 10/10 randomized pp512 blocks with no
rejected or retried cell. Exit-record overhead is 99.663% for gpubpf and
99.621% for matched NVBit; exit-count-histogram overhead is 4.007% and 10.301%.
The result is mixed: NVBit is 0.04185 percentage points lower-overhead for the
full record stream, whereas gpubpf is 6.29351 points lower-overhead for the
histogram. The two corrected rows are integrated into the 16-page paper build.

This does not complete the original three-tool/seven-arm campaign. The
host/device cross-clock `launchlate` comparison remains invalid and is omitted
from the paper. The performance runtime had GPU verification disabled; NVBit
uses custom matched adapters while both systems retain native transports. The
histogram arms close aggregate counts, but only gpubpf retained the complete
vector. The frozen plan named llama.cpp build 7101; every accepted preflight
and full arm consistently used build 7102, so the disclosed deviation creates
no cross-arm binary mismatch.

Remaining work is limited to these explicit boundaries:

1. Preserve every earlier failure and do not rerun or pool the accepted two-tool
   cells merely to seek a more favorable result.
2. Treat a future same-clock or principled RM-correlation `launchlate` repair as
   a separate experiment; until then the row remains absent, not zero-overhead.
3. Keep strict device-verifier enforcement separate from this verifier-off
   performance study.
4. Repeat the whole-paper build/placement review after any later source edits.

## 3. Agent prompts, original logs and harness release

The [public index](eval/agent/README.md) and benchmark/extractor sources exist.
The original study corpus does not: the recorded local corpus directory now has
one 1,460-byte JSONL, with **none of the ten required focus primary sessions**.
The historical [Q6 inventory](eval/agent/q6_precise_metrics.md) calls for 25 primary
and 259 nested transcripts. No backup was recovered in this check.

The [expanded inventory](experiment/revision-artifact-inventory.md) records all
25 primary prefixes and the checked locations. [Current reproduction templates](eval/agent/reproduction-prompts.md)
have now been authored and indexed separately; they are not recovered originals
or evidence that the historical agent trajectory can be replayed. The author
has been asked for the real backup location while other revision work continues.

Remaining work requires the real archive, not another GPU experiment:

1. Locate the original backup/export, preserve the primary/subagent layout, and
   reconcile all 25/259 entries with the historical inventory. The extractor only
   requires ten focus prefixes at startup; that check alone does not prove the
   complete original study corpus was recovered.
2. Prepare a release copy with credentials, unrelated private text and private
   service details removed; retain study prompts, tool interactions, failures
   and timestamps, and explicitly document omissions. Do not reconstruct missing
   prompts from summaries or substitute new conversations.
3. Recompute the reported aggregates and safety-event links from that release
   copy, then publish it together with the existing harness/index and ordinary
   file inventories. A new successful run cannot recreate historical evidence.

The existing CPU-only extractor is
`python3 scripts/analysis/extract_claude_q6_metrics.py --corpus ARCHIVE --output REPORT`.
`ARCHIVE` and `REPORT` are placeholders for the recovered release copy and a new
report path, not existing artifacts. GPU time required: **zero**. Recovery ETA is
unknown until the archive location is available; redaction/recomputation time
must be estimated from the real corpus, not the current 1,460-byte remnant.
