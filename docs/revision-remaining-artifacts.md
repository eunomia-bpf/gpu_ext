# Remaining revision artifacts — 2026-09-03

This is an execution checklist, not evidence that the missing experiments or
release have completed. It supplements the
[verbatim commitments](revision-shepherd-comment.md) and
[current experiment status](revision-experiment-status.md). FineMoE,
Hummingbird and POD-Attention have their own plans; do not rerun them merely to
fill this checklist. GPU work is exclusive and coordinated by the main thread.

## 1. LMCache local disk and storage-policy comparison

LMCache is active work. The first five-block performance-only campaign completed
15/15 cells. Recompute/CPU/disk median TTFT is
67.1691/72.6468/96.3280 ms, request throughput is
1.9151/1.8823/1.7858 req/s, and output throughput is
30.6422/30.1168/28.5723 token/s. Earlier correctness failures and their raw
records remain retained but no longer suspend performance work.

The next result extends this real local-disk path with two matched arms: the
same recoverability/slack-aware policy implemented natively and through
gpubpf. In parallel, a GPU-storage decision hook is being implemented so BPF
can select submit, defer, recompute, priority and batch hints while LMCache and
cuFile retain transport ownership. Direct P2PDMA, cuFile compatibility mode and
POSIX fallback are recorded as execution-path labels, not used to suppress
performance measurements. See the current
[recoverability plan](../workloads/lmcache-disk/recoverability-arbitration-plan.md).

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

### Historical harness preparation

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

### Historical full-protocol sequence

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

### Active native/BPF comparison

The existing recompute/CPU/disk arms remain the storage baselines. The active
implementation adds native and BPF variants of one integer-only policy over the
same inputs and trusted LMCache/cuFile executor. It must report both policy
benefit over plain transport and BPF mechanism cost relative to native. The
older vLLM UVM page-prefetch result stays separate and is not relabelled as a
disk-policy result.

## 2. RTX 5090 / NVBit Table 1

The current [three-tool result](../workloads/llama.cpp/observability_overhead/revision-rq4/results-table1-warp-plt-575-06/README.md)
completes the original seven arms across 10 rotated blocks / 70 successful
numeric cells. Baseline is 37,586.3225 token/s. gpubpf/NVBit overhead is
90.7051%/99.6210% for `kernelretsnoop`, 2.9653%/10.3501% for `threadhist`, and
0.2208%/8.7959% for `launchlate`. The submitted P40 values remain
8%/85%, 3%/87%, and 14%/93% respectively. Earlier 5090 attempts and verifier
studies remain retained as separate historical results.

Remaining Table 1 work is performance optimization, not row completion:

1. Preserve the complete 70-cell campaign and every earlier result.
2. Optimize `kernelretsnoop` using GPU-local preallocated event storage and a
   bulk drain while retaining its full per-warp record stream.
3. Run a one-block measurement, then a new ten-block campaign if it improves.
4. Repeat the paper placement review after the optimized numbers are integrated.

### Separate device-map placement result

The operation-matched [device-map campaign](../microbench/fig15-device/results-map-tier-full-575-06-20260904.md)
is complete and integrated. Across 16 balanced blocks and 128 fresh processes,
direct host mapping has 9.4307x the device-resident update latency (97.5% CI
[9.3789, 9.4896]) and 1.0904x the lookup latency ([1.0797, 1.1113]). This
replaces the old undifferentiated 6000x map claim. It is a one-block,
32-thread, scalar-runtime result; the much larger serialized-RPC measurements
diagnose that protocol rather than PCIe placement. Five earlier invalid full
attempts remain excluded.

The exact per-lane object is not STRICT-admissible: only its no-op passes, and
all six map callbacks violate current SIMT branch/key/value uniformity rules.
A deliberately separate [strict-uniform campaign](../microbench/fig15-device/strict-uniform-map/results-full-575-01-20260904.md)
closes the positive end-to-end boundary with 72/72 valid processes and 60/60
target-PID STRICT admissions. Host/device is 1.0008x [0.9891, 1.0143] for
constant-key update and 1.0778x [1.0644, 1.0833] for lookup. The update effect
is unresolved; the lookup result covers the complete host-map implementation,
including cache/coherence behavior, not pure PCIe latency. These semantically
different workloads remain separate.

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
