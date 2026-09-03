# LMCache local-NVMe experiment protocol — revision 2

## Current execution addendum — 2026-09-03

The user's subsequent execution request resumes the work on the prepared
575.57.08 host; the earlier 610 attempt history below is not rewritten or counted
as a successful 575 run. **No new 575 model load, disk preflight or performance
cell has run yet.** The model, eight prefixes, three native configurations and
ten complete paired blocks remain unchanged; there is no BPF arm.

The runner now holds the existing GPU/struct-ops leases, pins only the worker
(and optional strace wrapper) to CPU 8–15, and collects GPU telemetry every
200 ms plus a live kernel journal on CPU 16. Do not externally pin the coordinator
to 8–15: it must retain CPUs 8–16 in its allowed set. `execution.json` records the
actual boot, explicit driver, CPU placement, pre/post safety, monitor lifetime
and cleanup; 575 validation reparses those files and rejects mixed-driver/boot
formal blocks. The shared checks require the existing 400 W limit and reject
kernel errors or thermal/hardware throttling; normal fixed-power-cap activity
is reported. All cleanup targets owned process groups, never global processes
or shared-memory files.

Next, under the main thread's GPU schedule (the runner takes its own leases):

```sh
./current-venv/bin/python -B run_lmcache_disk.py run-cell \
  --expected-driver 575.57.08 --config lmcache_disk \
  --output raw/storage-575-preflight-01/disk --trace
```

Validate the full eight-prefix trace before the three correctness cells and
30 untraced formal cells. The safety changes have only CPU unit-test evidence;
they do not establish runtime compatibility or storage performance on 575.

## Historical revision-2 protocol

Historical status: **offline repair passed final independent review; no further GPU launch
is authorized because revision 1 exhausted the three-attempt preflight cap**.

This protocol addresses the explicit revision commitment to extend the LMCache
comparison to its local-disk backend. It characterizes the public LMCache
filesystem backend on one RTX 5090 and the workspace's local Samsung 9100 PRO
NVMe. It is standalone baseline evidence and is not a gpubpf storage result.

Planned role: **supporting**. A citation-only comparison cannot answer the
revision question because published LMCache results use different models,
hardware, and storage paths, while the submitted paper already measures the
CPU backend on this workload. The matched local-disk run would directly bound
that existing comparison and therefore has more decision value than citing an
unmatched published number; it does not independently establish the paper's
central mechanism claim.

## Question and configurations

For eight reusable 1,536-token prefixes from the paper's Qwen-30B workload,
what warm TTFT and sequential warm-request rate result when KV is (1)
recomputed, (2) retained in LMCache CPU DRAM, or (3) retrieved from LMCache's
local-NVMe backend?

Hypothesis: local-NVMe retrieval lowers warm TTFT relative to recomputation but
is slower than CPU-resident retrieval. The primary competing explanation is
that software and storage overhead erase the saved recomputation, making disk
neutral or slower; a second is that disk improves TTFT but reduces sequential
request rate by more than 5%. A valid negative or inconclusive result still
answers the revision's baseline question and will narrow, not inflate, the
paper's policy-versus-mechanism claim.

All cells use LMCache v0.5.4 at source revision
`3e11b8ed191631e6f098b8038235823f1a410b24`, official vLLM
`0.27.1+cu129`, CUDA 12.9, driver 610.43.02, and
`Qwen/Qwen3-30B-A3B-FP8` at revision
`d206ba732169f29bb77fbf80fc2c4b81d4d30782`. Every cell uses
`--enforce-eager`, `--max-model-len 4096`, `--max-num-seqs 1`,
`--gpu-memory-utilization 0.98`, native prefix caching disabled, and
`VLLM_USE_DEEP_GEMM=0`.

The 0.98 setting is the only new runtime repair. The preceding closed protocol
used 0.99, which requested 31.08 GiB after vLLM observed 30.89 GiB free and
failed before model loading. At 0.98 the nominal request is about 30.77 GiB,
below that observation. The seven checkpoint shards occupy about 30.22 GiB;
one 4,096-token sequence has a derived KV maximum of 384 MiB, leaving a small
but positive startup margin. This is a predeclared feasibility calculation,
not a successful-run claim.

The cells are:

1. `recompute`: no external KV connector.
2. `lmcache_cpu`: LMCacheConnectorV1 with an 8 GiB CPU tier and no disk.
3. `lmcache_disk`: CPU retention disabled, a 2 GiB staging allocator, a 16 GiB
   local-disk tier, and `use_odirect=true`.

## Workload and exact gates

`prompts.json` schema 3 stores exact prefix/cold/warm token arrays derived from
frozen ShareGPT row starts `[0,173,509,997,1499,2203,3109,4211]`. Admission
parses and validates the dataset, prompt structure, token-array lengths and
common prefixes, and all 15 entries of the fixed, position-balanced schedule.
The checked-in prompt artifact must exactly equal a fresh derivation from the
pinned dataset rows and tokenizer. The adapter validates exact
versions, import paths, dependency lines, source revisions, model filenames
and sizes, GPU exclusivity, driver, filesystem UUID, free space, and port.

No file or content fingerprints, checksums, or digests may be generated,
refreshed, compared, or recorded. Small structured artifacts are checked by
their parsed semantics. Responses are compared as exact text. Evidence-file
sets use logical absolute path, byte size, device, inode, and modification/change
times. Git commit IDs and upstream source revisions are used only for ordinary
version bookkeeping.

Each cold request must report zero external hits and an exact 1,536-token
store. Disk files must stabilize at six 24 MiB chunks per prefix; every file
and the directory are synchronized before the next request. Each warm request
must report exactly 1,536 hit and retrieved tokens. Every request must return
HTTP 200, the exact prompt-token count, and exactly 16 output tokens. Fatal,
fallback, eviction, partial-write, allocator, and native-prefix-cache evidence
invalidates the cell.

## Preflight, correctness smoke, and execution

The first real action is a disk-only preflight under
`strace -ff -e trace=open,openat`. All 48 cache-object write paths and the same
48 read paths must open successfully with `O_DIRECT`, remain under the run's
cache directory, and have no buffered `.pt` open. Raw traces are retained.

If higher-level authorization ever permits another launch, the real path is a
single traced `lmcache_disk` cell followed by ordinary revalidation of its raw
server log and strace. A later three-cell correctness check compares exact
response text. `schedule.json` contains five randomized three-order Latin
cycles: each configuration occurs exactly five times in every position over
15 attempts, and position counts differ by at most one in the first ten.

There are no pass markers, completion schemas, approval parser, promotion
gate, or custom resume protocol. Each invocation runs one official `vllm
serve` cell and preserves its raw output. Analysis reparses every cell's result,
server log, request usage and engagement, and any trace; it also regenerates
the prompt and schedule semantics. It requires contiguous attempt numbers,
position-prefix execution, strictly increasing launch-observation timestamps,
an ordinary nonempty `failure.md` for every incomplete attempt, no attempts
after the tenth complete block, and balanced positions in the cells that
actually completed. Technical failures remain ordinary named directories and
are not included as completed attempts.

## Metrics and interpretation

Primary: within-attempt median warm TTFT over eight prefixes, measured at the
client from request send to the first SSE choice carrying a generated token
(requested through vLLM completion logprobs). Secondary: warm P95/max TTFT,
sequential warm requests/s, and output tokens/s, all from client timestamps and
server usage counters. The timed phase
contains only eight contiguous warm requests and excludes startup, cold
population, persistence barriers, and shutdown.

These names and measurement points follow the official
[vLLM benchmark definitions](https://docs.vllm.ai/en/stable/benchmarking/cli/):
TTFT runs from sending a request to its first streamed output; request
throughput is successful requests divided by the timed duration; output-token
throughput is generated tokens divided by that duration. This experiment uses
the first streamed event that actually carries a generated token and records
the exact formulas in every cell result.

For disk minus recompute and disk minus CPU, report paired attempt differences
and fixed-seed percentile-bootstrap 95% intervals over ten complete blocks.
For disk versus recompute, classify mutually exclusively as beneficial,
latency-throughput tradeoff, not beneficial, or inconclusive using the runner's
predeclared TTFT and -5% rate rules. A tradeoff requires the rate interval's
upper bound below -5%; an interval crossing -5% is inconclusive. The claim is scoped to this model, prefix
set, runtime, GPU, and SSD. Any engagement failure is invalid/blocked, never a
performance result.

## Historical boundary

The three attempts under `raw/preflight-610-20260831-{01,02,03}` belong to the
closed revision-1 protocol. They served no request and produced no timing or
cache-I/O evidence. They remain failure provenance and cannot be relabeled as
revision-2 attempts. Neither a new namespace nor a new review resets the
three-attempt cap. The repaired adapter is therefore offline-only unless the
user explicitly grants a higher-level exception.

## Fixed commands (offline only while blocked)

From `workloads/lmcache-disk`, the ordinary commands are:

```text
./current-venv/bin/python run_lmcache_disk.py inspect --storage-root raw
./current-venv/bin/python run_lmcache_disk.py run-cell \
  --config lmcache_disk --output raw/revision2-preflight/attempt-00/position-2-lmcache_disk --trace
./current-venv/bin/python run_lmcache_disk.py validate-cell \
  raw/revision2-preflight/attempt-00/position-2-lmcache_disk --require-trace
./current-venv/bin/python run_lmcache_disk.py compare-outputs \
  raw/revision2-smoke/recompute raw/revision2-smoke/lmcache_cpu \
  raw/revision2-smoke/lmcache_disk
./current-venv/bin/python run_lmcache_disk.py analyze raw/revision2-full
```

The `run-cell` line is documented for reproducibility but must not be executed
under the current blocked status. If execution is authorized, each attempt's
three `position-N-CONFIG` paths must follow the exact order in `schedule.json`.
The analysis is paper-valuable only with ten fully revalidated attempts; setup,
inspection, and failed launches are not paper evidence.
