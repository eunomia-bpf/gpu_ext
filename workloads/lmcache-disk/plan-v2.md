# LMCache local-NVMe experiment protocol — revision 2

Status: **pending independent review; no GPU execution authorized yet**.

This protocol addresses the explicit revision commitment to extend the LMCache
comparison to its local-disk backend. It characterizes the public LMCache
filesystem backend on one RTX 5090 and the workspace's local Samsung 9100 PRO
NVMe. It is standalone baseline evidence and is not a gpubpf storage result.

## Question and configurations

For eight reusable 1,536-token prefixes from the paper's Qwen-30B workload,
what warm TTFT and sequential warm-request rate result when KV is (1)
recomputed, (2) retained in LMCache CPU DRAM, or (3) retrieved from LMCache's
local-NVMe backend?

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
common prefixes, and all 15 entries of the fixed schedule. It validates exact
versions, import paths, dependency lines, source revisions, model filenames
and sizes, GPU exclusivity, driver, filesystem UUID, free space, and port.

No file or content fingerprints, checksums, or digests may be generated,
refreshed, compared, or recorded. Small structured artifacts are checked by
their parsed semantics. Responses are compared as exact text. Evidence-file
sets use resolved path, byte size, device, inode, and modification/change
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

After preflight, an isolated three-cell smoke requires exact response-text
equality across all cells. Only then may the randomized full run start.
`schedule.json` fixes 15 cell orders with seed 2709. Collection stops at ten
valid complete blocks or 15 attempts. Technical failures are preserved and
classified without consulting performance. A complete block exists only when
all three cells pass engagement and exact-output gates.

Revision 2 permits at most three named real preflight attempts. An attempt is
consumed whenever the real model server is launched. A repair must target a
specific observed experiment-side defect, be tested offline, and be reviewed
before another attempt. Repetition of the same root cause is terminal. No
timing is analyzed before preflight and smoke both pass.

## Metrics and interpretation

Primary: within-block median warm TTFT over eight prefixes. Secondary: warm
P95/max TTFT, sequential warm requests/s, and output tokens/s. The timed phase
contains only eight contiguous warm requests and excludes startup, cold
population, persistence barriers, and shutdown.

For disk minus recompute and disk minus CPU, report paired block differences
and fixed-seed percentile-bootstrap 95% intervals over ten complete blocks.
For disk versus recompute, classify mutually exclusively as beneficial,
latency-throughput tradeoff, not beneficial, or inconclusive using the runner's
predeclared TTFT and -5% rate rules. The claim is scoped to this model, prefix
set, runtime, GPU, and SSD. Any engagement failure is invalid/blocked, never a
performance result.

## Historical boundary

The three attempts under `raw/preflight-610-20260831-{01,02,03}` belong to the
closed revision-1 protocol. They served no request and produced no timing or
cache-I/O evidence. They remain failure provenance and cannot be relabeled as
revision-2 attempts. Revision 2 uses a new fixed namespace and requires a new
independent final approval before its first launch.
