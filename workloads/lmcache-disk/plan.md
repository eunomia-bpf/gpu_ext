# Revision experiment plan: LMCache local-NVMe KV-cache tier

Status: **approved after round 3; runtime admission and GPU preflight remain
incomplete**. See `plan-review.md`; approval does not waive the frozen stack
or exclusive-GPU requirements below.

## 1. Revision requirement and claim boundary

The revision plan asks us to extend the submitted LMCache comparison to a
local-disk backend.  This experiment characterizes LMCache's public filesystem
backend on the machine's local NVMe SSD.  It does not claim that gpubpf
implements a storage tier, GDS, CXL, or raw-block storage, and its fresh numbers
will not be pooled with the submitted gpubpf measurements.

The primary implementation is LMCache `v0.5.4`, the current stable release at
protocol freeze.  It is paired with the official vLLM `0.27.1+cu129` wheel:
this is the newest public CUDA-12.9 wheel available before vLLM's CUDA-13
default, and therefore the strongest current triplet compatible with the
paper's 575-series driver and RTX 5090.  The exact historical LMCache
`58cae13ba` build recovered from the submitted logs is retained only as a
bridge/sensitivity artifact; it is not the primary experiment.

## 2. Question and hypothesis

Question: for reusable 1,536-token prefixes on the paper's Qwen-30B workload,
what latency and sequential warm-request throughput does LMCache provide when
KV is retrieved from local NVMe rather than retained in CPU DRAM or recomputed?

Primary hypothesis: disk retrieval lowers warm TTFT relative to recomputation,
but remains slower than LMCache's CPU tier.  No directional claim is made about
throughput before measurement.

## 3. Frozen environment and artifacts

- RTX 5090 (32 GiB), one GPU; NVIDIA driver exactly `610.43.02`.
- CUDA 12.9; Python 3.12; official vLLM `0.27.1+cu129` wheel.
- official LMCache `v0.5.4` source commit
  `3e11b8ed191631e6f098b8038235823f1a410b24`, rebuilt for `sm_120`.
- `Qwen/Qwen3-30B-A3B-FP8` at immutable Hugging Face revision
  `d206ba732169f29bb77fbf80fc2c4b81d4d30782`.
- checked-in ShareGPT dataset, `prompts.json`, `schedule.json`, wheel hashes,
  imported-module paths/hashes, and complete environment freeze.
- cache directory on `/dev/nvme1n1p1` (ext4, non-rotational Samsung 9100 PRO).

Admission fails closed on any version, source commit, imported path/hash,
dataset/prompt/schedule hash, model snapshot, driver, GPU occupancy, residual
GPU memory, mount, capacity, port, or tracing mismatch.  It records mount,
device model/serial/firmware, free space, filesystem block size, and available
NVMe temperature sensors.  It reports foreign GPU processes but never signals
them.

## 4. Configurations

Every configuration uses the same server binary, model, token arrays, request
IDs/order, `--enforce-eager`, max model length, memory utilization, one active
sequence, and explicit `--no-enable-prefix-caching`.

1. `recompute`: vLLM without an external KV connector.
2. `lmcache_cpu`: `LMCacheConnectorV1`, 8 GiB local CPU tier, no disk.
3. `lmcache_disk`: `LMCacheConnectorV1`, local CPU retention disabled, 2 GiB
   CPU staging allocator, 16 GiB local disk, `use_odirect=true`.

LMCache chunk size is 256 tokens, incomplete chunks are not saved, layerwise
storage is disabled, and `PYTHONHASHSEED=0`.  The Qwen KV footprint is
`48 layers * 2(K,V) * 4 KV heads * 128 head dimension * 2 bf16 bytes` = 98,304
bytes/token.  One 256-token file is exactly 24 MiB; six files per prefix and
eight prefixes total 48 files / 1.125 GiB.  Thus the complete resident CPU
tier fits 8 GiB, while the disk mode's 2 GiB CPU allocation is staging rather
than hot retention.

## 5. Public workload and exact engagement

Eight prefixes are deterministically constructed from frozen ShareGPT row
starts `[0,173,509,997,1499,2203,3109,4211]`.  `prompts.json` stores the exact
prefix, cold, and warm token-ID arrays and their hashes.  Each cold/warm pair
has a precomputed token LCP; after 256-token alignment the expected external
hit is exactly 1,536 tokens.

Requests use the token arrays directly, sequential streaming, temperature
zero, seed zero, `ignore_eos=true`, and exactly 16 output tokens.  Unique
`X-Request-Id` values map deterministically to vLLM/LMCache request IDs.

For each configuration the protocol is:

1. issue all eight cold requests, one at a time;
2. after each external-cache cold request, require a request-scoped store of
   exactly 1,536 tokens and zero external hit tokens;
3. for disk, require the cumulative exact file count and 24 MiB size for every
   file, then `fsync` every file and the directory before proceeding;
4. issue all eight warm requests contiguously with no persistence barrier in
   the timed interval;
5. require each warm request to report exactly 1,536 hit and retrieved tokens,
   no missing/partial/fallback/eviction/allocator error, and unchanged exact
   disk footprint.

The result retains TTFT, end-to-end latency, server token counts, exact input
token-array hashes, response hashes/text, request IDs, request-scoped log
evidence, cache footprint, command/environment, and server log hash.

## 6. Exhaustive O_DIRECT preflight

Before performance collection, a separate disk-only preflight runs the exact
model, all eight prompt pairs, and the exact disk configuration under
`strace -ff -e trace=open,openat`.  Every observed `.pt` read or write open must
contain `O_DIRECT`.  The 48 unique successfully opened write paths and 48
unique successfully opened read paths must be the same set, and every path
must be contained by the current run's cache directory.  Any failed or
buffered `.pt` open, alignment warning, path-set mismatch, escaped path,
absent direction, or missing trace invalidates the preflight.  Raw per-process
traces and their hashes are retained and authenticated by the pass marker.
Tracing is disabled for performance collection.

The plan reports cache footprint, not inferred disk bytes read/written; no I/O
byte metric is claimed without a block-device accounting source.

## 7. Metrics, repetitions, and stopping

Primary metric: within each complete block, median warm TTFT across the eight
fixed prefixes.  Secondary metrics: warm P95 and maximum TTFT, cold TTFT,
warm/cold ratio, and sequential warm requests/s and output tokens/s.  The warm
rate covers only the contiguous eight-request warm phase and excludes server
startup, cold population, durability barriers, and shutdown.

`schedule.json` precomputes 15 configuration orders using seed 2709.  The run
stops after ten technically valid complete blocks or after all 15 scheduled
attempts.  A retry is allowed only for a recorded correctness/engagement or
execution failure, before performance analysis; performance values are never
consulted.  Partial and invalid attempts are preserved and never overwritten.

Resume requires an exact manifest match.  New JSON and completion markers are
written atomically.  A complete block marker is created only after all three
configurations pass and their deterministic response hashes match the frozen
smoke.  Full execution also requires matching preflight and smoke artifacts
and a final independent `APPROVE` marker.  The gate requires the complete
known evidence-file set rather than accepting a caller-selected subset.  It
also matches model blob identities, runtime imports, canonical server
commands/environments, and the SHA-256 of this plan and the runner, so gates
and resumed attempts cannot cross a harness or protocol edit.

Each preflight, smoke, and full-run admission resolves its requested output
path (or nearest existing ancestor) and requires that actual target to be on
`/dev/nvme1n1p1` with ext4 and at least 100 GiB free.  The server starts from
a fixed allow-listed environment with `CUDA_VISIBLE_DEVICES=0`; caller
`PYTHONPATH`, `LD_PRELOAD`, and arbitrary `VLLM_*`, `LMCACHE_*`, or CUDA
overrides are not inherited.

## 8. Analysis and predeclared interpretation

For each complete block calculate paired differences in median warm TTFT for
disk minus recompute and disk minus CPU, and the paired relative sequential
warm-rate difference for disk versus recompute.  Report median effects and
paired percentile-bootstrap 95% intervals (10,000 draws; fixed seeds).  The
intervals quantify repeated-run variability only for this fixed model, prefix
set, and SSD; request samples are not treated as independent replicates.

The disk-versus-recompute interpretation is mutually exclusive:

- `beneficial`: TTFT-difference upper bound < 0 and throughput-relative lower
  bound >= -5%;
- `latency-throughput tradeoff`: TTFT-difference upper bound < 0 but throughput
  non-inferiority is not established;
- `not beneficial`: TTFT-difference lower bound >= 0;
- `inconclusive`: otherwise.

Disk versus CPU is descriptive unless its paired interval excludes zero.
Engagement failure is `invalid/blocked`, never a performance result.

## 9. Executable sequence and current blockers

After final plan approval, the frozen sequence is:

```text
python run_lmcache_disk.py admission
python run_lmcache_disk.py preflight --output raw/preflight-<id>
python run_lmcache_disk.py smoke --output raw/smoke-<id>
python run_lmcache_disk.py run --output raw/run-<id> \
  --preflight raw/preflight-<id> --smoke raw/smoke-<id>
python run_lmcache_disk.py analyze raw/run-<id>
```

Recorded pre-run deviation (2026-08-31): following the requested 610 port,
all three configurations will use the same installed Open Kernel Modules
610.43.02 stack on Linux 7.1.12-070112-generic instead of 575.57.08. No GPU
preflight or timed block has run, so no older-driver samples are retained.
The reviewed CUDA/Python/runtime versions, model, workload, correctness
checks, repetitions, and analysis remain unchanged. This storage experiment
does not require custom gpubpf modules. The first real preflight must qualify
the retained CUDA 12.9 runtime on this driver; its results cannot be pooled
with submitted 575 measurements or used for the separate NVBit comparison.

The current host is not admitted because an unrelated SGLang service owns
about 31 GiB. Those processes are outside this task's authority. No GPU launch
is authorized while the occupancy and isolation checks fail.
