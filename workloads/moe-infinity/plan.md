# MoE-Infinity head-to-head experiment plan — proposal 2

Status: proposal 2 revision 3, independently approved. Offline implementation
is authorized; GPU execution still requires admission.

## 1. Question, hypothesis, and claim boundary

On one RTX 5090, for the same public GPT-OSS-120B MXFP4 model and frozen
512-input/64-output workload, how do public MoE-Infinity's activation-aware
expert offload, llama.cpp's framework-managed expert offload, plain UVM, and a
gpubpf host-only page-granularity policy compare?

The preregistered directional hypothesis is that `gpubpf_host_stride_lfu` has
higher aggregate output-token throughput than public MoE-Infinity (paired
ratio > 1.0). Output-token throughput is total verified output tokens divided
by the measured request-batch duration, matching vLLM's serving benchmark
definition. This is a deployment comparison across different runtimes, not a
causal cache-algorithm comparison or a model-token decode-kernel benchmark.
In particular, `gpubpf_host_stride_lfu` is a new, runnable host-only ablation:
it does **not** include the paper's device-side observer and is **not** labeled
as the submitted final gpubpf policy. The result therefore answers the
revision's head-to-head baseline request at system level but cannot validate
the full submitted policy by itself.

## 2. Frozen artifacts

The host is one RTX 5090 (32 GiB), Intel Core Ultra 9 285K, 125 GiB DRAM, and
Samsung 9100 PRO NVMe `/dev/nvme1n1p1` (ext4). No multi-GPU, MIG, remote
storage, speculative decoding, request batching, or prefix caching is used.

MoE-Infinity:

- EfficientMoE/MoE-Infinity commit
  `b766f8f1f6379fac6cd23594713ba6f4c7650ad9`;
- CUTLASS commit `dc45f979ae336a235da1676b311f35efeb30149a`;
- Python 3.12.3, Torch 2.13.0+cu129, Transformers 5.16.1,
  `sglang-kernel` 0.4.6.post1+cu129, and the exact 108-package freeze;
- CUDA 12.9, `MOE_ENABLE_SM120=1`, `MOE_ENABLE_SM90=0`, `NVTX_DISABLE=1`;
- official `openai/gpt-oss-120b` revision
  `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`, root-level 15-shard MXFP4
  safetensors only; `original/*` and the unrelated `metal/*` serialization are
  excluded from the admitted model view.

llama.cpp:

- source commit `26836b27ae1ec9d6e94c6b56306cca75c7e86814`;
- `llama-server` SHA-256
  `d4fb5910a4c6f037f12d4e3b8dd4da66d7486ba22f4b136cf7af4007aae072e7`;
- `ggml-org/gpt-oss-120b-GGUF` revision
  `238abdd290bb874b90a5da1b4549881b7d05c091`, exact file
  `gpt-oss-120b-MXFP4.gguf`, 63,387,346,208 bytes, SHA-256
  `582bd40f6886200101f4c4ed9f25f3fe80cc14c86e9e2b37746cd8904a0c622d`;
  its pinned `.src_sha` records primary source revision
  `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`.

`artifacts-current.json` binds these commits, module hashes, model identities,
and build products. Before GPU preflight it is extended with every model shard
hash, tokenizer/config hash, GGUF provenance metadata, combined-policy source
and object hashes, runner/probe hashes, and exact resolved paths. The runner
rejects a missing, extra, size-mismatched, or hash-mismatched artifact.

## 3. Frozen configurations and commands

Every server process is pinned to P-cores `0-7`; concurrency and batch size are
one. `CUDA_VISIBLE_DEVICES=0` is explicit. Child environments are constructed
from an allow list and reject caller `PYTHONPATH`, `LD_PRELOAD`, and unrelated
CUDA, vLLM, llama, or MoE variables. All four configurations use the same
request JSON except for endpoint-specific fields.

The absolute paths below are resolved and frozen into the generated manifest.
The llama command common to configurations 1--3 is:

```text
taskset -c 0-7 <llama-server> --model <exact-MXFP4-GGUF> \
  --alias gpt-oss-120b --host 127.0.0.1 --port <owned-port> \
  --n-gpu-layers 99 --parallel 1 --ctx-size 4096 \
  --threads 8 --threads-batch 8 --cache-ram 0 --flash-attn on \
  --no-warmup --timeout 600
```

The four configurations are:

1. `llama_ncmoe32`: common command plus `--n-cpu-moe 32`; no UVM environment
   override and no struct_ops attachment.
2. `llama_uvm`: exact common command with
   `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`; no struct_ops attachment.
3. `gpubpf_host_stride_lfu`: byte-identical `llama_uvm` server command and
   environment, plus one owned combined struct_ops object. That one object
   merges the existing `prefetch_stride` hooks and `eviction_lfu` hooks and
   adds read-only engagement counters. There is one struct_ops map and one
   link, never two simultaneous policy objects.
4. `moe_infinity_075`: run from the parent of the exact HF snapshot so that
   `--model b5c939de8f754692c1647ca79fbf85e8c1e70f8a` remains a relative path;
   otherwise upstream `os.path.join(offload_dir, model)` discards the offload
   directory. The frozen command is:

```text
taskset -c 0-7 <venv-python> -m \
  moe_infinity.entrypoints.openai.revision_server \
  --model b5c939de8f754692c1647ca79fbf85e8c1e70f8a \
  --offload-dir <attempt>/moe-offload --host 127.0.0.1 \
  --port <owned-port> --device-memory-ratio 0.75 \
  --kv-cache-ratio 0 --max-batch-size 1 \
  --startup-timeout 1800 --decode-step-timeout 600
```

The MoE process additionally receives
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=8`.
Proposal 2 permits one measurement-only native patch: add
`ExpertDispatcher::GetCacheCounts() const`, which returns the existing
`cache_access_count_` and `cache_hit_count_` through two relaxed atomic loads,
and expose it as `get_cache_counts` in pybind. It adds no writes, resets,
topology traversal, synchronization, or calls from the dispatch path. The
minimal measurement-only change, rebuilt module, source diff, and hashes are frozen
and disclosed; a source audit and CPU unit test must prove repeated reads leave
both totals unchanged. The mutating Archer `get_hit_rate()` and
`GetNodeVisitCounts()` are forbidden.

The mandatory `moe_infinity.entrypoints.openai.revision_server` module sets
Torch intra-op threads to eight, imports the official v2 module, and registers
only `/revision/stats` on its `app`. Its `main` executes the official entry
sequence exactly: call `parse_args`; set `_max_waiting_requests` and `_max_n`;
call `_configure_auth`; set the three ContextPilot globals while holding
`_contextpilot_state_lock`; assign `_startup_args`; then call `uvicorn.run` on
the official `app`, host, port, info log level, and `TIMEOUT_KEEP_ALIVE`. The
wrapper source, this copied entry sequence, and the import graph are
hash-frozen. The JSON schema has monotonic totals
`engine_generated_tokens`, `engine_steps`, `expert_cache_accesses`,
`expert_cache_hits`, `expert_cache_misses` (accesses minus hits), and
`exposed_fetch_seconds_total`; the only gauge is `kv_cache_num_blocks`. The
wrapper reads cache totals directly from
`ContinuousBatchingEngine.engine.expert_dispatcher.get_cache_counts()`. It is
forbidden to call `clear_expert_cache_counts()`, old `get_hit_rate()`,
`num_offloaded_experts`, or `is_tensor_offloaded()`: all but the new getter
either reset or mutate native state. Totals are sampled after warm-up and after
all eight measured requests and must be non-decreasing. The KV gauge is never
differenced or required to be monotonic; it is checked only for integer type
and equality to 128. `kv_cache_ratio=0` is not called “no KV”: upstream falls
back to those 128 KV blocks (about 2,048-token capacity).

## 4. Frozen workload and tokenizer equivalence

Source data is
`workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json`, SHA-256
`35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`.
Nine source rows (one distinct warm-up and eight measured prompts) are selected
before execution by `np.random.default_rng(1797).permutation(94145)` with no
performance observation; the first nine eligible rows occur after 138 scanned
candidates. For each candidate, `source_text` is the raw UTF-8 conversation
text with no Unicode normalization. The HF tokenizer runs with
`add_special_tokens=false`; candidates shorter than 512 IDs are skipped and the
first 512 IDs become `prompt_token_ids`. Thus the canonical request contains no
automatically inserted BOS/EOS. `prompt_text` is the HF decoding of those IDs
with `skip_special_tokens=false` and `clean_up_tokenization_spaces=false`.

A candidate is accepted only if both the pinned HF tokenizer and the pinned
GGUF tokenizer (`llama-tokenize --no-bos --no-parse-special`) encode
`prompt_text` back to the identical 512 IDs. An offline GGUF metadata check
also requires identical ID-to-token pieces for every ID present in the nine
prompts and matching BOS/EOS IDs; llama's `/detokenize` endpoint must reproduce
`prompt_text` during correctness preflight. The artifact retains source
row/index, `source_text`, `prompt_text`, IDs, all byte/ID hashes, tokenizer
commands/options, skipped candidates and reasons. Actual requests pass
`prompt_token_ids` directly; no server re-tokenizes request text. The selection
script, tokenizer/config files, prompt artifact, and fixed seed are hash-bound
to every attempt.

The exact common `/v1/completions` keys are: `model="gpt-oss-120b"`;
`prompt` equal to the numeric 512-element `prompt_token_ids` array;
`max_tokens=64`; `temperature=0.0`; `top_p=1.0`; `stop=[]`; and boolean
`stream=false` for smoke or `stream=true` for measurement. llama additionally
receives `cache_prompt=false` and `return_tokens=true`;
MoE-Infinity receives `n=1`, `best_of=1`, and `echo=false`. No seed parameter
exists in MoE-Infinity's completion schema; greedy decoding and golden-output
checks enforce determinism. Requests are sequential. Both servers have prefix
caching and speculative decoding disabled. A precomputed seed fixes eight
four-configuration block orders and the eight prompt orders within each
configuration. The schedule uses `np.random.default_rng(1798)` and is frozen in
`schedule.json`; prompts, schedule, bootstrap indices, and their input hashes
are bound by `workload-manifest.json` before preflight.

## 5. Correctness smoke

Each configuration starts in isolation. The distinct fixed warm-up prompt is
sent once with the non-streaming payload and is excluded. The eight measured
prompts are then sent non-streaming in two complete passes. The smoke records
raw JSON, usage, finish reason, UTF-8 output bytes, and output hashes. Both
passes must report 512 prompt tokens, 64 completion tokens,
`finish_reason=length`, valid UTF-8, and identical output bytes per prompt; the
common result becomes that configuration's smoke golden. The three llama
configurations must have byte-identical goldens. Cross-runtime equality is
reported but not required because kernels and serialization paths differ.

The smoke also rejects CUDA/OOM/fallback/model-load/store/policy errors. Its
own post-warm-up snapshots span 16 non-streaming requests and therefore require
a MoE generated-token delta of exactly 1,024, positive engine-step and expert-
cache-access deltas, internally consistent hit/miss totals, positive direct-I/O
read bytes, and the same static gates as section 7. The timed-block value 512
does not apply to smoke. Equivalent positive hook/eviction engagement is
required for the combined gpubpf policy across the 16 smoke requests. It never
relies on MoE streaming token IDs or streaming usage, which that server does
not provide.

## 6. Measured event semantics

Each configuration within each block has this immutable lifecycle: admission;
optional owned policy attach; fresh server start and health/config gate; one
excluded request using the distinct fixed warm-up; warm-up completion and idle
barrier; counter/metrics/I/O snapshots; eight sequential measured streaming
requests in the frozen prompt order; final snapshots; owned shutdown and
cleanup. Server or policy state is never reused across configurations or
blocks.

For every streamed request, the client records `CLOCK_MONOTONIC_RAW` at request
start, receipt of every raw SSE frame, first frame containing non-empty visible
output text, terminal frame with `finish_reason=length`, `[DONE]`, and stream
EOF. Raw SSE bytes are retained. Decoded `choices[0].text` fragments are
concatenated in order; their UTF-8 bytes and hash must exactly match that
configuration/prompt's smoke golden. These are client-visible output-event
times, not claimed model-token timestamps.

Immediately before the first measured request and after the eighth stream EOF,
MoE `/metrics` and `/revision/stats` are sampled. The block is valid only when
their generated-token delta is exactly 512 and all eight requests terminate
with `finish_reason=length`; therefore each request produced its maximum 64
tokens. llama's returned token/usage accounting must report 64 for each request
and 512 in aggregate.

Per request, TTFT is first non-empty visible-output event minus request start,
and end-to-end latency is stream EOF minus request start. For each
configuration/block, measured duration starts immediately before sending the
first request and ends at the eighth stream EOF. Aggregate output-token
throughput is the 512 verified output tokens divided by that duration. This is
the definition in vLLM `vllm/benchmarks/serve.py` commit
`3ec7b051563670b3af9cf5c10bc8ba3295ec125f`, file SHA-256
`5dcfbc9cb735450d9399cc65d7a7fecad8e9b841c5f7ea0fad90f0eb0b768d97`.
The experiment does not report goodput, TPOT, or “decode throughput.”

Server/model loading, expert-store construction, attachment, warm-up, cooldown,
and shutdown are excluded. The raw evidence includes commands, controlled
environment, logs, SSE, counters, process-tree CPU, GPU clocks/power/memory and
throttle state, and per-process NVMe I/O.

## 7. Engagement gates

### Combined gpubpf policy

The combined object exports monotonic counters for page-fault hook calls,
stride detections, prefetches issued, LFU activations, LFU accesses, and
eviction-prepare calls. Actual completed UVM evictions are counted separately
from the driver's UVM Tools `UvmEventTypeEviction` event stream, not inferred
from `gpu_evict_prepare`. The excluded warm-up establishes a counter snapshot;
measured deltas require page-fault calls > 0, at least one stride detection and
prefetch, LFU activation/access > 0, eviction-prepare > 0, and completed UVM
evictions > 0. If the fixed workload causes no eviction, this configuration is
technically invalid and the experiment is reported inconclusive rather than
weakening the gate after seeing performance.

### MoE-Infinity

The endpoint returns official engine generated-token and step totals, the new
side-effect-free dispatcher access/hit totals, their derived miss total,
exposed-fetch seconds, and the KV-block gauge. Across the measured block,
engine steps and generated tokens must increase, generated tokens must equal
512, cache accesses must increase, hits and misses must remain bounded by
accesses, and at least one hit or miss must occur. Positive dispatcher accesses
together with a non-empty expert store and positive direct-I/O read bytes prove
expert-offload engagement without querying mutable native residency state. The
offload directory must remain under the admitted filesystem.

Direct I/O is established once in isolated preflight by tracing the official
expert-cache file opens and requiring `O_DIRECT`; tracing is disabled for timed
runs. Every measured block additionally requires positive `/proc/<pid>/io`
`read_bytes` delta attributable to the owned MoE process tree. The complete
process tree remains in cpuset `0-7`. These gates prove that expert offload was
used; they do not turn `/proc` byte counts into a cross-runtime traffic claim.

## 8. Exclusive lifecycle and failure safety

Admission acquires an exclusive filesystem lease for GPU 0 and a separate
struct_ops lease. It inventories GPU compute PIDs, GPU memory, open device
handles, existing `gpu_mem_ops` registrations/links, driver version, module
hashes, mount identity/free space, thermal state, and all artifact hashes.
Admission rejects any pre-existing GPU process, more than 256 MiB residual GPU
memory, or any pre-existing/ambiguous struct_ops registration. It never clears,
detaches, signals, or adopts unknown state.

For `gpubpf_host_stride_lfu`, the runner launches exactly one hash-bound loader,
records its PID plus link/map/program IDs, and confirms those IDs are the sole
registration before starting the owned server. Warm-up counters are snapshotted
and measured deltas retained. Shutdown order is: stop and reap the owned server
tree; send SIGINT only to the exact owned loader PID; wait for its recorded link
ID to disappear; verify no residual registration or owned GPU process; release
the leases. Every exception/signal path executes the same ID-checked `finally`
sequence. Generic `pkill`, cleanup helpers that remove “old” struct_ops, and
unregister-by-name are forbidden.

Between configurations, the runner waits for zero owned or unowned compute
processes, <=256 MiB residual memory, successful struct_ops inventory, then a
fixed 60-second idle/cooldown. Any throttling invalidates the complete block.

## 9. Repetitions, stopping, and analysis

There are at most eight attempted complete blocks. Attempts follow the frozen
schedule in order and stop immediately after the fifth valid complete block is
atomically finalized; attempts six through eight are used only as replacements
for earlier technically invalid blocks. No sixth valid block is collected or
discarded. A block is valid only when all four configurations pass
admission, correctness, protocol, deterministic-output, and engagement gates.
Invalid and partial attempts are retained atomically; retry decisions inspect
only technical gates, never performance values.

If fewer than five valid complete blocks remain after eight attempts, the
automatic outcome is `inconclusive`: all failures and partial measurements are
reported, but no directional interval claim is made. With five valid blocks,
the primary statistic is one aggregate output-token throughput value per
configuration per block. Secondary per-block statistics are median TTFT,
descriptive TTFT P95 and maximum, median and maximum end-to-end latency, peak
GPU memory, expert-store size, process-tree CPU time, and engagement counters.
Prompt-level samples are not treated as independent replicates.

With exactly five valid blocks, let `r_b` be gpubpf aggregate output-token
throughput divided by MoE-Infinity throughput in paired block `b`. The point
estimator is `exp(mean(log(r_b)))`, the geometric mean of all five paired
ratios. Bootstrap indices are generated exactly once with
`np.random.default_rng(1797).integers(0, 5, size=(10000, 5), endpoint=False,
dtype=np.int64)`, saved as a hash-bound `.npy` artifact, and reused for every
comparison. Each row samples five complete block pairs with replacement and
recomputes that exact geometric-mean estimator. The 95% interval is the 2.5th
and 97.5th quantiles using NumPy's `quantile(method="linear")`. Baseline/MoE
ratios use the same estimator and index matrix. Secondary TTFT uses, per block,
the median of the eight paired
per-prompt TTFT differences; its across-block point estimate is the arithmetic
mean of the five block medians with a separately labeled bootstrap interval
using the same block resamples and quantiles. No prompt is independently
resampled. P95 remains descriptive because eight prompts cannot support a
precise tail estimate.

The gpubpf/MoE interpretation uses the preregistered 1.0 threshold and is
mutually exclusive:

- `higher output-token throughput`: paired-ratio lower bound > 1;
- `lower output-token throughput`: paired-ratio upper bound < 1;
- `no resolved difference`: otherwise.

TTFT is a separately reported trade-off. Existing paper measurements are
context only and are not pooled with this experiment.

## 10. Gates and current blocker

After independent approval, implementation proceeds through CPU-only unit
tests, a no-launch admission test, isolated GPU correctness/engagement
preflight, and only then the full schedule. Resume markers bind plan, runner,
commands/environments, artifact manifest, prompt/schedule, counters, and raw
evidence schema.

The host is currently not admitted: driver 610.43.02 differs from the required
gpubpf/NVBit driver 575.57.08, and two unrelated SGLang processes own about 31
GiB of GPU memory. They are outside this task's authority. Offline construction
and tests may continue, but no GPU launch is valid until both conditions clear.

Offline implementation status (2026-08-31): the frozen workload, observational
MoE instrumentation, combined one-object policy, ownership-safe loader, UVM
Tools V2 completed-eviction monitor, exact command/environment manifest, and
fail-closed no-launch admission are implemented. The combined policy and both
userspace probes compile cleanly; 26 CPU-only tests pass. A full content audit
passed for all 15 HF shards, seven HF metadata files, and the public GGUF.
Admission records only the three external blockers above; artifact, source,
storage, port, and empty-struct_ops gates pass. GPU correctness preflight and
timed attempts remain prohibited until admission succeeds.
The required four custom 575.57.08 modules are already built for the running
6.15.11-061511-generic kernel; their exact hashes and vermagic are admission
gates. They have not been loaded or substituted while foreign GPU processes
remain active.
