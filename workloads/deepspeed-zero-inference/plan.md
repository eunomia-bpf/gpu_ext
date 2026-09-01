# Experiment Plan: RQ1 DeepSpeed ZeRO-Inference fallback

Status: **closed before execution after plan-review round 2**. The pinned
Transformers MXFP4 path removes the dominant expert projections from
`nn.Parameter` registration and replaces them with Triton tensor objects.
Official ZeRO-3 parameter offload therefore cannot manage those expert weights,
while custom offload glue would no longer be the DeepSpeed baseline. No GPU
preflight, correctness sample, or performance sample was run. See
`availability.md` and `plan-review.md`.

## Research Question

- RQ exactly as written in the paper: **RQ1 (Single-Tenant Management): How
  much performance gain can gpubpf's programmable memory and scheduling
  policies provide on oversubscribed single-tenant workloads?**
- Specific uncertainty tested here: whether the official DeepSpeed
  ZeRO-Inference path can execute the exact public GPT-OSS-120B MXFP4 model on
  one RTX 5090 by streaming parameters from CPU or NVMe, and, if it can,
  whether its matched request throughput differs from gpubpf's existing
  page-granular host policy and llama.cpp framework offload.
- Why the answer matters: Author Q1 promises runnable SOTA research baselines.
  MoE-Infinity exhausted its approved preflight budget without a valid sample.
  PowerInfer has no GPT-OSS model path. DeepSpeed is the only named fallback
  that may preserve the same model and workload instead of silently changing
  the question.

## Paper-Value Admission

- Planned role: **decisive** for whether the revision can include a runnable
  same-model MoE/model-weight-offload research baseline; **supporting** for RQ1
  performance because the currently runnable gpubpf policy is host-only and
  must not be mislabeled as the submitted device-observed policy.
- Largest credible paper story this experiment could unlock: an official
  ZeRO-Inference versus gpubpf versus framework-offload comparison on the same
  120B MXFP4 model, prompt tokens, output length, GPU, and host storage.
- Strongest reviewer reject argument or load-bearing uncertainty addressed:
  the current evaluation compares mostly against framework defaults and may
  omit a research system that already streams model weights from CPU/NVMe.
- Independent evidence added beyond existing runs and published results:
  DeepSpeed's published ZeRO-Inference evaluation uses dense OPT/BLOOM models
  and older datacenter GPUs; it does not establish GPT-OSS MXFP4 compatibility
  or performance on consumer Blackwell. MoE-Infinity never produced a valid
  request on this workload.
- Why the result is not tautological, already settled, or dominated:
  DeepSpeed may either accept the quantized parameter representation and
  stream it correctly or fail at the quantizer/ZeRO boundary. Both outcomes
  change the revision decision. A citation cannot answer that executable
  compatibility question.
- Paper decision if positive: run the complete matched comparison and report
  the official research baseline even if it matches or beats gpubpf.
- Paper decision if contradictory, mixed, or inconclusive: retain the exact
  failure and name DeepSpeed and PowerInfer with concrete unavailability
  reasons, as permitted by Author Q1; do not substitute Mixtral/Qwen or weaken
  GPT-OSS-120B after observing failure.
- Best alternative experiment and why this one has higher decision value:
  an Orion/XSched scheduling repair is also required, but its live path needs
  a display maintenance window. DeepSpeed can use the currently idle 5090
  without replacing display-owned core modules and directly repairs the failed
  MoE baseline commitment.

## Expected And Alternative Outcomes

- Current expected answer: DeepSpeed 0.19.5 will install on the current cu129
  Torch environment, but ZeRO-3 parameter offload may reject or materialize
  GPT-OSS's MXFP4 custom parameters rather than stream them layer-by-layer.
- Strongest competing explanation: recent Transformers and DeepSpeed releases
  may interoperate generically, in which case the exact model should complete
  and provide a valid external baseline.
- Result that would contradict the expectation: two deterministic 512-input,
  64-output GPT-OSS-120B requests complete through verified ZeRO-3 CPU offload
  without fallback, full-weight GPU materialization, or OOM.

## Published Precedent And Real Assets

- Closest published protocol: DeepSpeed's official
  [ZeRO-Inference description](https://www.deepspeed.ai/2022/09/09/zero-inference.html),
  which uses ZeRO stage 3 `offload_param` to stream weights from CPU or NVMe
  and evaluates end-to-end token generation.
- Official system/model/data/benchmark/tool and version:
  Python 3.12.3, DeepSpeed 0.19.5, Torch 2.13.0+cu129, Transformers 5.16.1,
  Triton 3.7.1, `kernels` 0.16.0, Accelerate 1.14.0, Safetensors 0.8.0, and
  Tokenizers 0.23.1 in a new isolated environment; official
  `openai/gpt-oss-120b` revision already retained as the 15-shard MXFP4
  snapshot; the existing nine frozen ShareGPT prompts; RTX 5090 with driver
  610.43.02; workspace Samsung 9100 PRO NVMe.
- What is reused: the exact HF snapshot and tokenizer, prompt-token artifact,
  schedule logic, correctness rules, lifecycle boundaries, and output-token
  throughput definition from `workloads/moe-infinity/plan.md`.
- Necessary deviations or custom glue: one minimal Python generation adapter
  that follows the official `HfDeepSpeedConfig`/ZeRO-3 initialization path and
  emits ordinary JSONL request results; CPU and NVMe DeepSpeed JSON configs;
  no custom offload implementation or experiment-control framework.

## Comparison

- Proposed system or method: `gpubpf_host_stride_lfu`, explicitly retained as
  a host-only page-policy ablation and not the submitted full policy.
- Main baseline: official DeepSpeed ZeRO-Inference with CPU parameter offload.
  It represents application/runtime-level layer streaming with model semantic
  knowledge. A matched run is required because its paper does not cover
  GPT-OSS MXFP4 or RTX 5090.
- Second configuration of the same baseline: ZeRO-Inference NVMe parameter
  offload. It is included only if the official CPU path first completes; it
  tests the artifact's other published storage tier, not a separate baseline
  count.
- Control: llama.cpp `--n-cpu-moe 32`, representing current framework-managed
  expert offload on the exact GGUF derived from the same primary model
  revision.
- Control: llama.cpp plain UVM, needed to interpret the page policy but not
  counted as a research baseline.
- Conclusion if the main baseline matches or wins: gpubpf's safe/general
  mechanism does not imply a performance advantage over semantic layer
  streaming; report the result and narrow any policy-superiority claim.
- Information, tuning, and compute fairness: batch/concurrency one, identical
  512 prompt token IDs and 64 generated tokens, greedy decoding, eight requests
  per configuration per block, same P-core affinity, one GPU, and isolated
  server/model lifecycle. There is no performance-guided tuning. CPU candidate
  A uses `stage3_max_live_parameters=1000000000` and
  `stage3_max_reuse_distance=1000000000`; candidate B changes only those two
  values to `500000000` and is allowed solely after a candidate-A CUDA OOM.
  NVMe uses one fixed configuration. The passing choice is frozen before any
  measured block.

## Workloads And Metrics

- Real workload: the existing one warm-up plus eight measured 512-token
  ShareGPT prompts, each generating exactly 64 tokens with GPT-OSS-120B.
- Primary metric: aggregate verified output-token throughput, 512 completion
  tokens divided by the duration from first request start through eighth
  request completion, matching the already approved vLLM serving-benchmark
  boundary.
- Secondary metrics: per-request TTFT and end-to-end latency, peak GPU memory,
  process-tree CPU time, process-tree storage read bytes, and DeepSpeed offload
  activity.
- Correctness: two untimed greedy passes per prompt must each report 512 input
  and 64 output tokens, length termination, valid UTF-8, and byte-identical
  output within a configuration. DeepSpeed CPU and NVMe must agree byte-for-byte
  on every prompt because they use the same runtime and model representation.
  The three llama configurations must also agree byte-for-byte. Cross-runtime
  equality is reported but not required.
- Repetitions and uncertainty: five valid complete paired blocks from at most
  eight attempts. Report the geometric mean of per-block throughput ratios and
  a block bootstrap 95% interval using the existing fixed resample indices.
- Cost estimate: no additional model download. NVMe admission requires at least
  200 GiB free and reserves 120 GiB for the host; the owned offload directory is
  capped at 100 GiB. Reaching either limit aborts the attempt and is evidence of
  representation expansion, not permission to consume the remaining disk. At
  most three real preflight attempts occur before any five-block run.

## Planned Runs

| Run group | Role | Workload | System/method | Repetitions | Decision consequence |
|---|---|---|---|---:|---|
| preflight | baseline feasibility | one excluded 512+64 request, then two identical 512+64 requests | DeepSpeed ZeRO-3 CPU offload | allocated from 3 total attempts | continue only if exact model, correctness, and engagement pass |
| preflight | storage-tier feasibility | one excluded 512+64 request, then two identical 512+64 requests | DeepSpeed ZeRO-3 NVMe offload | allocated from the same 3 attempts | the five-cell run exists only if NVMe also passes |
| main | external baseline | eight 512+64 requests | DeepSpeed ZeRO-3 CPU offload | 5 valid blocks | required external comparison |
| main | baseline storage tier | eight 512+64 requests | DeepSpeed ZeRO-3 NVMe offload | 5 valid blocks | quantify official NVMe path if CPU path works |
| main | proposed ablation | eight 512+64 requests | gpubpf host stride+LFU | 5 valid blocks | page-policy comparison, not full-policy claim |
| control | current practice | eight 512+64 requests | llama.cpp CPU-MoE offload | 5 valid blocks | deployment context |
| control | mechanism context | eight 512+64 requests | llama.cpp plain UVM | 5 valid blocks | isolate host page-policy effect |

## Execution

- Authoritative workflow: an isolated Python 3.12.3 environment with the exact
  versions above. The adapter imports `HfDeepSpeedConfig`, constructs and keeps
  it alive **before** `AutoModelForCausalLM.from_pretrained`, then calls
  `deepspeed.initialize(model=model, config=ds_config)`, `engine.eval()`, and
  `engine.module.generate` under `torch.inference_mode()`. It passes
  `local_files_only=True` and `torch_dtype="auto"`; `device_map`, Accelerate
  dispatch/offload, manual tensor movement, and MXFP4 dequantization are
  forbidden. The frozen launcher shape is:

  ```text
  CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 <venv>/bin/deepspeed --num_gpus 1 \
    --master_port <owned-port> run_zero_inference.py \
    --model <exact-local-snapshot> --config <cpu-or-nvme.json> \
    --requests <frozen-request-json> --output <owned-jsonl>
  ```

  The common non-`auto` configuration is
  `train_batch_size=1`, `train_micro_batch_size_per_gpu=1`,
  `gradient_accumulation_steps=1`, `steps_per_print=2000`, and
  `zero_optimization.stage=3` with
  `stage3_param_persistence_threshold=0`,
  `stage3_max_live_parameters=1000000000`, and
  `stage3_max_reuse_distance=1000000000`. CPU adds exactly
  `offload_param={device:cpu,pin_memory:true}`. NVMe adds exactly
  `offload_param={device:nvme,nvme_path:<owned-attempt-dir>,pin_memory:true,
  buffer_count:5,buffer_size:100000000,max_in_cpu:0}` and
  `aio={block_size:262144,queue_depth:32,thread_count:1,
  single_submit:false,overlap_events:true}`. Candidate B, if admitted by the
  OOM rule above, changes only the two declared live/reuse values.
- Real preflight allocation is deterministic. Attempt 1 is CPU candidate A. If
  it passes, attempt 2 is fixed NVMe and attempt 3 is available only for one
  diagnosed NVMe repair. If CPU attempt 1 fails, attempt 2 is its sole retry
  after one diagnosed repair; candidate B is used only when that diagnosis is
  CUDA OOM, otherwise the config remains A. If CPU first passes on attempt 2,
  attempt 3 is the fixed NVMe preflight with no retry. A second CPU failure or
  an unresolved NVMe failure after this allocation ends the axis with no
  timing; unused attempts are never spent on tuning.
- Every attempt has a 3,600-second model-start timeout, a 900-second timeout per
  request, and a 120-second owned shutdown timeout. Admission requires at least
  110 GiB `MemAvailable`; the process tree runs in an owned cgroup with
  `MemoryMax=118G`, `MemorySwapMax=0`, and kill-on-OOM. Any OOM, kernel kill,
  lower-than-8-GiB host-memory safety margin, directory growth above 100 GiB,
  or filesystem free space below 120 GiB invalidates and terminates the owned
  attempt.
- A DeepSpeed cell passes engagement only if the loaded config reports ZeRO
  stage 3 and the intended `offload_param.device`, every ordinary model
  parameter has ZeRO partition metadata and the intended final location, and
  read-only layer-entry/exit observations show positive transitions between
  unavailable-at-rest and materialized-for-forward states. The model must
  retain `model_type=gpt_oss`, `quant_method=mxfp4`, `dequantize=false`, and all
  36 `Mxfp4GptOssExperts` modules. Their packed expert weight objects must also
  be represented in the ZeRO-managed partition inventory. Missing expert
  coverage, Accelerate/device-map activity, BF16 expansion, non-ZeRO offload,
  or full-weight GPU materialization invalidates the cell. NVMe additionally
  requires owned offload files and positive process-tree storage reads during
  generation. Peak GPU allocation must stay below physical capacity without
  OOM or fallback.
- Real preflight loads the exact local 120B snapshot and completes one excluded
  512+64 request followed by two identical 512+64 requests on the first frozen
  prompt. Each tier must pass token counts, length termination, UTF-8,
  repeatability, and the engagement gates; after both tiers pass, their two
  outputs must be byte-identical. Before timing, the full nine-prompt two-pass
  smoke applies the same checks and establishes the cross-tier golden outputs.
- Full completion rule: five technically valid blocks containing every
  planned cell; if either ZeRO tier cannot pass within the shared three-attempt
  preflight budget, no timing is authorized and the result is an executable
  incompatibility report.
- Raw-result path: `workloads/deepspeed-zero-inference/raw/`.
- Checkpoint or recovery: preserve installation/build logs, server logs,
  request JSONL, DeepSpeed config/logs, process/GPU/storage observations, and
  every failed attempt. Only owned processes and temporary offload directories
  may be cleaned.

## Driver, Policy, And Block Order

- Every cell uses the same Open Kernel Modules 610.43.02 stack. At experiment
  admission, with zero GPU compute processes and `nvidia_uvm` reference count
  zero, unload only distribution `nvidia_uvm` and load the BTF-enabled custom
  UVM from `gpu_ext-kernel-610` commit `c4fd5655`. The matching distribution
  core, modeset, DRM, and display modules remain loaded. On final cleanup,
  detach only the owned struct_ops link, unload only the idle custom UVM, and
  restore distribution UVM. Any ambiguous ownership or nonzero UVM reference
  count aborts without mutation.
- The gpubpf cell uses the tracked combined stride-prefetch plus LFU object from
  the approved MoE protocol. Admission requires exactly one owned struct_ops
  link and the expected map/program IDs. Both smoke and each measured block
  require positive deltas for page-fault hook calls, stride detections,
  prefetches issued, LFU activations/accesses, eviction-prepare calls, and
  completed UVM Tools eviction events. A missing delta invalidates the entire
  paired block; the gate is not weakened after observing results.
- Five valid block slots use this frozen Latin-square order, where `DCPU`,
  `DNVME`, `GPUBPF`, `LLAMA`, and `UVM` name the five cells:

  ```text
  1: DCPU,  DNVME, GPUBPF, LLAMA, UVM
  2: DNVME, GPUBPF, LLAMA, UVM,    DCPU
  3: GPUBPF, LLAMA, UVM,   DCPU,   DNVME
  4: LLAMA, UVM,    DCPU,   DNVME, GPUBPF
  5: UVM,   DCPU,   DNVME,  GPUBPF, LLAMA
  ```

  The nine-prompt order within each cell is frozen from seed 1798 before any
  run. At most eight complete block attempts are allowed. A technically invalid
  attempt retries the same block slot and exact order; the slot advances only
  after a valid complete block, so the final five blocks remain position
  balanced. No sixth valid block is collected.
- The common request boundary begins immediately before the single-request
  adapter call (DeepSpeed `generate` entry or llama HTTP POST), TTFT is the
  first non-empty emitted output event, and completion is the adapter return or
  stream EOF after exactly 64 tokens. Aggregate duration begins at the first
  request call and ends at the eighth completion. Server/model initialization,
  policy attachment, warm-up, and smoke remain excluded for every cell.

## Interpretation

- Positive result: CPU ZeRO passes correctness and engagement and five paired
  blocks resolve the relative throughput interval.
- Negative or contradictory result: a valid DeepSpeed run matches or wins;
  report that outcome and remove any implied mechanism-caused superiority.
- Mixed or inconclusive result: exact-model incompatibility, fallback,
  nondeterminism, OOM, or fewer than five valid blocks prevents a performance
  conclusion. Preserve it as the named artifact's availability boundary.
- Target paper figure or table: one RQ1 MoE/model-weight-offload comparison
  panel, with host-only gpubpf clearly labeled; otherwise an artifact
  availability row and no performance bar.

## Reproducibility Notes

- Software and data versions: pin DeepSpeed 0.19.5, the installed compatible
  Torch/Transformers versions, exact upstream revisions where source checkout
  is needed, and the existing GPT-OSS/HF/GGUF Git revisions.
- Config and seed notes: reuse prompt seed 1797, schedule seed 1798, greedy
  decoding, and existing block-resample indices.
- Known deviations: current driver is 610.43.02 rather than the original 575
  environment; gpubpf is host-only until the deferred device-side migration
  lifetime interface is redesigned; DeepSpeed and llama.cpp use different
  runtimes and model serializations derived from the same primary revision.
- Never generate, refresh, compare, or record file/content hashes, checksums,
  digests, or fingerprints. Use explicit versions, file inventories and sizes,
  semantic configuration checks, build/tests, and real engagement evidence.
  Git commit IDs and upstream revisions remain ordinary version bookkeeping.
