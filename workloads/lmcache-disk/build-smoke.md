# LMCache build and compatibility smoke

Date: 2026-08-31

## Primary current triplet

- LMCache stable `v0.5.4`, official source commit
  `3e11b8ed191631e6f098b8038235823f1a410b24`.
- official vLLM `0.27.1+cu129` distribution from
  `https://wheels.vllm.ai/0.27.1/cu129`.
- official vLLM wheel:
  `dist-vllm/vllm-0.27.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl`.
- Python 3.12.3, torch `2.13.0+cu129`, CUDA runtime/toolkit 12.9.
- transformers 5.16.1, huggingface-hub 1.29.0, numpy 2.2.6.
- full resolved environment: `current-requirements.txt`.
- `uv pip check`: all 223 installed packages are compatible.

vLLM 0.28.0 is newer, but its default NVIDIA distribution moved to CUDA 13;
the paper's 575-series driver/consumer RTX 5090 environment requires the
public CUDA-12.9 variant.  vLLM 0.27.1 is the newest frozen release with an
official `cu129` wheel and is paired here with current stable LMCache rather
than with the historical paper environment.

LMCache was built from source with:

```text
CUDA_HOME=/usr/local/cuda-12.9
TORCH_CUDA_ARCH_LIST=12.0
ENABLE_CXX11_ABI=1
CXX=/usr/bin/g++
CC=/usr/bin/gcc
MAX_JOBS=4
LMCACHE_CUDA_MAJOR=12
```

The build succeeded. The CPython 3.12 wheel is
`dist-current-py312/lmcache-0.5.4-cp312-cp312-linux_x86_64.whl`.
`cuobjdump --list-elf` reports seven `sm_120` cubins in `lmcache.cuda_ops`.

With `CUDA_VISIBLE_DEVICES` empty, the following imports succeeded without a
GPU allocation against vLLM `0.27.1+cu129`:

- `lmcache` (`0.5.4`)
- `lmcache.lmcache_native`
- `lmcache.cuda_ops`
- `lmcache.integration.vllm.vllm_v1_adapter`
- `lmcache.integration.vllm.lmcache_connector_v1`
- `lmcache.v1.storage_backend.local_disk_backend`
- `vllm`

The exact imported paths are frozen in `artifacts-current.json` and rechecked
by admission. `vllm serve
--help=kv-transfer-config` also succeeds in this environment.  No model server,
CUDA kernel, or GPU benchmark was launched.

## Historical bridge artifact

The submitted LMCache logs identify source
`58cae13baee5b82163a6f279bac1786a75e88bc6` (`0.3.11.dev18`) with the paper's
vLLM fork `3ec7b051563670b3af9cf5c10bc8ba3295ec125f`.  That source was separately
built using Python 3.10, torch 2.8.0, CUDA 12.9, and `sm_120`:

- historical wheel and `c_ops` files retained under the historical build
  directories;
- five `sm_120` cubins; CPU-only connector import passed.

It is retained for provenance/sensitivity only and is not the primary
paper-facing storage-tier experiment.

## 610 runtime preflight — 2026-08-31

After the foreign GPU jobs exited, read-only admission passed on the uniform
610.43.02 stack. Attempt 1 (`raw/preflight-610-20260831-01`) stopped before
vLLM startup: strace could not open its relative output path after the child
changed to the vLLM workload directory. The original server log and manifest
are retained. No O_DIRECT or model-correctness result was obtained.

The repair resolves the trace directory before spawning the child; it does
not change the model, requests, I/O mode, or comparison. A regression test
checks the launch command with a relative output root and a different server
working directory. All seven CPU tests pass after the repair.

Attempt 2 (`raw/preflight-610-20260831-02`) then loaded all seven real Qwen
checkpoint shards. Engine initialization failed during DeepGEMM FP8 scale
conversion in `deep_gemm.py` / `fp8_utils.py`, ending at
`layout.hpp:60: Unknown SF transformation`. No cache request was served; the
server exited and its owned GPU memory was released.

Before attempt 3, every comparison cell disables DeepGEMM with the upstream
`VLLM_USE_DEEP_GEMM=0` setting and increases the common vLLM memory budget from
0.90 to 0.99. The capacity rationale and unchanged comparison are recorded in
`plan.md`. The repair uses source-native settings, not a patched vLLM fork;
eight CPU tests pass. See the official
[environment switch](https://docs.vllm.ai/en/latest/configuration/env_vars/)
and [memory-budget definition](https://docs.vllm.ai/en/stable/cli/bench/throughput/).

Attempt 3 (`raw/preflight-610-20260831-03`) failed before model loading. The
admission manifest had no compute applications and reported 15 MiB GPU memory
used. vLLM then measured 30.89/31.4 GiB free and rejected the 0.99 request for
31.08 GiB. Raising the budget to 0.99 was therefore an experiment-side startup
error. No request or cache I/O occurred, and this attempt did not test the
DeepGEMM-off fallback. The three-attempt real-preflight allowance is exhausted;
the protocol is closed without O_DIRECT, correctness, or performance evidence.

## Revision-2 offline qualification — 2026-08-31

Revision 2 uniformly lowers the vLLM startup budget from 0.99 to 0.98 and
replaces content-fingerprint gates with parsed semantic validation, exact
response-text comparison, and ordinary file metadata. `prompts.json` was
regenerated as schema 3 without fingerprints. Ten CPU-only structural tests
pass, including exact schedule semantics, prompt LCP validation, O_DIRECT
failure cases, request-scoped log parsing, uniform runtime settings, and a
source guard against reintroducing fingerprint logic. Read-only admission
passes on an idle RTX 5090 with 15 MiB reported used, the exact 610.43.02
driver, required model files, runtime imports, dependencies, and the workspace
NVMe mount. No revision-2 model server or CUDA workload has been launched.

Round-1 independent review blocked the custom control layer and another real
preflight. The active runner is now a thin one-cell adapter plus recomputable
analysis: it has no approval parser, promotion/pass markers, completion schema,
resume protocol, or attempt-budget controller. Prompt admission exactly
regenerates the pinned dataset/tokenizer derivation; the schedule consists of
five randomized Latin cycles; log parsing preserves store/retrieve denominators;
TTFT starts at the first SSE event carrying a generated token; analysis
re-parses logs/traces/usage, compares exact text, reports both rate metrics
against both baselines, and requires an interval upper bound below -5% before
calling a throughput regression. Fifteen CPU-only tests pass. The custom
control layer was removed, leaving only active low-level primitives behind the
thin adapter. The
three-attempt cap remains exhausted, so this is offline repair evidence only.
Final independent review passed all offline repairs and separately blocked a
GPU launch because no higher-level exception to that cap exists.

## Bounded local-disk dependency smoke — 2026-09-01

The user separately authorized a fast code-path check rather than another
paper experiment attempt. The adapter gained an explicit `--prefix-limit`;
its default remains eight. It still regenerates and validates all prompts
before selecting the leading subset. Per-run request, disk, optional trace,
and replay counts derive from `prefix_count`. All 18 offline tests pass.

The final single-prefix disk smoke used the unchanged 0.98 memory budget and
no trace. The exact seven-shard model loaded, the API became healthy, and both
requests returned HTTP 200 with 16 generated tokens. The cold request had
1,549 total tokens, zero hits, and stored all 1,536 cache-eligible tokens. It
created six fully allocated 24 MiB files (144 MiB total), synchronizing each
file and the directory. The warm request had 1,550 total tokens, hit and
retrieved all 1,536 cache-eligible tokens, and did not engage vLLM's native
prefix cache.

The configuration and runtime log confirm LocalDiskBackend engagement and
report O_DIRECT enabled. There was no syscall trace, so this does not prove
syscall-level O_DIRECT. Online validation exited successfully, offline
`validate-cell` passes, and cleanup left no compute process or port-18080
listener and only 15 MiB reported GPU memory.

This is dependency and code-path evidence only. Recorded timings and rates
are not results. The attempt cap remains exhausted; no recompute, CPU,
repeated, or gpubpf comparison cell was run.
