# LMCache build and compatibility smoke

Date: 2026-08-31

## Primary current triplet

- LMCache stable `v0.5.4`, official source commit
  `3e11b8ed191631e6f098b8038235823f1a410b24`.
- official vLLM `0.27.1+cu129` distribution from
  `https://wheels.vllm.ai/0.27.1/cu129`.
- official vLLM wheel SHA-256:
  `bf0d52faa2a51e7a01c6856a7a8a2d1307fd0ff711415d34168a67ffac0fa47b`.
- Python 3.12.3, torch `2.13.0+cu129`, CUDA runtime/toolkit 12.9.
- transformers 5.16.1, huggingface-hub 1.29.0, numpy 2.2.6.
- full resolved environment: `current-requirements.txt`, SHA-256
  `aad2fbf3e7ae84487e206d68888b5b290ceb4c3b6de221ef07542f9a0f1e9d9b`.
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

The build succeeded.  The CPython 3.12 wheel SHA-256 is
`9429740adfd73a554ac4bf1e46b169fadbdbbfe8a10bf5acc62522fcbae02fb5`.
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

The exact imported paths and SHA-256 values are frozen in
`artifacts-current.json` and rechecked by admission.  `vllm serve
--help=kv-transfer-config` also succeeds in this environment.  No model server,
CUDA kernel, or GPU benchmark was launched.

## Historical bridge artifact

The submitted LMCache logs identify source
`58cae13baee5b82163a6f279bac1786a75e88bc6` (`0.3.11.dev18`) with the paper's
vLLM fork `3ec7b051563670b3af9cf5c10bc8ba3295ec125f`.  That source was separately
built using Python 3.10, torch 2.8.0, CUDA 12.9, and `sm_120`:

- historical wheel SHA-256:
  `72270d243d4e81e190ac04355f7023ba93f335879589eb177f9b5189b19fa7b0`;
- historical `c_ops` SHA-256:
  `e5920da2598e830fae9b663976656029ab96f448dcc53b569925e670c82aeeae`;
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
